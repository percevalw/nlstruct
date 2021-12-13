import torch
import torch.nn.functional as F

from nlstruct.models.crf import BIOULDecoder
from nlstruct.models.ner import SpanScorer
from nlstruct.registry import *
from nlstruct.torch_utils import *


def compute_overlaps(begins1, ends1, begins2, ends2, end_included=True):
    if end_included:
        return ~((ends1[:, :, None] < begins2[:, None, :]) | (ends2[:, None, :] < begins1[:, :, None]))
    else:
        return ~((ends1[:, :, None] <= begins2[:, None, :]) | (ends2[:, None, :] <= begins1[:, :, None]))


def spans_to_tags(begins, ends, labels, mask, words_mask, aggregate_entities=True):
    # O - I1 B1 L1 U1  - I2 B2 L2 U2 - ...
    # begins: n_samples * n_entities
    positions = torch.arange(words_mask.shape[1], device=begins.device)  # _ * _ * n_words
    begins = begins[..., None]
    ends = ends[..., None]
    labels = labels[..., None]
    tags = torch.zeros_like(words_mask, dtype=torch.long)
    tags = (
          (((positions >= begins) & (positions <= ends)) * (4 * labels + 1)) +
          (positions == begins) * 1 +  # + 1 if begin
          (positions == ends) * 2  # + 2 if end
        # (if begin and end => 1+2 = +3 = U)
    )  # n_samples * n_entities * n_words
    tags = tags.masked_fill(~mask[..., None], 0)
    if aggregate_entities:
        tags = tags.max(-2).values if tags.shape[-2] else tags.sum(-2)
    return tags


def tags_to_spans(tag, mask=None, n_labels=1, return_flat=False):
    I, B, L, U = 0, 1, 2, 3

    device = tag.device

    if mask is not None:
        tag = tag.masked_fill(~mask, 0)

    unstrided_tags = (
          ((tag - 1) % 4).masked_fill(tag == 0, -1)[..., None, :]
          * (((tag - 1).masked_fill(tag == 0, -4)[..., None, :] // 4) == torch.arange(n_labels, device=device)[:, None]).long()
    )

    cs_not_I = (unstrided_tags != 0).long().cumsum(-1)
    has_no_hole = (cs_not_I.unsqueeze(-1) - cs_not_I.unsqueeze(-2)) == -1
    prediction = multi_dim_triu(
        ((unstrided_tags == U).unsqueeze(-1) &
         (unstrided_tags == U).unsqueeze(-2)).masked_fill(~torch.eye(unstrided_tags.shape[-1], device=device).bool(), False) |
        ((unstrided_tags == B).unsqueeze(-1) &
         (unstrided_tags == L).unsqueeze(-2) &
         has_no_hole),
    ).transpose(-1, -3).transpose(-2, -3)

    if mask is not None:
        prediction = prediction & mask[:, :, None, None] & mask[:, None, :, None]
    if return_flat:
        return multi_dim_nonzero(prediction, dim=-3)
    return prediction


@register("autoregressive")
class AutoregressiveSpanScorer(SpanScorer):
    def __init__(self,
                 input_size,
                 n_labels,
                 hidden_size=None,
                 contextualizer=None,
                 dropout_p=0.1,

                 encoding_size=None,
                 encoding_scheme="bioul",
                 decoding_scheme="bioul",
                 max_fragments_count=200,
                 initial_observed_p=0.1,

                 mode="short-to-large",

                 learnable_transitions=False,
                 max_iter=20):
        super().__init__()

        self.input_size = input_size
        self.n_labels = n_labels
        self.encoding_size = encoding_size
        self.dropout = torch.nn.Dropout(dropout_p)

        if encoding_size is None:
            lstm_input_size = input_size
            encoding_size = input_size
        else:
            lstm_input_size = input_size + encoding_size
        if hidden_size is None:
            hidden_size = lstm_input_size
        self.contextualizer = get_instance({**contextualizer, "input_size": lstm_input_size, "hidden_size": hidden_size})
        self.max_iter = max_iter
        self.max_fragments_count = max_fragments_count
        self.mode = mode
        self.initial_observed_p = initial_observed_p

        self.crf = {"bioul": BIOULDecoder}[decoding_scheme](num_labels=n_labels, with_start_end_transitions=True, allow_overlap=False, learnable_transitions=learnable_transitions)
        self.encoder_cls = {"bioul": BIOULDecoder}[encoding_scheme]

        n_tags = self.crf.transitions.shape[0]

        self.tag_proj = torch.nn.Linear(hidden_size, n_tags)
        self.tag_embedding = torch.nn.Embedding(n_tags - 1, encoding_size)

    def fast_params(self):
        return []

    def forward(self, words_embed, words_mask, batch, force_gold=False):
        n_samples, n_words = words_mask.shape
        device = words_embed.device
        words_embed = self.dropout(words_embed[-1])

        start_positions = torch.arange(n_words, device=device)[None, :, None]
        end_positions = torch.arange(n_words, device=device)[None, None, :]
        spans_lengths = (end_positions + 1 - start_positions).repeat_interleave(words_mask.shape[0], dim=0)
        spans_mask = (
              words_mask.unsqueeze(-1)
              & words_mask.unsqueeze(-2)
              & (0 < spans_lengths)
        )

        prediction = False

        loss_data = []
        if force_gold:
            remaining_mask = batch["fragments_mask"]
            initially_observed_mask = (torch.rand_like(remaining_mask, dtype=torch.float) < (self.initial_observed_p if self.training else 0.)) & remaining_mask
            gold_spans_label = batch["fragments_label"]
            gold_spans_begin = batch["fragments_begin"]
            gold_spans_end = batch["fragments_end"]
            gold_overlaps = compute_overlaps(
                batch["fragments_begin"], batch["fragments_end"],
                batch["fragments_begin"], batch["fragments_end"],
            )
            remaining_mask = remaining_mask & ~initially_observed_mask

            initially_observed_tags = spans_to_tags(
                gold_spans_begin,
                gold_spans_end,
                gold_spans_label,
                initially_observed_mask,
                words_mask,
            )
            observed_tags_embeds = self.tag_embedding((initially_observed_tags - 1).clamp_min(0)).masked_fill((initially_observed_tags == 0)[..., None], 0)
        else:
            observed_tags_embeds = torch.zeros(*words_embed.shape[:-1], self.tag_embedding.weight.shape[-1], dtype=torch.float, device=device)

        all_predictions_mask = torch.zeros_like(spans_mask).unsqueeze(-1).repeat_interleave(self.n_labels, dim=-1)
        sample_indices = torch.arange(n_samples, device=device)

        non_empty = torch.ones(n_samples, dtype=torch.bool, device=device)
        for step_i in range(self.max_iter):
            # Run the lstm + crf => viterbi prediction + tag log probabilities
            context_word_embeds = self.contextualizer(words_embed + observed_tags_embeds if self.encoding_size is None else torch.cat([words_embed, observed_tags_embeds], dim=-1), words_mask)
            emitted_tag_logits = self.tag_proj(context_word_embeds)  # n_samples * n_words * n_tags

            # Score and select the most likely gold predictions
            if force_gold:
                # crf_tag_logprobs: n_samples * n_words * n_tags
                crf_tag_logprobs = self.crf.marginal(emitted_tag_logits, words_mask)
                # prediction_tags = self.crf.tags_to_spans(predicted_tags, words_mask, return_flat=True)

                # self.tag_to_label_mapper: n_tags * n_labels * n_tags
                # label_tag_logprobs: n_samples * n_labels * n_words * n_tags
                # how do we do it: O I-x B-x L-x U-x I-y B-y L-y U-y => [I-x B-x L-x U-x] [I-y B-y L-y U-y]
                label_tag_logprobs = crf_tag_logprobs[..., 1:]
                label_tag_logprobs = label_tag_logprobs.view(*label_tag_logprobs.shape[:-1], self.n_labels, -1).permute(0, 2, 1, 3)

                is_not_empty_logprobs = label_tag_logprobs.logsumexp(-1)
                is_not_empty_cs_logprobs = torch.cat([
                    torch.zeros_like(is_not_empty_logprobs[..., :1]),
                    is_not_empty_logprobs,
                ], dim=-1).cumsum(-1)
                has_no_hole_inside = (multi_dim_triu(is_not_empty_cs_logprobs[..., None, :-1] - is_not_empty_cs_logprobs[..., 1:, None], diagonal=1))  # .clamp_max(-1e-14)

                if self.mode == "prob":
                    span_logprobs = torch.add(torch.add(
                        dclamp(has_no_hole_inside, max=-1e-14),
                        label_tag_logprobs[..., :, None, [1, 3]].logsumexp(-1),
                    ), label_tag_logprobs[..., None, :, [2, 3]].logsumexp(-1))
                    span_scores = inv_logsigmoid(span_logprobs)  # n_samples * n_words * n_words
                    gold_scores = span_scores[
                        torch.arange(len(sample_indices), device=device)[:, None],
                        gold_spans_label,
                        gold_spans_begin,
                        gold_spans_end]  # .view(n_samples, -1)
                elif self.mode == "short-to-large":
                    gold_scores = -(gold_spans_end - gold_spans_begin)
                elif self.mode == "large-to-short":
                    gold_scores = (gold_spans_end - gold_spans_begin)

                # gold_scores = (gold_spans_end - gold_spans_begin)

                gold_mask = remaining_mask.clone()
                selected_mask = torch.zeros_like(remaining_mask)
                while gold_mask.any():
                    any_remaining = gold_mask.any(-1)
                    best_gold_indices = gold_scores.masked_fill(~gold_mask, -10000).argmax(-1)
                    # gold_overlaps: n_samples * n_entities * n_entities
                    # gold_overlaps[sample_indices[:, None], best_gold_indices[:, None] : n_samples * n_entities
                    selected_overlaps = gold_overlaps[torch.arange(len(sample_indices), device=device), best_gold_indices]
                    selected_overlaps = selected_overlaps.masked_fill(~any_remaining[:, None], False)
                    gold_mask[selected_overlaps] = False
                    selected_mask[torch.arange(len(sample_indices), device=device), best_gold_indices] |= any_remaining
                remaining_mask = remaining_mask & ~selected_mask

                predicted_tags = spans_to_tags(
                    gold_spans_begin,
                    gold_spans_end,
                    gold_spans_label,
                    selected_mask,
                    words_mask,
                )

                loss_data.append((emitted_tag_logits, words_mask, predicted_tags))
                flat_mask = selected_mask
            else:
                predicted_tags = self.crf.decode(emitted_tag_logits, words_mask)
                # print("pred", predicted_tags)
                prediction_mask = tags_to_spans(predicted_tags, words_mask, n_labels=self.n_labels)
                (flat_begins, flat_ends, flat_labels), flat_mask = multi_dim_nonzero(prediction_mask, dim=1)
                # print("=>", flat_begins, flat_ends, flat_labels)

                # Store the predictions made at this step
                all_predictions_mask[sample_indices, :prediction_mask.shape[1], :prediction_mask.shape[2], :] |= prediction_mask

            # Embed the predictions for the next step
            observed_tags_embeds = observed_tags_embeds + self.tag_embedding((predicted_tags - 1).clamp_min(0)).masked_fill((predicted_tags == 0)[..., None], 0)

            # Slice the tensors to only keep non empty sentences
            non_empty = flat_mask.any(-1)

            if not non_empty.any():
                break

            sample_indices, words_embed, words_mask, observed_tags_embeds = sample_indices[non_empty], words_embed[non_empty], words_mask[non_empty], observed_tags_embeds[non_empty]

            if force_gold:
                gold_overlaps = gold_overlaps[non_empty]
                gold_mask = gold_mask[non_empty]
                gold_spans_label = gold_spans_label[non_empty]
                gold_spans_begin = gold_spans_begin[non_empty]
                gold_spans_end = gold_spans_end[non_empty]
                remaining_mask = remaining_mask[non_empty]

        spans_target = None
        # assert step_i <= 4

        ###############################################################
        # Sample the candidate spans/fragments (topk, sampling, gold)
        ###############################################################

        if force_gold:
            loss_data = tuple(map(torch.cat, zip(*loss_data)))

            fragments_label = batch["fragments_label"]
            fragments_begin = batch["fragments_begin"]
            fragments_end = batch["fragments_end"]
            fragments_mask = batch["fragments_mask"]
        else:
            (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_nonzero(
                all_predictions_mask,
                dim=1)
            fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
            fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
            fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
            fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)

        return {
            "flat_spans_label": fragments_label,
            "flat_spans_begin": fragments_begin,
            "flat_spans_end": fragments_end,
            "flat_spans_mask": fragments_mask,
            "flat_spans_logit": torch.ones_like(fragments_begin).float(),
            "loss_data": loss_data,
        }

    def loss(self, spans, batch):
        logits, mask, target = spans["loss_data"]
        loss = -self.crf(logits, mask, F.one_hot(target, num_classes=logits.shape[-1]).bool()).sum(-1)

        return {
            "loss": loss,
        }
