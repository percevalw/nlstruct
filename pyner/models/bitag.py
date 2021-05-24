import math
import torch
import torch.nn.functional as F
from pyner.models.common import register, Identity
from pyner.models.ner import SpanScorer, MarginalTagLoss, SpanLoss
from pyner.models.crf import BIOULDecoder
from pyner.torch_utils import repeat, dclamp, multi_dim_triu, multi_dim_topk, log1mexp, gather


@register("bitag")
class BiTagSpanScorer(SpanScorer):
    def __init__(self, input_size, hidden_size, n_labels,
                 max_length=100,
                 max_fragments_count=100,
                 share_bounds=False,
                 learn_bounds=True,
                 do_biaffine=True,
                 do_norm=True,
                 do_tagging=True,
                 do_length=True,
                 do_density=True,
                 do_tag_bounds=True,
                 allow_overlap=True,
                 learnable_transitions=True,
                 positive_tag_only_loss=True,
                 tag_loss_weight=0.2,
                 eps=1e-8,
                 keep_fragments_threshold=False):
        super().__init__()

        self.input_size = input_size
        self.n_labels = n_labels
        self.do_biaffine = do_biaffine
        self.do_length = do_length
        self.do_tagging = do_tagging
        self.do_tag_bounds = do_tag_bounds
        self.tag_loss_weight = tag_loss_weight

        self.keep_fragments_threshold = keep_fragments_threshold
        self.max_fragments_count = max_fragments_count
        if do_tagging:
            self.label_proj = torch.nn.Linear(input_size, n_labels)
            self.bound_proj = torch.nn.Linear(input_size, 2 * (1 if share_bounds else n_labels))
        self.crf = BIOULDecoder(num_labels=1, with_start_end_transitions=False, allow_overlap=allow_overlap, learnable_transitions=learnable_transitions)
        self.hidden_size = hidden_size
        self.begin_proj = torch.nn.Linear(input_size, hidden_size * (1 if share_bounds else n_labels), bias=True)
        self.end_proj = torch.nn.Linear(input_size, hidden_size * (1 if share_bounds else n_labels), bias=True)
        if do_norm:
            self.norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        else:
            self.norm = Identity()
        self.length_proj = torch.nn.Linear(hidden_size, max_length)
        self.max_length = max_length
        self.eps = eps
        # O, I, B, L, U
        self.register_buffer('bound_to_tag_proj', torch.tensor([
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
        ]).float() * (1 if learn_bounds else 0))
        self.register_buffer('label_to_tag_proj', torch.tensor([0, 1, 1, 1, 1]).float())
        if do_density:
            self.densities = torch.nn.Parameter(torch.zeros(n_labels) + 0.01)
        else:
            self.register_buffer('densities', torch.zeros(n_labels))
        self.compat_bias = torch.nn.Parameter(torch.zeros(()))
        self.span_loss = SpanLoss()
        self.marginal_tag_loss = MarginalTagLoss(positive_only=positive_tag_only_loss)

    def fast_params(self):
        return (
              [self.compat_bias] +
              ([self.densities] if isinstance(self.densities, torch.nn.Parameter) else []) +
              ([self.crf.transitions] if isinstance(self.crf.transitions, torch.nn.Parameter) else [])
        )

    def forward(self, words_embed, words_mask, batch, force_gold=False):
        n_samples, n_words = words_mask.shape
        device = words_embed.device
        n_repeat = 1

        start_positions = torch.arange(n_words, device=device)[None, :, None]
        end_positions = torch.arange(n_words, device=device)[None, None, :]
        spans_lengths = (end_positions + 1 - start_positions).repeat_interleave(words_mask.shape[0], dim=0)
        spans_mask = (
              words_mask.unsqueeze(-1)
              & words_mask.unsqueeze(-2)
              & (0 < spans_lengths)
              & (spans_lengths < self.max_length)
        )
        spans_lengths[~spans_mask] = 0

        if words_embed.ndim == 4:
            n_repeat = words_embed.shape[0]
            words_embed = words_embed.view(words_embed.shape[0] * words_embed.shape[1], words_embed.shape[2], words_embed.shape[3])
            words_mask = repeat(words_mask, n_repeat, 0)

        span_logprobs = 0
        compat_scores = 0
        tag_logprobs = label_logits = bound_logits = has_no_hole_inside = compat_logprobs = None

        if self.do_biaffine:
            begins_embed = self.norm(self.begin_proj(words_embed).view(words_embed.shape[0], words_embed.shape[1], -1, self.hidden_size)).transpose(1, 2)
            ends_embed = self.norm(self.end_proj(words_embed).view(words_embed.shape[0], words_embed.shape[1], -1, self.hidden_size)).transpose(1, 2)
            compat_begin_end_logits = torch.einsum('nlad,nlbd->nlab', begins_embed, ends_embed) / math.sqrt(begins_embed.shape[-1])  # sample * begin * end
            begin_length_logits = self.length_proj(begins_embed)  # sample * label * begin * length
            end_length_logits = self.length_proj(ends_embed)  # sample * label * end * length

            compat_scores = compat_begin_end_logits + self.compat_bias
            if self.do_length:
                compat_scores = compat_scores + (
                      gather(begin_length_logits, dim=-1, index=spans_lengths.unsqueeze(1)) +
                      gather(end_length_logits, dim=-1, index=spans_lengths.transpose(1, 2).unsqueeze(1))
                )
            if self.do_tagging:
                compat_logprobs = F.logsigmoid(compat_scores.double())
                span_logprobs = span_logprobs + dclamp(compat_logprobs, max=-self.eps)
            else:
                # spans_labels_score = compat_scores
                compat_logprobs = F.logsigmoid(compat_scores.double())
                span_logprobs = span_logprobs + dclamp(compat_logprobs, max=-self.eps)
                spans_labels_score = span_logprobs - log1mexp(span_logprobs)

        if self.do_tagging:
            label_logits = self.label_proj(words_embed)  # n_samples * n_words * n_labels
            bound_logits = self.bound_proj(words_embed)  # n_samples * n_words * begin/end
            tag_logits = (
                  torch.einsum('nwl,t->nlwt', label_logits, self.label_to_tag_proj) +
                  torch.einsum('nwlb,bt->nlwt', bound_logits.view(*bound_logits.shape[:-1], -1, 2), self.bound_to_tag_proj)
            )
            tag_logprobs = dclamp(self.crf.marginal(
                tag_logits.reshape(-1, *tag_logits.shape[2:]).double(),
                words_mask.repeat_interleave(self.n_labels, dim=0)
            ).reshape(tag_logits.shape), max=-self.eps)

            is_not_empty_logprobs = tag_logprobs[..., 1:].logsumexp(-1)
            is_not_empty_cs_logprobs = torch.cat([
                torch.zeros_like(is_not_empty_logprobs[..., :1]),
                is_not_empty_logprobs,
            ], dim=-1).cumsum(-1)

            has_no_hole_inside = (multi_dim_triu(is_not_empty_cs_logprobs[..., None, :-1] - is_not_empty_cs_logprobs[..., 1:, None], diagonal=1))  # .clamp_max(-1e-14)

            span_logprobs = span_logprobs + (
                dclamp((has_no_hole_inside / (1 + self.densities.unsqueeze(-1).unsqueeze(-1).abs() * repeat(spans_lengths, n_repeat, dim=0).unsqueeze(1).float().clamp_min(1))), max=-self.eps)
            )
            if self.do_tag_bounds:
                span_logprobs = span_logprobs + dclamp(tag_logprobs[..., :, None, [2, 4]].logsumexp(-1), max=-self.eps)
                span_logprobs = span_logprobs + dclamp(tag_logprobs[..., None, :, [3, 4]].logsumexp(-1), max=-self.eps)
            spans_labels_score = span_logprobs - log1mexp(span_logprobs)

        spans_labels_score = spans_labels_score.permute(0, 2, 3, 1)
        spans_mask = spans_mask.unsqueeze(1).permute(0, 2, 3, 1)

        last_spans_labels_score = spans_labels_score[-n_samples:]

        spans_target = None
        if force_gold:
            spans_target = torch.zeros(n_samples, n_words, n_words, self.n_labels, device=device).bool()
            if torch.is_tensor(batch["entities_mask"]) and batch["entities_mask"].any():
                spans_target[
                    (torch.arange(n_samples, device=device).unsqueeze(1) * batch["fragments_mask"].long())[batch["fragments_mask"]],
                    batch["fragments_begin"][batch["fragments_mask"]],
                    batch["fragments_end"][batch["fragments_mask"]],
                    batch["fragments_label"][batch["fragments_mask"]],
                ] = True

        ###############################################################
        # Sample the candidate spans/fragments (topk, sampling, gold)
        ###############################################################

        # Top k sampling
        fragments_entities = None
        entities_fragments = None

        if self.keep_fragments_threshold:
            # And keep the gold ones too
            if force_gold:
                # Trick to keep the gold targets in the sorted (label, begin, end) order at the top of the candidates
                # so we can do fragments[:n_gold] later to retrieve them in the right order
                # spans_target_forbidden = -spans_target.float().view(-1).cumsum(0).view(spans_target.shape) + 100000
                (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                    last_spans_labels_score.masked_fill(spans_target, 10000),
                    topk=self.max_fragments_count,
                    mask=((last_spans_labels_score.sigmoid() > self.keep_fragments_threshold) | spans_target) & spans_mask,
                )
                matches = (
                      (batch["fragments_begin"][:, None, :] == fragments_begin[:, :, None])
                      & (fragments_end[:, :, None] == batch["fragments_end"][:, None, :])
                      & (batch["fragments_label"][:, None, :] == fragments_label[:, :, None])
                      & batch["fragments_mask"][:, None, :]
                      & fragments_mask[:, :, None]
                )
                one_hot_gold_fragments_entities = F.one_hot(batch['fragments_entities'].clamp_min(0), num_classes=batch['entities_mask'].shape[1]).masked_fill(
                    batch['fragments_entities'][..., None] == -1, 0).bool().any(-2)
                one_hot_fragments_entities = (one_hot_gold_fragments_entities[:, None] & matches[..., None]).any(-2)
                gold_fragments_mask, fragments_entities = (-one_hot_fragments_entities.long()).sort(-1)
                gold_fragments_mask = gold_fragments_mask.bool()
                fragments_entities = fragments_entities.masked_fill(~gold_fragments_mask, -1)

                entities_fragments_mask, entities_fragments = (-one_hot_fragments_entities.long().transpose(-1, -2)).sort(-1)
                entities_fragments_mask = entities_fragments_mask.bool()
                entities_fragments_mask = entities_fragments_mask[..., :int(entities_fragments_mask.sum(-1).max())]
                entities_fragments = entities_fragments[..., :entities_fragments_mask.shape[-1]].masked_fill(~entities_fragments_mask, -1)
            # We don't look at the gold fragments here (for prediction)
            else:
                (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                    last_spans_labels_score,
                    topk=self.max_fragments_count,
                    mask=((last_spans_labels_score.sigmoid() > self.keep_fragments_threshold)) & spans_mask,
                )
                fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
                fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
                fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
                fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)
            # Not supposed to necessary but avoids some nans
            fragments_begin = torch.minimum(fragments_begin, fragments_end)
        # Gold sampling (only keep the gold ones for teacher forcing)
        elif force_gold:
            fragments_label = batch["fragments_label"]
            fragments_begin = batch["fragments_begin"]
            fragments_end = batch["fragments_end"]
            fragments_mask = batch["fragments_mask"]
            fragments_entities = batch["fragments_entities"]
            entities_fragments = batch["entities_fragments"]
        else:
            (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                last_spans_labels_score,
                topk=self.max_fragments_count,
                mask=(last_spans_labels_score > 0.) & spans_mask,
            )
            fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
            fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
            fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
            fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)

        if force_gold:
            same_entity_mask = (
                  (fragments_entities != -1).any(-1).unsqueeze(1)
                  & (fragments_entities != -1).any(-1).unsqueeze(2)
                # & ~torch.eye(n_fragments, device=device, dtype=torch.bool)
            )
            same_entity_target = (
                  (fragments_entities.unsqueeze(1).unsqueeze(-1) == fragments_entities.unsqueeze(2).unsqueeze(-2)) &
                  (fragments_entities.unsqueeze(1).unsqueeze(-1) != -1) &
                  (fragments_entities.unsqueeze(2).unsqueeze(-2) != -1)
            ).any(-1).any(-1)
        else:
            same_entity_mask = same_entity_target = None

        ##################################
        # Do something entity per entity #
        ##################################

        # Compute the keep score
        # keep_fragments_score = None
        # if self.keep_fragments_threshold:
        #    keep_fragments_score = self.ff_keep(all_fragments_embeds)[..., 0]

        return {
            "flat_spans_label": fragments_label,
            "flat_spans_begin": fragments_begin,
            "flat_spans_end": fragments_end,
            "flat_spans_mask": fragments_mask,
            "spans_scores": spans_labels_score.view(n_repeat, n_samples, *spans_labels_score.shape[1:]),
            "spans_target": spans_target,
            "spans_mask": spans_mask,
            "flat_spans_entities": fragments_entities,
            "flat_entities_spans": entities_fragments,
            "same_entity_mask": same_entity_mask,
            "same_entity_target": same_entity_target,
            "tag_logprobs": tag_logprobs.view(n_repeat, n_samples, *tag_logprobs.shape[1:]) if tag_logprobs is not None else None,
            "span_logprobs": span_logprobs.view(n_repeat, n_samples, *span_logprobs.shape[1:]) if span_logprobs is not None else None,
            "has_no_hole_inside": has_no_hole_inside.view(n_repeat, n_samples, *has_no_hole_inside.shape[1:]) if has_no_hole_inside is not None else None,
            "begin_bounds": tag_logprobs.view(n_repeat, n_samples, *tag_logprobs.shape[1:])[..., :, None, [2, 4]].logsumexp(-1) if tag_logprobs is not None else None,
            "end_bounds": tag_logprobs.view(n_repeat, n_samples, *tag_logprobs.shape[1:])[..., None, :, [3, 4]].logsumexp(-1) if tag_logprobs is not None else None,
            "compat_logprobs": compat_logprobs.view(n_repeat, n_samples, *compat_logprobs.shape[1:]) if compat_logprobs is not None else None,
            "label_logits": label_logits.view(n_repeat, n_samples, *label_logits.shape[1:]) if label_logits is not None else None,
            "bound_logits": bound_logits.view(n_repeat, n_samples, *bound_logits.shape[1:]) if bound_logits is not None else None,
            "flat_spans_scores": last_spans_labels_score[
                torch.arange(n_samples, device=device).unsqueeze(1), fragments_begin, fragments_end, fragments_label] if last_spans_labels_score is not None else None,
        }

    def loss(self, spans, batch):
        if "spans_scores" in spans and getattr(self, "span_loss", None):
            span_loss = self.span_loss(spans["spans_scores"], spans["spans_mask"], spans["spans_target"])
        if "tag_logprobs" in spans and getattr(self, "marginal_tag_loss", None):
            tag_loss = self.marginal_tag_loss(spans["tag_logprobs"], batch) * self.tag_loss_weight

        return {
            "loss": tag_loss + span_loss,
            "tag_loss": tag_loss,
            "span_loss": span_loss,
        }
