import math

import torch
import torch.nn.functional as F

from pyner.models.common import Identity
from pyner.models.crf import BIOULDecoder
from pyner.models.ner import SpanScorer
from pyner.registry import register
from pyner.torch_utils import multi_dim_triu, repeat, dclamp, multi_dim_topk, log1mexp, gather, nll, bce_with_logits, cross_entropy_with_logits


@register("bitag")
class BiTagSpanScorer(SpanScorer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_labels,
                 max_length=100,
                 max_fragments_count=100,
                 share_bounds=False,
                 learn_bounds=True,
                 do_biaffine=True,
                 do_viterbi_filtering=False,
                 threshold=0.5,
                 do_norm=True,
                 do_tagging=True,
                 do_length=True,
                 do_tag_bounds=True,
                 allow_overlap=True,
                 learnable_transitions=True,
                 positive_tag_only_loss=True,
                 detach_span_tag_logits=True,
                 tag_loss_weight=0.2,
                 multi_label=True,
                 combine_mode="and",
                 eps=1e-8):
        super().__init__()

        if not multi_label:
            n_labels = n_labels + 1
        self.input_size = input_size
        self.n_labels = n_labels
        self.do_biaffine = do_biaffine
        self.do_length = do_length
        self.do_tagging = do_tagging
        self.do_tag_bounds = do_tag_bounds
        self.tag_loss_weight = tag_loss_weight
        self.combine_mode = combine_mode
        assert do_biaffine or do_tagging
        assert not do_viterbi_filtering or do_tagging, "You must set do_tagging=True to perform viterbi filterinng"
        self.do_viterbi_filtering = do_viterbi_filtering
        self.threshold = threshold
        self.multi_label = multi_label

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
        self.compat_bias = torch.nn.Parameter(torch.zeros(()))
        self.span_loss = SpanLoss(multi_label=multi_label)
        self.marginal_tag_loss = MarginalTagLoss(positive_only=positive_tag_only_loss)
        self.detach_span_tag_logits = detach_span_tag_logits

    def fast_params(self):
        return (
              [self.compat_bias] +
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
        tag_logits = tag_logprobs = label_logits = bound_logits = has_no_hole_inside = compat_logprobs = None
        prediction = torch.zeros_like(spans_mask)

        if self.do_tagging:
            label_logits = self.label_proj(words_embed).transpose(-1, -2)  # n_samples * n_words * n_labels
            bound_logits = self.bound_proj(words_embed)  # n_samples * n_words * begin/end
            tag_logits = (
                  torch.einsum('nlw,t->nlwt', label_logits, self.label_to_tag_proj) +
                  torch.einsum('nwlb,bt->nlwt', bound_logits.view(*bound_logits.shape[:-1], -1, 2), self.bound_to_tag_proj)
            )
            tag_logprobs = self.crf.marginal(
                tag_logits.reshape(-1, *tag_logits.shape[2:]),
                words_mask.repeat_interleave(self.n_labels, dim=0)
            ).reshape(tag_logits.shape).double()

            is_not_empty_logprobs = tag_logprobs[..., 1:].logsumexp(-1)
            is_not_empty_cs_logprobs = torch.cat([
                torch.zeros_like(is_not_empty_logprobs[..., :1]),
                is_not_empty_logprobs,
            ], dim=-1).cumsum(-1)

            has_no_hole_inside = (multi_dim_triu(is_not_empty_cs_logprobs[..., None, :-1] - is_not_empty_cs_logprobs[..., 1:, None], diagonal=1))  # .clamp_max(-1e-14)

            span_logprobs = span_logprobs + dclamp(has_no_hole_inside, max=-self.eps)
            if self.do_tag_bounds:
                span_logprobs = span_logprobs + tag_logprobs[..., :, None, [2, 4]].logsumexp(-1)
                span_logprobs = span_logprobs + tag_logprobs[..., None, :, [3, 4]].logsumexp(-1)
            spans_labels_score = span_logprobs - log1mexp(dclamp(span_logprobs, max=-self.eps))
            if self.detach_span_tag_logits:
                spans_labels_score = spans_labels_score.detach()

            prediction = self.crf.tags_to_spans(self.crf.decode(
                tag_logits[:n_samples].reshape(-1, *tag_logits.shape[2:]),
                words_mask[:n_samples].repeat_interleave(self.n_labels, dim=0)
            ), words_mask[:n_samples].repeat_interleave(self.n_labels, dim=0)).reshape(n_samples, -1, n_words, n_words).permute(0, 2, 3, 1)

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
            if self.combine_mode in ("sum", "and") and self.do_tagging:
                if self.combine_mode == "and":
                    compat_logprobs = F.logsigmoid(compat_scores.double())
                    span_logprobs = dclamp(span_logprobs + compat_logprobs, max=-self.eps)
                    spans_labels_score = span_logprobs - log1mexp(span_logprobs)
                elif self.combine_mode == "sum":
                    spans_labels_score = spans_labels_score + compat_scores
            else:
                spans_labels_score = compat_scores
                # compat_logprobs = F.logsigmoid(compat_scores.double())
                # span_logprobs = span_logprobs + dclamp(compat_logprobs, max=-self.eps)
                # spans_labels_score = span_logprobs - log1mexp(span_logprobs)

        spans_labels_score = spans_labels_score.permute(0, 2, 3, 1)
        spans_mask = spans_mask.unsqueeze(1).permute(0, 2, 3, 1)

        last_spans_labels_score = spans_labels_score[-n_samples:]

        spans_target = None
        if force_gold:
            spans_target = torch.zeros(n_samples, n_words, n_words, self.n_labels, device=device).bool()
            if torch.is_tensor(batch["fragments_mask"]) and batch["fragments_mask"].any():
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

        if force_gold:
            fragments_label = batch["fragments_label"]
            fragments_begin = batch["fragments_begin"]
            fragments_end = batch["fragments_end"]
            fragments_mask = batch["fragments_mask"]
        elif not self.multi_label:
            begin_end_label_scores, begin_end_label = last_spans_labels_score.max(-1)
            (fragments_begin, fragments_end), fragments_mask = multi_dim_topk(
                begin_end_label_scores,
                mask=(begin_end_label > 0) & spans_mask.any(-1) & (gather(prediction, begin_end_label, dim=-1) if self.do_viterbi_filtering else True),
                topk=self.max_fragments_count,
                dim=1,
            )
            fragments_label = begin_end_label[torch.arange(n_samples)[..., None], fragments_begin, fragments_end] - 1
            fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
            fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
            fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
            fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)
        else:
            (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                last_spans_labels_score,
                topk=self.max_fragments_count,
                mask=(last_spans_labels_score.sigmoid() > self.threshold) & spans_mask & (prediction if self.do_viterbi_filtering else True),
                dim=1,
            )
            fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
            fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
            fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
            fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)

        return {
            "flat_spans_label": fragments_label,
            "flat_spans_begin": fragments_begin,
            "flat_spans_end": fragments_end,
            "flat_spans_mask": fragments_mask,
            "spans_scores": spans_labels_score.view(n_repeat, n_samples, *spans_labels_score.shape[1:]),
            "spans_target": spans_target,
            "spans_mask": spans_mask,
            "tag_logprobs": tag_logprobs.view(n_repeat, n_samples, *tag_logprobs.shape[1:]) if tag_logprobs is not None else None,
            # "span_logprobs": span_logprobs.view(n_repeat, n_samples, *span_logprobs.shape[1:]) if span_logprobs is not None else None,
            "has_no_hole_inside": has_no_hole_inside.view(n_repeat, n_samples, *has_no_hole_inside.shape[1:]) if has_no_hole_inside is not None else None,
            "begin_bounds": tag_logprobs.view(n_repeat, n_samples, *tag_logprobs.shape[1:])[..., :, None, [2, 4]].logsumexp(-1) if tag_logprobs is not None else None,
            "end_bounds": tag_logprobs.view(n_repeat, n_samples, *tag_logprobs.shape[1:])[..., None, :, [3, 4]].logsumexp(-1) if tag_logprobs is not None else None,
            "compat_logprobs": compat_logprobs.view(n_repeat, n_samples, *compat_logprobs.shape[1:]) if compat_logprobs is not None else None,
            "label_logits": label_logits.view(n_repeat, n_samples, *label_logits.shape[1:]) if label_logits is not None else None,
            "tag_logits": tag_logits,
            "bound_logits": bound_logits.view(n_repeat, n_samples, *bound_logits.shape[1:]) if bound_logits is not None else None,
            "flat_spans_scores": last_spans_labels_score[
                torch.arange(n_samples, device=device).unsqueeze(1), fragments_begin, fragments_end, fragments_label] if last_spans_labels_score is not None else None,
        }

    def loss(self, spans, batch):
        tag_loss = 0.
        span_loss = 0.
        if "spans_scores" in spans and getattr(self, "span_loss", None):
            span_loss = self.span_loss(spans["spans_scores"], spans["spans_mask"], spans["spans_target"])
        if "tag_logprobs" in spans and getattr(self, "marginal_tag_loss", None):
            tag_loss = self.marginal_tag_loss(spans["tag_logprobs"], spans["label_logits"], batch) * self.tag_loss_weight

        return {
            "loss": tag_loss + span_loss,
            "tag_loss": tag_loss,
            "span_loss": span_loss,
        }


class SpanLoss(torch.nn.Module):
    def __init__(self, multi_label=True):
        super().__init__()
        self.multi_label = multi_label

    def forward(self, scores, mask, target):
        if target.ndim == scores.ndim - 1:
            target = target.unsqueeze(0).repeat_interleave(scores.shape[0], dim=0)
        if self.multi_label:
            loss = (bce_with_logits(
                scores,
                target, reduction='none').masked_fill(~mask, 0))
            loss = loss.mean(-1).sum()
        else:
            is_empty, single_label_target = target.max(-1)
            single_label_target[is_empty == 0] = -1
            single_label_target += 1
            loss = cross_entropy_with_logits(
                scores,
                single_label_target,
                reduction='none'
            ).masked_fill(~mask.any(-1), 0)
            loss = loss.sum()
        # loss = loss.sum()#(loss / mask.rename(None).sum(-1, keepdim=True).sum(-2, keepdim=True).sum(-3, keepdim=True)).sum() * 10
        return loss


class MarginalTagLoss(torch.nn.Module):
    def __init__(self, positive_only=False):
        super().__init__()
        self.positive_only = positive_only

    def forward(self, tag_logprobs, label_logits, batch):
        if tag_logprobs is None:
            return 0
        if tag_logprobs.ndim == 5:
            shape = tag_logprobs.shape[1:-1]
        else:
            shape = tag_logprobs.shape[:-1]
        O, I, B, L, U = 0, 1, 2, 3, 4
        begins = batch["fragments_begin"]  # n_samples * n_fragments
        ends = batch["fragments_end"]  # n_samples * n_fragments
        labels = batch["fragments_label"]  # n_samples * n_fragments
        fragment_mask = batch["fragments_mask"]  # n_samples * n_fragments
        i_tags = torch.zeros(shape, dtype=torch.long)
        bl_tags = torch.zeros(shape, dtype=torch.long)
        u_tags = torch.zeros(shape, dtype=torch.long)
        mask = batch['words_mask']
        for sample_idx, b, e, l in zip(fragment_mask.nonzero(as_tuple=True)[0].tolist(), begins[fragment_mask].tolist(), ends[fragment_mask].tolist(), labels[fragment_mask].tolist()):
            if b < e:
                i_tags[sample_idx, l, b:e + 1] = I
                bl_tags[sample_idx, l, b] = B
                bl_tags[sample_idx, l, e] = L
            else:
                u_tags[sample_idx, l, b] = U
        tags = torch.maximum(torch.maximum(i_tags, bl_tags), u_tags).to(tag_logprobs.device)
        if self.positive_only:
            mask = mask & (tags > 0).any(1)  # & shift(tags_target.any(-1), dim=1, n=1) & shift(tags_target.any(-1), dim=1, n=-1)
        loss = nll(
            tag_logprobs,
            tags if tag_logprobs.ndim == 4 else tags.unsqueeze(0).repeat_interleave(tag_logprobs.shape[0], dim=0),
            reduction='none',
        ).masked_fill(~mask.unsqueeze(1), 0).mean(-2).sum()

        loss = loss + bce_with_logits(
            label_logits,
            tags if label_logits.ndim == 3 else tags.unsqueeze(0).repeat_interleave(label_logits.shape[0], dim=0) > 0,
            reduction='none',
        ).masked_fill(~mask.unsqueeze(1), 0).mean(-2).sum()
        # loss = loss.mean(-1).sum()
        return loss
