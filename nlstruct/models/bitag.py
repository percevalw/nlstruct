from nlstruct.models.common import *
from nlstruct.models.crf import BIOULDecoder
from nlstruct.models.ner import *
from nlstruct.torch_utils import multi_dim_triu, repeat, dclamp, multi_dim_topk, gather, nll, bce_with_logits, cross_entropy_with_logits, inv_logsigmoid, fork_rng


class TagFFN(torch.nn.Module):
    def __init__(self, size, n_labels, n_tags, dropout_p=0.1):
        super().__init__()

        self.bias = torch.nn.Parameter(torch.zeros(n_labels * n_tags))
        self.label_embeddings = torch.nn.Embedding(n_labels, size).weight
        self.linear1 = torch.nn.Linear(size, size * n_tags)
        self.linear2 = torch.nn.Linear(size, size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.n_tags = n_tags
        self.n_labels = n_labels

    def forward(self, inputs):
        #        tag_embeds = self.linear2(self.dropout(F.gelu(self.linear1(self.dropout(self.label_embeddings)))).reshape(self.n_labels * self.n_tags, -1))
        tag_embeds = self.linear1(self.label_embeddings).reshape(self.n_labels * self.n_tags, -1)
        return F.linear(inputs, weight=tag_embeds, bias=self.bias)


@register("bitag")
class BiTagSpanScorer(SpanScorer):
    ENSEMBLE = "ensemble_bitag"

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_labels,
                 do_biaffine=True,
                 do_tagging=True,
                 do_length=True,
                 multi_label=True,

                 threshold=0.5,
                 max_length=100,
                 max_fragments_count=100,
                 detach_span_tag_logits=True,
                 tag_loss_weight=0.2,
                 biaffine_loss_weight=0.2,
                 dropout_p=0.2,

                 mode="seq",

                 allow_overlap=True,
                 learnable_transitions=False,

                 marginal_tagger_loss=False,
                 eps=1e-8):
        super().__init__()

        assert do_biaffine or do_tagging

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_labels = n_labels

        if not multi_label:
            effective_n_labels = n_labels + 1
        else:
            effective_n_labels = n_labels

        self.do_biaffine = do_biaffine
        self.do_tagging = do_tagging
        self.do_length = do_length
        self.multi_label = multi_label

        self.tag_loss_weight = tag_loss_weight
        self.biaffine_loss_weight = biaffine_loss_weight
        self.detach_span_tag_logits = detach_span_tag_logits

        self.threshold = threshold
        self.max_length = max_length
        self.max_fragments_count = max_fragments_count
        self.dropout = torch.nn.Dropout(dropout_p)

        with fork_rng():
            if do_tagging:
                if self.do_tagging is True:
                    self.do_tagging = "full"
                if self.do_tagging.startswith("full"):
                    n_tags = 5
                    self.register_buffer('tag_combinator', torch.tensor([
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                    ]).float())
                elif self.do_tagging.startswith("positive"):
                    n_tags = 4
                    self.register_buffer('tag_combinator', torch.tensor([
                        [-1, 1, 0, 0, 0],
                        [-1, 0, 1, 0, 0],
                        [-1, 0, 0, 1, 0],
                        [-1, 0, 0, 0, 1],
                    ]).float())
                elif self.do_tagging.startswith("shared_label_unary"):
                    n_tags = 3
                    self.register_buffer('tag_combinator', torch.tensor([
                        [-1, 1, 1, 1, 1],
                        [-1, -1, 1, 0, 1],
                        [-1, -1, 0, 1, 1],
                    ]).float())
                elif self.do_tagging.startswith("shared_label"):
                    n_tags = 4
                    self.register_buffer('tag_combinator', torch.tensor([
                        [-1, 1, 1, 1, 1],
                        [-1, 0, 1, 0, 0],
                        [-1, 0, 0, 1, 0],
                        [-1, 0, 0, 0, 1],
                    ]).float())

                if self.do_tagging.endswith(":ffn"):
                    self.tag_proj = TagFFN(input_size, n_labels, n_tags=n_tags, dropout_p=dropout_p)
                else:
                    self.tag_proj = torch.nn.Linear(input_size, n_labels * n_tags)

                self.crf = BIOULDecoder(
                    num_labels=1,
                    with_start_end_transitions=True,
                    allow_overlap=allow_overlap,
                    learnable_transitions=learnable_transitions,
                )
                self.tagger_loss = TaggerLoss(marginal=marginal_tagger_loss, _crf=self.crf)

        self.eps = eps

        with fork_rng():
            if do_biaffine:
                self.length_proj = torch.nn.Linear(hidden_size, max_length) if self.do_length else None
                self.begin_proj = torch.nn.Linear(input_size, hidden_size * effective_n_labels, bias=True)
                self.end_proj = torch.nn.Linear(input_size, hidden_size * effective_n_labels, bias=True)
                self.biaffine_bias = torch.nn.Parameter(torch.zeros(()))
                self.biaffine_loss = BiaffineLoss(multi_label=self.multi_label)

        self.mode = mode

    def fast_params(self):
        return (
              ([self.biaffine_bias] if hasattr(self, 'biaffine_bias') else []) +  # [self.rescale] +
              ([self.crf.transitions] if hasattr(self, 'crf') and isinstance(self.crf.transitions, torch.nn.Parameter) else [])
        )

    def forward(self, words_embed, words_mask, batch, force_gold=False, do_tagging=None, do_biaffine=None):
        do_tagging = do_tagging if do_tagging is not None else self.do_tagging
        do_biaffine = do_biaffine if do_biaffine is not None else self.do_biaffine
        n_samples, n_words = words_mask.shape
        device = words_embed.device
        n_repeat = 1

        words_embed = self.dropout(words_embed)

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

        tagger_logprobs = 0
        crf_tag_logits = crf_tag_logprobs = label_logits = bound_logits = spans_logits = has_no_hole_inside = biaffine_logits = tagger_logits = None
        viterbi_mask = None

        if do_tagging:
            raw_logits = (
                self.tag_proj(words_embed)  # n_samples * n_words * (n_labels * ...)
                    .view(n_samples, n_words, (self.n_labels + (0 if self.multi_label else 1)), self.tag_combinator.shape[0])
                    .permute(0, 2, 1, 3)  # n_samples * n_labels * n_words * (...)
            )
            crf_tag_logits = raw_logits @ self.tag_combinator

            crf_tag_logprobs = self.crf.marginal(
                crf_tag_logits.reshape(-1, *crf_tag_logits.shape[2:]),
                words_mask.repeat_interleave((self.n_labels + (0 if self.multi_label else 1)), dim=0)
            ).reshape(crf_tag_logits.shape).double()

            is_not_empty_logprobs = crf_tag_logprobs[..., 1:].logsumexp(-1)
            is_not_empty_cs_logprobs = torch.cat([
                torch.zeros_like(is_not_empty_logprobs[..., :1]),
                is_not_empty_logprobs,
            ], dim=-1).cumsum(-1)
            has_no_hole_inside = (multi_dim_triu(is_not_empty_cs_logprobs[..., None, :-1] - is_not_empty_cs_logprobs[..., 1:, None], diagonal=1))  # .clamp_max(-1e-14)

            tagger_logprobs = torch.minimum(torch.minimum(
                dclamp(has_no_hole_inside, max=-self.eps),
                crf_tag_logprobs[..., :, None, [2, 4]].logsumexp(-1),
            ), crf_tag_logprobs[..., None, :, [3, 4]].logsumexp(-1))
            tagger_logits = inv_logsigmoid(tagger_logprobs, eps=self.eps)
            if self.detach_span_tag_logits:
                tagger_logits = tagger_logits.detach()
            crf_decoded_tags = self.crf.decode(
                crf_tag_logits[-n_samples:].reshape(-1, *crf_tag_logits.shape[2:]),
                words_mask[-n_samples:].repeat_interleave((self.n_labels + (0 if self.multi_label else 1)), dim=0)
            )
            viterbi_mask = self.crf.tags_to_spans(
                crf_decoded_tags,
                words_mask[-n_samples:].repeat_interleave((self.n_labels + (0 if self.multi_label else 1)), dim=0),
                do_overlap_disambiguation=not self.do_biaffine,
            ).reshape(n_samples, -1, n_words, n_words).permute(0, 2, 3, 1)

        if do_biaffine:
            begins_embed = self.begin_proj(words_embed).view(words_embed.shape[0], words_embed.shape[1], -1, self.hidden_size).transpose(1, 2)
            ends_embed = self.end_proj(words_embed).view(words_embed.shape[0], words_embed.shape[1], -1, self.hidden_size).transpose(1, 2)
            biaffine_logits = torch.einsum('nlad,nlbd->nlab', begins_embed, ends_embed) / math.sqrt(begins_embed.shape[-1]) + self.biaffine_bias  # sample * begin * end

            if self.do_length:
                begin_length_logits = self.length_proj(begins_embed)  # sample * label * begin * length
                end_length_logits = self.length_proj(ends_embed)  # sample * label * end * length

                biaffine_logits = biaffine_logits + (
                      gather(begin_length_logits, dim=-1, index=spans_lengths.unsqueeze(1)) +
                      gather(end_length_logits, dim=-1, index=spans_lengths.transpose(1, 2).unsqueeze(1))
                )

        if not do_biaffine:
            spans_logits = tagger_logits
        elif not do_tagging:
            spans_logits = biaffine_logits
        else:
            spans_logits = tagger_logits + biaffine_logits

        spans_mask = spans_mask.unsqueeze(1).permute(0, 2, 3, 1)
        if spans_logits is not None:
            spans_logits = spans_logits.permute(0, 2, 3, 1)
            spans_logits = spans_logits[-n_samples:]
        if biaffine_logits is not None:
            biaffine_logits = biaffine_logits.permute(0, 2, 3, 1)
            biaffine_logits = biaffine_logits[-n_samples:]
        if tagger_logits is not None:
            tagger_logits = tagger_logits.permute(0, 2, 3, 1)
            tagger_logits = tagger_logits[-n_samples:]

        spans_target = None
        if force_gold:
            spans_target = torch.zeros(n_samples, n_words, n_words, (self.n_labels + (0 if self.multi_label else 1)), device=device).bool()
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
        fragments_logit = None

        if force_gold:
            fragments_label = batch["fragments_label"]
            fragments_begin = batch["fragments_begin"]
            fragments_end = batch["fragments_end"]
            fragments_mask = batch["fragments_mask"]
            fragments_logit = spans_logits[torch.arange(n_samples, device=device).unsqueeze(1), fragments_begin, fragments_end, fragments_label] if spans_logits is not None else None
        elif self.multi_label:
            if do_tagging and do_biaffine:
                if self.mode == "seq":
                    bounds_match_logits = biaffine_logits.masked_fill(~viterbi_mask, -100000)
                    (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                        spans_logits,
                        topk=self.max_fragments_count,
                        mask=(
                              spans_mask & viterbi_mask & (
                              (
                                    (bounds_match_logits >= bounds_match_logits.max(-2, keepdim=True).values - 1e-14) &
                                    (bounds_match_logits >= bounds_match_logits.max(-3, keepdim=True).values - 1e-14)
                              ) | (bounds_match_logits.sigmoid() > self.threshold))),
                        dim=1,
                    )
                elif self.mode == "sum":
                    (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                        spans_logits,
                        topk=self.max_fragments_count,
                        mask=(
                              spans_mask & (
                            ((viterbi_mask | (tagger_logits > 0.)) & (biaffine_logits > 0))
                        )
                        ),
                        dim=1,
                    )

            elif do_tagging:
                (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                    spans_logits,
                    topk=self.max_fragments_count,
                    mask=spans_mask & viterbi_mask,
                    dim=1,
                )
            elif do_biaffine:
                (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                    spans_logits,
                    topk=self.max_fragments_count,
                    mask=spans_mask & (biaffine_logits.sigmoid() > self.threshold),
                    dim=1,
                )
            else:
                raise NotImplementedError()

            fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
            fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
            fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
            fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)
            fragments_logit = spans_logits[torch.arange(n_samples, device=device).unsqueeze(1), fragments_begin, fragments_end, fragments_label]
        else:
            if do_biaffine and not do_tagging:
                best_begin_end_scores, best_begin_end_labels = spans_logits.log_softmax(-1).max(-1)
                (fragments_begin, fragments_end), fragments_mask = multi_dim_topk(
                    best_begin_end_scores,
                    topk=self.max_fragments_count,
                    mask=spans_mask.any(-1) & (best_begin_end_labels > 0),
                    dim=1,
                )
                fragments_label = best_begin_end_labels[torch.arange(n_samples, device=device)[:, None], fragments_begin, fragments_end] - 1

                fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
                fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
                fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
                fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)
                fragments_logit = best_begin_end_scores[torch.arange(n_samples, device=device).unsqueeze(1), fragments_begin, fragments_end]

        return {
            "flat_spans_label": fragments_label,
            "flat_spans_begin": fragments_begin,
            "flat_spans_end": fragments_end,
            "flat_spans_mask": fragments_mask,
            "flat_spans_logit": fragments_logit,

            "spans_logits": spans_logits.view(n_repeat, n_samples, *spans_logits.shape[1:]) if spans_logits is not None else None,
            "biaffine_logits": biaffine_logits.view(n_repeat, n_samples, *spans_logits.shape[1:]) if biaffine_logits is not None else None,
            "spans_target": spans_target,
            "spans_mask": spans_mask,
            "raw_logits": raw_logits.view(n_repeat, n_samples, *raw_logits.shape[1:]) if raw_logits is not None else None,
            "crf_tag_logprobs": crf_tag_logprobs.view(n_repeat, n_samples, *crf_tag_logprobs.shape[1:]) if crf_tag_logprobs is not None else None,
            "crf_tag_logits": crf_tag_logits.view(n_repeat, n_samples, *crf_tag_logits.shape[1:]) if crf_tag_logits is not None else None,
        }

    def loss(self, spans, batch):
        tag_loss = 0.
        span_loss = 0.
        if getattr(self, "biaffine_loss", None) is not None and spans.get("biaffine_logits", None) is not None:
            span_loss = self.biaffine_loss(spans["biaffine_logits"], spans, batch) * self.biaffine_loss_weight
        if getattr(self, 'tagger_loss', None) is not None and ("crf_tag_logprobs" in spans or "crf_tag_logits" in spans):
            tag_loss = self.tagger_loss(spans, batch) * self.tag_loss_weight

        return {
            "loss": tag_loss + span_loss,
            "tag_loss": tag_loss,
            "span_loss": span_loss,
        }


@register("ensemble_bitag")
class EnsembleBiTagSpanScorer(SpanScorer):
    def __init__(self, models):
        super().__init__()

        models = [get_instance(model) for model in models]
        main = models[0]

        assert len({model.multi_label for model in models}) == 1, "All models must be multi-label or none"

        self.input_size = main.input_size
        self.hidden_size = main.hidden_size
        self.n_labels = main.n_labels
        self.do_biaffine = main.do_biaffine
        self.do_tagging = main.do_tagging
        self.do_length = main.do_length
        self.multi_label = main.multi_label
        self.biaffine_loss_weight = main.biaffine_loss_weight
        self.detach_span_tag_logits = main.detach_span_tag_logits
        self.threshold = main.threshold
        self.max_length = main.max_length
        self.max_fragments_count = main.max_fragments_count
        self.eps = main.eps
        self.mode = main.mode

        self.crf = main.crf
        self.register_buffer('ensemble_tag_combinator', torch.stack([getattr(m, 'tag_combinator', None) for m in models], dim=0))
        self.ensemble_tag_proj = torch.nn.ModuleList([getattr(m, 'tag_proj', None) for m in models])
        self.ensemble_label_proj = torch.nn.ModuleList([getattr(m, 'length_proj', None) for m in models])
        self.ensemble_begin_proj = torch.nn.ModuleList([getattr(m, 'begin_proj', None) for m in models])
        self.ensemble_end_proj = torch.nn.ModuleList([getattr(m, 'end_proj', None) for m in models])
        self.register_buffer('biaffine_bias', torch.stack([m.biaffine_bias for m in models], dim=0).mean(0))

    def fast_params(self):
        raise NotImplementedError()

    def forward(self, ensemble_words_embed, words_mask, batch, force_gold=False, do_tagging=None, do_biaffine=None):
        do_tagging = do_tagging if do_tagging is not None else self.do_tagging
        do_biaffine = do_biaffine if do_biaffine is not None else self.do_biaffine

        n_samples, n_words = words_mask.shape
        device = ensemble_words_embed[0].device
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

        if ensemble_words_embed[0].ndim == 4:
            n_repeat = ensemble_words_embed[0].shape[0]
            ensemble_words_embed = [words_embed.view(words_embed.shape[0] * words_embed.shape[1], words_embed.shape[2], words_embed.shape[3]) for words_embed in ensemble_words_embed]
            words_mask = repeat(words_mask, n_repeat, 0)

        crf_tag_logits = crf_tag_logprobs = biaffine_logits = tagger_logits = None
        viterbi_mask = None

        if do_tagging:
            crf_tag_logits = torch.stack([(
                                              tag_proj(words_embed)  # n_samples * n_words * (n_labels * ...)
                                                  .view(n_samples, n_words, (self.n_labels + (0 if self.multi_label else 1)), tag_combinator.shape[0])
                                                  .permute(0, 2, 1, 3)  # n_samples * n_labels * n_words * (...)
                                          ) @ tag_combinator for words_embed, tag_proj, tag_combinator in zip(ensemble_words_embed, self.ensemble_tag_proj, self.tag_combinator.unbind(0))],
                                         dim=0).mean(0)
            crf_tag_logprobs = self.crf.marginal(
                crf_tag_logits.reshape(-1, *crf_tag_logits.shape[2:]),
                words_mask.repeat_interleave((self.n_labels + (0 if self.multi_label else 1)), dim=0)
            ).reshape(crf_tag_logits.shape).double()

            is_not_empty_logprobs = crf_tag_logprobs[..., 1:].logsumexp(-1)
            is_not_empty_cs_logprobs = torch.cat([
                torch.zeros_like(is_not_empty_logprobs[..., :1]),
                is_not_empty_logprobs,
            ], dim=-1).cumsum(-1)
            has_no_hole_inside = (multi_dim_triu(is_not_empty_cs_logprobs[..., None, :-1] - is_not_empty_cs_logprobs[..., 1:, None], diagonal=1))  # .clamp_max(-1e-14)

            tagger_logprobs = torch.minimum(torch.minimum(
                dclamp(has_no_hole_inside, max=-self.eps),
                crf_tag_logprobs[..., :, None, [2, 4]].logsumexp(-1),
            ), crf_tag_logprobs[..., None, :, [3, 4]].logsumexp(-1))
            tagger_logits = inv_logsigmoid(tagger_logprobs, eps=self.eps)
            if self.detach_span_tag_logits:
                tagger_logits = tagger_logits.detach()

            if not force_gold:
                crf_decoded_tags = self.crf.decode(
                    crf_tag_logits[-n_samples:].reshape(-1, *crf_tag_logits.shape[2:]),
                    words_mask[-n_samples:].repeat_interleave((self.n_labels + (0 if self.multi_label else 1)), dim=0)
                )
                viterbi_mask = self.crf.tags_to_spans(
                    crf_decoded_tags, words_mask[-n_samples:].repeat_interleave((self.n_labels + (0 if self.multi_label else 1)), dim=0)
                ).reshape(n_samples, -1, n_words, n_words).permute(0, 2, 3, 1)

        if do_biaffine:
            ensemble_biaffine_logits = []
            for words_embed, norm, begin_proj, end_proj, length_proj in zip(
                  ensemble_words_embed,
                  self.ensemble_begin_proj,
                  self.ensemble_end_proj,
                  self.ensemble_length_proj,
            ):
                begins_embed = begin_proj(words_embed).view(words_embed.shape[0], words_embed.shape[1], -1, self.hidden_size).transpose(1, 2)
                ends_embed = end_proj(words_embed).view(words_embed.shape[0], words_embed.shape[1], -1, self.hidden_size).transpose(1, 2)
                biaffine_logits = torch.einsum('nlad,nlbd->nlab', begins_embed, ends_embed) / math.sqrt(begins_embed.shape[-1])
                if self.do_length:
                    begin_length_logits = self.length_proj(begins_embed)  # sample * label * begin * length
                    end_length_logits = self.length_proj(ends_embed)  # sample * label * end * length

                    biaffine_logits = biaffine_logits + (
                          gather(begin_length_logits, dim=-1, index=spans_lengths.unsqueeze(1)) +
                          gather(end_length_logits, dim=-1, index=spans_lengths.transpose(1, 2).unsqueeze(1))
                    )

                ensemble_biaffine_logits.append(biaffine_logits)
            biaffine_logits = torch.stack(ensemble_biaffine_logits, dim=0).mean(0) + self.compat_bias

        if not do_biaffine:
            spans_logits = tagger_logits
        elif not do_tagging:
            spans_logits = biaffine_logits
        else:
            spans_logits = tagger_logits + biaffine_logits

        spans_mask = spans_mask.unsqueeze(1).permute(0, 2, 3, 1)
        spans_logits = spans_logits.permute(0, 2, 3, 1)
        spans_logits = spans_logits[-n_samples:]
        if biaffine_logits is not None:
            biaffine_logits = biaffine_logits.permute(0, 2, 3, 1)
            biaffine_logits = biaffine_logits[-n_samples:]
        if tagger_logits is not None:
            tagger_logits = tagger_logits.permute(0, 2, 3, 1)
            tagger_logits = tagger_logits[-n_samples:]

        spans_target = None
        if force_gold:
            spans_target = torch.zeros(n_samples, n_words, n_words, (self.n_labels + (0 if self.multi_label else 1)), device=device).bool()
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
        fragments_logit = None

        if force_gold:
            fragments_label = batch["fragments_label"]
            fragments_begin = batch["fragments_begin"]
            fragments_end = batch["fragments_end"]
            fragments_mask = batch["fragments_mask"]
        elif self.multi_label:
            if do_tagging and do_biaffine:
                if self.mode == "seq":
                    bounds_match_logits = biaffine_logits.masked_fill(~viterbi_mask, -100000)
                    (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                        spans_logits,
                        topk=self.max_fragments_count,
                        mask=(
                              spans_mask & viterbi_mask & (
                              (
                                    (bounds_match_logits >= bounds_match_logits.max(-2, keepdim=True).values - 1e-14) &
                                    (bounds_match_logits >= bounds_match_logits.max(-3, keepdim=True).values - 1e-14)
                              ) | (bounds_match_logits.sigmoid() > self.threshold))),
                        dim=1,
                    )
                elif self.mode == "sum":
                    (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                        spans_logits,
                        topk=self.max_fragments_count,
                        mask=(
                              spans_mask & (
                            ((viterbi_mask | (tagger_logits > 0.)) & (biaffine_logits > 0))
                        )
                        ),
                        dim=1,
                    )

            elif do_tagging:
                (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                    spans_logits,
                    topk=self.max_fragments_count,
                    mask=spans_mask & viterbi_mask,
                    dim=1,
                )
            elif do_biaffine:
                (fragments_begin, fragments_end, fragments_label), fragments_mask = multi_dim_topk(
                    spans_logits,
                    topk=self.max_fragments_count,
                    mask=spans_mask & (biaffine_logits.sigmoid() > self.threshold),
                    dim=1,
                )
            else:
                raise NotImplementedError()

            fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
            fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
            fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
            fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)
            fragments_logit = spans_logits[torch.arange(n_samples, device=device).unsqueeze(1), fragments_begin, fragments_end, fragments_label]
        else:
            if do_biaffine and not do_tagging:
                best_begin_end_scores, best_begin_end_labels = spans_logits.log_softmax(-1).max(-1)
                (fragments_begin, fragments_end), fragments_mask = multi_dim_topk(
                    best_begin_end_scores,
                    topk=self.max_fragments_count,
                    mask=spans_mask.any(-1) & (best_begin_end_labels > 0),
                    dim=1,
                )
                fragments_label = best_begin_end_labels[torch.arange(n_samples, device=device)[:, None], fragments_begin, fragments_end] - 1

                fragments_sorter = torch.argsort(fragments_begin.masked_fill(~fragments_mask, 100000), dim=-1)
                fragments_begin = gather(fragments_begin, dim=-1, index=fragments_sorter)
                fragments_end = gather(fragments_end, dim=-1, index=fragments_sorter)
                fragments_label = gather(fragments_label, dim=-1, index=fragments_sorter)
                fragments_logit = best_begin_end_scores[torch.arange(n_samples, device=device).unsqueeze(1), fragments_begin, fragments_end]

        return {
            "flat_spans_label": fragments_label,
            "flat_spans_begin": fragments_begin,
            "flat_spans_end": fragments_end,
            "flat_spans_mask": fragments_mask,
            "flat_spans_logit": fragments_logit,

            "spans_logits": spans_logits.view(n_repeat, n_samples, *spans_logits.shape[1:]),
            "biaffine_logits": biaffine_logits.view(n_repeat, n_samples, *spans_logits.shape[1:]) if biaffine_logits is not None else None,
            "spans_target": spans_target,
            "spans_mask": spans_mask,
            "crf_tag_logprobs": crf_tag_logprobs.view(n_repeat, n_samples, *crf_tag_logprobs.shape[1:]) if crf_tag_logprobs is not None else None,
            "crf_tag_logits": crf_tag_logits.view(n_repeat, n_samples, *crf_tag_logprobs.shape[1:]) if crf_tag_logprobs is not None else None,
        }


class BiaffineLoss(torch.nn.Module):
    def __init__(self, multi_label=True, true_bounds_only=False):
        super().__init__()
        self.multi_label = multi_label
        self.true_bounds_only = true_bounds_only

    def forward(self, scores, res, batch):
        mask = res["spans_mask"]
        target = res["spans_target"]
        if self.true_bounds_only:
            mask = mask & target.any(-2, keepdim=True) & target.any(-3, keepdim=True)
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

        return loss


class TaggerLoss(torch.nn.Module):
    def __init__(self, _crf=None, marginal=False):
        super().__init__()
        self.marginal = marginal
        self._crf = _crf

    def forward(self, res, batch):
        crf_tag_logprobs = res.get("crf_tag_logprobs", None)
        crf_tag_logits = res.get("crf_tag_logits", None)
        raw_logits = res.get("raw_logits", None)

        scores = crf_tag_logprobs if crf_tag_logprobs is not None else crf_tag_logits
        if scores.ndim == 5:
            shape = scores.shape[1:-1]
        else:
            shape = scores.shape[:-1]
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
        tags = torch.maximum(torch.maximum(i_tags, bl_tags), u_tags).to(scores.device)
        tags = (tags if scores.ndim == 4 else tags.unsqueeze(0).repeat_interleave(scores.shape[0], dim=0))
        if self._crf is not None:
            if self.marginal:
                loss = nll(
                    crf_tag_logprobs,
                    tags,
                    reduction='none',
                ).masked_fill(~mask.unsqueeze(1), 0).mean(-2).sum()
            else:
                tags_1hot = F.one_hot(tags, 5).bool()
                loss = -self._crf(
                    crf_tag_logits.reshape(-1, crf_tag_logits.shape[-2], 5),
                    mask.unsqueeze(1).repeat_interleave(crf_tag_logits.shape[-3], dim=1).view(-1, crf_tag_logits.shape[-2]),
                    tags_1hot.reshape(-1, crf_tag_logits.shape[-2], 5),
                ).view(*crf_tag_logits.shape[:-2]).mean(-1).sum()
        else:
            loss = cross_entropy_with_logits(
                crf_tag_logits,
                tags,
                reduction='none',
            ).masked_fill(~mask.unsqueeze(1), 0).mean(-2).sum()

        # loss = loss + bce_with_logits(
        #    raw_logits[..., 0],
        #    tags > 0,
        #    reduction='none',
        # ).masked_fill(~mask.unsqueeze(1), 0).mean(-2).sum()

        return loss
