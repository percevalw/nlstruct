from typing import Optional

import torch
import torch.nn.functional as F

from nlstruct.models.common import Pooler, register

IMPOSSIBLE = -1000000


@register("span_embedding")
class SpanEmbedding(torch.nn.Module):
    def __init__(self, input_size, n_labels, mode="max", n_heads=None, do_label_embedding=True, do_value_proj=False, label_dropout_p=0.1):
        super().__init__()
        self.mode = mode
        if mode != "bounds_cat":
            self.pooler = Pooler(mode=mode, input_size=input_size, n_heads=n_heads, do_value_proj=do_value_proj)
        if do_label_embedding:
            self.label_embedding = torch.nn.Embedding(n_labels, input_size).weight
        self.label_dropout_p = label_dropout_p
        self.output_size = input_size

    def forward(self, words_embed, spans_begin, spans_end, spans_label):
        n_samples, n_words, size = words_embed.shape
        if self.mode == "bounds_cat":
            # return words_embed[torch.arange(n_samples)[:, None], spans_begin]
            # noinspection PyArgumentList
            pooled = torch.cat([
                words_embed[..., size // 2:][torch.arange(n_samples)[:, None], spans_begin],
                words_embed[..., :size // 2][torch.arange(n_samples)[:, None], spans_end],
            ], size=-1)
        else:
            pooled = self.pooler(words_embed, (spans_begin, spans_end + 1))

        if hasattr(self, 'label_embedding'):
            label_embeds = torch.einsum('ld,...l->...d', self.label_embedding, spans_label.float())
            if self.training:
                label_embeds = label_embeds.masked_fill(torch.rand_like(label_embeds[..., [0]]) < self.label_dropout_p, 0)
            pooled = pooled + label_embeds

        return pooled


@register("qualification", do_not_serialize=["ner_label_to_qualifiers"])
class Qualification(torch.nn.Module):
    ENSEMBLE = "ensemble_qualification"

    def __init__(
          self,
          input_size: int,
          n_qualifiers: Optional[int],
          ner_label_to_qualifiers: Optional[torch.BoolTensor],
          qualifiers_combinations: Optional[torch.BoolTensor],
          pooler_mode: str = "max",
          classifier_mode: str = "dot",
    ):
        super().__init__()

        if classifier_mode == "dot":
            self.classifier = torch.nn.Linear(input_size, n_qualifiers)
        else:
            raise Exception("Only scalar product is supported for qualifier classification.")
        self.pooler = Pooler(mode=pooler_mode)

        if ner_label_to_qualifiers is not None:
            self.register_buffer('ner_label_to_qualifiers', ner_label_to_qualifiers)  # n_ner_label * n_qualifiers
        else:
            self.ner_label_to_qualifiers = None
        if qualifiers_combinations is not None:
            self.register_buffer('qualifiers_combinations', qualifiers_combinations)  # n_combinations * n_qualifiers
        else:
            self.qualifiers_combinations = None

    def forward(self, words_embed, spans_begin, spans_end, spans_label, spans_mask, batch=None, return_loss=True):
        pooled = self.pooler(words_embed, (spans_begin, spans_end + 1))

        scores = self.classifier(pooled)
        if self.ner_label_to_qualifiers is not None:
            span_allowed_qualifiers = self.ner_label_to_qualifiers[spans_label]  # [b]atch_size * [e]ntities * [q]ualifiers
            scores = scores.masked_fill(~span_allowed_qualifiers, IMPOSSIBLE)  # [b]atch_size * [e]ntities * [q]ualifiers
        if self.qualifiers_combinations is not None:
            combination_scores = torch.einsum('beq,qc->bec')
        else:
            combination_scores = None

        if return_loss:
            gold_labels = batch['entities_label']  # [b]atch * [e]nts * [q]ualifiers
            if self.qualifiers_combinations is not None:
                gold_combinations = (
                    gold_labels[..., None, :] ==  # [b]atch * [e]nts * 1 * [q]ualifiers
                    self.qualifiers_combinations  # [c]ombinations * [q]ualifiers
                ).all(-1)  # [b]atch * [e]nts * [c]ombinations
                assert gold_combinations.any(-1)[spans_mask].all()
                mention_err = torch.einsum(
                    'bec,bec->be',  # [b]atch * [e]nts * [c]ombinations
                    -combination_scores.log_softmax(-1),
                    gold_combinations.float()  # [b]atch * [e]nts * [c]ombinations
                )  # [b]atch * [e]nts * [c]ombinations
                loss = mention_err[spans_mask].sum()
            else:
                mention_err = F.binary_cross_entropy_with_logits(
                    scores,
                    gold_labels.float(),
                    reduction='none',
                ).sum(-1)  # [b]atch * [e]nts
                loss = mention_err[spans_mask].sum()
            pred = batch['entities_label']
        else:
            loss = mention_err = gold_outputs = None
            if combination_scores is not None:
                pred = self.qualifiers_combinations[combination_scores.argmax(-1)]
            else:
                pred = scores > 0

        return {
            "prediction": pred,
            "loss": loss,
        }


@register("ensemble_qualification", do_not_serialize=["ner_label_to_qualifiers"])
class EnsembleQualification(torch.nn.Module):
    def __init__(
          self,
          models,
    ):
        super().__init__()

        self.register_buffer('ner_label_to_qualifiers', models[0].ner_label_to_qualifiers)
        self.register_buffer('qualifiers_combinations', models[0].qualifiers_combinations)
        self.poolers = torch.nn.ModuleList([m.pooler for m in models])
        self.classifiers = torch.nn.ModuleList([m.classifier for m in models])

    def forward(self, ensemble_words_embed, spans_begin, spans_end, spans_label, spans_mask, batch=None, return_loss=True):

        ensemble_scores = []
        for pooler, classifier, words_embed in zip(self.poolers, self.classifiers, ensemble_words_embed):
            pooled = pooler(words_embed, (spans_begin, spans_end + 1))
            scores = classifier(pooled)
            ensemble_scores.append(scores)
        scores = torch.stack(ensemble_scores, 0).mean(0)

        if self.ner_label_to_qualifiers is not None:
            span_allowed_qualifiers = self.ner_label_to_qualifiers[spans_label]  # [b]atch_size * [e]ntities * [q]ualifiers
            scores = scores.masked_fill(~span_allowed_qualifiers, IMPOSSIBLE)  # [b]atch_size * [e]ntities * [q]ualifiers
        if self.qualifiers_combinations is not None:
            combination_scores = torch.einsum('beq,qc->bec')
        else:
            combination_scores = None

        if return_loss:
            loss = None
            pred = batch['entities_label']
        else:
            loss = mention_err = gold_outputs = None
            if combination_scores is not None:
                pred = self.qualifiers_combinations[combination_scores.argmax(-1)]
            else:
                pred = scores > 0

        return {
            "prediction": pred,
            "loss": loss,
        }
