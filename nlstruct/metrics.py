import torch
from pytorch_lightning.metrics import Metric

from nlstruct.data_utils import regex_tokenize, split_spans, dedup
from nlstruct.registry import register
from nlstruct.torch_utils import pad_to_tensor
from collections import defaultdict


class MetricsCollection(torch.nn.ModuleDict):
    def forward(self, pred_docs, gold_docs):
        results = {}
        for key, module in self.items():
            module.reset()
            module(pred_docs, gold_docs)
            results[key] = {k: float(v) for k, v in module.compute().items()}
        return results


@register("precision_recall_f1")
class PrecisionRecallF1Metric(Metric):
    def __init__(
          self,
          compute_on_step=True,
          dist_sync_on_step=False,
          process_group=None,
          dist_sync_fn=None,
          prefix="",
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.prefix = prefix
        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("gold_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        def rec(p, g):
            if (len(p) and isinstance(p[0], tuple)) or (len(g) and isinstance(g[0], tuple)):
                self.true_positive += len(set(p) & set(g))
                self.pred_count += len(set(p))
                self.gold_count += len(set(g))
            else:
                for p_list, g_list in zip(p, g):
                    rec(p_list, g_list)

        rec(preds, target)

    def compute(self):
        """
        Computes accuracy over state.
        """
        if self.gold_count == 0 and self.pred_count == 0:
            return {
                self.prefix + "precision": 1,
                self.prefix + "recall": 1,
                self.prefix + "f1": 1,
            }
        return {
            self.prefix + "precision": (self.true_positive) / max(1, self.pred_count),
            self.prefix + "recall": (self.true_positive) / max(1, self.gold_count),
            self.prefix + "f1": (self.true_positive * 2) / (self.pred_count + self.gold_count),
        }


def entity_match_filter(labels, matcher):
    labels = labels if isinstance(labels, (tuple, list)) else (labels,)
    if isinstance(matcher, (tuple, list)):
        return any(label in matcher for label in labels)
    eval_locals = defaultdict(lambda: False)
    eval_locals.update({k: True for k in labels})
    return eval(matcher, None, eval_locals)


@register("dem")
class DocumentEntityMetric(Metric):
    def __init__(
          self,
          word_regex=r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]',
          filter_entities=None,
          joint_matching=False,
          binarize_label_threshold=1.,
          binarize_tag_threshold=0.5,
          eval_attributes=False,
          eval_fragments_label=False,
          compute_on_step=True,
          dist_sync_on_step=False,
          process_group=None,
          dist_sync_fn=None,
          explode_fragments=False,
          prefix="",
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.joint_matching = joint_matching
        self.filter_entities = filter_entities
        self.prefix = prefix
        self.eval_attributes = eval_attributes
        self.eval_fragments_label = eval_fragments_label
        self.explode_fragments = explode_fragments
        self.word_regex = word_regex
        self.binarize_label_threshold = float(binarize_label_threshold) if binarize_label_threshold is not False else binarize_label_threshold
        self.binarize_tag_threshold = float(binarize_tag_threshold) if binarize_tag_threshold is not False else binarize_tag_threshold
        self.add_state("true_positive", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pred_count", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("gold_count", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds, targets):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        for pred_doc, gold_doc in zip(preds, targets):
            tp, pc, gc = self.compare_two_samples(pred_doc, gold_doc)
            self.true_positive += tp
            self.pred_count += pc
            self.gold_count += gc

    def compare_two_samples(self, pred_doc, gold_doc, return_match_scores=False):
        assert pred_doc["text"] == gold_doc["text"]
        pred_doc_entities = list(pred_doc["entities"])
        gold_doc_entities = list(gold_doc["entities"])

        pred_doc_entities = [entity for entity in pred_doc_entities
                             if self.filter_entities is None
                             or entity_match_filter(entity["label"], self.filter_entities)]
        gold_doc_entities = [entity for entity in gold_doc_entities
                             if self.filter_entities is None
                             or entity_match_filter(entity["label"], self.filter_entities)]
        if self.explode_fragments:
            pred_doc_entities = [{"label": f.get("label", "main"), "fragments": [f]} for f in
                                 dedup((f for entity in pred_doc_entities for f in entity["fragments"]), key=lambda x: (x['begin'], x['end'], x.get('label', None)))]
            gold_doc_entities = [{"label": f.get("label", "main"), "fragments": [f]} for f in
                                 dedup((f for entity in gold_doc_entities for f in entity["fragments"]), key=lambda x: (x['begin'], x['end'], x.get('label', None)))]

        words = regex_tokenize(gold_doc["text"], reg=self.word_regex, do_unidecode=True, return_offsets_mapping=True)

        all_fragment_labels = set()
        all_entity_labels = set()
        fragments_begin = []
        fragments_end = []
        pred_entities_fragments = []
        for entity in pred_doc_entities:
            all_entity_labels.update(set(entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],)))
            #                                          *(("{}:{}".format(att["name"], att["label"]) for att in entity["attributes"]) if self.eval_attributes else ()))))
            pred_entities_fragments.append([])
            for fragment in entity["fragments"]:
                pred_entities_fragments[-1].append(len(fragments_begin))
                fragments_begin.append(fragment["begin"])
                fragments_end.append(fragment["end"])
                all_fragment_labels.add(fragment["label"] if self.eval_fragments_label else "main")

        gold_entities_fragments = []
        for entity in gold_doc_entities:
            all_entity_labels.update(set(entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],)))
            #                                          *(("{}:{}".format(att["name"], att["value"]) for att in entity["attributes"]) if self.eval_attributes else ()))))
            gold_entities_fragments.append([])
            for fragment in entity["fragments"]:
                gold_entities_fragments[-1].append(len(fragments_begin))
                fragments_begin.append(fragment["begin"])
                fragments_end.append(fragment["end"])
                all_fragment_labels.add(fragment["label"] if self.eval_fragments_label else "main")
        all_fragment_labels = list(all_fragment_labels)
        all_entity_labels = list(all_entity_labels)
        if len(all_fragment_labels) == 0:
            all_fragment_labels = ["main"]
        if len(all_entity_labels) == 0:
            all_entity_labels = ["main"]

        fragments_begin, fragments_end = split_spans(fragments_begin, fragments_end, words["begin"], words["end"])
        pred_entities_labels = [[False] * len(all_entity_labels)] * max(len(pred_doc_entities), 1)
        pred_tags = [[[False] * len(words["begin"]) for _ in range(len(all_fragment_labels))] for _ in range(max(len(pred_doc_entities), 1))]  # n_entities * n_token_labels * n_tokens
        gold_entities_labels = [[False] * len(all_entity_labels)] * max(len(gold_doc_entities), 1)
        gold_entities_optional_labels = [[False] * len(all_entity_labels)] * max(len(gold_doc_entities), 1)
        gold_tags = [[[False] * len(words["begin"]) for _ in range(len(all_fragment_labels))] for _ in range(max(len(gold_doc_entities), 1))]  # n_entities * n_token_labels * n_tokens

        for entity_idx, (entity_fragments, entity) in enumerate(zip(pred_entities_fragments, pred_doc_entities)):
            for fragment_idx, fragment in zip(entity_fragments, entity["fragments"]):
                begin = fragments_begin[fragment_idx]
                end = fragments_end[fragment_idx]
                label = all_fragment_labels.index(fragment["label"] if self.eval_fragments_label else "main")
                pred_tags[entity_idx][label][begin:end] = [True] * (end - begin)
            entity_labels = list(entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],))
            # *(("{}:{}".format(att["name"], att["label"]) for att in entity["attributes"]) if self.eval_attributes else ())]
            pred_entities_labels[entity_idx] = [label in entity_labels for label in all_entity_labels]

        for entity_idx, (entity_fragments, entity) in enumerate(zip(gold_entities_fragments, gold_doc_entities)):
            for fragment_idx, fragment in zip(entity_fragments, entity["fragments"]):
                begin = fragments_begin[fragment_idx]
                end = fragments_end[fragment_idx]
                label = all_fragment_labels.index(fragment["label"] if self.eval_fragments_label else "main")
                gold_tags[entity_idx][label][begin:end] = [True] * (end - begin)
            entity_labels = list(entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],))
            entity_optional_labels = entity.get("complete_labels", entity["label"])
            if not isinstance(entity_optional_labels, (tuple, list)):
                entity_optional_labels = [entity_optional_labels]
            # *(("{}:{}".format(att["name"], att["value"]) for att in entity["attributes"]) if self.eval_attributes else ())]
            gold_entities_labels[entity_idx] = [label in entity_labels for label in all_entity_labels]
            gold_entities_optional_labels[entity_idx] = [label in entity_optional_labels for label in all_entity_labels]

        gold_tags = pad_to_tensor(gold_tags)
        pred_tags = pad_to_tensor(pred_tags)
        gold_entities_labels = pad_to_tensor(gold_entities_labels)
        gold_entities_optional_labels = pad_to_tensor(gold_entities_optional_labels)
        pred_entities_labels = pad_to_tensor(pred_entities_labels)

        # score = 0.
        tag_denom_match_scores = (
              pred_tags.float().sum(-1).sum(-1).unsqueeze(1) +
              gold_tags.float().sum(-1).sum(-1).unsqueeze(0)
        )
        tag_match_scores = 2 * torch.einsum("pkt,gkt->pg", pred_tags.float(), gold_tags.float()) / tag_denom_match_scores.clamp_min(1)
        #tag_match_scores[(tag_denom_match_scores == 0.) & (tag_match_scores == 0.)] = 1.

        score = 0.

        # tag_denom_match_scores = (
        #      pred_tags.float().sum(-1).unsqueeze(1) + # pkt -> p:k
        #      gold_tags.float().sum(-1).unsqueeze(0)   # gkt -> :gk
        # )
        # tag_match_scores = (2 * torch.einsum("pkt,gkt->pgk", pred_tags.float(), gold_tags.float()) / tag_denom_match_scores.clamp_min(1)) > 0.5
        # tag_match_scores[tag_denom_match_scores == 0.] = 1.

        # label_match_scores = 2 * torch.einsum("pk,gk->pg", pred_entities_labels.float(), gold_entities_labels.float()) / (
        #      pred_entities_labels.float().sum(-1).unsqueeze(1) +
        #      gold_entities_labels.float().sum(-1).unsqueeze(0)
        # ).clamp_min(1)

        label_match_precision = torch.einsum("pk,gk->pg", pred_entities_labels.float(), gold_entities_optional_labels.float()) / pred_entities_labels.float().sum(-1).unsqueeze(1).clamp_min(1.)
        label_match_recall = torch.einsum("pk,gk->pg", pred_entities_labels.float(), gold_entities_labels.float()) / gold_entities_labels.float().sum(-1).unsqueeze(0).clamp_min(1.)
        label_match_scores = 2 / (1. / label_match_precision + 1. / label_match_recall)
        match_scores = label_match_scores * tag_match_scores
        if self.binarize_tag_threshold is not False:
            tag_match_scores = (tag_match_scores >= self.binarize_tag_threshold).float()
        if self.binarize_label_threshold is not False:
            label_match_scores = (label_match_scores >= self.binarize_tag_threshold).float()
        effective_scores = tag_match_scores * label_match_scores

        pred_count, gold_count = len(pred_doc_entities), len(gold_doc_entities)
        matched_scores = torch.zeros_like(match_scores) - 1.
        for pred_idx in range(match_scores.shape[0]):
            if self.joint_matching:
                pred_idx = match_scores.max(-1).values.argmax()
            gold_idx = match_scores[pred_idx].argmax()

            #match_score = match_scores[pred_idx, gold_idx].float()
            match_score = match_scores[pred_idx, gold_idx].float()
            effective_score = effective_scores[pred_idx, gold_idx].float()
            matched_scores[pred_idx, gold_idx] = max(matched_scores[pred_idx, gold_idx], effective_score)
            if match_score >= 0 and effective_score > 0:
                score += effective_score
                match_scores[:, gold_idx] = -1
                match_scores[pred_idx, :] = -1

        if return_match_scores:
            print(float(score), pred_count, gold_count)
            return matched_scores

        return float(score), pred_count, gold_count

    def compute(self):
        """
        Computes accuracy over state.
        """
        if self.gold_count == 0 and self.pred_count == 0:
            return {
                self.prefix + "tp": 0,
                self.prefix + "precision": 1,
                self.prefix + "recall": 1,
                self.prefix + "f1": 1,
            }
        return {
            self.prefix + "tp": self.true_positive,
            self.prefix + "precision": self.true_positive / max(1, self.pred_count),
            self.prefix + "recall": self.true_positive / max(1, self.gold_count),
            self.prefix + "f1": (self.true_positive * 2) / (self.pred_count + self.gold_count),
        }
