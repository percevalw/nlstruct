import torch
from pytorch_lightning.metrics import Metric

from pyner.data_utils import regex_tokenize, split_spans
from pyner.registry import register
from pyner.torch_utils import pad_to_tensor


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
        self.word_regex = word_regex
        self.binarize_label_threshold = binarize_label_threshold
        self.binarize_tag_threshold = binarize_tag_threshold
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

        pred_doc_entities = [entity for entity in pred_doc["entities"]
                             if self.filter_entities is None
                             or any(label in self.filter_entities for label in (entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],)))]
        gold_doc_entities = [entity for entity in gold_doc["entities"]
                             if self.filter_entities is None
                             or any(label in self.filter_entities for label in (entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],)))]
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
            # *(("{}:{}".format(att["name"], att["value"]) for att in entity["attributes"]) if self.eval_attributes else ())]
            gold_entities_labels[entity_idx] = [label in entity_labels for label in all_entity_labels]

        gold_tags = pad_to_tensor(gold_tags)
        pred_tags = pad_to_tensor(pred_tags)
        gold_entities_labels = pad_to_tensor(gold_entities_labels)
        pred_entities_labels = pad_to_tensor(pred_entities_labels)

        score = 0.
        tag_match_scores = 2 * torch.einsum("pkt,gkt->pg", pred_tags.float(), gold_tags.float()) / (
              pred_tags.float().sum(-1).sum(-1).unsqueeze(1) +
              gold_tags.float().sum(-1).sum(-1).unsqueeze(0)
        ).clamp_min(1)

        if self.binarize_tag_threshold is not False:
            tag_match_scores = (tag_match_scores >= self.binarize_tag_threshold).float()
        label_match_scores = 2 * torch.einsum("pk,gk->pg", pred_entities_labels.float(), gold_entities_labels.float()) / (
              pred_entities_labels.float().sum(-1).unsqueeze(1) +
              gold_entities_labels.float().sum(-1).unsqueeze(0)
        ).clamp_min(1)
        if self.binarize_label_threshold is not False:
            label_match_scores = (label_match_scores >= self.binarize_label_threshold).float()
        match_scores = label_match_scores * tag_match_scores

        if return_match_scores:
            return match_scores

        pred_count, gold_count = len(pred_doc_entities), len(gold_doc_entities)
        for pred_idx in range(match_scores.shape[0]):
            if self.joint_matching:
                pred_idx = match_scores.max(-1).values.argmax()
            # pred_idx = gold_scores.argmax()
            gold_idx = match_scores[pred_idx].argmax()
            best_score = max(0, match_scores[pred_idx, gold_idx])
            if best_score > 0:
                score += best_score
                match_scores[:, gold_idx] = 0.
                match_scores[pred_idx, :] = 0.
        return float(score), pred_count, gold_count

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