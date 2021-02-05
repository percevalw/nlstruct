import torch
from pytorch_lightning.metrics import Metric


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

    def update(self, preds: torch.Tensor, target: torch.Tensor):
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
