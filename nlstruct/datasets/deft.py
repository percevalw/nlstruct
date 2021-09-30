import os

from nlstruct.datasets.brat import BRATDataset


class DEFT(BRATDataset):
    def __init__(self, path, val=0.2, kept_entity_label=None, dropped_entity_label=("duree", "frequence"), seed=False, preprocess_fn=None):
        super().__init__(
            train=os.path.join(path, "t3-appr/"),
            test=os.path.join(path, "t3-test/"),
            val=val,
            kept_entity_label=kept_entity_label,
            dropped_entity_label=dropped_entity_label,
            seed=seed,
            preprocess_fn=preprocess_fn,
        )
