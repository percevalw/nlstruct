from pyner.dataloaders.brat import BRATDataset


class DEFT(BRATDataset):
    def __init__(self, train, test=None, val=0.2, dropped_entity_label=("duree", "frequence"), seed=False):
        super().__init__(train=train, test=test, val=val, dropped_entity_label=dropped_entity_label, seed=seed)
