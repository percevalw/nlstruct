import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets._base import RemoteFileMetadata

from nlstruct.collections.dataset import Dataset
from nlstruct.dataloaders.brat import load_from_brat
from nlstruct.environment.path import root
from nlstruct.train import fork_rng
from nlstruct.utils.network import ensure_files, NetworkLoadMode

remote = RemoteFileMetadata(
    url="https://quaerofrenchmed.limsi.fr/QUAERO_FrenchMed_brat.zip",
    checksum="2cf8b5715d938fdc1cd02be75c4eaccb5b8ee14f4148216b8f9b9e80b2445c10",
    filename="QUAERO_FrenchMed_brat.zip")


def load_quaero(resource_path="quaero", version="2016", dev_split=0.2, seed=42):
    """
    Loads the Quaero dataset
    
    Parameters
    ----------
    resource_path: str
        Location of the Quaero files
    version: str
        Version to load, either '2015' or '2016'
    dev_split: float
        Will only be used if version is '2015' since no dev set was defined for this version
    seed: int
        Will only be used if version is '2015' since no dev set was defined for this version

    Returns
    -------
    Dataset
    """
    assert version in ("2015", "2016")
    assert 0 <= dev_split <= 1

    path = root.resource(resource_path)
    [file] = ensure_files(path, [remote], mode=NetworkLoadMode.AUTO)
    zip_ref = zipfile.ZipFile(path / "QUAERO_FrenchMed_brat.zip", "r")
    zip_ref.extractall(path)
    zip_ref.close()
    dataset = Dataset.concat([
        load_from_brat(path / "QUAERO_FrenchMed/corpus/train/EMEA", doc_attributes={"source": "EMEA", "split": "train"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/train/MEDLINE", doc_attributes={"source": "MEDLINE", "split": "train"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/dev/EMEA", doc_attributes={"source": "EMEA", "split": "dev"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/dev/MEDLINE", doc_attributes={"source": "MEDLINE", "split": "dev"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/test/EMEA", doc_attributes={"source": "EMEA", "split": "test"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/test/MEDLINE", doc_attributes={"source": "MEDLINE", "split": "test"}),
    ])

    if version == "2015":
        dataset = dataset.query('split != "test"', propagate=True).copy()
        dataset["docs"].loc[dataset["docs"]["split"] == "dev", "split"] = "test"
        with fork_rng(seed):
            perm = np.random.permutation(np.flatnonzero(dataset["docs"]["split"] == "train"))
            dataset["docs"] = dataset["docs"].reset_index(drop=True)
            dataset["docs"].loc[perm[:int(len(perm) * dev_split)], "split"] = "dev"
            dataset["docs"].loc[perm[int(len(perm) * dev_split):], "split"] = "train"

    labels = dataset["comments"].rename({"comment_id": "label_id", "comment": "cui"}, axis=1)
    labels["cui"] = labels["cui"].apply(lambda x: x.split(" "))
    labels = labels.nlstruct.flatten("cui_id", tile_index=True)
    return Dataset(
        **dataset[["docs", "fragments", "mentions"]],
        labels=labels[["doc_id", "mention_id", "cui_id", "cui"]],
    )


def describe_quaero(dataset):
    stats = []
    for source in ("EMEA", "MEDLINE"):
        for split in ["train", "dev", "test"]:
            dataset_split = dataset.query(f'split == "{split}" and source == "{source}"', propagate=True)
            res = {
                "split": split,
                "source": source,
                "files": len(dataset_split["docs"]),
                "mentions": len(dataset_split["mentions"]),
                "unique_mentions": len(dataset_split["mentions"].drop_duplicates("text")),
                "labels": len(dataset_split["labels"]["cui"].drop_duplicates()),
            }
            stats.append(res)
    stats = pd.DataFrame(stats).set_index(["source", "split"]).stack().unstack(0).unstack(0)
    stats = stats[[('EMEA', "train"), ('EMEA', "dev"), ('EMEA', "test"), ('MEDLINE', "train"), ('MEDLINE', "dev"), ('MEDLINE', "test")]]
    return stats
