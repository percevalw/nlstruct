import enum
import os
import shutil
import tempfile
import urllib.request

import pytorch_lightning as pl
import torch
import tqdm as tqdm
from sklearn.datasets._base import _sha256


class NetworkLoadMode(enum.Enum):
    AUTO = 0
    CACHE_ONLY = 1
    FETCH_ONLY = 2


def download_file(url, path, checksum=None):
    bar = None

    def reporthook(_, chunk, total):
        nonlocal bar
        if bar is None:
            bar = tqdm.tqdm(desc=os.path.basename(url), total=total, unit="B", unit_scale=True)
        bar.update(min(chunk, bar.total - bar.n))

    urllib.request.urlretrieve(url, filename=path, reporthook=reporthook)
    if bar is not None:
        bar.close()
    if checksum is not None:
        computed_checksum = _sha256(path)
        if computed_checksum != checksum:
            raise IOError("{} has an SHA256 checksum ({}) "
                          "differing from expected ({}), "
                          "file may be corrupted.".format(path, computed_checksum,
                                                          checksum))


def ensure_files(path, remotes, mode):
    os.makedirs(path, exist_ok=True)
    file_paths = []
    for remote in remotes:
        file_path = os.path.join(path, remote.filename)
        file_exist = os.path.exists(str(file_path))

        tmp_dir = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_file_path = os.path.join(tmp_dir, remote.filename)

        if not file_exist and mode == NetworkLoadMode.CACHE_ONLY:
            raise IOError("Could not find cached file {} in {}".format(file_path, tmp_file_path, path))
        elif mode == NetworkLoadMode.FETCH_ONLY or (not file_exist and mode == NetworkLoadMode.AUTO):
            download_file(remote.url, tmp_file_path, remote.checksum)
            shutil.copy(tmp_file_path, file_path)
            os.remove(tmp_file_path)
        file_paths.append(file_path)
    return file_paths


class BaseDataset(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data)


class NERDataset(BaseDataset):
    def describe(self, as_dataframe=True):
        counts = {
            split: {"documents": 0, "entities": 0, "unique_entities_label": set(), "unique_entities_concept": set(), "unique_entities_text": set(), "fragments": 0, "fragmented_entities": 0}
            for split in ["train", "val", "test"]
        }
        for split, split_docs in (("train", self.train_data), ("val", self.val_data), ("test", self.test_data)):
            for doc in split_docs:
                counts[split]["documents"] += 1
                for entity in doc["entities"]:
                    text = ""
                    for fragment in entity["fragments"]:
                        text = text + " " + doc["text"][fragment["begin"]:fragment["end"]]
                    text = " ".join(text.split()).strip(" ")
                    counts[split]["entities"] += 1

                    if "label" in entity:
                        if isinstance(entity["label"], (list, tuple)):
                            for label in entity["label"]:
                                counts[split]["unique_entities_label"].add(label)
                        else:
                            counts[split]["unique_entities_label"].add(entity["label"])

                    if "concept" in entity:
                        if isinstance(entity["concept"], (list, tuple)):
                            for concept in entity["concept"]:
                                counts[split]["unique_entities_concept"].add(concept)
                        else:
                            counts[split]["unique_entities_concept"].add(entity["concept"])
                    if len(entity["fragments"]) > 1:
                        counts[split]["fragmented_entities"] += 1
                    counts[split]["unique_entities_text"].add(text)
                    counts[split]["fragments"] += len(entity["fragments"])
        counts = {
            split: {key: len(value) if hasattr(value, '__len__') and not isinstance(value, str) else value for key, value in doc_counts.items()}
            for split, doc_counts in counts.items()
        }
        if as_dataframe:
            import pandas as pd
            return pd.DataFrame.from_dict(counts, orient='index')
        return counts
