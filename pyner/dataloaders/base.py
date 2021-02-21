import enum
import os
import re
import shutil
import tempfile
import urllib.request
from collections import defaultdict
from copy import copy
from itertools import chain

import pytorch_lightning as pl
import torch
from sklearn.datasets._base import _sha256
from tqdm import tqdm
from unidecode import unidecode


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
    def __init__(self, train_data, val_data, test_data):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data)


class Terminology:
    def __init__(self,
                 concept_synonym_pairs,
                 concept_mapping={},
                 concept_semantic_types={},
                 build_synonym_concepts_mapping=False,
                 synonym_preprocess_fn=None,
                 do_unidecode=False,
                 subs=()):
        if do_unidecode or len(subs):
            res = {}
            for concept, synonyms in self.concept_synonyms:
                concept_res = []
                for synonym in synonyms:
                    if synonym_preprocess_fn is not None:
                        synonym = synonym_preprocess_fn(synonym)
                    if do_unidecode:
                        synonym = unidecode(synonym)
                    if subs:
                        for pattern, replacement in subs:
                            synonym = re.sub(pattern, replacement, synonym)
                    concept_res.append(synonym)
                res[concept] = concept_res
            self.concept_synonyms = res
        else:
            self.concept_synonyms = dict(concept_synonym_pairs)
        self.concept_mapping = concept_mapping
        self.concept_semantic_types = concept_semantic_types
        if build_synonym_concepts_mapping:
            self.build_synonym_concepts_mapping_()

    @property
    def all_preferred_synonyms(self):
        return [synonyms[0] for synonyms in self.concept_synonyms.values()]

    def __getitem__(self, concept):
        return self.get_concept_synonyms(concept)

    def get_concept_synonyms(self, concept, missing='raise'):
        try:
            return self.concept_synonyms[concept]
        except KeyError:
            try:
                return self.concept_synonyms[self.map_concept(concept)]
            except KeyError:
                pass
        if missing == "raise":
            raise KeyError(f"Could not find synonyms for {concept}")
        if missing == "null":
            return None

    def get_concept_preferred_synonym(self, concept, missing='raise'):
        try:
            return self.concept_synonyms[concept][0]
        except KeyError:
            try:
                return self.concept_synonyms[self.map_concept(concept)][0]
            except KeyError:
                pass
        if missing == "raise":
            raise KeyError(f"Could not find preferred synonym for {concept}")
        if missing == "null":
            return None

    def map_concept(self, concept, missing='raise'):
        try:
            return self.concept_mapping[concept]
        except KeyError:
            pass
        if missing == "raise":
            raise KeyError(f"Could not find concept mapping for {concept}")
        if missing == "null":
            return None

    def get_concept_semantic_type(self, concept, missing='raise'):
        try:
            return self.concept_semantic_types[concept]
        except KeyError:
            try:
                return self.concept_semantic_types[self.map_concept(concept)]
            except KeyError:
                pass
        if missing == "raise":
            raise KeyError(f"Could not find coarse label for {concept}")
        if missing == "null":
            return None

    @property
    def concepts(self):
        return list(self.concept_synonyms.keys())

    @property
    def synonyms(self):
        return list(chain.from_iterable(self.concept_synonyms.values()))

    @property
    def coarse_labels(self):
        return sorted(set(self.concept_semantic_types.values()))

    @property
    def preferred_synonyms(self):
        return [synonyms[0] for synonyms in self.concept_synonyms.values()]

    def build_synonym_concepts_mapping_(self):
        synonym_concepts = defaultdict(lambda: [])
        for concept, synonyms in tqdm(self.concept_synonyms.items()):
            for synonym in synonyms:
                synonym_concepts[synonym].append((concept, synonyms[0]))
        self.synonym_concepts = dict(synonym_concepts)

    def get_synonym_concepts(self, synonym):
        assert self.synonym_concepts is not None, "You must call build_synonym_concepts_mapping() once before calling get_synonym_concepts"
        return [pair[0] for pair in self.synonym_concepts[synonym]]

    def get_synonym_preferred(self, synonym):
        assert self.synonym_concepts is not None, "You must call build_synonym_concepts_mapping() once before calling get_synonym_preferred"
        return [pair[1] for pair in self.synonym_concepts[synonym]]

    def __or__(self, other):
        concept_synonym_pairs = defaultdict(lambda: {})

        for cui, synonyms in other.concept_synonyms.items():
            concept_synonym_pairs[cui].update(dict.fromkeys(synonyms))

        for cui, synonyms in self.concept_synonyms.items():
            concept_synonym_pairs[cui].update(dict.fromkeys(synonyms))

        concept_semantic_types = {**self.concept_semantic_types, **other.concept_semantic_types}

        return Terminology({concept: list(syns) for concept, syns in concept_synonym_pairs.items()},
                           concept_mapping={**self.concept_mapping, **other.concept_mapping},
                           concept_semantic_types=concept_semantic_types,
                           build_synonym_concepts_mapping=self.synonym_concepts is not None and other.synonym_concepts is not None)


class NERDataset(BaseDataset):
    def describe(self, as_dataframe=True):
        counts = {
            split: {"documents": 0, "entities": 0, "unique_entities_label": set(), "unique_entities_concept": set(), "unique_entities_text": set(), "fragments": 0, "fragmented_entities": 0}
            for split in ["train", "val", "test"]
        }
        for split, split_docs in (("train", self.train_data), ("val", self.val_data), ("test", self.test_data)):
            if split_docs is None:
                continue
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

    def labels(self, splits=("train", "val")):
        labels = set()
        if splits == "all":
            splits = ["train", "val", "test"]
        elif isinstance(splits, str):
            splits = [splits]
        for split in splits:
            docs = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[split]
            if docs is None:
                continue
            for doc in docs:
                for entity in doc["entities"]:
                    if isinstance(entity["label"], (list, tuple)):
                        labels.update(set(entity["label"]))
                    else:
                        labels.add(entity["label"])
        return sorted(labels)


class NormalizationDataset(NERDataset):
    def __init__(self,
                 train_data, val_data, test_data,
                 terminology=None,
                 map_concepts=False,
                 unmappable_concepts="raise",
                 relabel_with_semantic_type=False):
        super().__init__(train_data, val_data, test_data)

        if relabel_with_semantic_type:
            assert terminology is not None, "You need a terminology to relabel entities with coarse labels"
            self.relabel_with_semantic_type(terminology, unmappable_concepts=unmappable_concepts, inplace=True)
        if map_concepts:
            map_concepts = "cui" if map_concepts is True else map_concepts
            assert terminology is not None, "You need a terminology to normalize concepts"
            self.map_concepts(terminology, mode=map_concepts, unmappable_concepts=unmappable_concepts, inplace=True)

    def relabel_with_semantic_type(self, terminology, unmappable_concepts="raise", inplace=False):
        new_splits = {}
        if unmappable_concepts == "raise":
            missing = "raise"
        elif unmappable_concepts == "drop" or unmappable_concepts == "default":
            missing = "null"
        else:
            raise ValueError("unmappable_concepts parameter must be 'raise', 'drop' or 'default'")
        fn = lambda x: terminology.get_concept_semantic_type(x, missing=missing)

        for split, docs in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            new_docs = []
            for doc in docs:
                new_entities = []
                for entity in doc["entities"]:
                    if isinstance(entity["concept"], (list, tuple)):
                        new_label = tuple(fn(concept) for concept in entity["concept"])
                        if unmappable_concepts == "drop":
                            new_label = tuple(part for part in new_label if part is not None)
                            if not len(new_label):
                                continue
                    else:
                        new_label = fn(entity["concept"])
                        if unmappable_concepts == "drop" and new_label is None:
                            continue

                    if new_label is None and unmappable_concepts == "default":
                        new_label = entity["label"]

                    new_entities.append({**entity, "label": new_label})
                new_docs.append({**doc, "entities": new_entities})
            new_splits[split] = new_docs

        new_self = self if inplace else copy(self)
        new_self.train_data = new_splits["train"]
        new_self.val_data = new_splits["val"]
        new_self.test_data = new_splits["test"]
        if not inplace:
            return new_self

    def map_concepts(self, terminology, mode=True, unmappable_concepts="raise", inplace=False):
        new_splits = {}
        if unmappable_concepts == "raise":
            missing = "raise"
        elif unmappable_concepts == "default" or unmappable_concepts == "drop":
            missing = "null"
        else:
            raise ValueError("unmappable_concepts parameter must be 'raise', 'drop' or 'default'")
        if mode == "cui" or mode is True:
            fn = lambda x: terminology.map_concept(x, missing=missing)
        elif mode == "preferred_synonym" or mode is True:
            fn = lambda x: terminology.get_concept_preferred_synonym(x, missing=missing)
        else:
            raise ValueError(f"Unrecognized argument mode={mode}")

        for split, docs in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            new_docs = []
            for doc in docs:
                new_entities = []
                for entity in doc["entities"]:
                    if isinstance(entity["concept"], (list, tuple)):
                        new_concept = tuple(fn(concept) for concept in entity["concept"])
                        if unmappable_concepts == "drop":
                            new_concept = tuple(part for part in new_concept if part is not None)
                            if not len(new_concept):
                                continue
                        elif unmappable_concepts == "default":
                            if mode == "cui":
                                new_concept = tuple(old_part if new_part is None else new_part for new_part, old_part in zip(new_concept, entity["concept"]))
                            else:
                                text = " ".join([doc["text"][frag["begin"]:frag["end"]] for frag in entity["fragments"]])
                                new_concept = tuple(text if new_part is None else new_part for new_part in new_concept)
                    else:
                        new_concept = fn(entity["concept"])
                        if unmappable_concepts == "drop" and new_concept is None:
                            continue

                        if new_concept is None and unmappable_concepts == "default":
                            if mode == "cui":
                                new_concept = entity["concept"]
                            else:
                                new_concept = " ".join([doc["text"][frag["begin"]:frag["end"]] for frag in entity["fragments"]])

                    new_entities.append({**entity, "concept": new_concept})
                new_docs.append({**doc, "entities": new_entities})
            new_splits[split] = new_docs

        new_self = self if inplace else copy(self)
        new_self.train_data = new_splits["train"]
        new_self.val_data = new_splits["val"]
        new_self.test_data = new_splits["test"]
        if not inplace:
            return new_self
