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
import warnings

from ..data_utils import mix, loop


class NetworkLoadMode(enum.Enum):
    AUTO = 0
    CACHE_ONLY = 1
    FETCH_ONLY = 2


def download_file(url, path, checksum=None):
    bar = None

    def reporthook(_, chunk, total):
        nonlocal bar
        if bar is None:
            bar = tqdm(desc=os.path.basename(url), total=total, unit="B", unit_scale=True)
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
    def __init__(self, train_data, val_data, test_data, preprocess_fn=None):
        super().__init__()
        self.train_data = train_data if preprocess_fn is None or train_data is None else preprocess_fn(train_data)
        self.val_data = val_data if preprocess_fn is None or val_data is None else preprocess_fn(val_data)
        self.test_data = test_data if preprocess_fn is None or test_data is None else preprocess_fn(test_data)

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
        self.concept_synonyms = dict(concept_synonym_pairs)
        if do_unidecode or len(subs) or synonym_preprocess_fn is not None:
            res = {}
            for concept, synonyms in tqdm(self.concept_synonyms.items(), desc="Preprocessing synonyms"):
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
        self.concept_synonyms = {concept: list(dict.fromkeys(synonyms)) for concept, synonyms in self.concept_synonyms.items()}
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
            if concept in self.concept_synonyms:
                return concept
            return self.concept_mapping[concept]
        except KeyError:
            pass
        if missing == "raise":
            raise KeyError(f"Could not find concept mapping for {concept}")
        if missing == "null":
            warnings.warn("Missing concept: {}".format(concept))
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

    def build_synonym_concepts_mapping_(self, mode="remove_duplicates"):
        synonym_concepts = defaultdict(lambda: [])
        for concept, synonyms in tqdm(self.concept_synonyms.items(), desc="Building synonym to concept mapping"):
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
        concept_mapping = {**self.concept_mapping, **other.concept_mapping}
        concept_semantic_types = {}

        for cui, synonyms in other.concept_synonyms.items():
            cui = concept_mapping.get(cui, cui)
            concept_synonym_pairs[cui].update(dict.fromkeys(synonyms))
            sty = other.concept_semantic_types.get(cui, None)
            if sty is not None:
                concept_semantic_types[cui] = sty

        for cui, synonyms in self.concept_synonyms.items():
            cui = concept_mapping.get(cui, cui)
            concept_synonym_pairs[cui].update(dict.fromkeys(synonyms))
            sty = self.concept_semantic_types.get(cui, None)
            if sty is not None:
                concept_semantic_types[cui] = sty

        return Terminology({concept: list(syns) for concept, syns in concept_synonym_pairs.items()},
                           concept_mapping=concept_mapping,
                           concept_semantic_types=concept_semantic_types,
                           build_synonym_concepts_mapping=self.synonym_concepts is not None and other.synonym_concepts is not None)

    def filter_concepts(self, concepts=None, semantic_types=None):
        if concepts is None:
            concepts = self.concepts
        if semantic_types is not None:
            semantic_types = set(semantic_types)
            concepts = [concept for concept, sty in self.concept_semantic_types.items() if sty in semantic_types]
        concepts_set = set(concepts)
        return Terminology({concept: self.concept_synonyms[concept] for concept in concepts if concept in self.concept_synonyms},
                           concept_mapping={concept: dest for concept, dest in self.concept_mapping.items() if dest in concepts_set} if self.concept_mapping is not None else None,
                           concept_semantic_types={concept: self.concept_semantic_types[concept] for concept in concepts
                                                   if concept in self.concept_semantic_types} if self.concept_semantic_types is not None else None,
                           build_synonym_concepts_mapping=self.synonym_concepts is not None)


class NERDataset(BaseDataset):
    def describe(self, as_dataframe=True):
        counts = {
            split: {"documents": 0,
                    "entities": 0,
                    "length": 0,
                    "unique_labels": set(),
                    "unique_concepts": set(),
                    "unique_texts": set(),
                    "fragments": 0,
                    "nestings": 0,
                    "same_label_nestings": 0,
                    "overlaps": 0,
                    "same_label_overlaps": 0,
                    "superpositions": 0,
                    "fragmentations": 0}
            for split in ["train", "val", "test"]
        }
        for split, split_docs in (("train", self.train_data), ("val", self.val_data), ("test", self.test_data)):
            if split_docs is None:
                continue
            for doc in split_docs:
                counts[split]["documents"] += 1
                for i, entity in enumerate(doc["entities"]):
                    text = ""
                    for fragment in entity["fragments"]:
                        text = text + " " + doc["text"][fragment["begin"]:fragment["end"]]
                    text = " ".join(text.split()).strip(" ")
                    counts[split]["length"] += len(text.split())
                    counts[split]["entities"] += 1

                    if "label" in entity:
                        if isinstance(entity["label"], (list, tuple)):
                            for label in entity["label"]:
                                counts[split]["unique_labels"].add(label)
                        else:
                            counts[split]["unique_labels"].add(entity["label"])

                    if "concept" in entity:
                        if isinstance(entity["concept"], (list, tuple)):
                            for concept in entity["concept"]:
                                counts[split]["unique_concepts"].add(concept)
                        else:
                            counts[split]["unique_concepts"].add(entity["concept"])
                    if len(entity["fragments"]) > 1:
                        counts[split]["fragmentations"] += 1
                    counts[split]["unique_texts"].add(text)
                    counts[split]["fragments"] += len(entity["fragments"])
                    for other in doc["entities"][i + 1:]:
                        b1, e1, b2, e2 = entity['fragments'][0]['begin'], entity['fragments'][0]['end'], other['fragments'][-1]['begin'], other['fragments'][-1]['end']
                        if b1 == b2 and e1 == e2:
                            counts[split]["superpositions"] += 1
                        if not (e1 <= b2 or e2 <= b1):
                            if (b1 >= b2 and e1 <= e2) or (b2 >= b1 and e2 <= e1):
                                counts[split]["nestings"] += 1
                                if entity["label"] == other['label']:
                                    counts[split]["same_label_nestings"] += 1
                            else:
                                counts[split]["overlaps"] += 1
                                if entity["label"] == other['label']:
                                    counts[split]["same_label_overlaps"] += 1

            counts[split]["length"] /= (counts[split]["entities"] or 1)
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

    def __or__(self, other):
        def merge(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return x + y

        return NERDataset(
            merge(self.train_data, other.train_data),
            merge(self.val_data, other.val_data),
            merge(self.test_data, other.test_data),
        )

    def export_to_brat(self, path, overwrite_ann=False, overwrite_txt=False):
        for data, split in [(self.train_data, "train"), (self.val_data, "val"), (self.test_data, "test")]:
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
            if data is not None:
                from .brat import export_to_brat
                export_to_brat(data, os.path.join(path, split), overwrite_txt=overwrite_txt, overwrite_ann=overwrite_ann)


class NormalizationDataset(NERDataset):
    def __init__(self,
                 train_data, val_data, test_data,
                 terminology=None,
                 map_concepts=False,
                 unmappable_concepts="raise",
                 relabel_with_semantic_type=False,
                 preprocess_fn=None):
        super().__init__(train_data, val_data, test_data, preprocess_fn=preprocess_fn)

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

    def map_concepts(self, terminology, mode=True, unmappable_concepts="raise", inplace=False, deduplicate=True):
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
                        if deduplicate:
                            new_concept = list(dict.fromkeys(new_concept))
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

    def to_terminology(self, splits=['train'], label_as_semantic_type=False, multi_concepts="drop", **kwargs):
        concept_synonyms = defaultdict(lambda: [])
        if multi_concepts == "duplicate":
            multi_concepts = lambda text, e: (text, [{**e, "concept": c} for c in e["concept"]])
        concept_type = dict()
        for docs, split in ((self.train_data, 'train'), (self.val_data, 'val'), (self.test_data, 'test')):
            if split in splits:
                for doc in docs:
                    for entity in doc["entities"]:
                        concept = entity["concept"] if not isinstance(entity["concept"], str) else entity["concept"].split("+")
                        if len(concept) > 1 and multi_concepts != "drop":
                            if hasattr(multi_concepts, '__call__'):
                                text, split_entities = multi_concepts(doc["text"], entity)
                                original_begin = entity["fragments"][0]["begin"]
                                if len(split_entities) == len(concept):
                                    for composite_entity in split_entities:
                                        entity_text = " ".join(text[fragment["begin"] - original_begin:fragment["end"] - original_begin] for fragment in composite_entity["fragments"])
                                        concept = composite_entity["concept"]
                                        concept_synonyms[concept].append(entity_text)
                                        if label_as_semantic_type:
                                            concept_type[concept] = composite_entity["label"]
                        elif len(concept) == 1:
                            concept = concept[0]
                            entity_text = " ".join(doc["text"][fragment["begin"]:fragment["end"]] for fragment in entity["fragments"])
                            concept_synonyms[concept].append(entity_text)
                            if label_as_semantic_type:
                                concept_type[concept] = entity["label"]
        return Terminology(concept_synonym_pairs=concept_synonyms, concept_semantic_types=concept_type, **kwargs)

    def __or__(self, other):
        def merge(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return x + y

        return NormalizationDataset(
            merge(self.train_data, other.train_data),
            merge(self.val_data, other.val_data),
            merge(self.test_data, other.test_data),
        )


class MixDataset(BaseDataset):
    def __init__(self, datasets, rates=None):
        super().__init__()
        self.datasets = datasets
        self.rates = rates if rates is not None else [1 / len(datasets) for _ in datasets]

    @property
    def train_data(self):
        return mix(*[loop(d.train_data, shuffle=True) for d in self.datasets if d.train_data is not None], rates=self.rates)

    @property
    def val_data(self):
        return list(chain.from_iterable([d.val_data for d in self.datasets if d.val_data is not None]))

    @property
    def test_data(self):
        return list(chain.from_iterable([d.test_data for d in self.datasets if d.test_data is not None]))
