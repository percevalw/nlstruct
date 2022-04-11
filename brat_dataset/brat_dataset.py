# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""

import os
import random
import re
import glob
from collections import defaultdict

import datasets


REGEX_ENTITY = re.compile('^(T\d+)\t([^\s]+)([^\t]+)\t(.*)$')
REGEX_NOTE = re.compile('^(#\d+)\tAnnotatorNotes ([^\t]+)\t(.*)$')
REGEX_RELATION = re.compile('^(R\d+)\t([^\s]+) Arg1:([^\s]+) Arg2:([^\s]+)')
REGEX_ATTRIBUTE = re.compile('^([AM]\d+)\t(.+)$')
REGEX_EVENT = re.compile('^(E\d+)\t(.+)$')
REGEX_EVENT_PART = re.compile('([^\s]+):([TE]\d+)')

logger = datasets.logging.get_logger(__name__)


_CITATION = """"""

_DESCRIPTION = """"""

_URL = ""

def load_from_brat(path, merge_spaced_fragments=True):
    """
    Load a brat dataset into a Dataset object
    Parameters
    ----------
    path: str or pathlib.Path
    merge_spaced_fragments: bool
        Merge fragments of a entity that was splited by brat because it overlapped an end of line
    Returns
    -------
    Dataset
    """

    # Extract annotations from path and make multiple dataframe from it

    root_path = path
    path = str(path)
    if os.path.isdir(path):
        path = path + "/**/*.txt"
    elif path.endswith('.ann'):
        path = path.replace(".ann", ".txt")
    elif path.endswith(".txt"):
        pass
    else:
        path = path + "*.txt"

    filenames = {os.path.relpath(filename, root_path).rsplit(".", 1)[0]: {"txt": filename, "ann": []} for filename in glob.glob(path, recursive=True)}
    for filename in glob.glob(path.replace(".txt", ".a*"), recursive=True):
        filenames[os.path.relpath(filename, root_path).rsplit(".", 1)[0]]["ann"].append(filename)
    for doc_id, files in filenames.items():
        entities = {}
        relations = []
        events = {}

        # doc_id = filename.replace('.txt', '').split("/")[-1]

        with open(files["txt"]) as f:
            text = f.read()

        if not len(files["ann"]):
            yield {
                "doc_id": doc_id,
                "text": text,
            }
            continue

        for ann_file in files["ann"]:
            with open(ann_file) as f:
                for line_idx, line in enumerate(f):
                    try:
                        if line.startswith('T'):
                            match = REGEX_ENTITY.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            entity = match.group(2)
                            span = match.group(3)
                            mention_text = match.group(4)
                            entities[ann_id] = {
                                "text": mention_text,
                                "entity_id": ann_id,
                                "fragments": [],
                                "attributes": [],
                                "comments": [],
                                "label": entity,
                            }
                            last_end = None
                            fragment_i = 0
                            begins_ends = sorted([(int(s.split()[0]), int(s.split()[1])) for s in span.split(';')])

                            for begin, end in begins_ends:
                                # If merge_spaced_fragments, merge two fragments that are only separated by a newline (brat automatically creates
                                # multiple fragments for a entity that spans over more than one line)
                                if merge_spaced_fragments and last_end is not None and len(text[last_end:begin].strip()) == 0:
                                    entities[ann_id]["fragments"][-1]["end"] = end
                                    continue
                                entities[ann_id]["fragments"].append({
                                    "begin": begin,
                                    "end": end,
                                })
                                fragment_i += 1
                                last_end = end
                        elif line.startswith('A') or line.startswith('M'):
                            match = REGEX_ATTRIBUTE.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            parts = match.group(2).split(" ")
                            if len(parts) >= 3:
                                entity, entity_id, value = parts
                            elif len(parts) == 2:
                                entity, entity_id = parts
                                value = None
                            else:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            (entities[entity_id] if entity_id.startswith('T') else events[entity_id])["attributes"].append({
                                "attribute_id": ann_id,
                                "label": entity,
                                "value": value,
                            })
                        elif line.startswith('R'):
                            match = REGEX_RELATION.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            ann_name = match.group(2)
                            arg1 = match.group(3)
                            arg2 = match.group(4)
                            relations.append({
                                "relation_id": ann_id,
                                "relation_label": ann_name,
                                "from_entity_id": arg1,
                                "to_entity_id": arg2,
                            })
                        elif line.startswith('E'):
                            match = REGEX_EVENT.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            arguments_txt = match.group(2)
                            arguments = []
                            for argument in REGEX_EVENT_PART.finditer(arguments_txt):
                                arguments.append({"entity_id": argument.group(2), "label": argument.group(1)})
                            events[ann_id] = {
                                "event_id": ann_id,
                                "attributes": [],
                                "arguments": arguments,
                            }
                        elif line.startswith('#'):
                            match = REGEX_NOTE.match(line)
                            if match is None:
                                raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                            ann_id = match.group(1)
                            entity_id = match.group(2)
                            comment = match.group(3)
                            entities[entity_id]["comments"].append({
                                "comment_id": ann_id,
                                "comment": comment,
                            })
                    except:
                        raise Exception("Could not parse line {} from {}: {}".format(line_idx, filename.replace(".txt", ".ann"), repr(line)))
        yield {
            "doc_id": doc_id,
            "text": text,
            "entities": list(entities.values()),
            "relations": relations,
            "events": list(events.values()),
        }

class BRATConfig(datasets.BuilderConfig):
    """BuilderConfig for BRAT."""

    def __init__(self, name, brat_path=None, merge_spaced_fragments=True, seed=42, **kwargs):
        """BuilderConfig for BRAT.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        if isinstance(brat_path, str):
            brat_path = {"train": brat_path, "val": None, "test": None}
        if brat_path is None:
            brat_path = {"train": None, "val": None, "test": None}
        self.train_path = brat_path["train"]
        self.val_path = brat_path["val"]
        self.test_path = brat_path["test"]
        self.name = name
        self.seed = seed
        self.merge_spaced_fragments = merge_spaced_fragments


def shuffle(items, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items
        
class BRAT(datasets.GeneratorBasedBuilder):
    """BRAT"""

    BUILDER_CONFIG_CLASS = BRATConfig
    BUILDER_CONFIGS = [
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "entities": [({
                            "entity_id": datasets.Value("string"),
                            "fragments": [({
                                "begin": datasets.Value("int16"),
                                "end": datasets.Value("int16"),
                            })],
                            "attributes": [({
                                "label": datasets.Value("string"),
                                "value": datasets.Value("string"),
                            })],
                            "label": datasets.Value("string"),
                            "concept": datasets.Value("string"),
                            "comment": datasets.Value("string"),
                        }
                    )],
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation="",
            task_templates=[
            ],
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"split": "train"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"split": "val"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"split": "test"}),
        ]

    def _generate_examples(self, split):
        
        """This function returns the examples in the raw (text) form."""
        path = None
        if split == "train":
            path = self.config.train_path
        elif split == "test":
            path = self.config.test_path
        elif split == "val":
            if isinstance(self.config.val_path, float):
                path = self.config.train_path
            else:
                path = self.config.val_path
        else:
            raise Exception()
        
        if path is None:
            return

        logger.info(f"generating {split} examples from {path}")

        root_path = path
        path = str(path)
        if os.path.isdir(path):
            path = path + "/**/*.txt"
        elif path.endswith('.ann'):
            path = path.replace(".ann", ".txt")
        elif path.endswith(".txt"):
            pass
        else:
            path = path + "*.txt"

        filenames = {os.path.relpath(filename, root_path).rsplit(".", 1)[0]: {"txt": filename, "ann": []} for filename in glob.glob(path, recursive=True)}
        for filename in glob.glob(path.replace(".txt", ".a*"), recursive=True):
            filenames[os.path.relpath(filename, root_path).rsplit(".", 1)[0]]["ann"].append(filename)
            
        filenames_items = shuffle(sorted(filenames.items(), key=lambda kv: kv[0]), seed=self.config.seed)
        if split == "train" and isinstance(self.config.val_path, float):
            filenames = dict(filenames_items[:len(filenames_items) - int(len(filenames_items) * self.config.val_path)])
        if split == "val" and isinstance(self.config.val_path, float):
            filenames = dict(filenames_items[len(filenames_items) - int(len(filenames_items) * self.config.val_path):])
        
        for doc_id, files in filenames.items():
            entities = {}
            relations = []
            events = {}

            # doc_id = filename.replace('.txt', '').split("/")[-1]

            with open(files["txt"]) as f:
                text = f.read()

            if not len(files["ann"]):
                yield doc_id, {
                    "doc_id": doc_id,
                    "text": text,
                    "source": self.config.name,
                    "entities": {
                        "entity_id": [],
                        "fragments": {
                            "begin": [],
                            "end": [],
                        },
                        "attributes": {
                            "label": [],
                            "value": [],
                        },
                        "label": [],
                        "concept": "",
                        "comment": "",
                    },
                }
                continue

            for ann_file in files["ann"]:
                with open(ann_file) as f:
                    for line_idx, line in enumerate(f):
                        try:
                            if line.startswith('T'):
                                match = REGEX_ENTITY.match(line)
                                if match is None:
                                    raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                                ann_id = match.group(1)
                                entity = match.group(2)
                                span = match.group(3)
                                mention_text = match.group(4)
                                entities[ann_id] = {
                                    #"text": mention_text,
                                    "entity_id": ann_id,
                                    "fragments": [],
                                    "attributes": [],
                                    "concept": "",
                                    "comment": "",
                                    "label": entity,
                                }
                                last_end = None
                                fragment_i = 0
                                begins_ends = sorted([(int(s.split()[0]), int(s.split()[1])) for s in span.split(';')])

                                for begin, end in begins_ends:
                                    # If merge_spaced_fragments, merge two fragments that are only separated by a newline (brat automatically creates
                                    # multiple fragments for a entity that spans over more than one line)
                                    if self.config.merge_spaced_fragments and last_end is not None and len(text[last_end:begin].strip()) == 0:
                                        entities[ann_id]["fragments"][-1]["end"] = end
                                        continue
                                    entities[ann_id]["fragments"].append({
                                        "begin": begin,
                                        "end": end,
                                    })
                                    fragment_i += 1
                                    last_end = end
                            elif line.startswith('A') or line.startswith('M'):
                                match = REGEX_ATTRIBUTE.match(line)
                                if match is None:
                                    raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                                ann_id = match.group(1)
                                parts = match.group(2).split(" ")
                                if len(parts) >= 3:
                                    entity, entity_id, value = parts
                                elif len(parts) == 2:
                                    entity, entity_id = parts
                                    value = None
                                else:
                                    raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                                (entities[entity_id] if entity_id.startswith('T') else events[entity_id])["attributes"].append({
                                    #"attribute_id": ann_id,
                                    "label": entity,
                                    "value": value,
                                })
                            elif line.startswith('R'):
                                match = REGEX_RELATION.match(line)
                                if match is None:
                                    raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                                ann_id = match.group(1)
                                ann_name = match.group(2)
                                arg1 = match.group(3)
                                arg2 = match.group(4)
                                relations.append({
                                    "relation_id": ann_id,
                                    "relation_label": ann_name,
                                    "from_entity_id": arg1,
                                    "to_entity_id": arg2,
                                })
                            elif line.startswith('E'):
                                match = REGEX_EVENT.match(line)
                                if match is None:
                                    raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                                ann_id = match.group(1)
                                arguments_txt = match.group(2)
                                arguments = []
                                for argument in REGEX_EVENT_PART.finditer(arguments_txt):
                                    arguments.append({"entity_id": argument.group(2), "label": argument.group(1)})
                                events[ann_id] = {
                                    "event_id": ann_id,
                                    "attributes": [],
                                    "arguments": arguments,
                                }
                            elif line.startswith('#'):
                                match = REGEX_NOTE.match(line)
                                if match is None:
                                    raise ValueError(f'File {ann_file}, unrecognized Brat line {line}')
                                ann_id = match.group(1)
                                entity_id = match.group(2)
                                comment = match.group(3)
                                entities[entity_id]["comment"] += comment + ";"
                                entities[entity_id]["concept"] = "|".join(filter(None, (
                                    *entities[entity_id]["concept"].split("|"),
                                    *("C" + s for s in re.findall("(?:C|^)([0-9]+)", comment.strip().replace(".", "0")))
                                )))
                        except:
                            raise Exception("Could not parse line {} from {}: {}".format(line_idx, filename.replace(".txt", ".ann"), repr(line)))
            entities = list(entities.values())
            yield doc_id, {
                "doc_id": doc_id,
                "source": self.config.name,
                "text": text,
                "entities": entities,
                # "relations": relations,
                # "events": list(events.values()),
            }
