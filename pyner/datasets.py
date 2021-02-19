import os
import torch
import random
import pytorch_lightning as pl
import re

REGEX_ENTITY = re.compile('^(T\d+)\t([^ ]+)([^\t]+)\t(.*)$')
REGEX_NOTE = re.compile('^(#\d+)\tAnnotatorNotes ([^\t]+)\t(.*)$')
REGEX_RELATION = re.compile('^(R\d+)\t([^ ]+) Arg1:([^ ]+) Arg2:([^ ]+)')
REGEX_ATTRIBUTE = re.compile('^(A\d+)\t(.+)$')

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
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            entities = {}
            relations = []
            if filename.endswith('.txt'):
                doc_id = filename.replace('.txt', '').split("/")[-1]

                with open(filename) as f:
                    text = f.read()

                try:
                    with open(filename.replace(".txt", ".ann")) as f:
                        for line in f:
                            if line.startswith('T'):
                                match = REGEX_ENTITY.match(line)
                                if match is None:
                                    raise ValueError(f'File {filename}, unrecognized Brat line {line}')
                                ann_id = match.group(1)
                                entity = match.group(2)
                                span = match.group(3)
                                mention_text = match.group(4)
                                entities[ann_id] = {
                                    "entity_id": ann_id,
                                    "fragments": [],
                                    "attributes": [],
                                    "comments": [],
                                    "label": entity,
                                }
                                last_end = None
                                fragment_i = 0
                                for s in span.split(';'):
                                    begin, end = int(s.split()[0]), int(s.split()[1])
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
                            elif line.startswith('A'):
                                REGEX_ATTRIBUTE = re.compile('^(A\d+)\t(.+)$')
                                match = REGEX_ATTRIBUTE.match(line)
                                if match is None:
                                    raise ValueError(f'File {filename}, unrecognized Brat line {line}')
                                ann_id = match.group(1)
                                parts = match.group(2).split(" ")
                                if len(parts) >= 3:
                                    entity, entity_id, value = parts
                                elif len(parts) == 2:
                                    entity, entity_id = parts
                                    value = None
                                else:
                                    raise ValueError(f'File {filename}, unrecognized Brat line {line}')
                                entities[entity_id]["attributes"].append({
                                    "attribute_id": ann_id,
                                    "label": entity,
                                    "value": value,
                                })
                            elif line.startswith('R'):
                                match = REGEX_RELATION.match(line)
                                if match is None:
                                    raise ValueError(f'File {filename}, unrecognized Brat line {line}')
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
                            elif line.startswith('#'):
                                match = REGEX_NOTE.match(line)
                                if match is None:
                                    raise ValueError(f'File {filename}, unrecognized Brat line {line}')
                                ann_id = match.group(1)
                                entity_id = match.group(2)
                                comment = match.group(3)
                                entities[entity_id]["comments"].append({
                                    "comment_id": ann_id,
                                    "comment": comment,
                                })
                except FileNotFoundError:
                    yield {
                        "doc_id": doc_id,
                        "text": text,
                    }
                else:
                    yield {
                        "doc_id": doc_id,
                        "text": text,
                        "entities": list(entities.values()),
                        "relations": relations,
                    }


def export_to_brat(samples, filename_prefix="", overwrite_txt=False, overwrite_ann=False):
    if filename_prefix:
        try:
            os.mkdir(filename_prefix)
        except FileExistsError:
            pass
    for doc in samples:
        txt_filename = os.path.join(filename_prefix, doc["doc_id"] + ".txt")
        if not os.path.exists(txt_filename) or overwrite_txt:
            with open(txt_filename, "w") as f:
                f.write(doc["text"])

        ann_filename = os.path.join(filename_prefix, doc["doc_id"] + ".ann")
        attribute_idx = 1
        if not os.path.exists(ann_filename) or overwrite_ann:
            with open(ann_filename, "w") as f:
                if "entities" in doc:
                    for entity in doc["entities"]:
                        idx = None
                        spans = []
                        brat_entity_id = "T" + str(entity["entity_id"] + 1)
                        for fragment in sorted(entity["fragments"], key=lambda frag: frag["begin"]):
                            idx = fragment["begin"]
                            entity_text = doc["text"][fragment["begin"]:fragment["end"]]
                            for part in entity_text.split("\n"):
                                begin = idx
                                end = idx + len(part)
                                idx = end + 1
                                if begin != end:
                                    spans.append((begin, end))
                        print("T{}\t{} {}\t{}".format(
                            brat_entity_id,
                            str(entity["label"]),
                            ";".join(" ".join(map(str, span)) for span in spans),
                            entity_text.replace("\n", " ")), file=f)
                        if "attributes" in entity:
                            for attribute in entity["attributes"]:
                                if "value" in attribute and attribute["value"] is not None:
                                    print("A{}\t{} T{} {}".format(
                                        attribute_idx,
                                        str(attribute["label"]),
                                        brat_entity_id,
                                        attribute["value"]), file=f)
                                else:
                                    print("A{}\t{} T{}".format(
                                        i + 1,
                                        str(attribute["label"]),
                                        brat_entity_id), file=f)
                                attribute_idx += 1
                if "relations" in doc:
                    for i, relation in enumerate(doc["relations"]):
                        entity_from = relation["from_entity_id"] + 1
                        entity_to = relation["to_entity_id"] + 1
                        print("R{}\t{} Arg1:T{} Arg2:T{}\t".format(
                            i + 1,
                            str(relation["label"]),
                            entity_from,
                            entity_to), file=f)


class BRATDataset(pl.LightningDataModule):
    def __init__(self, train, test=None, val=None, kept_entity_label=None, dropped_entity_label=(), seed=False):
        super().__init__()
        self.train_source = train
        self.val_source = val
        self.test_source = test
        self.seed = seed
        self.dropped_entity_label = dropped_entity_label
        self.kept_entity_label = kept_entity_label

    def filter_entities(self, data):
        return [
            {**doc, "entities": [entity
                                 for entity in doc["entities"]
                                 if (self.dropped_entity_label is None or entity["label"] not in self.dropped_entity_label) and
                                 (self.kept_entity_label is None or entity["label"] in self.kept_entity_label)]}
            for doc in data
        ]

    def setup(self, stage='fit'):
        if isinstance(self.train_source, (str, list, tuple)):
            self.train_data = list(load_from_brat(self.train_source))
            if len(self.train_data) == 0:
                raise ValueError(f'No Brat file found in {self.train_source}')
        else:
            raise ValueError("train source for BRATDataset must be str or list of str")
            
        if len(self.train_data[0]['entities']) == 0:
            raise ValueError('No entity have been found in the training set')

        if self.train_data is not None:
            self.train_data = self.filter_entities(self.train_data)
            
        if isinstance(self.test_source, (str, list, tuple)):
            self.test_data = list(load_from_brat(self.test_source))
            if len(self.test_data) == 0:
                raise ValueError(f'No Brat file found in {self.test_source}')
        else:
            assert self.test_source is None
            self.test_data = None
        if self.test_data is not None:
            self.test_data = self.filter_entities(self.test_data)

        if isinstance(self.val_source, (str, list, tuple)):
            self.val_data = list(load_from_brat(self.val_source))
        elif isinstance(self.val_source, (int, float)):
            shuffled_data = list(self.train_data)
            if self.seed is not False:
                random.Random(self.seed).shuffle(shuffled_data)
            offset = self.val_source if isinstance(self.val_source, int) else int(self.val_source * len(shuffled_data))
            self.val_data = shuffled_data[:offset]
            self.train_data = shuffled_data[offset:]
        else:
            assert self.val_source is None
            self.val_data = None
        if self.val_data is not None:
            self.val_data = self.filter_entities(self.val_data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data) if self.train_data is not None else None

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data) if self.val_data is not None else None

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data) if self.test_data is not None else None


class DEFT(BRATDataset):
    def __init__(self, train, test=None, val=0.2, dropped_entity_label=("duree", "frequence"), seed=False):
        super().__init__(train=train, test=test, val=val, dropped_entity_label=dropped_entity_label, seed=seed)
