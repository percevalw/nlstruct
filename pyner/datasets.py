import os
import torch
import random
import pytorch_lightning as pl


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
    docs = []
    for filename in ((os.path.join(path, name) for name in sorted(os.listdir(path))) if isinstance(path, str) else path):
        entities = {}
        relations = []
        if filename.endswith('.txt'):
            doc_id = filename.replace('.txt', '').split("/")[-1]

            with open(filename) as f:
                text = f.read()

            try:
                with open(filename.replace(".txt", ".ann")) as f:
                    for line in f:
                        ann_parts = line.strip('\n').split('\t', 1)
                        ann_id, remaining = ann_parts
                        if ann_id.startswith('T'):
                            remaining, entity_text = remaining.split("\t", 1)
                            entity, span = remaining.split(" ", 1)
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
                        elif ann_id.startswith('A'):
                            parts = remaining.split(" ")
                            if len(parts) >= 3:
                                entity, entity_id, value = parts
                            else:
                                entity, entity_id = parts
                                value = None
                            entities[entity_id]["attributes"].append({
                                "attribute_id": ann_id,
                                "label": entity,
                                "value": value,
                            })
                        elif ann_id.startswith('R'):
                            [ann_name, *parts] = remaining.strip("\t").split(" ")
                            relations.append({
                                "relation_id": ann_id,
                                "relation_label": ann_name,
                                "from_entity_id": parts[0].split(":")[1],
                                "to_entity_id": parts[1].split(":")[1],
                            })
                        elif ann_id.startswith('#'):
                            remaining = remaining.strip(" \t").split("\t")
                            [entity_id, comment] = remaining + ([""] if len(remaining) < 2 else [])
                            ann_type, entity_id = entity_id.split(" ")
                            if ann_type == "AnnotatorNotes":
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
    return docs


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


class BaseDataset(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data)


class BRATDataset(BaseDataset):
    def __init__(self, train, test=None, val=None, kept_entity_label=None, dropped_entity_label=(), seed=False):
        super().__init__()
        self.train_source = train
        self.val_source = val
        self.test_source = test
        self.seed = seed
        self.dropped_entity_label = dropped_entity_label
        self.kept_entity_label = kept_entity_label

        if isinstance(self.train_source, (str, list, tuple)):
            self.train_data = list(load_from_brat(self.train_source))
        else:
            raise ValueError("train source for BRATDataset must be str or list of str")
        if self.train_data is not None:
            self.train_data = self.filter_entities(self.train_data)

        if isinstance(self.test_source, (str, list, tuple)):
            self.test_data = list(load_from_brat(self.test_source))
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

    def filter_entities(self, data):
        return [
            {**doc, "entities": [entity
                                 for entity in doc["entities"]
                                 if entity["label"] not in self.dropped_entity_label and
                                 (self.kept_entity_label is None or entity["label"] in self.kept_entity_label)]}
            for doc in data
        ]


class DEFT(BRATDataset):
    def __init__(self, train, test=None, val=0.2, dropped_entity_label=("duree", "frequence"), seed=False):
        super().__init__(train=train, test=test, val=val, dropped_entity_label=dropped_entity_label, seed=seed)
