import xml.etree.ElementTree as ET
import glob
import os
import tarfile
import tempfile
from pathlib import Path
import random

from nlstruct.datasets.base import NERDataset


class I2B2Temporal(NERDataset):
    def __init__(self, path, val_split=False, seed=False, debug=False, preprocess_fn=None):
        train_data, val_data, test_data = self.extract(path, val_split, seed, debug)
        super().__init__(train_data, val_data, test_data, preprocess_fn=preprocess_fn)

    @staticmethod
    def extract(path, val_split=False, seed=False, debug=False):
        train_path = os.path.join(path, "2012-07-15.original-annotation.release.tar.gz")
        test_path = os.path.join(path, "2012-08-23.test-data.groundtruth.tar.gz")
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                "You should download I2B2 temporal dataset ('Training Data: Full training set with original temporal relations' and 'Test Data: Test Data Groundtruth') "
                "from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/#collapse5 and place it in under {}".format(
                    path))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for file in [train_path, test_path]:
                with tarfile.open(file, "r:gz") as tar:
                    tar.extractall(tmpdir)

            splits = {
                "train": [],
                "test": [],
            }
            for split, files in [
                ("train", sorted(glob.glob(str(tmpdir / "2012-07-15.original-annotation.release/*.xml")))),
                ("test", sorted(glob.glob(str(tmpdir / "ground_truth/merged_xml/*.xml"))))
            ]:
                for filename in sorted(files):
                    doc_id = filename.split('/')[-1]

                    with open(filename, 'r', encoding='utf-8') as f:
                        s = f.read()

                    root_node = ET.fromstring(s.replace("&", "___AND___"))

                    entities = []
                    tags = root_node.find('TAGS')

                    for big_type, attr_names in [
                        ("EVENT", ["modality", "polarity"]),
                        ("TIMEX3", ["mod"]),
                    ]:
                        for event in tags.findall(big_type):
                            attributes = []
                            entities.append({
                                "entity_id": event.attrib["id"],
                                "fragments": [{
                                    "begin": int(event.attrib["start"]),
                                    "end": int(event.attrib["end"]),
                                }],
                                "text": event.attrib["text"].replace("___AND___", "&"),
                                "label": event.attrib["type"],
                                "attributes": attributes
                            })
                            for attr_name in attr_names:
                                attributes.append({
                                    "attribute_id": event.attrib["id"] + '-' + attr_name,
                                    "label": attr_name,
                                    "value": event.attrib[attr_name],
                                })
                    relations = []
                    for link in tags.findall('TLINK'):
                        relations.append({
                            "relation_id": link.attrib["fromID"] + "-" + link.attrib["toID"],
                            "from_entity_id": link.attrib["fromID"],
                            "to_entity_id": link.attrib["toID"],
                            "label": link.attrib["type"],
                        })

                    splits[split].append({
                        "doc_id": doc_id,
                        "text": root_node.findall('TEXT')[0].text.replace("___AND___", "&"),  # .lstrip('\n')
                        "entities": entities,
                        "relations": relations,
                    })

        train_data = splits["train"]
        test_data = splits["test"]

        val_data = []
        if val_split is not None and val_split:
            shuffled_data = list(train_data)
            if seed is not False:
                random.Random(seed).shuffle(shuffled_data)
            offset = val_split if isinstance(val_split, int) else int(val_split * len(shuffled_data))
            val_data = shuffled_data[:offset]
            train_data = shuffled_data[offset:]

        subset = slice(None) if not debug else slice(0, 50)
        train_data = train_data[subset]
        val_data = val_data[subset]
        test_data = test_data  # Never subset the test set, we don't want to give false hopes

        return train_data, val_data, test_data
