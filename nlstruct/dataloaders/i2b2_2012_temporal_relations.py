import glob
import os
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from nlstruct.collections import Dataset
from nlstruct.environment import root


def load_i2b2_2012_temporal_relations():
    train_path = root.resource('i2b2_temporal/2012-07-15.original-annotation.release.tar.gz')
    test_path = root.resource('i2b2_temporal/2012-08-23.test-data.groundtruth.tar.gz')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "You should download I2B2 temporal dataset ('Training Data: Full training set with original temporal relations' and 'Test Data: Test Data Groundtruth') "
            "from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/#collapse5 and place it in under {}".format(
                root.resource('i2b2_temporal')))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for file in [train_path, test_path]:
            with tarfile.open(file, "r:gz") as tar:
                tar.extractall(tmpdir)

        parser = ET.XMLParser(encoding="utf-8")
        all_mentions = []
        all_attributes = []
        all_docs = []
        all_relations = []

        for split, files in [
            ("train", glob.glob(str(tmpdir / "2012-07-15.original-annotation.release/*.xml"))),
            ("test", glob.glob(str(tmpdir / "ground_truth/merged_xml/*.xml")))
        ]:
            for filename in sorted(files):
                doc_id = filename.split('/')[-1]

                with open(filename, 'r', encoding='utf-8') as f:
                    s = f.read()

                root_node = ET.fromstring(s.replace("&", "___AND___"))

                all_docs.append({
                    "doc_id": doc_id,
                    "text": root_node.findall('TEXT')[0].text.replace("___AND___", "&"),  # .lstrip('\n')
                    "split": split,
                })

                tags = root_node.find('TAGS')
                for event in tags.findall('EVENT'):
                    all_mentions.append({
                        "doc_id": doc_id,
                        "mention_id": event.attrib["id"],
                        "begin": int(event.attrib["start"]),
                        "end": int(event.attrib["end"]),
                        "text": event.attrib["text"].replace("___AND___", "&"),
                        "label": event.attrib["type"],
                    })
                    for attr_name in ["modality", "polarity"]:
                        all_attributes.append({
                            "doc_id": doc_id,
                            "mention_id": event.attrib["id"],
                            "attribute_id": event.attrib["id"] + '-' + attr_name,
                            "attribute": attr_name,
                            "value": event.attrib[attr_name],
                        })
                for link in tags.findall('TLINK'):
                    all_relations.append({
                        "doc_id": doc_id,
                        "relation_id": link.attrib["fromID"] + "-" + link.attrib["toID"],
                        "from_mention_id": link.attrib["fromID"],
                        "to_mention_id": link.attrib["toID"],
                        "label": link.attrib["type"],
                    })
    mentions = pd.DataFrame(all_mentions)
    return Dataset(
        docs=pd.DataFrame(all_docs),
        mentions=mentions[["doc_id", "mention_id", "label", "text"]],
        fragments=mentions[["doc_id", "mention_id", "begin", "end"]],
        relations=pd.DataFrame(all_relations),
        attributes=pd.DataFrame(all_attributes),
    )
