import os
import re
import tarfile
import xml.etree.ElementTree as ET
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.datasets._base import RemoteFileMetadata

from nlstruct.collections import Dataset
from nlstruct.dataloaders import load_from_brat
from nlstruct.environment import root
from nlstruct.utils import ensure_files, NetworkLoadMode

GENIA_EVENTS_REMOTE_FILES = [
    RemoteFileMetadata(
        url="http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_train_data_rev1.tar.gz",
        checksum="9ffb580b191d4df3d54ecd9fa83db6c81bec32faec09ea539a093123eb0511ff",
        filename="BioNLP-ST_2011_genia_train_data_rev1.tar.gz"),
    RemoteFileMetadata(
        url="http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_devel_data_rev1.tar.gz",
        checksum="f70e5f6d6e2a7f7fcdb5c8671715f3909a77662a6238015b2916ce939f2a890f",
        filename="BioNLP-ST_2011_genia_devel_data_rev1.tar.gz"),
    RemoteFileMetadata(
        url="http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_test_data.tar.gz",
        checksum="8b5b3184e68199ec4c03f7ef1abcf60405b7c5e7ee289453be85545fecac1df4",
        filename="BioNLP-ST_2011_genia_test_data.tar.gz"),
]


def load_genia_events(resource_path="genia", doc_attributes={}):
    path = root.resource(resource_path)
    files = ensure_files(path, GENIA_EVENTS_REMOTE_FILES, mode=NetworkLoadMode.AUTO)
    for file in files:
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall(path)

    # It's important to sort to ensure reproducibility of everything, and listdir can give different results depending on the machine
    for folder in [
        path / "BioNLP-ST_2011_genia_train_data_rev1",
        path / "BioNLP-ST_2011_genia_devel_data_rev1",
        path / "BioNLP-ST_2011_genia_test_data",
    ]:
        for filename in sorted(os.listdir(folder)):
            if filename.endswith('.a1') or filename.endswith('.a2'):
                dest = filename.replace('.a1', '.ann').replace('.a2', '.ann')
                with open(folder / dest, 'w') as cat_file:
                    with open(folder / filename, 'r') as input_file:
                        cat_file.write(input_file.read())

    dataset = Dataset.concat([
        load_from_brat(path / "BioNLP-ST_2011_genia_train_data_rev1", doc_attributes={"split": "train", **doc_attributes}),
        load_from_brat(path / "BioNLP-ST_2011_genia_devel_data_rev1", doc_attributes={"split": "val", **doc_attributes}),
        load_from_brat(path / "BioNLP-ST_2011_genia_test_data", doc_attributes={"split": "test", **doc_attributes}),
    ])
    return dataset


GENIA_NER_REMOTE_FILES = {
    "3.02p": RemoteFileMetadata(
        url="http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Part-of-speech/GENIAcorpus3.02p.tgz",
        checksum="8faa7813641cf41d24dfa13f24382ef73a94654bff2297f763fc6b19102c8b74",
        filename="GENIAcorpus3.02p.tgz"),
    "3.02": RemoteFileMetadata(
        url="http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Term/GENIAcorpus3.02.tgz",
        checksum="85b2aede313308c33783ba508d054d0af0ff9c7e48490f0db5168a9f88d74d1c",
        filename="GENIAcorpus3.02.tgz"
    ),
}


def process_xml(root_node, idx=0):
    text = ""
    if root_node.text is not None:
        text = root_node.text
        idx += len(root_node.text)
    mentions = []
    for elem in root_node:
        sub_text, next_idx, sub_mentions = process_xml(elem, idx)
        mentions.extend(sub_mentions)
        if elem.tag == "cons":
            mentions.append({"attrib": elem.attrib, "begin": idx, "end": next_idx, "text": sub_text})
        tail = elem.tail or ""
        text += sub_text + tail
        next_idx += len(tail)
        idx = next_idx
    return text, idx, mentions


def agg_type(x, merge_composite_types=True):
    if merge_composite_types:
        x = next(iter(re.findall(r"G#[a-zA-Z_]+", x)), None)
    if x.startswith("G#DNA_"):
        return "DNA"
    elif x.startswith("G#RNA_"):
        return "RNA"
    elif x.startswith("G#protein_"):
        return "protein"
    elif x.startswith("G#cell_type"):
        return "cell_type"
    elif x.startswith("G#cell_line"):
        return "cell_line"
    return None


def load_genia_ner(resource_path="genia_ner", version="3.02p", doc_attributes={}, raw=False, merge_composite_types=True, drop_duplicates=False, test_split=0.1):
    path = root.resource(resource_path)
    [file] = ensure_files(path, [GENIA_NER_REMOTE_FILES[version]], mode=NetworkLoadMode.AUTO)
    with tarfile.open(file, "r:gz") as tar:
        tar.extractall(path)

    filename = {"3.02": "GENIAcorpus3.02.merged.xml", "3.02p": "GENIAcorpus3.02.merged.xml"}

    root_node = ET.parse(path / filename[version]).getroot()
    all_mentions = []
    all_docs = []
    seen_docs = set()
    for article in root_node.findall('article'):
        doc_id = article.findall('articleinfo/bibliomisc')[0].text
        if doc_id in seen_docs:
            if drop_duplicates:
                continue
            else:
                doc_id = doc_id + "-bis"
        seen_docs.add(doc_id)
        title = article.findall('title/sentence')
        abstract = article.findall('abstract/sentence')
        offset = 0
        mentions = []
        text = ""
        for sent in chain(title, [None], abstract):
            # Double line break between title and abstract
            if sent is None:
                text += "\n"
                offset += 1
                continue
            sent_text, offset, sent_mentions = process_xml(sent, offset)
            mentions.extend(sent_mentions)
            text += sent_text
            text += "\n"
            offset += 1
        if len(mentions):
            text = text[:-1]
            offset -= 1
        all_mentions.extend({"doc_id": doc_id, "begin": m["begin"], "end": m["end"], "text": m["text"], **m["attrib"]} for m in mentions)
        all_docs.append({"doc_id": doc_id, "text": text, **doc_attributes})
    mentions = pd.DataFrame(all_mentions)
    del mentions["lex"]
    docs = pd.DataFrame(all_docs)
    docs["split"] = ["train"] * (len(docs) - int(len(docs) * 0.1)) + ["test"] * int(len(docs) * 0.1)

    if not raw:
        mentions = mentions.query('~sem.isna()').copy()
        mentions["label"] = mentions["sem"].apply(lambda x: agg_type(x, merge_composite_types=merge_composite_types))
        mentions = mentions.query("~label.isna()").copy()
    else:
        mentions["label"] = mentions["sem"]

    mentions["mention_id"] = np.arange(len(mentions))
    mentions["fragment_id"] = np.arange(len(mentions))

    attributes = pd.DataFrame({"doc_id": [], "mention_id": [], "attribute_id": [], "label": [], "value": []}).astype(str)

    return Dataset(docs=docs,
                   mentions=mentions[["doc_id", "mention_id", "label", "text"]],
                   fragments=mentions[["doc_id", "mention_id", "fragment_id", "begin", "end"]],
                   attributes=attributes)
