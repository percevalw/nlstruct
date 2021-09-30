import os
import re
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from itertools import chain

from sklearn.datasets._base import RemoteFileMetadata

from nlstruct.data_utils import sentencize, regex_sentencize, slice_document
from nlstruct.datasets.base import NetworkLoadMode, ensure_files, NERDataset


class GENIA(NERDataset):
    REMOTE_FILES = {
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

    def __init__(self, path, test_split=0.1, val_split=0.1, version="3.02p", debug=False, preprocess_fn=None, split_by_sentences=True, merge_composite_types=True):
        train_data, val_data, test_data = self.download_and_extract(
            path, version, debug, test_split=test_split, val_split=val_split, split_by_sentences=split_by_sentences, merge_composite_types=merge_composite_types)
        super().__init__(train_data, val_data, test_data, preprocess_fn=preprocess_fn)

    def process_xml(self, root_node, idx=0):
        text = ""
        if root_node.text is not None:
            text = root_node.text
            idx += len(root_node.text)
        mentions = []
        for elem in root_node:
            sub_text, next_idx, sub_mentions = self.process_xml(elem, idx)
            mentions.extend(sub_mentions)
            if elem.tag == "cons":
                mentions.append({"attrib": elem.attrib, "begin": idx, "end": next_idx, "text": sub_text})
            tail = elem.tail or ""
            text += sub_text + tail
            next_idx += len(tail)
            idx = next_idx
        return text, idx, mentions

    def agg_type(self, x, merge_composite_types=True):
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

    def download_and_extract(self, path, version, debug=False, raw=False, merge_composite_types=True, drop_duplicates=False,
                             test_split=0.1, val_split=0.1, split_by_sentences=True):
        remote = self.REMOTE_FILES[version]
        [file] = ensure_files(path, [remote], mode=NetworkLoadMode.AUTO)
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall(path)

        filename = os.path.join(path, {"3.02": "GENIA_term_3.02/GENIAcorpus3.02.xml", "3.02p": "GENIAcorpus3.02.merged.xml"}[version])

        temp = tempfile.NamedTemporaryFile(mode='w', suffix=".xml", delete=False)

        with open(filename) as rawtext:
            rawtext = rawtext.read()
            rawtext = rawtext.replace("HMG-I(Y)</cons>", '<w c="NN">HMG-I(Y)</w></cons>')

        temp.write(rawtext)
        temp.close()
        root_node = ET.parse(temp.name).getroot()
        seen_docs = set()
        genia_docs = []
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
                sent_text, offset, sent_mentions = self.process_xml(sent, offset)
                mentions.extend(sent_mentions)
                text += sent_text
                text += "\n"
                offset += 1
            if len(mentions):
                text = text[:-1]
                offset -= 1
            genia_docs.append({
                "doc_id": doc_id,
                "text": text,
                "entities": list(filter(lambda x: x["label"] is not None, (
                    {
                        "entity_id": doc_id + "-" + str(i),
                        "fragments": [{"begin": m["begin"], "end": m["end"]}],
                        "text": m["text"],
                        **m["attrib"],
                        "label": self.agg_type(m["attrib"]["sem"], merge_composite_types=merge_composite_types) if not raw else m["attrib"]["sem"]
                    }
                    for i, m in enumerate(mentions)
                    if m["attrib"].get("sem", None)
                ))),
            })

        if split_by_sentences:
            n_sentences = len(list(sentencize(genia_docs, reg_split="\n+", balance_chars=(), chain=True)))
            n_train_sentences = 0
            n_val_sentences = 0
            n_test_sentences = 0
            train_data = []
            val_data = []
            test_data = []
            target_n_test_sentences = int(n_sentences * test_split)
            target_n_val_sentences = int((n_sentences - target_n_test_sentences) * val_split)
            target_n_train_sentences = n_sentences - target_n_test_sentences - target_n_val_sentences
            current_queue = None
            current_count = 0
            current_target = 0
            total = 0
            while len(genia_docs):
                doc = genia_docs.pop(0)
                if n_train_sentences != target_n_train_sentences:
                    current_queue = train_data
                    current_target = target_n_train_sentences
                elif n_val_sentences != target_n_val_sentences:
                    current_queue = val_data
                    current_target = target_n_val_sentences
                elif n_test_sentences != target_n_test_sentences:
                    current_queue = test_data
                    current_target = target_n_test_sentences

                sents = list(regex_sentencize(doc["text"], "(\n+)", balance_chars=()))

                if current_count + len(sents) <= current_target:
                    current_queue.append(doc)
                    current_count += len(sents)
                    total += len(sents)
                    continue
                elif current_count + len(sents) > current_target:
                    total += len(sents[:current_target - current_count])
                    current_queue.append(slice_document(doc, 0, sents[current_target - current_count - 1][1]))
                    genia_docs.insert(0, slice_document(doc, sents[current_target - current_count][0], len(doc["text"])))
                    current_count = current_target
                if current_queue is train_data:
                    n_train_sentences = current_count
                if current_queue is val_data:
                    n_val_sentences = current_count
                if current_queue is test_data:
                    n_test_sentences = current_count
                if current_count == current_target:
                    current_count = 0
        else:
            target_n_test_docs = int(len(genia_docs) * test_split)
            target_n_val_docs = int((len(genia_docs) - target_n_test_docs) * val_split)
            target_n_train_docs = len(genia_docs) - target_n_test_docs - target_n_val_docs

            train_data = genia_docs[:target_n_train_docs]
            val_data = genia_docs[target_n_train_docs:target_n_train_docs + target_n_val_docs]
            test_data = genia_docs[target_n_train_docs + target_n_val_docs:]

        if debug:
            train_data = train_data[:10]
            val_data = val_data[:10]

        return train_data, val_data, test_data
