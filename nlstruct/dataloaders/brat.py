import os

import pandas as pd
from sklearn.utils import check_random_state

from nlstruct.core.environment import RelativePath
from nlstruct.core.dataset import Dataset


def load_from_brat(path, validation_split=0.2, random_state=42, merge_newlines=True):
    path = RelativePath(path)
    mentions = []
    fragments = []
    attributes = []
    relations = []

    # Extract texts from path and make a dataframe from it
    texts = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.txt'):
            with open(path / filename) as f:
                texts.append({"doc_id": filename.replace('.txt', ''), "text": f.read()})
    docs = pd.DataFrame(texts).astype({"text": "category"})
    rng = check_random_state(random_state)
    docs['split'] = rng.choice(['train', 'val'], size=len(docs),
                               p=[1 - validation_split, validation_split])
    docs = docs.astype({
        "doc_id": "category",
        "split": "category",
    })

    # Extract annotations from path and make multiple dataframe from it
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.ann'):
            doc_id = filename.replace('.ann', '')
            with open(path / filename) as f:
                for line in f:
                    ann_parts = line.strip('\n').split('\t', 1)
                    ann_id, remaining = ann_parts
                    if ann_id.startswith('T'):
                        remaining, text = remaining.split("\t")
                        mention, span = remaining.split(" ", 1)
                        mentions.append({
                            "doc_id": doc_id,
                            "mention_id": ann_id,
                            "label": mention,
                            "text": text,
                        })
                        last = None
                        fragment_i = 0
                        for s in span.split(';'):
                            begin, end = int(s.split()[0]), int(s.split()[1])
                            # If merge_newlines, merge two fragments that are only separated by a newline (brat automatically creates
                            # multiple fragments for a mention that spans over more than one line)
                            if merge_newlines and begin - 1 == last and text[last:begin] == '\n':
                                fragments[-1]["end"] = end
                                continue
                            fragments.append({
                                "doc_id": doc_id,
                                "mention_id": ann_id,
                                "fragment_id": "{}-{}".format(ann_id, fragment_i),
                                "begin": begin,
                                "end": end,
                            })
                            fragment_i += 1
                    elif ann_id.startswith('A'):
                        parts = remaining.split(" ")
                        if len(parts) >= 3:
                            mention, mention_id, value = parts
                        else:
                            mention, mention_id = parts
                            value = None
                        attributes.append({
                            "doc_id": doc_id,
                            "attribute_id": ann_id,
                            "mention_id": mention_id,
                            "label": mention,
                            "value": value,
                        })
                    elif ann_id.startswith('R'):
                        [ann_name, *parts] = remaining.strip("\t").split(" ")
                        relations.append({
                            "doc_id": doc_id,
                            "relation_id": ann_id,
                            "relation_label": ann_name,
                            "from_mention_id": parts[0].split(":")[1],
                            "to_mention_id": parts[1].split(":")[1],
                        })
    mentions = pd.DataFrame(mentions)
    fragments = pd.DataFrame(fragments)
    mentions = mentions[["doc_id", "mention_id", "label", "text"]]
    if len(attributes):
        attributes = pd.DataFrame(attributes)[["doc_id", "mention_id", "attribute_id", "label", "value"]]
    else:
        attributes = pd.DataFrame(columns=["doc_id", "mention_id", "attribute_id", "label", "value"])
    if len(relations):
        relations = pd.DataFrame(relations)[["doc_id", "relation_id", "relation_label", "from_mention_id", "to_mention_id"]]
    else:
        relations = pd.DataFrame(columns=["doc_id", "relation_id", "relation_label", "from_mention_id", "to_mention_id"])

    return Dataset(docs=docs, mentions=mentions, fragments=fragments, attributes=attributes, relations=relations)
