import os

import pandas as pd
from sklearn.utils import check_random_state

from nlstruct.core.environment import RelativePath
from nlstruct.core.dataset import Dataset


def load_from_brat(path, validation_split=0.2, random_state=42):
    path = RelativePath(path)
    mentions = []
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
                            "begin": tuple(int(s.split()[0]) for s in span.split(";")),
                            "end": tuple(int(s.split()[1]) for s in span.split(";")),
                            "text": text,
                        })
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
                            "label": ann_name,
                            "from": parts[0].split(":")[1],
                            "to": parts[1].split(":")[1],
                        })
    mentions = pd.DataFrame(mentions).astype({
        "doc_id": docs["doc_id"].dtype,
        "mention_id": "category",
        "label": "category",
    })
    fragments = mentions[["doc_id", "mention_id", "begin", "end"]].nlp.flatten("fragment_id").astype({
        "fragment_id": "category",
    })[["doc_id", "mention_id", "fragment_id", "begin", "end"]]
    mentions = mentions[["doc_id", "mention_id", "label", "text"]]
    attributes = pd.DataFrame(attributes).astype({
        "doc_id": docs["doc_id"].dtype,
        "mention_id": mentions.dtypes["mention_id"],
        "attribute_id": "category",
        "label": "category",
        "value": "category",
    })[["doc_id", "mention_id", "attribute_id", "label", "value"]]
    relations = pd.DataFrame(relations).astype({
        "doc_id": docs["doc_id"].dtype,
        "relation_id": "category",
        "label": "category",
        "from": mentions.dtypes["mention_id"],
        "to": mentions.dtypes["mention_id"],
    })[["doc_id", "relation_id", "label", "from", "to"]]

    return Dataset(docs=docs, mentions=mentions, fragments=fragments, attributes=attributes, relations=relations)
