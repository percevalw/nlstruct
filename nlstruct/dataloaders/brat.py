import os

import pandas as pd
from sklearn.utils import check_random_state

from nlstruct.collections.dataset import Dataset
from nlstruct.environment.path import RelativePath


def load_from_brat(path, validation_split=0.2, random_state=42, merge_newlines=True, doc_attributes=None):
    """
    Load a brat dataset into a Dataset object

    Parameters
    ----------
    path: str or pathlib.Path
    validation_split: float
    random_state: int
        Seed
    merge_newlines: bool
        Merge fragments of a mention that was splited by brat because it overlapped an end of line
    doc_attributes: dict
        Attributes that will be added in the docs dataframe for all entries

    Returns
    -------
    Dataset
    """

    path = RelativePath(path)
    mentions = []
    fragments = []
    attributes = []
    relations = []
    comments = []

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

    if doc_attributes is not None:
        for key, val in doc_attributes.items():
            docs[key] = val

    # Test if path contains any .ann files
    # If not: only return docs
    if not any(fname.endswith('.ann') for fname in os.listdir(path)):
        return Dataset(docs=docs)
    
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
                    elif ann_id.startswith('#'):
                        remaining = remaining.strip(" \t").split("\t")
                        [mention_id, comment] = remaining + ([""] if len(remaining) < 2 else [])
                        ann_type, mention_id = mention_id.split(" ")
                        if ann_type == "AnnotatorNotes":
                            comments.append({
                                "doc_id": doc_id,
                                "comment_id": ann_id,
                                "mention_id": mention_id,
                                "comment": comment,
                            })
    mentions = pd.DataFrame(mentions, columns=["doc_id", "mention_id", "label", "text"])
    fragments = pd.DataFrame(fragments, columns=["doc_id", "mention_id", "fragment_id", "begin", "end"])
    attributes = pd.DataFrame(attributes, columns=["doc_id", "mention_id", "attribute_id", "label", "value"])
    relations = pd.DataFrame(relations, columns=["doc_id", "relation_id", "relation_label", "from_mention_id", "to_mention_id"])
    comments = pd.DataFrame(comments, columns=["doc_id", "comment_id", "mention_id", "comment"])

    return Dataset(docs=docs, mentions=mentions, fragments=fragments, attributes=attributes, relations=relations, comments=comments)
