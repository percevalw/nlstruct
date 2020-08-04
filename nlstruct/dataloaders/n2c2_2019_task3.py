import os

import pandas as pd
from sklearn.utils import check_random_state

from nlstruct.collections.dataset import Dataset
from nlstruct.environment.path import root
from nlstruct.utils.pandas import merge_with_spans


def load_n2c2_2019_task3_split(validation_split=0.2, random_state=42, split="train"):
    path = root.resource("n2c2/".format(split))
    dataset = []
    for filename in sorted(os.listdir(path / '{}_norm'.format(split))):
        if filename.endswith('.norm'):
            with open(path / '{}_norm'.format(split) / filename) as f:
                for line in f:
                    (mention_id, label, *spans) = line.strip('\n').split('||')
                    begins, ends = [int(b) for b in spans[::2]], [int(e) for e in spans[1::2]]
                    dataset.append({
                        "doc_id": filename.replace('.norm', ''),
                        "mention_id": mention_id,
                        "label": label,
                        "begin": begins,
                        "end": ends,
                    })
    texts = []
    for filename in sorted(os.listdir(path / '{}_note'.format(split))):
        if filename.endswith('.txt'):
            with open(path / '{}_note'.format(split) / filename) as f:
                texts.append({"doc_id": filename.replace('.txt', ''), "text": f.read().strip('\n')})
    with open(path / '{}_file_list.txt'.format(split)) as f:
        train_files = pd.Series([n.strip('\n') for n in f.readlines()])
        train_files.name = 'doc_id'

    docs = merge_with_spans(train_files, pd.DataFrame(texts), on='doc_id')
    rng = check_random_state(random_state)
    if split == "train":
        docs['split'] = rng.choice(['train', 'val'], size=len(docs),
                                   p=[1 - validation_split, validation_split])
    else:
        docs['split'] = 'test'

    mentions = pd.DataFrame(dataset)
    fragments = mentions[['doc_id', 'mention_id', 'begin', 'end']].nlstruct.flatten("fragment_id", tile_index=False).astype({"fragment_id": object})
    return Dataset(
        docs=docs[["doc_id", "text", "split"]],
        mentions=mentions[["doc_id", "mention_id", "label"]],
        fragments=fragments[["doc_id", "mention_id", "fragment_id", "begin", "end"]]
    )


def load_n2c2_2019_task3(validation_split=0.2, random_state=42):
    return Dataset.concat([
        load_n2c2_2019_task3_split(validation_split=validation_split, random_state=random_state, split="train"),
        load_n2c2_2019_task3_split(split="test"),
    ])
