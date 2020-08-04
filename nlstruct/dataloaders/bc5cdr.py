import pandas as pd

from nlstruct.collections.dataset import Dataset
from nlstruct.environment.path import root


# remote_files = [
#     RemoteFileMetadata(
#         url="https://raw.githubusercontent.com/openbiocorpora/biocreative-v-cdr/master/original-data/train/"
#             "CDR_TrainingSet.PubTator.PubTator.txt",
#         checksum="a37e416daf5b3042ff98a1f3110e1efdb4454ee76ebc0dd7f5f4c2c77c4adb75",
#         filename="CDR_TrainingSet.PubTator.PubTator.txt"),
#     RemoteFileMetadata(
#         url="https://raw.githubusercontent.com/openbiocorpora/biocreative-v-cdr/master/original-data/devel/"
#             "CDR_DevelopmentSet.PubTator.txt",
#         checksum="9e18d8e700168887a7919e62f67ac2ce2357e3f409a48ee992906abdd80fe3e5",
#         filename="CDR_DevelopmentSet.PubTator.txt"),
# ]


def load_bc5cdr():
    train_file = root.resource(root['BC5CDR_TRAIN_PATH'])
    dev_file = root.resource(root['BC5CDR_DEV_PATH'])
    test_file = root.resource(root['BC5CDR_TEST_PATH'])

    entries = []
    # Load full datasets with concept annotations
    for split, file in [("train", train_file),
                        ("test", test_file),
                        ("dev", dev_file)]:
        with open(str(file), "r") as cursor:
            entry = {"doc_id": None,
                     "begin": [],
                     "end": [],
                     "synonym": [],
                     "category": [],
                     "label": [],
                     "source": [],
                     "code": [],
                     "split": split}  # accumulate sample info here
            counter = -1  # count the line number in the current sample

            for line in cursor.readlines():
                counter += 1
                # end of sample, yield it
                if not line.strip():
                    if entry["doc_id"]:
                        entries.append(entry)
                        entry = {"doc_id": None,
                                 "begin": [],
                                 "end": [],
                                 "synonym": [],
                                 "category": [],
                                 "label": [],
                                 "source": [],
                                 "code": [],
                                 "split": split}  # accumulate sample info here
                    counter = -1
                elif counter == 0:
                    sample_id, title = line.split("|t|")
                    entry["doc_id"] = sample_id
                    entry["title"] = title
                elif counter == 1:
                    entry["abstract"] = line.split("|a|")[1].strip("\n")
                else:
                    _, begin, end, synonym, category, cui_set = line[:-1].split("\t")
                    entry["begin"].append(int(begin))
                    entry["end"].append(int(end))
                    entry["synonym"].append(synonym)
                    entry["category"].append(category)
                    labels = [c2.strip() for c1 in cui_set.split("|") for c2 in c1.split('+')]
                    entry["source"].append(['OMIM' if l.startswith('OMIM:') else "MSH" if l != "-1" else ""
                                            for l in labels])
                    entry["code"].append([l.split(":")[-1] if l != "-1" else ""
                                          for l in labels])
                    entry["label"].append([":".join((source, code)) if code != "" else ""
                                           for source, code in zip(entry["source"][-1], entry["code"][-1])])
            entries.append(entry)
    res = pd.DataFrame(entries)
    res['text'] = res['title'] + res['abstract']
    raw = res[~res['doc_id'].duplicated()].reset_index(drop=True)

    # Transform the raw dataset as a Dataset instance
    raw = raw
    mentions = raw[["doc_id", "begin", "end", "label", "category", "text"]].nlstruct.flatten("mention_id", tile_index=True)
    mentions["text"] = mentions.apply(lambda row: row["text"][row["begin"]:row["end"]], axis=1)
    mentions["mention_id"] = (mentions["doc_id"].astype("str") + "-" + mentions["mention_id"].astype(str))
    mentions_label = mentions[["doc_id", "mention_id", "label"]].nlstruct.flatten("label_id", tile_index=True).astype({"label_id": object})
    mentions_label["label_id"] = (mentions_label["mention_id"].astype("str") + "-" + mentions_label["label_id"].astype(str))
    mentions_label.loc[mentions_label["label"] == "", "label"] = None
    fragments = mentions[["doc_id", "mention_id", "begin", "end"]].copy()
    fragments["fragment_id"] = fragments["mention_id"]
    return Dataset(
        docs=raw[["doc_id", "text"]],
        mentions=mentions[["doc_id", "mention_id", "category", "text"]],
        labels=mentions_label,
        fragments=fragments,
    )
