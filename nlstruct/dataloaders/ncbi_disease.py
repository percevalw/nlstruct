import zipfile

import pandas as pd
from sklearn.datasets._base import RemoteFileMetadata

from nlstruct.environment.cache import get_cache
from nlstruct.utils.network import ensure_files, NetworkLoadMode
from nlstruct.collections.dataset import Dataset

remote_files = [
    RemoteFileMetadata(
        url="https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip",
        checksum="26157233d70aeda0b2ac1dda4fc9369b0717bd888f5afe511d0c1c6a5ad307a0",
        filename="NCBItrainset_corpus.zip"),
    RemoteFileMetadata(
        url="https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip",
        checksum="b978442f39c739deb6619c70e7b07327d9eb1b71aff64996c02a592435583f46",
        filename="NCBItestset_corpus.zip"),
    RemoteFileMetadata(
        url="https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip",
        checksum="61681ad09356619c8f4f3e1738663dd007ad0135720fdc90195120fea944cbe9",
        filename="NCBIdevelopset_corpus.zip"),
]


def load_ncbi_disease():
    download_path = get_cache("ncbi_raw_files")
    train_file, test_file, dev_file = ensure_files(download_path, remote_files, mode=NetworkLoadMode.AUTO)

    entries = []
    # Load full datasets with concept annotations
    for split, file in [("train", train_file),
                        ("test", test_file),
                        ("dev", dev_file)]:
        zip_ref = zipfile.ZipFile(file, "r")
        zip_ref.extractall(download_path)
        zip_ref.close()
        with open(str(file).replace(".zip", ".txt"), "r") as cursor:
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
                    entry["source"].append(['OMIM' if l.startswith('OMIM:') else "MSH"
                                            for l in labels])
                    entry["code"].append([l.split(":")[-1]
                                          for l in labels])
                    entry["label"].append([":".join((source, code))
                                           for source, code in zip(entry["source"][-1], entry["code"][-1])])
            entries.append(entry)
    res = pd.DataFrame(entries)
    res['text'] = res['title'] + res['abstract']
    raw = res[~res['doc_id'].duplicated()].reset_index(drop=True)

    # Transform the raw dataset as a Dataset instance
    raw = raw
    mentions = raw[["doc_id", "begin", "end", "label", "category", "text"]].nlstruct.flatten("mention_id", tile_index=True)
    mentions["mention_id"] = (mentions["doc_id"].astype("str") + "-" + mentions["mention_id"].astype(str))
    mentions["text"] = mentions.apply(lambda row: row["text"][row["begin"]:row["end"]], axis=1)
    mentions_label = mentions[["doc_id", "mention_id", "label"]].nlstruct.flatten("label_id", tile_index=True).astype({"label_id": object})
    mentions_label["label_id"] = (mentions_label["mention_id"].astype("str") + "-" + mentions_label["label_id"].astype(str))
    fragments = mentions[["doc_id", "mention_id", "begin", "end"]].copy()
    fragments["fragment_id"] = fragments["mention_id"]
    return Dataset(
        docs=raw[["doc_id", "text", "split"]],
        mentions=mentions[["doc_id", "mention_id", "category", "text"]],
        labels=mentions_label,
        fragments=fragments,
    )
