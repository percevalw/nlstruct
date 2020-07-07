import os

import pandas as pd

from nlstruct.utils import encode_ids


def export_to_brat(dataset, dest=None, filename_prefix=""):
    doc_id_to_text = dict(zip(dataset["docs"]["doc_id"], dataset["docs"]["text"]))
    counter = 0
    mention_counter = 0
    mentions = dataset["mentions"]
    attributes = dataset["attributes"]
    relations = dataset["relations"]
    assert mentions is not None, "Dataset must contain a 'docs' frame, a 'mentions' frame, and a 'fragments' frame or 'begin' and 'end' columns in those mentions"
    if "begin" not in mentions:
        mentions = mentions.merge(dataset["fragments"])
    if dest is not None:
        try:
            os.mkdir(dest)
        except FileExistsError:
            pass
    for doc_id, text in doc_id_to_text.items():
        if not os.path.exists("{}/{}.txt"):
            with open("{}/{}.txt".format(dest, filename_prefix + doc_id), "w") as f:
                f.write(text)

        doc_mentions = mentions.query(f'doc_id == "{doc_id}"').sort_values(["doc_id", "begin"])
        doc_attributes = attributes.query(f'doc_id == "{doc_id}"')
        doc_relations = relations.query(f'doc_id == "{doc_id}"')

        # encode_ids([doc_attributes], ("doc_id", "mention_id", "attribute_id"))
        encode_ids([doc_mentions, doc_attributes, doc_relations, doc_relations],
                   [("doc_id", "mention_id"), ("doc_id", "mention_id"), ("doc_id", "from_mention_id"), ("doc_id", "to_mention_id")])
        counter += 1
        f = None
        if dest is not None:
            f = open("{}/{}.ann".format(dest, filename_prefix + doc_id), "w")
        try:
            for _, row in doc_mentions.iterrows():
                mention_text = text[row["begin"]:row["end"]]
                idx = row["begin"]
                mention_i = row["mention_id"] + 1
                spans = []
                for part in mention_text.split("\n"):
                    begin = idx
                    end = idx + len(part)
                    idx = end + 1
                    if begin != end:
                        spans.append((begin, end))
                    else:
                        print("!!!!!!!!!!!!!!!!!!!")
                print("T{}\t{} {}\t{}".format(
                    mention_i,
                    str(row["label"]),
                    ";".join(" ".join(map(str, span)) for span in spans),
                    mention_text.replace("\n", " ")), file=f)
            for i, (_, row) in enumerate(doc_attributes.iterrows()):
                mention_i = row["mention_id"] + 1
                if not pd.isna(row["value"]):
                    print("A{}\t{} T{} {}".format(
                        i + 1,
                        str(row["label"]),
                        mention_i,
                        row["value"]), file=f)
                else:
                    print("A{}\t{} T{}".format(
                        i + 1,
                        str(row["label"]),
                        mention_i), file=f)
            for i, (_, row) in enumerate(doc_relations.iterrows()):
                mention_from = row["from_mention_id"] + 1
                mention_to = row["to_mention_id"] + 1
                print("R{}\t{} Arg1:T{} Arg2:T{}\t".format(
                    i + 1,
                    str(row["relation_label"]),
                    mention_from,
                    mention_to), file=f)
        finally:
            if f is not None:
                f.close()
