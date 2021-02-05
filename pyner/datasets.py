import os


def load_from_brat(path, merge_spaced_fragments=True):
    """
    Load a brat dataset into a Dataset object
    Parameters
    ----------
    path: str or pathlib.Path
    merge_spaced_fragments: bool
        Merge fragments of a mention that was splited by brat because it overlapped an end of line
    Returns
    -------
    Dataset
    """

    # Extract annotations from path and make multiple dataframe from it
    docs = []
    for filename in ((path + name for name in sorted(os.listdir(path))) if isinstance(path, str) else path):
        mentions = {}
        relations = []
        if filename.endswith('.txt'):
            doc_id = filename.replace('.txt', '').split("/")[-1]

            with open(filename) as f:
                text = f.read()

            try:
                with open(filename.replace(".txt", ".ann")) as f:
                    for line in f:
                        ann_parts = line.strip('\n').split('\t', 1)
                        ann_id, remaining = ann_parts
                        if ann_id.startswith('T'):
                            remaining, mention_text = remaining.split("\t")
                            mention, span = remaining.split(" ", 1)
                            mentions[ann_id] = {
                                "mention_id": ann_id,
                                "fragments": [],
                                "attributes": [],
                                "comments": [],
                                "label": mention,
                                "text": mention_text,
                            }
                            last_end = None
                            fragment_i = 0
                            for s in span.split(';'):
                                begin, end = int(s.split()[0]), int(s.split()[1])
                                # If merge_newlines, merge two fragments that are only separated by a newline (brat automatically creates
                                # multiple fragments for a mention that spans over more than one line)
                                if merge_spaced_fragments and begin - 1 == last_end and len(mention_text[last_end:begin].strip()) == 0:
                                    mentions[ann_id]["fragments"][-1]["end"] = end
                                    continue
                                mentions[ann_id]["fragments"].append({
                                    "begin": begin,
                                    "end": end,
                                })
                                fragment_i += 1
                                last_end = end
                        elif ann_id.startswith('A'):
                            parts = remaining.split(" ")
                            if len(parts) >= 3:
                                mention, mention_id, value = parts
                            else:
                                mention, mention_id = parts
                                value = None
                            mentions[mention_id]["attributes"].append({
                                "attribute_id": ann_id,
                                "label": mention,
                                "value": value,
                            })
                        elif ann_id.startswith('R'):
                            [ann_name, *parts] = remaining.strip("\t").split(" ")
                            relations.append({
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
                                mentions[mention_id]["comments"].append({
                                    "comment_id": ann_id,
                                    "comment": comment,
                                })
            except FileNotFoundError:
                yield {
                    "doc_id": doc_id,
                    "text": text,
                }
            else:
                yield {
                    "doc_id": doc_id,
                    "text": text,
                    "mentions": list(mentions.values()),
                    "relations": relations,
                }
    return docs
