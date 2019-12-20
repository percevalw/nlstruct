import re
from itertools import repeat

import numpy as np
import pandas as pd
from tqdm import tqdm

from nlstruct.core.cache import cached
from nlstruct.core.pandas import make_merged_names_map, merge_with_spans, make_id_from_merged


def make_tag_scheme(length, entity, scheme='bio'):
    if scheme == "bio":
        return [f"B-{entity}", *(f"I-{entity}" for _ in range(length - 1))]
    raise ValueError(f"'{scheme}' scheme is not supported")


class DeltaCollection(object):
    def __init__(self, begins, ends, deltas):
        self.begins = np.asarray(begins, dtype=int)
        self.ends = np.asarray(ends, dtype=int)
        self.deltas = np.asarray(deltas, dtype=int)

    @classmethod
    def from_absolute(cls, begins, ends, deltas):
        deltas = np.asarray(deltas)
        shift = np.roll(deltas, 1)
        shift[0] = 0
        deltas -= shift
        return DeltaCollection(begins, ends, deltas)

    def __repr__(self):
        return "DeltaCollection([{}], [{}], [{}])".format(", ".join(map(str, self.begins)),
                                                          ", ".join(map(str, self.ends)),
                                                          ", ".join(map(str, self.deltas)))

    def apply(self, positions, side='left'):
        positions = np.asarray(positions)
        to_add = ((positions.reshape(-1, 1) >= self.ends.reshape(1, -1)) * self.deltas).sum(axis=1)
        between = np.logical_and(self.begins.reshape(1, -1) < positions.reshape(-1, 1),
                                 positions.reshape(-1, 1) < self.ends.reshape(1, -1))
        between_mask = between.any(axis=1)
        between = between[between_mask]
        between_i = between.argmax(axis=1)
        if side == 'right':
            to_add[between_mask] += self.ends[between_i] - positions[between_mask] + self.deltas[between_i]
        elif side == 'left':
            to_add[between_mask] += self.begins[between_i] - positions[between_mask]
        return positions + to_add

    def unapply(self, positions, side='left'):
        positions = np.asarray(positions)
        begins = self.apply(self.begins, side='left')
        ends = self.apply(self.ends, side='right')
        to_remove = -((positions.reshape(-1, 1) >= ends.reshape(1, -1)) * self.deltas).sum(axis=1)
        between = np.logical_and(begins.reshape(1, -1) < positions.reshape(-1, 1),
                                 positions.reshape(-1, 1) < ends.reshape(1, -1))
        between_mask = between.any(axis=1)
        between = between[between_mask]
        between_i = between.argmax(axis=1)
        if side == 'right':
            to_remove[between_mask] += ends[between_i] - positions[between_mask] - self.deltas[between_i]
        elif side == 'left':
            to_remove[between_mask] += begins[between_i] - positions[between_mask]
        pos = positions + to_remove
        return pos

    def __add__(self, other):
        if len(self.begins) == 0:
            return other
        if len(other.begins) == 0:
            return self
        begins = self.unapply(other.begins, side='left')
        ends = self.unapply(other.ends, side='right')
        new_begins = np.concatenate([begins, self.begins])
        new_ends = np.concatenate([ends, self.ends])
        new_deltas = np.concatenate([other.deltas, self.deltas])
        sorter = np.lexsort((new_ends, new_begins))
        return DeltaCollection(new_begins[sorter], new_ends[sorter], new_deltas[sorter])


def make_str_from_groups(replacement, groups):
    for i, group in enumerate(groups):
        replacement = replacement.replace(f"\\{i+1}", group)
    return replacement


def regex_sub_with_spans(pattern, replacement, text):
    needed_groups = [int(i) for i in re.findall(r"\\([0-9]+)", replacement)]
    begins = []
    ends = []
    deltas = []
    for match in reversed(list(re.finditer(pattern, text))):
        middle = make_str_from_groups(replacement, [match.group(i) for i in needed_groups])
        start = match.start()
        end = match.end()
        text = text[:start] + middle + text[end:]
        begins.append(start)
        ends.append(end)
        deltas.append(len(middle) - end + start)
    return text, DeltaCollection(begins, ends, deltas)


def regex_multisub_with_spans(patterns, replacements, text):
    deltas = DeltaCollection([], [], [])
    for pattern, replacement in zip(patterns, replacements):
        text, new_deltas = regex_sub_with_spans(pattern, replacement, text)
        if deltas is not None:
            deltas += new_deltas
        else:
            deltas = new_deltas
    return text, deltas


@cached
def transform_text(dataset,
                   global_patterns=None,
                   global_replacements=None, return_deltas=True, with_tqdm=False):
    assert (global_patterns is None) == (global_replacements is None)
    expand_deltas = lambda x: (x[0], tuple(x[1].begins), tuple(x[1].ends), tuple(x[1].deltas))
    if global_patterns is None:
        global_patterns = []
        global_replacements = []
    if return_deltas:
        text, delta_begins, delta_ends, deltas = zip(*[
            expand_deltas(regex_multisub_with_spans(
                [*doc_patterns, *global_patterns],
                [*doc_replacements, *global_replacements],
                text
            )) for text, doc_patterns, doc_replacements in
            (tqdm if with_tqdm else lambda x: x)(zip(
                dataset["text"],
                dataset["patterns"] if "patterns" in dataset.columns else repeat([]),
                dataset["replacements"] if "replacements" in dataset.columns else repeat([])))
        ])
        dataset = pd.DataFrame({
            "text": text,
            "begin": delta_begins,
            "end": delta_ends,
            "delta": deltas,
            **{c: dataset[c] for c in dataset.columns if c not in ("text", "begin", "end", "delta")}
        })
        return (
            dataset[[c for c in dataset.columns if c not in ("begin", "end", "delta")]],
            dataset[["doc_id", "begin", "end", "delta"]].nlp.flatten())
    else:
        new_texts = []
        for text, doc_patterns, doc_replacements in (tqdm if with_tqdm else lambda x: x)(zip(
              dataset["text"],
              dataset["patterns"] if "patterns" in dataset.columns else repeat([]),
              dataset["replacements"] if "replacements" in dataset.columns else repeat([]))):
            for pattern, replacement in zip([*doc_patterns, *global_patterns], [*doc_replacements, *global_replacements]):
                text = re.sub(pattern, replacement, text)
            new_texts.append(text)
        dataset = pd.DataFrame({"text": new_texts,
                                **{c: dataset[c] for c in dataset.columns if c not in ("text",)}})
        return dataset[[c for c in dataset.columns if c not in ("begin", "end", "delta")]]


def apply_deltas(positions, deltas, on, position_columns=None):
    if not isinstance(on, (tuple, list)):
        on = [on]
    if position_columns is None:
        position_columns = {'begin': 'left', 'end': 'right'}

    positions = positions.copy()
    positions['_id_col'] = np.arange(len(positions))

    mention_deltas = merge_with_spans(positions[[*position_columns, *on, '_id_col']], deltas, on=on,
                                      suffixes=('_pos', '_delta'), how='inner')
    # To be faster, we remove categorical columns (they may only be in 'on') before the remaining ops
    mention_deltas = mention_deltas[[c for c in mention_deltas.columns if c not in on]]
    positions = positions.set_index('_id_col')
    mention_deltas = mention_deltas.set_index('_id_col')

    delta_col_map, positions_col_map = make_merged_names_map(deltas.columns, [*position_columns, *on, '_id_col'],
                                                             left_on=on, right_on=on, suffixes=('_delta', '_pos'))
    for col, side in position_columns.items():
        mention_deltas.eval(f"shift = ({delta_col_map['end']} <= {positions_col_map[col]}) * {delta_col_map['delta']}",
                            inplace=True)
        mention_deltas.eval(
            f"between_magnet = {delta_col_map['begin']} < {positions_col_map[col]} and {positions_col_map[col]} < {delta_col_map['end']}",
            inplace=True)
        if side == "left":
            mention_deltas.eval(
                f"between_magnet = between_magnet * ({delta_col_map['begin']} - {positions_col_map[col]})",
                inplace=True)
        elif side == "right":
            mention_deltas.eval(
                f"between_magnet = between_magnet * ({delta_col_map['end']} + {delta_col_map['delta']} - {positions_col_map[col]})",
                inplace=True)
        order = "first" if side == "left" else "last"
        tmp = mention_deltas.sort_values(['_id_col', delta_col_map['begin' if side == 'left' else 'end']]).groupby(
            '_id_col').agg({
            "shift": "sum",
            **{n: order for n in mention_deltas.columns if n not in ("shift", "_id_col")}})
        positions[col] = positions[col].add(tmp['shift'] + tmp['between_magnet'], fill_value=0)
    positions = positions.reset_index(drop=True)
    return positions


def reverse_deltas(positions, deltas, on, position_columns=None):
    if not isinstance(on, (tuple, list)):
        on = [on]
    if position_columns is None:
        position_columns = {'begin': 'left', 'end': 'right'}

    positions = positions.copy()
    positions['_id_col'] = np.arange(len(positions))

    deltas = apply_deltas(deltas, deltas, on, position_columns={'begin': 'left', 'end': 'right'})
    mention_deltas = merge_with_spans(positions[[*position_columns, *on, '_id_col']], deltas, on=on,
                                      suffixes=('_pos', '_delta'), how='left')

    positions = positions.set_index('_id_col')
    mention_deltas = mention_deltas.set_index('_id_col')

    # To be faster, we remove categorical columns (they may only be in 'on') before the remaining ops
    # mention_deltas = mention_deltas[[c for c in mention_deltas.columns if c not in on]]
    delta_col_map, positions_col_map = make_merged_names_map(deltas.columns, [*position_columns, *on, '_id_col'],
                                                             left_on=on, right_on=on, suffixes=('_delta', '_pos'))
    for col, side in position_columns.items():
        mention_deltas.eval(
            f"shift = ({delta_col_map['end']} <= {positions_col_map[col]}) * (-{delta_col_map['delta']})",
            inplace=True)
        mention_deltas.eval(
            f"between_magnet = {delta_col_map['begin']} < {positions_col_map[col]} and {positions_col_map[col]} < {delta_col_map['end']}",
            inplace=True)
        if side == "left":
            mention_deltas.eval(
                f"between_magnet = between_magnet * ({delta_col_map['begin']} - {positions_col_map[col]})",
                inplace=True)
        elif side == "right":
            mention_deltas.eval(
                f"between_magnet = between_magnet * ({delta_col_map['end']} - {delta_col_map['delta']} - {positions_col_map[col]})",
                inplace=True)
        order = "first" if side == "left" else "last"

        tmp = mention_deltas.sort_values(['_id_col', delta_col_map['begin' if side == 'left' else 'end']])

        tmp = tmp.groupby('_id_col').agg({
            "shift": "sum",
            **{n: order for n in mention_deltas.columns if n not in ("shift", "_id_col")}})
        positions[col] = positions[col].add(tmp['shift'] + tmp['between_magnet'], fill_value=0)
    positions = positions.reset_index(drop=True)
    return positions


def preprocess_ids(large, small, large_id_cols=None, small_id_cols=None):
    # Define on which columns we're going to operate
    if small_id_cols is None:
        small_id_cols = [c for c in small.columns if (c.endswith("_id") or c.endswith("_pos")) and c not in ("begin", "end")]
    if large_id_cols is None:
        large_id_cols = [c for c in large.columns if (c.endswith("_id") or c.endswith("_pos")) and c not in ("begin", "end")]
    doc_id_cols = [c for c in small.columns if c in large.columns and c not in ("begin", "end")]
    return (
        doc_id_cols,
        [c for c in small_id_cols if c not in doc_id_cols],
        [c for c in large_id_cols if c not in doc_id_cols],
        [c for c in small.columns if c not in small_id_cols and c not in ("begin", "end") and c not in doc_id_cols],
        [c for c in large.columns if c not in large_id_cols and c not in ("begin", "end") and c not in doc_id_cols])


def encode_as_tag(small, large, label_cols=None, tag_scheme="bio"):
    """

    Parameters
    ----------
    small: tokens
    large: mentions
    small_id_cols: token id cols (doc_id, token_pos)
    large_id_cols: mention id cols (doc_id, mention_id, mention_part_id)
    label_cols: "label"

    Returns
    -------
    pd.DataFrame
    """
    assert tag_scheme in ("bio", "bioul", "raw")

    doc_id_cols, small_id_cols, large_id_cols, small_val_cols, large_val_cols = preprocess_ids(large, small)
    # assert len(large_val_cols) < 2, "Cannot encode more than one column as tags"
    assert len(large_val_cols) > 0, "Must have a column to encode as tags"
    if label_cols is None:
        label_cols = large_val_cols
    if isinstance(label_cols, str):
        label_cols = [label_cols]

    # Map mentions to small as a tag
    large = large.sort_values([*doc_id_cols, "begin", "end"])
    merged = merge_with_spans(large, small, span_policy='partial_strict', on=[*doc_id_cols, ("begin", "end")], suffixes=('_large', ''))
    # If a token overlap multiple mentions, assign it to the last mention
    merged = merged.drop_duplicates([*doc_id_cols, *small_id_cols], keep='last')
    merged_id_cols = doc_id_cols + large_id_cols + small_id_cols

    # Encode mention labels as a tag
    tags = (merged[merged_id_cols + large_val_cols]
            .sort_values(merged_id_cols))
    if tag_scheme != "raw":
        tags = tags.nlp.unflatten(doc_id_cols + large_id_cols + large_val_cols)
        for label_col in label_cols:
            tags[label_col] = tags.apply(lambda row: make_tag_scheme(len(row[small_id_cols[0]]), row[label_col], tag_scheme), axis=1)
        tags = tags.nlp.flatten()

    merged = merged[[*merged_id_cols, *small_val_cols, "begin", "end"]].merge(tags)
    merged = small.merge(merged, how="left")
    if tag_scheme != "raw":
        for label_col in label_cols:
            merged[label_col] = merged[label_col].fillna("O")
            merged[label_col] = merged[label_col].astype(pd.CategoricalDtype(
                ["O", *(tag for label in large[label_col].cat.categories for tag in ("B-" + label, "I-" + label))] if tag_scheme == "bio" else
                ["O", *(tag for label in large[label_col].cat.categories for tag in ("B-" + label, "I-" + label, "U-" + label, "L-" + label))]
            ))
    return merged.sort_values([*doc_id_cols, "begin", "end"])


def partition_spans(smalls, large, overlap_policy="merge", new_id_name="sample_id", span_policy="partial_strict"):
    assert len(smalls) >= 1
    if not isinstance(smalls, (list, tuple)):
        smalls = [smalls]

    merged_id_cols = doc_id_cols = None
    original_new_id_name = new_id_name
    while new_id_name in large.columns:
        new_id_name = "_" + new_id_name
    if overlap_policy == "merge":
        large = large.copy()
        old_to_new = None
        has_created_new_id_col = False
        for small in smalls:
            doc_id_cols, small_id_cols, large_id_cols, small_val_cols, large_val_cols = preprocess_ids(large, small)
            large_id_cols = [c for c in large_id_cols if c != new_id_name]
            # Merge sentences and mentions
            merged = merge_with_spans(small, large, span_policy=span_policy, how='right', on=[*doc_id_cols, ("begin", "end")])
            # If a mention overlap multiple sentences, assign it to the last sentence
            small_ids = merged[doc_id_cols + small_id_cols].nlp.factorize(group_nans=False)
            if has_created_new_id_col:
                large_ids = merged[doc_id_cols + [new_id_name]].nlp.factorize(group_nans=False)
            else:
                large_ids = merged[doc_id_cols + large_id_cols].nlp.factorize(group_nans=False)

            merged[new_id_name] = make_id_from_merged(
                large_ids,
                small_ids,
                apply_on=[(0, large_ids)])[0]
            merged["begin"] = merged[['begin_x', 'begin_y']].min(axis=1)
            merged["end"] = merged[['end_x', 'end_y']].max(axis=1)
            large = (merged
                     .groupby(new_id_name, as_index=False)
                     .agg({**{n: 'first' for n in doc_id_cols}, 'begin': 'min', 'end': 'max'})
                     .astype({"begin": int, "end": int, **large[doc_id_cols].dtypes}))
            large = large[doc_id_cols + [new_id_name] + ["begin", "end"]]
            old_to_new = large[doc_id_cols + [new_id_name]].drop_duplicates().reset_index(drop=True)
            merged_id_cols = [new_id_name]
        # large[original_new_id_name] = large[doc_id_cols + [new_id_name]].apply(lambda x: "/".join(map(str, x[doc_id_cols])) + "/" + str(x[new_id_name]), axis=1).astype("category")
        # large = large.drop(columns={*doc_id_cols, new_id_name} - {original_new_id_name})
    elif overlap_policy == "none":
        # merged = merged.drop_duplicates([*doc_id_cols, *small_id_cols], keep=overlap_policy)
        doc_id_cols, small_id_cols, large_id_cols, small_val_cols, large_val_cols = preprocess_ids(large, smalls[0])
        merged_id_cols = large_id_cols
        new_id_name = None
        old_to_new = None
    else:
        raise Exception()

    # Merge sentences and mentions
    new_smalls = []
    for small in smalls:
        doc_id_cols, small_id_cols, large_id_cols, small_val_cols, large_val_cols = preprocess_ids(large, small)
        merged = merge_with_spans(small, large[doc_id_cols + large_id_cols + ['begin', 'end']],
                                  how='inner', span_policy=span_policy, on=[*doc_id_cols, ("begin", "end")])
        new_small = (
            merged.assign(begin=merged["begin_x"] - merged["begin_y"], end=merged["end_x"] - merged["begin_y"])
                .astype({"begin": int, "end": int})[[*doc_id_cols, *merged_id_cols, *small_id_cols, *small_val_cols, "begin", "end"]])
        if new_id_name:
            new_small[original_new_id_name] = new_small[list(set((*doc_id_cols, new_id_name)))].apply(
                lambda x: "/".join([str(x[c]) for c in list(doc_id_cols) + ([new_id_name] if new_id_name not in doc_id_cols else [])]), axis=1).astype("category")
            new_small = new_small.drop(columns={*doc_id_cols, new_id_name} - {original_new_id_name})
        new_smalls.append(new_small)
    if new_id_name:
        large[original_new_id_name] = large[doc_id_cols + [new_id_name]].apply(lambda x: "/".join(map(str, x[doc_id_cols])) + "/" + str(x[new_id_name]), axis=1).astype("category")
        large = large.drop(columns={*doc_id_cols, new_id_name} - {original_new_id_name})
        old_to_new[original_new_id_name] = old_to_new[doc_id_cols + [new_id_name]].apply(lambda x: "/".join(map(str, x[doc_id_cols])) + "/" + str(x[new_id_name]), axis=1).astype("category")
    return new_smalls, large, old_to_new


def split_into_spans(large, small, pos_col=None):
    if pos_col is None:
        pos_col = next(iter(c for c in small.columns if c.endswith("_pos")))
    small = small.nlp.partition(large)
    doc_id_cols, small_id_cols, large_id_cols, small_val_cols, large_val_cols = preprocess_ids(large, small)
    return large[[*doc_id_cols, *large_id_cols, *large_val_cols]].merge(
        small
            .eval(f"""
            begin={pos_col}
            end={pos_col} + 1""")
            .groupby(doc_id_cols, as_index=False)
            .agg({"begin": "min", "end": "max"})
    )
    # .sort_values([*doc_id_cols, "begin", *large_id_cols]))
