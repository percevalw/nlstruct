import re
from itertools import repeat

import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

from nlstruct.utils.pandas import flatten
from nlstruct.utils.pandas import make_merged_names_map, merge_with_spans


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
    for match in reversed(list(re.finditer(pattern, text, flags=re.DOTALL))):
        middle = make_str_from_groups(replacement, [match.group(i) for i in needed_groups])
        start = match.start()
        end = match.end()
        text = text[:start] + middle + text[end:]
        begins.append(start)
        ends.append(end)
        deltas.append(len(middle) - end + start)
    return text, DeltaCollection(begins, ends, deltas)


def regex_multisub_with_spans(patterns, replacements, text, deltas=None):
    if deltas is None:
        deltas = DeltaCollection([], [], [])
    for pattern, replacement in zip(patterns, replacements):
        text, new_deltas = regex_sub_with_spans(pattern, replacement, text)
        if deltas is not None:
            deltas += new_deltas
        else:
            deltas = new_deltas
    return text, deltas


def run_unidecode(text):
    begins, ends, deltas = [], [], []
    new_text = ""
    for i, (old_char, new_char) in enumerate((char, unidecode(char)) for char in text):
        if len(old_char) != len(new_char):
            begins.append(i)
            ends.append(i + 1)
            deltas.append(len(new_char) - 1)
        new_text += new_char
    return new_text, DeltaCollection(begins, ends, deltas)


def apply_substitutions(
      dataset,
      global_patterns=None,
      global_replacements=None,
      text_col="text",
      pattern_col="patterns",
      replacements_col="replacements",
      doc_cols=("doc_id",),
      apply_unidecode=False,
      return_deltas=True, with_tqdm=False):
    assert (global_patterns is None) == (global_replacements is None)

    if isinstance(doc_cols, str):
        doc_cols = [doc_cols]

    if global_patterns is None:
        global_patterns = []
        global_replacements = []

    def process_text(text_, doc_patterns, doc_replacements):
        deltas_ = None
        if apply_unidecode:
            text_, deltas_ = run_unidecode(text_)
        text_, deltas_ = regex_multisub_with_spans(
            [*doc_patterns, *global_patterns],
            [*doc_replacements, *global_replacements],
            text_,
            deltas=deltas_,
        )
        return text_, deltas_.begins, deltas_.ends, deltas_.deltas

    if return_deltas:
        text, delta_begins, delta_ends, deltas = zip(*[
            process_text(text, doc_patterns, doc_replacements) for text, doc_patterns, doc_replacements in
            tqdm(zip(
                dataset[text_col],
                dataset[pattern_col] if pattern_col in dataset.columns else repeat([]),
                dataset[replacements_col] if replacements_col in dataset.columns else repeat([])), total=len(dataset), disable=not with_tqdm, desc="Performing regex subs")
        ])
        dataset = pd.DataFrame({
            text_col: text,
            "begin": delta_begins,
            "end": delta_ends,
            "delta": deltas,
            **{c: dataset[c] for c in dataset.columns if c not in (text_col, "begin", "end", "delta")}
        })
        return (
            dataset[[c for c in dataset.columns if c not in ("begin", "end", "delta")]],
            flatten(dataset[[*doc_cols, "begin", "end", "delta"]]))
    else:
        new_texts = []
        for text, doc_patterns, doc_replacements in tqdm(zip(
              dataset[text_col],
              dataset[pattern_col] if pattern_col in dataset.columns else repeat([]),
              dataset[replacements_col] if replacements_col in dataset.columns else repeat([])), total=len(dataset), disable=not with_tqdm, desc="Performing regex subs"):
            if apply_unidecode:
                text = unidecode(text)
            for pattern, replacement in zip([*doc_patterns, *global_patterns], [*doc_replacements, *global_replacements]):
                text = re.sub(pattern, replacement, text)
            new_texts.append(text)
        dataset = pd.DataFrame({text_col: new_texts,
                                **{c: dataset[c] for c in dataset.columns if c not in (text_col,)}})
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
        tmp = mention_deltas.sort_values(['_id_col', delta_col_map['begin' if side == 'left' else 'end']]).groupby('_id_col').agg({
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
        positions[col] = positions[col].add(tmp['shift'] + tmp['between_magnet'], fill_value=0).astype(int)
    positions = positions.reset_index(drop=True)
    return positions
