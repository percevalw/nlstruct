import random
import re
import string
from copy import copy

import numpy as np
from unidecode import unidecode


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
        pos = positions + to_remove
        if side == 'right':
            pos[between_mask] = self.ends[between_i]
        elif side == 'left':
            pos[between_mask] = self.begins[between_i]
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


class batchify:
    def __init__(self, samples, batch_size):
        self.samples = samples
        self.buffer = []
        self.batch_size = batch_size

    def __iter__(self):
        new_self = batchify(iter(self.samples), self.batch_size)
        new_self.buffer = list(self.buffer)
        return new_self

    def state_dict(self):
        return {
            "samples": self.samples.state_dict() if hasattr(self.samples, 'state_dict') else None,
            "buffer": list(self.buffer),
        }

    def load_state_dict(self, dico):
        if dico['samples'] is not None:
            self.samples.load_state_dict(dico['samples'])
        self.buffer = dico["buffer"]

    def __next__(self):
        try:
            while True:
                sample = next(self.samples)
                self.buffer.append(sample)
                if len(self.buffer) >= self.batch_size:
                    res = self.buffer
                    self.buffer = []
                    return res
        except StopIteration:
            if len(self.buffer):
                res = self.buffer
                self.buffer = []
                return res
            else:
                raise

    def __repr__(self):
        return f"<batchify samples={self.samples} batch_size={self.batch_size}>"


class mix:
    def __init__(self, *datasets, rates, rng=None):
        self.rng = np.random.default_rng(rng if rng is not None else random.randint(0, 2 ** 32 - 1))
        self.rates = np.asarray(rates)
        self.rates_idx = np.arange(len(rates))
        self.datasets = datasets

    def __iter__(self):
        return mix(*(iter(dataset) for dataset in self.datasets), rates=self.rates, rng=self.rng)

    def state_dict(self):
        return {
            "rng": copy(self.rng),
            "datasets": [dataset.state_dict() if hasattr(dataset, 'state_dict') else None for dataset in self.datasets]
        }

    def load_state_dict(self, dico):
        self.rng = dico['rng']
        for dataset, dataset_state in zip(self.datasets, dico["datasets"]):
            if dataset_state is not None:
                dataset.load_state_dict(dataset_state)

    def __next__(self):
        dataset_idx = np.random.choice(self.rates_idx, p=self.rates)
        return next(self.datasets[dataset_idx])


class loop:
    def __init__(self, samples, shuffle=False, rng=None):
        self.samples = samples
        self.indices = None
        self.idx = len(samples)
        self.rng = np.random.default_rng(rng if rng is not None else random.randint(0, 2 ** 32 - 1)) if shuffle else None

    def __iter__(self):
        return self

    def state_dict(self):
        return {
            "idx": self.idx,
            "rng": copy(self.rng),
            "indices": self.indices
        }

    def load_state_dict(self, dico):
        self.__dict__.update(dico)

    def __next__(self):
        if self.idx >= len(self.samples):
            self.idx = 0
            if self.rng is not None:
                self.indices = self.rng.permutation(len(self.samples))
        sample = self.samples[self.indices[self.idx] if self.indices is not None else self.idx]
        self.idx += 1
        return sample


class sentencize:
    def __init__(self, samples, reg_split=r"(?<=[.])(?:\s+)(?=[A-Z])", balance_chars=()):
        self.samples = samples
        self.reg_split = reg_split
        self.balance_chars = balance_chars
        self.current_doc = None
        self.current_idx = -1
        self.remaining_sentences = []

    def __iter__(self):
        new_self = sentencize(iter(self.samples), self.reg_split, balance_chars=self.balance_chars)
        return new_self

    def state_dict(self):
        return {
            "samples": self.samples.state_dict() if hasattr(self.samples, 'state_dict') else None,
            "current_doc": self.current_doc,
            "current_idx": self.current_idx,
            "remaining_sentences": self.remaining_sentences,
        }

    def load_state_dict(self, dico):
        if dico['samples'] is not None:
            self.samples.load_state_dict(dico['samples'])
        self.current_doc = dico["current_doc"]
        self.current_idx = dico["current_idx"]
        self.remaining_sentences = dico["remaining_sentences"]

    def __next__(self):
        if not len(self.remaining_sentences):
            self.current_doc = next(self.samples)
            self.current_idx = -1
            self.remaining_sentences = regex_sentencize(self.current_doc["text"], self.reg_split, balance_chars=self.balance_chars)

        [(sentence_begin, sentence_end), *self.remaining_sentences] = self.remaining_sentences
        self.current_idx += 1
        new_mentions = []
        if "mentions" in self.current_doc:
            for mention in self.current_doc["mentions"]:
                min_begin = min(fragment["begin"] for fragment in mention["fragments"])
                max_end = max(fragment["end"] for fragment in mention["fragments"])
                if min_begin <= sentence_end and sentence_begin <= max_end:
                    assert sentence_begin <= min_begin and max_end <= sentence_end
                    new_mentions.append({**mention, "fragments": [{"begin": fragment["begin"] - sentence_begin, "end": fragment["end"] - sentence_begin} for fragment in mention["fragments"]]})
        return {
            "doc_id": self.current_doc["doc_id"] + "/" + str(self.current_idx),
            "text": self.current_doc["text"][sentence_begin:sentence_end],
            "begin": sentence_begin,
            "end": sentence_end,
            "mentions": new_mentions
        }

    def __repr__(self):
        return f"<sentencize samples={self.samples} reg_split={self.reg_split}>"


def reshape_variable_sequences(sequences, indices_map):
    return sequences.reshape(-1, *sequences.shape[2:])[indices_map]


def make_str_from_groups(replacement, groups):
    for i, group in enumerate(groups):
        replacement = replacement.replace(f"\\{i + 1}", group)
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


def split_spans(span_begins, span_ends, token_begins, token_ends):
    token_begins = np.asarray(token_begins).reshape(1, -1)
    token_ends = np.asarray(token_ends).reshape(1, -1)
    span_begins = np.asarray(span_begins).reshape(-1, 1)
    span_ends = np.asarray(span_ends).reshape(-1, 1)
    token_span_overlap = (
          ((token_begins != token_ends) &
           ((token_begins < span_ends) & (span_begins < token_ends))) |
          ((token_begins == token_ends) &
           ((token_begins == span_begins) & (token_ends == span_ends)))
    )
    token_span_overlap = np.concatenate([token_span_overlap, np.zeros_like(token_span_overlap[:, [-1]])], axis=1)

    next_token_span_overlap = np.roll(token_span_overlap, 1, 1)
    next_token_span_overlap[:, 0] = 0

    diff = token_span_overlap != next_token_span_overlap
    flat_idx = np.flatnonzero(diff).reshape(-1, 2)
    return tuple(flat_idx.T % token_span_overlap.shape[1])


def huggingface_tokenize(text, tokenizer, subs=(), do_unidecode=True, text_col="text", **kwargs):
    deltas = None
    if do_unidecode:
        text, deltas = run_unidecode(text)
    if len(subs):
        text, deltas = regex_multisub_with_spans(*zip(*subs), text, deltas=deltas)

    res = tokenizer.encode_plus(text, return_offsets_mapping=True)
    begins, ends = zip(*res['offset_mapping'][:-1], (len(text), len(text)))
    words = tokenizer.convert_ids_to_tokens(res['input_ids'])

    # Apply substitutions on tokens
    if deltas is not None and len(deltas.begins):
        dc = DeltaCollection(deltas.begins, deltas.ends, deltas.deltas)
        begins = dc.unapply(np.asarray(begins), side="left").tolist()
        ends = dc.unapply(np.asarray(ends), side="right").tolist()

    return {
        "begin": begins,
        "end": ends,
        "word": words,
    }


def regex_tokenize(text, reg=r"[\w']+|[{}]".format(string.punctuation), do_unidecode=True, lower=False, subs=()):
    tokens = []
    begins = []
    ends = []
    token_idx = []

    if lower:
        text = text.lower()

    deltas = None
    if do_unidecode:
        text, deltas = run_unidecode(text)
    if len(subs):
        text, deltas = regex_multisub_with_spans(*zip(*subs), text, deltas=deltas)

    i = 0
    token_id = 0

    for match in re.finditer(reg, text):
        tokens.append(match.group())
        begins.append(match.start())
        ends.append(match.end())
        token_idx.append(token_id)
        token_id += 1

    # Apply substitutions on tokens
    if deltas is not None and len(deltas.begins):
        dc = DeltaCollection(deltas.begins, deltas.ends, deltas.deltas)
        begins = dc.unapply(np.asarray(begins), side="left").tolist()
        ends = dc.unapply(np.asarray(ends), side="right").tolist()

    return {
        "begin": begins,
        "end": ends,
        "token_idx": token_idx,
        "word": tokens,
    }


def regex_sentencize(text, reg_split, balance_chars=('()', '[]')):
    begin = 0
    for match in re.finditer(reg_split, text):
        end = match.start()
        if all(text[begin:end].count(chars[0]) <= text[begin:end].count(chars[1]) for chars in balance_chars):
            yield begin, end
            begin = match.end()
    yield begin, len(text)
