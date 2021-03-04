import random
import re
import string
from copy import copy

import numpy as np
from unidecode import unidecode
import warnings

import functools
import textwrap


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


class StatefulMap():
    def __init__(self, data, fn, args, kwargs):
        self.fn = fn
        self.data = data
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return StatefulMap(iter(self.data), self.fn, self.args, self.kwargs)

    def state_dict(self):
        return {
            "data": self.data.state_dict() if hasattr(self.data, 'state_dict') else None,
        }

    def __len__(self):
        return len(self.data)

    def load_state_dict(self, state):
        data_state = state.get("data", None)
        if data_state is not None:
            self.data.load_state_dict(data_state)

    def __next__(self):
        obj = next(self.data)
        return self.fn(obj, *self.args, **self.kwargs)

    def __repr__(self):
        return "<map\n" + textwrap.indent("fn={}({})\ndata={}".format(
            self.fn.__name__,
            ", ".join((*map(repr, self.args), *("{}={}".format(k, repr(v)) for k, v in self.kwargs.items()))),
            "{}({})".format(type(self.data).__name__, len(self.data) if isinstance(self.data, (list, tuple)) else repr(self.data)),
        ), "  ") + "\n>"


class StatefulChain():
    def __init__(self, data):
        self.data = data
        self.current = []

    def __iter__(self):
        return StatefulChain(iter(self.data))

    def state_dict(self):
        return {
            "data": self.data.state_dict() if hasattr(self.data, 'state_dict') else None,
            "current": self.current,
        }

    def load_state_dict(self, state):
        data_state = state.get("data", None)
        if data_state is not None:
            self.data.load_state_dict(data_state)
        self.current = state["current"]

    def __next__(self):
        if len(self.current) == 0:
            self.current = next(self.data)
        [res, *self.current] = self.current
        return res

    def __repr__(self):
        return "<chain\n" + textwrap.indent("data={}".format(
            "{}({})".format(type(self.data).__name__, len(self.data) if isinstance(self.data, (list, tuple)) else repr(self.data))
        ), "  ") + "\n>"


def mappable(fn):
    class wrap():
        def __new__(cls, fn):
            instance = super().__new__(cls)
            return functools.wraps(fn)(instance)

        def __init__(self, fn):
            self.fn = fn

        # @functools.wraps(fn)
        def __call__(self, data, *args, **kwargs):
            if hasattr(data, '__iter__') and not isinstance(data, (dict, str)):
                iterator = StatefulMap(data, self.fn, args, kwargs)
                chain = kwargs.pop("chain", False)
                if chain:
                    iterator = StatefulChain(iterator)
                return iterator
            else:
                return self.fn(data, *args, **kwargs)

            return self.fn(data, *args, **kwargs)

        def __get__(self, instance, owner):
            return wrap(self.fn.__get__(instance, owner))

    return wrap(fn)


class batchify:
    def __init__(self, data, batch_size):
        self.data = data
        self.buffer = []
        self.batch_size = batch_size

    def __iter__(self):
        new_self = batchify(iter(self.data), self.batch_size)
        new_self.buffer = list(self.buffer)
        return new_self

    def state_dict(self):
        return {
            "data": self.samples.state_dict() if hasattr(self.data, 'state_dict') else None,
            "buffer": list(self.buffer),
        }

    def load_state_dict(self, dico):
        if dico['data'] is not None:
            self.data.load_state_dict(dico['data'])
        self.buffer = dico["buffer"]

    def __next__(self):
        try:
            while True:
                sample = next(self.data)
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
        return "<batchify\n" + textwrap.indent("batch_size={}\ndata={}".format(
            self.batch_size,
            "{}({})".format(type(self.data).__name__, len(self.data)) if isinstance(self.data, (list, tuple)) else repr(self.data),
        ), "  ") + "\n>"


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


class OverlappingEntityException(Exception):
    pass

def slice_document(doc, begin, end, entity_overlap='raise'):
    assert entity_overlap in ("raise", "split")
    absolute_begin = doc.get("begin", 0)
    new_entities = []
    sentence_size = end - begin
    if "entities" in doc:
        for entity in doc["entities"]:
            min_begin = min(fragment["begin"] for fragment in entity["fragments"])
            max_end = max(fragment["end"] for fragment in entity["fragments"])
            if min_begin < end and begin < max_end:
                if begin <= min_begin and max_end <= end:
                    new_entities.append({**entity, "fragments": [{"begin": fragment["begin"] - begin,
                                                                   "end": fragment["end"] - begin}
                                                                  for fragment in entity["fragments"]]})
                else:
                    if entity_overlap == "raise":
                        raise OverlappingEntityException(
                            "Entity {} spans more than one sentence in document {}. "
                            "Use sentence_entity_overlap='split' in preprocessor to handle such cases.".format(
                                repr(doc["text"][min_begin:max_end]), doc["doc_id"]))
                    else:
                        new_entities.append({**entity, "fragments": [{"begin": min(max(fragment["begin"] - begin, 0), sentence_size),
                                                                       "end": max(min(fragment["end"] - begin, sentence_size), 0)}
                                                                      for fragment in entity["fragments"]
                                                                      if fragment["begin"] <= end and begin <= fragment["end"]]})
    return {
        **doc,
        "doc_id": doc["doc_id"] + "/{}-{}".format(absolute_begin + begin, absolute_begin + end),
        "text": doc["text"][begin:end],
        "begin": absolute_begin + begin,
        "end": absolute_begin + end,
        "entities": new_entities
    }


@mappable
def sentencize(doc, reg_split=r"(?<=[.])(?:\s+)(?=[A-Z])", balance_chars=('()', '[]'), multi_sentence_entities="raise", chain=False):
    for begin, end in regex_sentencize(doc["text"], reg_split=reg_split, balance_chars=balance_chars):
        yield slice_document(doc, begin, end, multi_sentence_entities=multi_sentence_entities)


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
    new_begins = np.full(len(span_begins), fill_value=-1)
    new_ends = np.full(len(span_ends), fill_value=-1)
    matched_spans = diff.any(1)
    new_begins[matched_spans], new_ends[matched_spans] = tuple(flat_idx.T % token_span_overlap.shape[1])
    return new_begins, new_ends


def huggingface_tokenize(text, tokenizer, subs=(), do_unidecode=True, text_col="text", **kwargs):
    deltas = None
    if do_unidecode:
        text, deltas = run_unidecode(text)
    if len(subs):
        text, deltas = regex_multisub_with_spans(*zip(*subs), text, deltas=deltas)

    res = tokenizer.encode_plus(text, return_offsets_mapping=True)
    begins, ends = zip(*res['offset_mapping'][:-1], (len(text), len(text)))
    words = tokenizer.convert_ids_to_tokens(res['input_ids'])
    
    model_max_length = getattr(tokenizer, 'model_max_length', None)
    #if model_max_length is not None and len(words) > model_max_length:
    #    warnings.warn("Sentence {} is longer than maximum length of {} (count is {} wordpieces)".format(repr(text), model_max_length, len(words)))

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
            if begin != end:
                yield begin, end
            begin = match.end()
    if begin != len(text):
        yield begin, len(text)
