from copy import copy
from math import ceil
from random import randint

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import _utils
from torch.utils.data.dataloader import _DatasetKind

from nlstruct.train import fork_rng


def compute_batches(
      ids,
      batch_size,
      offset=None,
      incomplete_is_last=True,
      shuffle=True,
      sorting_keys=None,
      sorting_noise=0.,
):
    perm = None
    if shuffle:
        ids = np.asarray(ids) if hasattr(ids, '__len__') else np.arange(ids)
        perm = np.random.permutation(len(ids))
        ids = np.asarray(ids)[perm]
    else:
        ids = np.asarray(ids) if hasattr(ids, '__len__') else np.arange(ids)
    length = len(ids)
    if sorting_keys is not None:
        sort = True
        sorting_keys = sorting_keys[ids]
        if shuffle:
            sorting_keys = sorting_keys + np.random.poisson(sorting_noise, size=sorting_keys.shape)

        sorter = np.lexsort(np.asarray(sorting_keys).reshape(len(sorting_keys), -1).T, axis=0)
        ids = ids[sorter]

    steps = ceil(length / batch_size)
    rest = length % batch_size

    if offset == 0:
        block_begins = (np.arange(steps) * batch_size).round().astype(int)  # - np.random.randint(length % batch_size)
    else:
        block_begins = np.concatenate([[0], offset + np.arange(steps - (1 if rest <= offset else 0)) * batch_size]).round().astype(int)  # - np.random.randint(length % batch_size)
    block_begins[0] = 0

    if shuffle:
        block_perm = np.random.permutation(len(block_begins))
        shift = np.random.choice(block_begins)  # + (offset if offset is not None else 0)
        block_begins = ((block_begins - shift + length) % length)
        if offset is not None:
            block_perm[0] -= len(block_begins)
        if incomplete_is_last:
            block_perm[-1] += len(block_begins)
        block_perm = np.argsort(block_perm)
        block_ends = np.roll(block_begins, -1)[block_perm]
        block_begins = block_begins[block_perm]
    else:
        block_ends = np.roll(block_begins, -1)

    block_ends = ((np.clip(block_ends, 0, length)) - 1) % length + 1

    batches = [list(ids[begin:end]) for begin, end in zip(block_begins, block_ends)]

    return batches, (offset - length) % batch_size


class RepeatIterator(object):
    def __init__(self, ids):
        self.ids = ids

    def __iter__(self):
        return self

    def __next__(self):
        return self.ids

    def state_dict():
        return {}

    def load_state_dict(state):
        pass

class BatchIterator(object):
    def __init__(
          self,
          ids,
          batch_size,
          shuffle=False,

          sorting_keys=None,
          sorting_noise=0.,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sorting_keys = sorting_keys
        self.sorting_noise = sorting_noise if sorting_keys is not None else None
        self.seed = 0

        self.ids = ids

        self.step = 0
        self.next_offset = 0
        self.batch_idx = 0
        self.batches = []

    def __len__(self):
        if not self.is_loop:
            return ceil((len(self.ids) if hasattr(self.ids, '__len__') else self.ids) / self.batch_size)
        raise TypeError("Loop iterator has no length")

    def __iter__(self):
        new_self = copy(self)
        new_self.seed = randint(0, 2 ** 32 - 1)

        if hasattr(new_self.ids, '__next__'):
            new_self.ids = iter(new_self.ids)

        with fork_rng((new_self.seed + new_self.step) % (2 ** 32 - 1)):
            new_self.step += 1
            new_self.batches, new_self.next_offset = compute_batches(
                ids=new_self.next_ids(),
                batch_size=new_self.batch_size,
                offset=new_self.next_offset,
                incomplete_is_last=new_self.is_loop,
                shuffle=new_self.shuffle,

                sorting_keys=new_self.sorting_keys,
                sorting_noise=new_self.sorting_noise,
            )

        return new_self

    def state_dict(self):
        state = dict(self.__dict__)

        for key in ["batch_size", "shuffle", "sorting_keys", "sorting_noise", "seed"]:
            state.pop(key, None)
        if hasattr(state["ids"], 'state_dict'):
            state["ids"] = state["ids"].state_dict()
        return state

    def load_state_dict(self, state):
        ids_state = state.pop("ids", None)
        if ids_state is not None:
            self.ids.load_state_dict(ids_state)
        self.__dict__.update(state)

    def next_ids(self):
        if hasattr(self.ids, '__next__'):
            data = next(self.ids)
        else:
            data = self.ids
        return data

    @property
    def is_loop(self):
        return hasattr(self.ids, '__next__') or isinstance(self.ids, tuple)

    def __next__(self):
        # select the next precomputed batch
        if self.batch_idx >= len(self.batches):
            raise StopIteration()
        batch = self.batches[self.batch_idx]

        self.batch_idx += 1

        # if we no longer have batches, compute the next ones
        while self.batch_idx == len(self.batches) and self.is_loop:
            self.batch_idx = 0
            must_complete_batch = self.is_loop and len(batch) < self.batch_size
            self.last_offset = self.next_offset
            with fork_rng((self.seed + self.step) % (2 ** 32 - 1)):
                self.step += 1
                new_batches, new_offset = compute_batches(
                    ids=self.next_ids(),
                    batch_size=self.batch_size,
                    offset=self.next_offset,
                    incomplete_is_last=self.is_loop,
                    shuffle=self.shuffle,

                    sorting_keys=self.sorting_keys,
                    sorting_noise=self.sorting_noise,
                )
            if must_complete_batch:
                batch += new_batches[0]
                self.batches = new_batches[1:]
            else:
                self.batches = new_batches
            self.next_offset = new_offset
        return batch


def compute_chunk_sizes(raw_counts, rests, mix, eps):
    raw_mix = rests / rests.sum()
    fractions = (mix / raw_mix)
    fractions /= fractions.max()
    chunk_sizes = fractions * rests + eps
    int_chunk_sizes = chunk_sizes.round().astype(int)
    return int_chunk_sizes, (((rests - int_chunk_sizes) + raw_counts - 1) % raw_counts) + 1, chunk_sizes - int_chunk_sizes


class MultiBatchIterator(object):
    def __init__(self, ids, mix, shuffle=False):
        self.ids = ids
        self.mix = np.asarray(mix)
        self.shuffle = shuffle

        self.seed = 0

        self.offsets = np.zeros(len(ids), dtype=int)
        self.current_ids = [[] for _ in ids]
        self.lengths = np.zeros(len(ids), dtype=int)
        self.rests = np.zeros(len(ids), dtype=int)
        self.eps = np.zeros(len(ids), dtype=float)
        self.steps = np.zeros(len(ids), dtype=int)

    def __iter__(self):
        new_self = copy(self)
        new_self.seed = randint(0, 2 ** 32 - 1)
        new_self.ids = [iter(x) for x in new_self.ids if hasattr(x, '__next__')]
        return new_self

    def state_dict(self):
        state = dict(self.__dict__)

        state.pop("mix", None)
        state.pop("seed", None)
        state.pop("shuffle", None)

        state["ids"] = []
        for ids in self.ids:
            if hasattr(ids, 'state_dict'):
                state["ids"].append(ids.state_dict())
            else:
                state["ids"].append(None)
        return state

    def load_state_dict(self, state):
        ids_state = state.pop("ids", None)
        if ids_state is not None:
            for seq, seq_state in zip(self.ids, ids_state):
                if seq_state is not None:
                    seq.load_state_dict(seq_state)
        self.__dict__.update(state)

    def get_ids(self, i):
        if self.offsets[i] >= len(self.current_ids[i]):
            if hasattr(self.ids[i], '__next__'):
                self.current_ids[i] = next(self.ids[i])
            else:
                self.current_ids[i] = self.ids[i]
            if self.shuffle:
                with fork_rng((self.seed + self.steps[i]) % (2 ** 32 - 1)):
                    self.current_ids[i] = np.random.permutation(self.current_ids[i])
            self.offsets[i] = 0
            self.steps[i] += 1
            self.lengths[i] = self.rests[i] = len(self.current_ids[i])
        return self.current_ids[i]

    def __next__(self):
        ids = [self.get_ids(i) for i in range(len(self.ids))]
        batch_sizes, self.rests, self.eps = compute_chunk_sizes(self.lengths, self.rests, self.mix, self.eps)
        batch_ids = np.concatenate([seq[offset:offset + size] for seq, offset, size in zip(ids, self.offsets, batch_sizes)])
        self.offsets += batch_sizes
        return batch_ids


class StatefulDataLoader(DataLoader):
    def __iter__(self):
        if self.num_workers > 0:
            raise NotImplementedError()
        return StatefulDataLoaderIter(self)


class StatefulDataLoaderIter(object):
    def __init__(self, loader):
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_().item()

        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def __next__(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def __len__(self):
        return len(self._index_sampler)

    def state_dict(self):
        """Returns the state of the dataloader as a :class:`dict`.
        """
        return {"_sampler_iter": self._sampler_iter.state_dict()}

    def load_state_dict(self, state_dict):
        """Loads the dataloader state.

        Arguments:
            state_dict (dict): dataloader state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        if "_sampler_iter" in state_dict:
            self._sampler_iter.load_state_dict(state_dict["_sampler_iter"])
