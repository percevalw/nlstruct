import re
from collections import defaultdict, Sequence
from string import ascii_letters

import einops as ops
import torch
import torch.nn.functional as F
import inspect

# Parts of this file was adapted from "https://github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp/random.py"

import functools
import random
from collections import namedtuple
from contextlib import contextmanager

import numpy as np

RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])

if "registry" not in globals():
    registry = {}


def register(name):
    def fn(cls):
        old_init = cls.__init__

        @functools.wraps(old_init)
        def cls_init(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            args = inspect.getcallargs(old_init, self, *args, **kwargs)
            for arg, value in args.items():
                if arg != "self" and not arg.startswith('_') and not hasattr(self, arg):
                    setattr(self, arg, value)

        cls_init.fn = old_init
        cls.__init__ = cls_init
        cls.registry_name = name
        registry[name] = cls
        return cls

    return fn


def get_module(name):
    return registry[name]


def get_config(self, max_sequence_size=20, path=()):
    config = {"module": getattr(self.__class__, "registry_name", self.__class__.__name__)}
    for key in inspect.getfullargspec(getattr(self.__init__, 'fn', self.__init__)).args[1:]:
        if key.startswith('_'):
            continue
        value = getattr(self, key)
        if hasattr(value, 'to_diff_dict'):
            config[key] = value.to_diff_dict()
        elif hasattr(value, 'to_dict'):
            config[key] = value.to_dict()
        elif isinstance(value, torch.nn.ModuleList):
            config[key] = {i: get_config(item, max_sequence_size=max_sequence_size) for i, item in enumerate(value)}
        elif isinstance(value, torch.nn.ModuleDict):
            config[key] = {name: get_config(item, max_sequence_size=max_sequence_size, path=(*path, key)) for name, item in value.items()}
        elif isinstance(value, torch.nn.Module):
            config[key] = get_config(value)
        elif isinstance(value, torch.Tensor):
            pass
        elif isinstance(value, (int, float, str)):
            config[key] = value
        elif isinstance(value, (tuple, list)) and len(value) <= max_sequence_size:
            config[key] = value
        elif isinstance(value, type):
            config[key] = getattr(value, "__name__", str(value))
        else:
            config[key] = "#HASHED: {}".format(hash(str(value)))

    return config


def list_factorize(values, reference_values=None, freeze_reference=None):
    if freeze_reference is None:
        freeze_reference = reference_values is not None
    if reference_values is not None:
        reference_values = dict(zip(list(reference_values), range(len(reference_values))))
    else:
        reference_values = dict()

    def rec(obj):
        if hasattr(obj, '__len__') and not isinstance(obj, str):
            return list(item for item in (rec(item) for item in obj) if item != -1)
        if not freeze_reference:
            return reference_values.setdefault(obj, len(reference_values))
        return reference_values.get(obj, -1)

    return rec(values), list(reference_values.keys())


def get_nested_properties(nested, dtype=None):
    def explore(obj):
        nonlocal dtype
        if isinstance(obj, Sequence) and not isinstance(obj, str):
            for sub in obj:
                depth = explore(sub)
                if depth >= 0:
                    return depth + 1
            return -1
        if dtype is None and isinstance(obj, (int, bool, float)):
            dtype = torch.tensor(obj).dtype
        return 0

    n_depth = explore(nested) - 1
    return n_depth, dtype


def pad_to_tensor(y, dtype=None, device=None):
    n_depth, dtype = get_nested_properties(y, dtype=dtype)

    if n_depth == 0:
        return torch.as_tensor(y, device=device, dtype=dtype)

    def find_max_len(obj, depth=0):
        if depth >= n_depth:
            return (len(obj),)
        return (len(obj), *(max(l) for l in zip(*[find_max_len(item, depth + 1) for item in obj])))

    max_len = find_max_len(y)

    block_sizes = [1]
    for l in max_len[::-1]:
        block_sizes.insert(0, block_sizes[0] * l)
    [total, *block_sizes] = block_sizes

    array = torch.zeros(total, dtype=dtype, device=device)

    def flat_rec(sequence, parent_idx, depth=0):
        for i, obj in enumerate(sequence):
            current_idx = parent_idx + i * block_sizes[depth]
            if depth + 1 >= n_depth:
                array[current_idx:current_idx + len(obj)] = torch.as_tensor(obj, dtype=dtype, device=device)
            else:
                flat_rec(obj, current_idx, depth + 1)

    flat_rec(y, 0)
    return array.reshape(max_len)


def batch_to_tensors(batch, ids_mapping={}, device=None):
    if isinstance(batch, (list, tuple)):
        batch = {key: [row[key] for row in batch] for key in batch[0]}
    result = {}
    for key, rows in batch.items():
        dtype = get_nested_properties(rows)[1]
        if dtype is None and key.endswith("_id"):
            reference_id = ids_mapping.get(key, None)
            factorized_rows = list_factorize(rows, reference_values=batch[reference_id] if reference_id is not None else None)[0]
            result['@' + key] = pad_to_tensor(factorized_rows, device=device)
            result[key] = rows
        elif dtype is None:
            result[key] = rows
        else:
            result[key] = pad_to_tensor(rows, dtype=dtype, device=device)
    return result


def simple_factorize(seqs):
    index = defaultdict(lambda: len(index))
    return [[index[item] for item in seq] for seq in seqs]


def infer_names(tensor, expr, only_ellipsis=False):
    expr = tuple(expr.split()) if isinstance(expr, str) else tuple(expr)
    if '...' in expr:
        start = expr.index('...')
        end = -(len(expr) - expr.index('...') - 1)
    else:
        start = end = 0
    ellipsis_names = tensor.names[start:tensor.ndim + end]
    return expr[:start] + ellipsis_names + expr[len(expr) + end:] if not only_ellipsis else ellipsis_names


def einsum(*tensors):
    [*tensors, expr] = tensors
    assert isinstance(expr, str)
    before, after = expr.split("->")
    before = [part.split() for part in before.strip().split(',')]
    after = after.split()
    [_, *before_indices, after_indices] = simple_factorize([['...'], *before, after])
    expr = "{}->{}".format(
        ",".join("".join(ascii_letters[i - 1] if i > 0 else '...' for i in part) for part in before_indices),
        "".join(ascii_letters[i - 1] if i > 0 else '...' for i in after_indices)
    )
    res = torch.einsum(expr, *(t.rename(None) for t in tensors))
    if '...' in after:
        ellipsis_names = next(infer_names(tensor, before_item, only_ellipsis=True) for tensor, before_item in zip(tensors, before))
        new_names = tuple(after[:after.index('...')]) + ellipsis_names + tuple(after[after.index('...') + 1:])
    else:
        new_names = after
    return res.refine_names(*new_names)


def complete_expr(*tensors, dims=()):
    [*tensors, expr] = tensors
    if expr.strip() == '':
        expr = '...'
    if '->' in expr:
        left, right = expr.split('->')
        left = left.split(",")
    elif '...' in expr:
        right_parts = [part for part in expr.split() if not part.split("=")[-1].isdigit()]
        right = expr
        left = [" ".join(tensor.names[:right_parts.index('...')]) + " ... " + " ".join(tensor.names[tensor.ndim - (len(right_parts) - right_parts.index('...') - 1):])
                for tensor in tensors]
    else:
        right = expr
        left = [" ".join(tensor.names) for tensor in tensors]
    new_left_names, old_left_names = zip(*[
        zip(*[name.split('=') if '=' in name else (None, name) if " " in name or name.isdigit() else (name.strip('()'), name)
              for name in re.findall('(?:\S*=)?(?:\([^\)]*\)|\S+)', left_part)])
        for left_part in left
    ])
    new_right_names, old_right_names = zip(*[name.split('=') if '=' in name else (None, name) if " " in name or name.isdigit() else (name.strip('()'), name)
                                             for name in re.findall('(?:\S*=)?(?:\([^\)]*\)|\S+)', right)])
    expr = ",".join(" ".join(part) for part in old_left_names) + "->" + " ".join(old_right_names)

    if '...' in new_right_names:
        ellipsis = next(infer_names(tensor, part, only_ellipsis=True) for tensor, part in zip(tensors, old_left_names) if '...' in part)
        new_right_names = new_right_names[:new_right_names.index('...')] + ellipsis + new_right_names[new_right_names.index('...') + 1:]
        if '...' in old_right_names:
            old_right_names = old_right_names[:old_right_names.index('...')] + ellipsis + old_right_names[old_right_names.index('...') + 1:]
    return expr, new_right_names, [old_right_names.index(dim) for dim in dims]


def rearrange(tensor, expr, *args, **kwargs):
    expr, new_names, _ = complete_expr(tensor, expr)
    return ops.rearrange(tensor.rename(None), expr, *args, **kwargs).refine_names(*new_names)


def reduce(tensor, expr, *args, **kwargs):
    expr, new_names, _ = complete_expr(tensor, expr)
    return ops.reduce(tensor.rename(None), expr, *args, **kwargs).refine_names(*new_names)


def wrap_repeat():
    fn = torch.Tensor.repeat
    if hasattr(fn, 'back'):
        fn = fn.back

    def repeat(self, expr, *args, **kwargs):
        if isinstance(expr, tuple):
            return fn(self, *expr, *args, **kwargs)
        elif isinstance(expr, int):
            return fn(self, expr, *args, **kwargs)
        expr, new_names, _ = complete_expr(self, expr)
        return ops.repeat(self.rename(None), expr, *args, **kwargs).refine_names(*new_names)

    repeat.back = fn
    torch.Tensor.repeat = repeat


def einsum(*tensors, **kwargs):
    [*tensors, expr] = tensors
    expr, new_names, _ = complete_expr(*tensors, expr)
    before, after = expr.split("->")
    before = [part.split() for part in before.strip().split(',')]
    after = after.split()
    [_, *before_indices, after_indices] = simple_factorize([['...'], *before, after])
    expr = "{}->{}".format(
        ",".join("".join(ascii_letters[i - 1] if i > 0 else '...' for i in part) for part in before_indices),
        "".join(ascii_letters[i - 1] if i > 0 else '...' for i in after_indices)
    )
    res = torch.einsum(expr, *(t.rename(None) for t in tensors))
    return res.refine_names(*new_names)


def bce_with_logits(input, target, **kwargs):
    dim = input.shape[-1]
    res = F.binary_cross_entropy_with_logits(input.rename(None).reshape(-1, dim), target.float().rename(None).reshape(-1, dim), **kwargs)
    if kwargs.get('reduction', 'mean') == 'none':
        res = res.view(input.shape).rename(*input.names)
    return res


def cross_entropy_with_logits(input, target, **kwargs):
    dim = input.shape[-1]
    res = F.cross_entropy(input.rename(None).reshape(-1, dim), target.rename(None).reshape(-1), **kwargs)
    if kwargs.get('reduction', 'mean') == 'none':
        res = res.view(target.shape).rename(*target.names)
    return res


def pad(tensor, expr='', value=0, **kwargs):
    expr, new_names, dim_indices = complete_expr(tensor, expr, dims=kwargs)
    dims_to_pad = dict(zip(dim_indices, kwargs.values()))
    assert max(dims_to_pad) < tensor.ndim, "Unknown dim name"
    min_dim_to_pad = min(dims_to_pad)
    padding = [item for i in range(tensor.ndim - 1, min_dim_to_pad - 1, -1) for item in dims_to_pad.get(i, (0, 0))]
    return F.pad(tensor.rename(None), padding).refine_names(*new_names)


def wrap_unary_op(op_name):
    fn = getattr(torch.Tensor, op_name)
    if hasattr(fn, 'back'):
        fn = fn.back

    def wrapper(self, dim=None):
        if dim is None:
            return fn(self.rename(None))
        names = list(self.names)
        if dim in self.names:
            dim = self.names.index(dim)
        names.pop(dim)
        return fn(self.rename(None), dim).rename(*names)

    wrapper.back = fn
    setattr(torch.Tensor, op_name, wrapper)


def wrap_bin_op(op_name):
    fn = getattr(torch.Tensor, op_name)
    if hasattr(fn, 'back'):
        fn = fn.back

    def wrapper(self, other):
        if hasattr(other, 'names') and hasattr(self, 'names') and None not in self.names and None not in other.names:
            other = other.rearrange(" ".join(name if name in other.names else name + "=1" for name in self.names))
            return fn(self.rename(None), other.rename(None)).rename(*self.names)
        else:
            return fn(self, other)

    wrapper.back = fn
    setattr(torch.Tensor, op_name, wrapper)


def wrap_argsort():
    fn = torch.Tensor.argsort
    if hasattr(fn, 'back'):
        fn = fn.back

    def argsort(self, dim=0):
        names = list(self.names)
        if dim in self.names:
            dim = self.names.index(dim)
        return fn(self.rename(None), dim).rename(*names)

    argsort.back = fn
    torch.Tensor.argsort = argsort


def wrap_sort():
    fn = torch.Tensor.sort
    if hasattr(fn, 'back'):
        fn = fn.back

    def sort(self, dim):
        names = list(self.names)
        if dim in self.names:
            dim = self.names.index(dim)
        values, indices = fn(self.rename(None), dim)
        return values.rename(*names), indices.rename(*names)

    sort.back = fn
    torch.Tensor.sort = sort


def wrap_nonzero():
    fn = torch.Tensor.nonzero
    if hasattr(fn, 'back'):
        fn = fn.back

    def nonzero(self, *args, **kwargs):
        return fn(self.rename(None), *args, **kwargs)

    nonzero.back = fn
    torch.Tensor.nonzero = nonzero


def wrap_setitem():
    fn = torch.Tensor.__setitem__
    if hasattr(fn, 'back'):
        fn = fn.back

    def __setitem__(self, index, other):
        index = index.rename(None) if torch.is_tensor(index) else index
        other = other.rename(None) if torch.is_tensor(other) else other
        names = self.names
        self = self.rename(None)
        if not isinstance(index, tuple):
            index = index,
        fn(self, index, other)
        self.names = names

    __setitem__.back = fn
    torch.Tensor.__setitem__ = __setitem__


def wrap_getitem():
    fn = torch.Tensor.__getitem__
    if hasattr(fn, 'back'):
        fn = fn.back

    def __getitem__(self, index):
        if isinstance(index, int):
            return fn(self, index)
        torch.Tensor.__getitem__ = fn
        try:
            names = (None,)
            new_names = None
            if torch.is_tensor(index):
                if index.dtype == torch.bool:
                    if self.names[:index.ndim] != index.names:
                        new_names = [name for name in self.names if name not in index.names[:-1]]
                        self = self.align_to(*index.names, *(name for name in self.names if name not in index.names))
                    names = (index.names[-1], *self.names[index.ndim:])
                    index = index.rename(None)
                else:
                    index = index,
            elif isinstance(index, int):
                index = index,
            elif isinstance(index, slice):
                index = index,
            if isinstance(index, tuple):
                first_names = []
                last_names = []
                if any(part is Ellipsis for part in index):
                    complete_index = list(index)
                    complete_index[index.index(...):index.index(...) + 1] = [slice(None)] * (len(self.names) - len(index) + 1)
                else:
                    complete_index = index + (slice(None),) * (len(self.names) - len(index))
                for item, name in zip(complete_index, self.names):
                    if hasattr(item, 'names'):
                        first_names = list(item.names)
                    elif hasattr(item, '__len__'):
                        first_names = [None]
                    elif isinstance(item, slice):
                        last_names.append(name)
                names = tuple(first_names + last_names)
                index = tuple(item.rename(None) if torch.is_tensor(item) else item for item in index)
            if not isinstance(index, tuple):
                index = index,
            self = self.rename(None)
            self = fn.__get__(self)(index).rename(*names)
            if new_names is not None:
                self = self.align_to(*new_names)
            return self
        finally:
            torch.Tensor.__getitem__ = __getitem__

    __getitem__.back = fn
    torch.Tensor.__getitem__ = __getitem__


def one_hot(tensor, num_classes, new_name):
    return F.one_hot(tensor.rename(None), num_classes).rename(*tensor.names, new_name)


def wrap_masked_fill():
    fn = torch.Tensor.masked_fill
    if hasattr(fn, 'back'):
        fn = fn.back

    def masked_fill(self, mask, value):
        if hasattr(mask, 'names') and hasattr(self, 'names') and None not in self.names and None not in mask.names:
            tensor_names = [name for name in self.names if name not in mask.names] + list(mask.names)
            return fn(self.align_to(*tensor_names).rename(None), mask.rename(None), value).rename(*tensor_names).align_to(*self.names)
        else:
            return fn(self, mask, value)

    masked_fill.back = fn
    torch.Tensor.masked_fill = masked_fill


def smart_gather(tensor, index, dim):
    def arange_at_dim(n, dim, ndim):
        view = [1] * ndim
        view[dim] = -1
        return torch.arange(n, device=tensor.device).view(view)

    common = [name for name in tensor.names if name in index.names]
    missing_tensor = [name for name in tensor.names if name not in common and name != dim]
    missing_index = [name for name in index.names if name not in common]
    tensor = tensor.align_to(*common, dim, *missing_tensor)
    index = index.align_to(*common, *missing_index)
    return tensor[tuple([arange_at_dim(dim, i, index.ndim) for i, dim in enumerate(tensor.shape[:len(common)])]) + (index,)].rename(*common, *missing_index, *missing_tensor)


def repeat_like(tensor, other):
    n_repeats = {}
    name_to_dim = dict(zip(tensor.names, tensor.shape))
    for other_name, other_dim in zip(other.names, other.shape):
        self_dim = name_to_dim.get(other_name, None)
        if other_dim != self_dim:
            n_repeats[other_name] = other_dim // (self_dim if self_dim is not None else 1)
    return tensor.repeat(" ".join(other.names), **n_repeats)


torch.Tensor.rearrange = rearrange
torch.Tensor.reduce = reduce
torch.Tensor.pad = pad
torch.Tensor.one_hot = one_hot
torch.Tensor.smart_gather = smart_gather
torch.Tensor.repeat_like = repeat_like
wrap_nonzero()
wrap_repeat()
wrap_setitem()
wrap_getitem()
wrap_unary_op('all')
wrap_unary_op('any')
wrap_unary_op('argmin')
wrap_unary_op('argmax')
wrap_bin_op("__truediv__")
wrap_bin_op("__floordiv__")
wrap_bin_op("__le__")
wrap_bin_op("__ge__")
wrap_bin_op("__lt__")
wrap_bin_op("__gt__")
wrap_bin_op("__eq__")
wrap_bin_op("__add__")
wrap_bin_op("__and__")
wrap_bin_op("__or__")
wrap_bin_op("__sub__")
wrap_bin_op("__mul__")
wrap_masked_fill()
wrap_argsort()
wrap_sort()


def get_random_generator_state(cuda=torch.cuda.is_available()):
    """ Get the `torch`, `numpy` and `random` random generator state.
    Parameters
    ----------
    cuda: bool  If `True` saves the `cuda` seed also. Note that getting and setting
            the random generator state for CUDA can be quite slow if you have a lot of GPUs.
    Returns
    -------
    RandomGeneratorState
    """
    return RandomGeneratorState(random.getstate(), torch.random.get_rng_state(),
                                np.random.get_state(),
                                torch.cuda.get_rng_state_all() if cuda else None)


def set_random_generator_state(state):
    """
    Set the `torch`, `numpy` and `random` random generator state.
    Parameters
    ----------
    state: RandomGeneratorState
    """
    random.setstate(state.random)
    torch.random.set_rng_state(state.torch)
    np.random.set_state(state.numpy)
    if state.torch_cuda is not None and torch.cuda.is_available() and len(
          state.torch_cuda) == torch.cuda.device_count():  # pragma: no cover
        torch.cuda.set_rng_state_all(state.torch_cuda)


@contextmanager
def fork_rng(seed=None, cuda=torch.cuda.is_available()):
    """
    Forks the `torch`, `numpy` and `random` random generators, so that when you return, the
    random generators are reset to the state that they were previously in.
    Parameters
    ----------
    seed: int or None
        If defined this sets the seed values for the random
        generator fork. This is a convenience parameter.
    cuda: bool
        If `True` saves the `cuda` seed also. Getting and setting the random
        generator state can be quite slow if you have a lot of GPUs.
    """
    state = get_random_generator_state(cuda)
    if seed is not None:
        set_seed(seed, cuda)
    try:
        yield
    finally:
        set_random_generator_state(state)


def fork_rng_wrap(function=None, **kwargs):
    """ Decorator alias for `fork_rng`.
    """
    if not function:
        return functools.partial(fork_rng_wrap, **kwargs)

    @functools.wraps(function)
    def wrapper():
        with fork_rng(**kwargs):
            return function()

    return wrapper


def set_seed(seed, cuda=torch.cuda.is_available()):
    """ Set seed values for random generators.
    Parameters
    ----------
    seed: int
        Value used as a seed.
    cuda: bool
        If `True` sets the `cuda` seed also.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:  # pragma: no cover
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_all = set_seed
