import functools
import random
import re
import math
from collections import defaultdict, Sequence
from collections import namedtuple
from contextlib import contextmanager
from itertools import zip_longest
from string import ascii_letters

import einops as ops
import numpy as np
import torch
import torch.nn.functional as F

try:
    import networkx as nx
except ImportError:
    pass

# Parts of this file was adapted from "https://github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp/random.py"

RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


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
    max_depth = 0

    def explore(obj):
        nonlocal dtype
        if isinstance(obj, Sequence) and not isinstance(obj, str):
            if len(obj) == 0:
                return 1, True
            max_depth = -1
            for sub in obj:
                depth, was_empty = explore(sub)
                max_depth = max(max_depth, depth)
                if depth >= 0 and not was_empty:
                    return depth + 1, False
            return max_depth, True
        if dtype is None and isinstance(obj, (int, bool, float)):
            dtype = torch.tensor(obj).dtype
        return 0, False

    n_depth = explore(nested)[0] - 1
    return n_depth, dtype


def pad_to_tensor(y, dtype=None, device=None, pad=0):
    def find_max_len(obj, depth=0):
        if len(obj) == 0 or not hasattr(obj[0], '__len__'):
            return ((len(obj), True),)
        # print(list(zip(*[find_max_len(item, depth+1) for item in obj])))
        return ((len(obj), True), *((max(all_lengths), len(set(all_lengths)) == 1 and all(all_is_unique))
                                    for zipped_by_depth in zip_longest(*[find_max_len(item, depth + 1) for item in obj], fillvalue=(0, True))
                                    for all_lengths, all_is_unique in (zip(*zipped_by_depth),)))

    max_len, is_unique = zip(*find_max_len(y))
    n_depth = len(max_len) - 1
    if n_depth == 0:
        return torch.as_tensor(y, device=device, dtype=dtype)

    block_sizes = [1]
    for l in max_len[::-1]:
        block_sizes.insert(0, block_sizes[0] * l)
    [total, *block_sizes] = block_sizes
    n_depth = n_depth - (is_unique[::-1].index(False) if False in is_unique else n_depth)

    array = None

    def flat_rec(sequence, parent_idx, depth=0):
        nonlocal array
        for i, obj in enumerate(sequence):
            current_idx = parent_idx + i * block_sizes[depth]
            if depth + 1 >= n_depth:
                if len(obj):
                    obj = torch.as_tensor(obj, dtype=dtype, device=device).view(-1)
                    if array is None:
                        array = torch.full((total,), fill_value=pad, dtype=obj.dtype)
                    array[current_idx:current_idx + obj.numel()] = obj
            else:
                flat_rec(obj, current_idx, depth + 1)

    flat_rec(y, 0)
    if array is None:
        array = torch.full((total,), fill_value=pad, dtype=dtype)
    array = array.reshape(max_len)
    if device is not None:
        array = array.to(device)
    return array


def batch_to_tensors(batch, dtypes={}, ids_mapping={}, device=None, pad=0):
    if isinstance(batch, (list, tuple)):
        batch = {key: [row[key] for row in batch] for key in batch[0]}
    result = {}
    try:
        for key, rows in batch.items():
            pad_value = pad.get(key, 0) if isinstance(pad, dict) else pad
            dtype = dtypes.get(key, None)
            if rows is None or all(row is None for row in rows):
                result[key] = None
            else:
                if dtype is None:
                    dtype = get_nested_properties(rows)[1]
                if dtype is None and key.endswith("_id"):
                    reference_id = ids_mapping.get(key, None)
                    factorized_rows = list_factorize(rows, reference_values=batch[reference_id] if reference_id is not None else None)[0]
                    result['@' + key] = pad_to_tensor(factorized_rows, device=device, pad=pad_value)
                    result[key] = rows
                elif dtype is None:
                    result[key] = rows
                else:
                    result[key] = pad_to_tensor(rows, dtype=dtype, device=device, pad=pad_value)
    except (ValueError, IndexError) as e:
        raise Exception(f"Error during padding of {key}")
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
    res = torch.einsum(expr, *(t for t in tensors))
    if '...' in after:
        ellipsis_names = next(infer_names(tensor, before_item, only_ellipsis=True) for tensor, before_item in zip(tensors, before))
        new_names = tuple(after[:after.index('...')]) + ellipsis_names + tuple(after[after.index('...') + 1:])
    else:
        new_names = after
    return res  # .refine_names(*new_names)


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
    return ops.rearrange(tensor, expr, *args, **kwargs)  # .refine_names(*new_names)


def reduce(tensor, expr, *args, **kwargs):
    expr, new_names, _ = complete_expr(tensor, expr)
    return ops.reduce(tensor, expr, *args, **kwargs)  # .refine_names(*new_names)


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
        return ops.repeat(self, expr, *args, **kwargs)  # .refine_names(*new_names)

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
    res = torch.einsum(expr, *(t for t in tensors))
    return res  # .refine_names(*new_names)


def bce_with_logits(input, target, **kwargs):
    dim = input.shape[-1]
    names = input.names
    input = input
    target = target.float()

    max_val = (-input).clamp_min(0);
    pos_weight = kwargs.get('pos_weight', None)
    weight = kwargs.get('weight', None)
    if pos_weight is not None:
        # pos_weight need to be broadcasted, thus mul(target) is not inplace.
        log_weight = ((pos_weight - 1) * target) + 1;
        loss = ((1 - target) * input) + (log_weight * ((((-max_val).exp() + ((-input - max_val).exp())).log()) + max_val));
    else:
        loss = ((1 - target) * input) + (max_val) + (((-max_val).exp() + ((-input - max_val).exp())).log());

    if weight is not None:
        loss = loss * weight;

    # res = F.binary_cross_entropy_with_logits(input.reshape(-1, dim), target.float().reshape(-1, dim), **kwargs)
    reduction = kwargs.get('reduction', 'mean')
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        assert reduction == 'none'
        return loss
        # res = res.view(input.shape).rename(*input.names)
    return res


def nll(input, target, **kwargs):
    size = input.shape[-1]
    res = F.nll_loss(input.reshape(-1, size), target.reshape(-1), **kwargs)
    if kwargs.get('reduction', 'mean') == 'none':
        res = res.view(target.shape)
    return res


def cross_entropy_with_logits(input, target, **kwargs):
    dim = input.shape[-1]
    res = F.cross_entropy(input.reshape(-1, dim), target.reshape(-1), **kwargs)
    if kwargs.get('reduction', 'mean') == 'none':
        res = res.view(target.shape)
    return res


def pad(tensor, expr='...', value=0, **kwargs):
    expr, new_names, dim_indices = complete_expr(tensor, expr if "->" in expr else f"{expr} -> {expr}", dims=kwargs)
    dims_to_pad = dict(zip(dim_indices, kwargs.values()))
    assert max(dims_to_pad) < tensor.ndim, "Unknown dim name"
    min_dim_to_pad = min(dims_to_pad)
    padding = [item for i in range(tensor.ndim - 1, min_dim_to_pad - 1, -1) for item in dims_to_pad.get(i, (0, 0))]
    if tensor.shape[-1] == 0:
        padding = ([0, 0] * (tensor.ndim - len(padding) // 2)) + list(reversed(padding))
        new_shape = list(tensor.shape)
        for i, p in enumerate(padding):
            new_shape[i // 2] += p
        new_tensor = torch.full(new_shape, fill_value=value, device=tensor.device)
        new_tensor[tuple(
            slice(before, before + size)
            for before, size in zip(padding[::2], tensor.shape)
        )] = tensor
        return new_tensor
    return F.pad(tensor, padding, value=value)  # .refine_names(*tensor.names)


def wrap_unary_op(op_name):
    fn = getattr(torch.Tensor, op_name)
    if hasattr(fn, 'back'):
        fn = fn.back

    def wrapper(self, dim=None):
        if dim is None:
            return fn(self)
        names = list(self.names)
        if dim in self.names:
            dim = self.names.index(dim)
        names.pop(dim)
        return fn(self, dim)

    wrapper.back = fn
    setattr(torch.Tensor, op_name, wrapper)


def wrap_bin_op(op_name):
    fn = getattr(torch.Tensor, op_name)
    if hasattr(fn, 'back'):
        fn = fn.back

    def wrapper(self, other):
        if hasattr(other, 'names') and hasattr(self, 'names') and None not in self.names and None not in other.names:
            other = other.rearrange(" ".join(name if name in other.names else name + "=1" for name in self.names))
            return fn(self, other)
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
        return fn(self, dim)

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
        values, indices = fn(self, dim)
        return values, indices

    sort.back = fn
    torch.Tensor.sort = sort


def wrap_nonzero():
    fn = torch.Tensor.nonzero
    if hasattr(fn, 'back'):
        fn = fn.back

    def nonzero(self, *args, **kwargs):
        return fn(self, *args, **kwargs)

    nonzero.back = fn
    torch.Tensor.nonzero = nonzero


def wrap_setitem():
    fn = torch.Tensor.__setitem__
    if hasattr(fn, 'back'):
        fn = fn.back

    def __setitem__(self, index, other):
        index = index if torch.is_tensor(index) else index
        other = other if torch.is_tensor(other) else other
        names = self.names
        self = self
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
        if all(n is None for n in self.names):
            return fn(self, index if isinstance(index, tuple) else (index,))
        torch.Tensor.__getitem__ = fn
        try:
            names = (None,)
            new_names = None
            if torch.is_tensor(index):
                if index.dtype == torch.bool:
                    if self.names[:index.ndim] != index.names:
                        index_names = list(index.names)
                        self_names = list(self.names)
                        self_names[:min(index.ndim, self.ndim)], index_names[:min(index.ndim, self.ndim)] = zip(*[
                            (self_name if self_name is not None else index_name, index_name if index_name is not None else self_name)
                            for self_name, index_name in zip(self.names, index.names)])
                        assert set(index_names) <= set(self_names), "Index names must exist in indexed tensor: tensor = {}, index = {} ".format(self_names, index_names)
                        self = self.rename(*self_names).align_to(*index_names, *(name for name in self_names if name not in index_names))
                    else:
                        assert set(index.names) <= set(self.names), "Index names must exist in indexed tensor: tensor = {}, index = {} ".format(self_names, index_names)
                    names = (index.names[-1], *self.names[index.ndim:])
                    index = index
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
                index = tuple(item if torch.is_tensor(item) else item for item in index)
            if not isinstance(index, tuple):
                index = index,
            self = self
            self = fn.__get__(self)(index)
            if new_names is not None:
                self = self.align_to(*new_names)
            return self
        finally:
            torch.Tensor.__getitem__ = __getitem__

    __getitem__.back = fn
    torch.Tensor.__getitem__ = __getitem__


def one_hot(tensor, num_classes, new_name):
    return F.one_hot(tensor, num_classes).rename(*tensor.names, new_name)


def wrap_masked_fill():
    fn = torch.Tensor.masked_fill
    if hasattr(fn, 'back'):
        fn = fn.back

    def masked_fill(self, mask, value):
        if hasattr(mask, 'names') and hasattr(self, 'names') and None not in self.names and None not in mask.names:
            tensor_names = [name for name in self.names if name not in mask.names] + list(mask.names)
            return fn(self.align_to(*tensor_names), mask, value).rename(*tensor_names).align_to(*self.names)
        else:
            return fn(self, mask, value)

    masked_fill.back = fn
    torch.Tensor.masked_fill = masked_fill


def smart_gather(expr, tensor, index,):
    def arange_at_dim(n, dim, ndim):
        view = [1] * ndim
        view[dim] = -1
        return torch.arange(n, device=tensor.device).view(view)

    tensor_names = expr.split("->")[0].split(",")[0].split()
    index_names = expr.split("->")[0].split(",")[1].split()
    common = [name for name in tensor_names if name in index_names]
    result_names = expr.split("->")[1].split()
    dim = next(n for n in index_names if n.startswith("@"))[1:]
    missing_tensor = [name for name in tensor_names if name not in common and name != dim]
    missing_index = [name for name in index_names if name not in common]
    tensor = tensor.permute([tensor_names.index(n) for n in (*common, dim, *missing_tensor)])
    index = index.permute([index_names.index(n) for n in (*common, *missing_index)])
    res = tensor[tuple([arange_at_dim(dim, i, index.ndim) for i, dim in enumerate(tensor.shape[:len(common)])]) + (index,)]
    res.permute([result_names.index(n) for n in (*common, *missing_index, *missing_tensor)])
    return res

def gather(tensor, index, dim):
    def arange_at_dim(n, dim, ndim):
        view = [1] * ndim
        view[dim] = -1
        return torch.arange(n, device=tensor.device).view(view)

    dim = (dim + tensor.ndim) % tensor.ndim
    indices = [(arange_at_dim(size, i, index.ndim) if i != dim else index)
               for i, size in enumerate(tensor.shape)]
    return tensor[tuple(indices)]


def repeat_like(tensor, other):
    n_repeats = {}
    name_to_dim = dict(zip(tensor.names, tensor.shape))
    for other_name, other_dim in zip(other.names, other.shape):
        self_dim = name_to_dim.get(other_name, None)
        if other_dim != self_dim:
            n_repeats[other_name] = other_dim // (self_dim if self_dim is not None else 1)
    return tensor.repeat(" ".join(other.names), **n_repeats)


def multi_dim_triu(x, diagonal=0):
    return x.masked_fill(~torch.ones(x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device).triu(diagonal=diagonal), 0)


def unsqueeze_around(n, dim):
    shape = [1] * n
    shape[dim] = -1
    return shape


def multi_dim_topk(x, topk, mask=None, dim=0):
    if x.dtype == torch.bool:
        mask = x
        x = x.long()
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    flat = x.masked_fill(~mask, -1000000).reshape(*x.shape[:dim], -1)
    flat_mask = mask.reshape(*x.shape[:dim], -1)
    actual_top_k = min(topk, flat_mask.sum(-1).max().item())
    top_values, flat_top_indices = flat.topk(actual_top_k, dim=-1)
    top_indices = []
    for dim_size in reversed(x.shape[dim:]):
        top_indices.insert(0, flat_top_indices % dim_size)
        flat_top_indices = flat_top_indices // dim_size
    if mask is not None:
        top_mask = mask[(*(
            torch.arange(n).view(*unsqueeze_around(top_indices[0].ndim, i))
            for i, n in enumerate(top_indices[0].shape[:-1])),
                         *top_indices,
                         )][..., :actual_top_k]
        return top_indices, top_mask
    return top_indices


def multi_dim_nonzero(x, mask=None, dim=-1):
    return multi_dim_topk(x, topk=x.numel() // x.shape[0], mask=x, dim=dim)


def log1mexp(a):
    # log (1 - exp(x))
    return torch.where(a < math.log(.5), torch.log1p(-a.exp()), (-torch.expm1(a)).log())


def shift(x, dim, n, pad=0):
    shape = list(x.shape)
    shape[dim] = abs(n)

    slices = [slice(None)] * x.ndim
    slices[dim] = slice(n, None) if n >= 0 else slice(None, n)
    pad = torch.full(shape, fill_value=pad, dtype=x.dtype, device=x.device)
    x = torch.cat(([pad] if n > 0 else []) + [x] + ([pad] if n < 0 else []), dim=dim).roll(dims=dim, shifts=n)
    return x[tuple(slices)]


def identity(x):
    return x


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == None:
        return identity
    raise RuntimeError(F"activation should be relu/gelu/glu/None, not {activation}.")


def masked_flip(x, mask, dim_x=-2):
    flipped_x = torch.zeros_like(x)
    flipped_x[mask] = x.flip(dim_x)[mask.flip(-1)]
    return flipped_x


from torch.cuda.amp import custom_bwd, custom_fwd


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp_min(min) if max is None else input.clamp_max(max) if min is None else input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min=None, max=None):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


def repeat(t, n, dim):
    return t.unsqueeze(dim).repeat_interleave(n, dim).view(tuple(-1 if (i - dim + t.ndim) % t.ndim == 0 else s for i, s in enumerate(t.shape)))


def monkey_patch():
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
    if seed is True:
        seed = random.randint(1, 2 ** 16)
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


def get_cliques(adj, mask, indices=None, must_contain=None):
    """
    Extract cliques from a torch adjacency matrix
    :param adj: torch.FloatTensor
        Adjacency matrix: n_samples * n_items * n_items of floats
    :param mask: torch.BoolTensor
        Mask of the items: n_samples * n_items
    :param indices: torch.LongTensor
        Mapping to apply to the extracted indices
    :param must_contain: torch.BoolTensor
        If not None, at least one of these items must be contained in a clique to extract it: n_samples * n_items

    :return: torch.LongTensor
        Cliques, with -1 as pad value n_samples * n_cliques * n_items
    """
    device = adj.device
    cliques_items_indices = []
    for sample_idx, (sample_relations_scores, sample_items_mask) in enumerate(zip(adj.cpu().numpy(), mask.cpu())):
        sample_items_indices = sample_items_mask.nonzero(as_tuple=True)[0].numpy()
        sample_items_mask = sample_items_mask.numpy()
        graph = nx.from_numpy_matrix(sample_relations_scores[sample_items_indices][:, sample_items_indices])
        cliques_items_indices.append([])
        for (entity_idx, clique), _ in zip(enumerate(nx.find_cliques(graph)), range(100)):
            clique_items_indices = sample_items_indices[clique]
            if indices is None:
                cliques_items_indices[-1].append(clique_items_indices[sample_items_mask[clique_items_indices]].tolist())
            else:
                cliques_items_indices[-1].append([
                    idx
                    for i in clique_items_indices[sample_items_mask[clique_items_indices]].tolist()
                    for idx in indices[sample_idx][i][indices[sample_idx][i] != -1].tolist()
                ])
            if must_contain is not None and not must_contain[sample_idx, cliques_items_indices[-1][-1]].any():
                cliques_items_indices[-1].pop(-1)
    cliques_items_indices = pad_to_tensor(cliques_items_indices, pad=-1, dtype=torch.long, device=device)
    if cliques_items_indices.ndim == 2:
        cliques_items_indices = cliques_items_indices.unsqueeze(-1)[..., :0]
    return cliques_items_indices


def inv_logsigmoid(x, eps=1e-8):
    return x - log1mexp(dclamp(x, max=-eps))


seed_all = set_seed
