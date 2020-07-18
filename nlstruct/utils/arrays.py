from itertools import chain

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse, vstack


def flatten_array(array, mask=None):
    """
    Flatten array to get the list of active entries
    If not mask is provided, it's just array.view(-1)
    If a mask is given, then it is array[mask]
    If the array or the mask are sparse, some optimizations are possible, justifying this function

    Parameters
    ----------
    array: scipy.sparse.spmatrix or np.ndarray or torch.Tensor
    mask: scipy.sparse.spmatrix or np.ndarray or torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
    """
    if issparse(array):
        array = array.tocsr(copy=True)
        col_bis = array.copy()
        col_bis.data = np.ones(len(array.data), dtype=bool)
        if mask is not None and issparse(mask):
            if 0 in mask.shape:
                return np.asarray([], dtype=array.dtype)
            res = array[mask]
            # If empty mask, scipy returns a sparse matrix: we use toarray to densify
            if hasattr(res, 'toarray'):
                return res.toarray().reshape(-1)
            # else, scipy returns a 2d matrix, we use asarray to densify
            return np.asarray(res).reshape(-1)
        array = array.toarray()
        if mask is not None:
            mask = as_numpy_array(mask)
    if isinstance(array, (list, tuple)):
        if mask is None:
            return array
        array = np.asarray(array)
    if isinstance(array, np.ndarray):
        if mask is not None:
            if not isinstance(array, np.ndarray):
                raise Exception(f"Mask type {repr(type(mask))} should be the same as array type {repr(type(array))}")
            return array[mask]
        else:
            return array.reshape(-1)
    elif torch.is_tensor(array):
        if mask is not None:
            if not torch.is_tensor(mask):
                raise Exception(f"Mask type {repr(type(mask))} should be the same as array type {repr(type(array))}")
            return array[mask]
        else:
            return array.reshape(-1)
    else:
        raise Exception(f"Unrecognized array type {repr(type(array))} during array flattening (mask type is {repr(type(mask))}')")


def factorize(values, mask=None, reference_values=None, freeze_reference=True):
    """
    Express values in "col" as row numbers in a reference list of values
    The reference values list is the deduplicated concatenation of preferred_unique_values (if not None) and col

    Ex:
    >>> factorize(["A", "B", "C", "D"], None, ["D", "B", "C", "A", "E"])
    ... [3, 2, 1, 0], None, ["D", "B", "C", "A", "E"]
    >>> factorize(["A", "B", "C", "D"], None, None)
    ... [0, 1, 2, 3], None, ["A", "B", "C", "D"]

    Parameters
    ----------
    values: np.ndarray or scipy.sparse.spmatrix or torch.Tensor or list of (np.ndarray or scipy.sparse.spmatrix or torch.Tensor)
        values to factorize
    mask: np.ndarray or scipy.sparse.spmatrix or torch.Tensor or list of (np.ndarray or scipy.sparse.spmatrix or torch.Tensor) or None
        optional mask on col, useful for multiple dimension values arrays
    freeze_reference: bool
        Should we throw out values out of reference values (if given).
        Then we need a mask to mark those rows as disabled
        TODO: handle cases when a mask is not given
    reference_values: np.ndarray or scipy.sparse.spmatrix or torch.Tensor or list or None
        If given, any value in col that is not in prefered_unique_values will be thrown out
        and the mask will be updated to be False for this value

    Returns
    -------
    col, updated mask, reference values
    """
    if isinstance(values, list) and not hasattr(values[0], '__len__'):
        values = np.asarray(values)
    return_as_list = isinstance(values, list)
    all_values = values if isinstance(values, list) else [values]
    del values
    all_masks = mask if isinstance(mask, list) else [None for _ in all_values] if mask is None else [mask]
    del mask

    assert len(all_values) == len(all_masks), "Mask and values lists must have the same length"

    if reference_values is None:
        freeze_reference = False

    all_flat_values = []
    for values, mask in zip(all_values, all_masks):
        assert (
              (isinstance(mask, np.ndarray) and isinstance(values, np.ndarray)) or
              (issparse(mask) and issparse(values)) or
              (torch.is_tensor(mask) and torch.is_tensor(values)) or
              (mask is None and (isinstance(values, (list, tuple, np.ndarray)) or issparse(values) or torch.is_tensor(values)))), (
            f"values and (optional mask) should be of same type torch.tensor, numpy.ndarray or scipy.sparse.spmatrix. Given types are values: {repr(type(values))} and mask: {repr(type(mask))}")
        all_flat_values.append(flatten_array(values, mask))
        # return all_values[0], all_masks[0], all_values[0].tocsr(copy=True).data if hasattr(all_values[0], 'tocsr') else all_values#col.tocsr(copy=True).data if hasattr(col, 'tocsr')

    device = all_flat_values[0].device if torch.is_tensor(all_flat_values[0]) else None
    if sum(len(vec) for vec in all_flat_values) == 0:
        relative_values = all_flat_values[0]
        unique_values = all_flat_values[0]
    # elif torch.is_tensor(all_flat_values[0]):
    #     device = all_flat_values[0].device
    #     if reference_values is None:
    #         unique_values, relative_values = torch.unique(torch.cat(all_flat_values).unsqueeze(0), dim=1, sorted=False, return_inverse=True)
    #     elif freeze_reference:
    #         relative_values, unique_values = torch.unique(torch.cat((reference_values, *all_flat_values)).unsqueeze(0), dim=1, sorted=False, return_inverse=True)[1], reference_values
    #     else:
    #         unique_values, relative_values = torch.unique(torch.cat((reference_values, *all_flat_values)).unsqueeze(0), dim=1, sorted=False, return_inverse=True)
    #     relative_values = relative_values.squeeze(0)
    else:
        was_tensor = False
        if torch.is_tensor(all_flat_values[0]):
            was_tensor = True
            all_flat_values = [as_numpy_array(v) for v in all_flat_values]
            reference_values = as_numpy_array(reference_values)
        if reference_values is None:
            relative_values, unique_values = pd.factorize(np.concatenate(all_flat_values))
        elif freeze_reference:
            relative_values, unique_values = pd.factorize(np.concatenate((reference_values, *all_flat_values)))[0], reference_values
        else:
            relative_values, unique_values = pd.factorize(np.concatenate((reference_values, *all_flat_values)))
        if was_tensor:
            relative_values = as_tensor(relative_values, device=device)
            unique_values = as_tensor(unique_values, device=device)

    if freeze_reference:
        all_unk_masks = relative_values < len(reference_values)
    else:
        all_unk_masks = None

    offset = 0 if reference_values is None else len(reference_values)
    new_flat_values = []
    new_flat_values = []
    unk_masks = []
    for flat_values in all_flat_values:
        indexer = slice(offset, offset + len(flat_values))
        new_flat_values.append(relative_values[indexer])
        unk_masks.append(all_unk_masks[indexer] if all_unk_masks is not None else None)
        offset = indexer.stop
    all_flat_values = new_flat_values
    del new_flat_values

    if freeze_reference:
        unique_values = unique_values[:len(reference_values)]
    new_values = []
    new_masks = []
    for values, mask, flat_relative_values, unk_mask in zip(all_values, all_masks, all_flat_values, unk_masks):
        if issparse(values):
            new_data = flat_relative_values + 1
            if unk_mask is not None:
                new_data[~unk_mask] = 0
            if mask is None:
                values = values.tocsr(copy=True)
                values.data = new_data
                values.eliminate_zeros()
                new_mask = values.copy()
                values.data -= 1
                new_mask.data = np.ones(len(new_mask.data), dtype=bool)
            else:
                values = mask.tocsr(copy=True)
                values.data = new_data
                values.eliminate_zeros()
                new_mask = values.copy()
                values.data -= 1
                new_mask.data = np.ones(len(new_mask.data), dtype=bool)
            new_values.append(values.tolil())
            new_masks.append(new_mask.tolil())
        elif isinstance(values, (list, tuple)):
            mask = unk_mask
            if mask is not None:
                values = [v for v, valid in zip(flat_relative_values, mask) if valid]
                new_values.append(values)
                new_masks.append(None)
            else:
                values = list(flat_relative_values)
                new_values.append(values)
                new_masks.append(None)
        elif isinstance(values, np.ndarray):
            new_mask = mask
            if freeze_reference:
                if mask is None:
                    new_mask = unk_mask.reshape(values.shape)
                else:
                    new_mask = mask.copy()
                    mask[mask] = unk_mask
            if mask is not None:
                values = np.zeros(values.shape, dtype=int)
                values[mask] = flat_relative_values[unk_mask] if unk_mask is not None else flat_relative_values
                new_values.append(values)
                new_masks.append(new_mask)
            else:
                values = flat_relative_values.reshape(values.shape)
                new_values.append(values)
                new_masks.append(new_mask)
        else:  # torch
            new_mask = mask
            if freeze_reference:
                if mask is None:
                    new_mask = unk_mask.view(*values.shape)
                else:
                    new_mask = mask.clone()
                    mask[mask] = unk_mask
            if mask is not None:
                values = torch.zeros(values.shape, dtype=torch.long, device=device)
                values[mask] = flat_relative_values[unk_mask] if unk_mask is not None else flat_relative_values
                new_values.append(values)
                new_masks.append(mask)
            else:
                values = flat_relative_values.view(*values.shape)
                new_values.append(values)
                new_masks.append(new_mask)
    if return_as_list:
        return new_values, new_masks, unique_values
    return new_values[0], new_masks[0], unique_values


def get_deduplicator(values):
    if isinstance(values, np.ndarray):
        perm = values.argsort()
        sorted_values = values[perm]
        mask = np.ones_like(values, dtype=bool)
        mask[1:] = sorted_values[1:] != sorted_values[:-1]
        return perm[mask]
    elif torch.is_tensor(values):
        perm = values.argsort()
        sorted_values = values[perm]
        mask = torch.ones_like(values, dtype=torch.bool)
        mask[1:] = sorted_values[1:] != sorted_values[:-1]
        return perm[mask]
    else:
        raise Exception()


def index_slice(values, indices):
    if issparse(values):
        return values[indices].astype(values.dtype)
    if hasattr(values, 'shape'):
        return values[indices]
    elif hasattr(indices, 'shape'):
        return as_array(values, type(indices), device=getattr(indices, 'device', None))[indices]
    else:
        return type(values)(as_numpy_array(values)[as_numpy_array(indices)])


def as_numpy_array(array, dtype=None):
    if array is None:
        return None
    if isinstance(array, np.ndarray):
        pass
    elif hasattr(array, 'toarray'):
        if dtype is None or np.issubdtype(array.dtype, dtype):
            return array.toarray()
        return array.astype(dtype).toarray()
    elif torch.is_tensor(array):
        array = array.cpu().numpy()
    else:
        array = np.asarray(array, dtype=dtype)
        return array
    if dtype is None or np.issubdtype(array.dtype, dtype):
        return array
    return array.astype(dtype)


def as_tensor(array, device=None, dtype=None):
    if array is None:
        return None
    if torch.is_tensor(array):
        device = device if device is not None else torch.device('cpu')
        return array.to(device)
    elif isinstance(array, np.ndarray):
        return torch.as_tensor(array, device=device, dtype=dtype)
    elif hasattr(array, 'toarray'):
        return torch.as_tensor(array.toarray(), device=device, dtype=dtype)
    else:
        return torch.as_tensor(array, device=device, dtype=dtype)


def as_array(array, t, device, dtype=None):
    if array is None:
        return None
    if issubclass(t, torch.Tensor):
        return as_tensor(array, device=device, dtype=dtype)
    else:
        return as_numpy_array(array, dtype=dtype)


def concat(arrays):
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays)
    elif torch.is_tensor(arrays[0]):
        return torch.cat(arrays)
    elif issparse(arrays[0]):
        max_width = max(a.shape[1] for a in arrays)
        for a in arrays:
            a.resize(a.shape[0], max_width)
        return vstack(arrays, format='csr')
    elif isinstance(arrays[0], (tuple, list)):
        return type(arrays[0])(chain.from_iterable(arrays))
    else:
        raise Exception()


def as_same(array, t, device):
    if issubclass(t, np.ndarray):
        return as_numpy_array(array)
    elif issubclass(t, torch.Tensor):
        return as_tensor(array, device=device)
    elif issubclass(t, list):
        return list(array)
    else:
        raise Exception()
