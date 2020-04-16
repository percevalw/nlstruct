import logging
import pprint

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse, vstack, csr_matrix
from torch.utils.data import DataLoader, BatchSampler


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
        array = array.tocsr()
        col_bis = array.copy()
        col_bis.data = np.ones(len(array.data), dtype=bool)
        if mask is not None and issparse(mask):
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
    col: np.ndarray or scipy.sparse.spmatrix or torch.Tensor or list of (np.ndarray or scipy.sparse.spmatrix or torch.Tensor)
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
        # return all_values[0], all_masks[0], all_values[0].tocsr().data if hasattr(all_values[0], 'tocsr') else all_values#col.tocsr().data if hasattr(col, 'tocsr')
    was_torch = False
    if torch.is_tensor(all_flat_values[0]):
        was_torch = True
        device = all_flat_values[0].device
        all_flat_values = [v.cpu() for v in all_flat_values]
        # if reference_values is None:
        #     unique_values, relative_values = torch.unique(torch.cat(all_flat_values), sorted=False, return_inverse=True)
        # elif freeze_reference:
        #     relative_values, unique_values = torch.unique(torch.cat((reference_values, *all_flat_values)), sorted=False, return_inverse=True)[1], reference_values
        # else:
        #     unique_values, relative_values = torch.unique(torch.cat((reference_values, *all_flat_values)), sorted=False, return_inverse=True)
    if True:
        if reference_values is None:
            relative_values, unique_values = pd.factorize(np.concatenate(all_flat_values))
        elif freeze_reference:
            relative_values, unique_values = pd.factorize(np.concatenate((reference_values, *all_flat_values)))[0], reference_values
        else:
            relative_values, unique_values = pd.factorize(np.concatenate((reference_values, *all_flat_values)))
    if was_torch:
        relative_values = torch.as_tensor(relative_values, device=device)
        unique_values = torch.as_tensor(unique_values, device=device)
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
            values = values.tocsr()
            values.data = flat_relative_values + 1
            if unk_mask is not None:
                values.data[~unk_mask] = 0
            values.eliminate_zeros()
            new_mask = values.copy()
            values.data -= 1
            new_mask.data = np.ones(len(new_mask.data), dtype=bool)

            if mask is not None:
                new_mask = new_mask.multiply(mask.tocsr())
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


def as_numpy_array(array):
    if isinstance(array, np.ndarray):
        return array
    elif hasattr(array, 'toarray'):
        return array.toarray()
    elif torch.is_tensor(array):
        return array.cpu().numpy()
    else:
        return np.asarray(array)


class BatcherPrinter(pprint.PrettyPrinter):
    def format_batcher(self, obj, stream, indent, allowance, context, level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write("Batcher(")
        object_dict = obj.tables
        length = len(object_dict)
        items = [(obj.main_table, obj.tables[obj.main_table]),
                 *list((k, v) for k, v in object_dict.items() if k != obj.main_table)]
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            self.format_tables(items, stream, indent, allowance + 1, context, level, inline=False)
        write('\n' + ' ' * indent + ')')

    def format_array(self, obj, stream, indent, allowance, context, level):
        dtype_str = (
              ("ndarray" if isinstance(obj, np.ndarray) else "tensor" if torch.is_tensor(obj) else str(obj.__class__.__name__)) +
              "[{}]".format(str(obj.dtype) if hasattr(obj, 'dtype') else str(obj.dtypes.values[0]) if len(set(obj.dtypes.values)) == 1 else 'multiple')
        )
        stream.write(dtype_str + str(tuple(obj.shape)))

    def format_columns(self, items, stream, indent, allowance, context, level, inline=False):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(items) - 1
        if inline:
            delimnl = ' '
        else:
            delimnl = '\n' + ' ' * indent
            write('\n' + ' ' * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write("({})".format(key) + ': ')
            self._format(ent, stream, indent,  # + len(key) + 4,
                         allowance if last else 1,
                         context, level)
            if not last:
                write(delimnl)

    def format_tables(self, items, stream, indent, allowance, context, level, inline=False):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(items) - 1
        if inline:
            delimnl = ' '
        else:
            delimnl = '\n' + ' ' * indent
            write('\n' + ' ' * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write("[{}]".format(key) + ': ')
            self.format_columns(ent.items(), stream, indent,  # + len(key) + 4,
                                allowance if last else 1,
                                context, level)
            if not last:
                write(delimnl)

    def _repr(self, obj, context, level):
        """Format object for a specific context, returning a string
        and flags indicating whether the representation is 'readable'
        and whether the object represents a recursive construct.
        """
        if isinstance(obj, Batcher) or hasattr(obj, 'shape'):
            return " " * (self._width + 1)
        return super()._repr(obj, context, level)

    def _format(self, obj, stream, indent, allowance, context, level):
        # We dynamically add the types of our namedtuple and namedtuple like
        # classes to the _dispatch object of pprint that maps classes to
        # formatting methods
        # We use a simple criteria (_asdict method) that allows us to use the
        # same formatting on other classes but a more precise one is possible
        if isinstance(obj, Batcher) and type(obj).__repr__ not in self._dispatch:
            self._dispatch[type(obj).__repr__] = BatcherPrinter.format_batcher
        elif hasattr(obj, 'shape') and type(obj).__repr__ not in self._dispatch:
            self._dispatch[type(obj).__repr__] = BatcherPrinter.format_array
        super()._format(obj, stream, indent, allowance, context, level)


class SparseBatchSampler(BatchSampler):
    def __init__(self, batcher, on, batch_size=32, shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batcher = batcher
        self.on = on

    def __iter__(self):
        length = len(self.batcher)
        block_begins = np.arange(len(self)) * self.batch_size
        block_ends = np.roll(block_begins, -1)
        block_ends[-1] = block_begins[-1] + self.batch_size
        if self.shuffle:
            init_permut = np.random.permutation(length)
            sorter = np.argsort(
                (getattr(self.batcher[self.on], "getnnz", self.batcher[self.on].sum)(1) + np.random.poisson(1, size=length))[init_permut])
            for i in np.random.permutation(len(block_begins)):
                yield init_permut[sorter[block_begins[i]:block_ends[i]]]
        else:
            sorter = np.argsort(getattr(self.batcher[self.on], "getnnz", self.batcher[self.on].sum)(1))
            for i in range(len(block_begins)):
                yield sorter[block_begins[i]:block_ends[i]]

    def __len__(self):
        if self.drop_last:
            return len(self.batcher) // self.batch_size
        else:
            return (len(self.batcher) + self.batch_size - 1) // self.batch_size


class Batcher:
    def __init__(self, tables, main_table=None, masks=None, subcolumn_names=None, foreign_ids=None, primary_ids=None, check=True):
        self.subcolumn_names = subcolumn_names or {}
        if check:
            self.tables = {}
            for table_name, table in tables.items():
                for col_name, col in table.items():
                    if isinstance(col, pd.DataFrame):
                        self.tables.setdefault(table_name, {}).setdefault(col_name, col.values)
                        self.subcolumn_names.setdefault(table_name, {}).setdefault(col_name, list(col.columns))
                    elif isinstance(col, pd.Series):
                        self.tables.setdefault(table_name, {}).setdefault(col_name, col.values)
                    elif isinstance(col, pd.Categorical):
                        self.tables.setdefault(table_name, {}).setdefault(col_name, col.codes)
                    else:
                        assert col is not None, f"Column {repr(table_name)}{repr(col_name)} cannot be None"
                        self.tables.setdefault(table_name, {}).setdefault(col_name, col)
        else:
            self.tables = tables
        self.main_table = main_table or next(iter(tables.keys()))
        self.masks = masks or {}
        if primary_ids is not None:
            self.primary_ids = primary_ids
        else:
            self.primary_ids = {table_name: f"{table_name}_id" for table_name in self.tables if f"{table_name}_id" in self.tables[table_name]}
            if check:
                for table_name, primary_id in self.primary_ids.items():
                    uniques, counts = np.unique(tables[table_name][primary_id], return_counts=True)
                    duplicated = uniques[counts > 1]
                    assert len(duplicated) == 0, f"Primary id {repr(primary_id)} of {repr(table_name)} has {len(duplicated)} duplicate{'s' if len(duplicated) > 0 else ''}, " \
                                                 f"when it should be unique: {repr(list(duplicated[:5]))} (first 5 shown)"
        if isinstance(foreign_ids, dict):
            self.foreign_ids = foreign_ids
        else:
            if foreign_ids == "relative":
                mode = "relative"
            elif foreign_ids == "absolute":
                mode = "absolute"
            else:
                mode = "absolute"
                assert foreign_ids is None, f"Unrecognized format for foreign_ids: {type(foreign_ids)}"
            self.foreign_ids = {}
            for table_name, table_columns in self.tables.items():
                # ex: table_name = "mention", table_columns = ["sample_id", "begin", "end", "idx_in_sample"]
                for col_name in table_columns:
                    if col_name.endswith('_id') and col_name != self.primary_ids.get(table_name, None):
                        prefix = col_name[:-3]
                        foreign_table_name = next((table_name for table_name in self.tables if prefix.endswith(table_name)), None)
                        # foreign_table_id = f"{table_name}_id"
                        if foreign_table_name is not None:
                            self.foreign_ids.setdefault(table_name, {})[col_name] = (foreign_table_name, mode)
        if masks is not None:
            self.masks = masks
        else:
            self.masks = {}
            for table_name, table_columns in self.tables.items():
                # ex: table_name = "mention", table_columns = ["sample_id", "begin", "end", "idx_in_sample"]
                for col_name in table_columns:
                    if col_name.endswith('_mask') and col_name != self.primary_ids.get(table_name, None):
                        id_name = col_name[:-5] + '_id'
                        # foreign_table_id = f"{table_name}_id"
                        if id_name in table_columns:
                            self.masks.setdefault(table_name, {})[id_name] = col_name

        if check:
            # Check that all tables / columns exist in subcolumn names
            for table_name, cols in self.subcolumn_names.items():
                assert table_name in self.tables, f"Unknown table {repr(table_name)} in `subcolumn_names`"
                for col_name in cols:
                    assert col_name in self.tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `subcolumn_names`"
            # Check that all tables / columns exist in masks
            for table_name, cols in self.masks.items():
                assert table_name in self.tables, f"Unknown table {repr(table_name)} in `masks`"
                for col_name, mask_name in cols.items():
                    assert col_name in self.tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `masks`"
                    assert mask_name in self.tables[table_name], f"Unknown mask {repr(mask_name)} for column {table_name}/{col_name} in `masks`"
            # Check that all tables / columns exist in foreign_ids
            for table_name, cols in self.foreign_ids.items():
                assert table_name in self.tables, f"Unknown table {repr(table_name)} in `foreign_ids`"
                for col_name, (foreign_table_name, mode) in cols.items():
                    assert mode in ('relative', 'absolute'), f"Unknown mode {repr(mode)} for {table_name}/{col_name} in `foreign_ids`"
                    assert col_name in self.tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `foreign_ids`"
                    assert foreign_table_name in self.tables, f"Unknown foreign table {repr(foreign_table_name)} for column {table_name}/{col_name} in `foreign_ids`"
            # Check that all tables / columns exist in primary_ids
            for table_name, col_name in self.primary_ids.items():
                assert table_name in self.tables, f"Unknown table {repr(table_name)} in `primary_ids`"
                assert col_name in self.tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `primary_ids`"

    def __len__(self):
        return next(iter(self.tables[self.main_table].values())).shape[0]

    @property
    def device(self):
        return getattr(next(iter(self.tables[self.main_table].values())), 'device', None)

    @property
    def shape(self):
        return next(iter(self.tables[self.main_table].values())).shape

    def keys(self):
        return self.tables[self.main_table].keys()

    def values(self):
        return self.tables[self.main_table].values()

    def items(self):
        return self.tables[self.main_table].items()

    def copy(self):
        return Batcher(
            {key: dict(table) for key, table in self.tables.items()},
            main_table=self.main_table,
            masks=dict(self.masks),
            subcolumn_names=dict(self.subcolumn_names),
            foreign_ids={table_name: dict(vals) for table_name, vals in self.foreign_ids.items()},
            primary_ids=dict(self.primary_ids),
            check=False,
        )

    def set_main_table(self, name, inplace=False):
        if not inplace:
            self = self.copy()
        self.main_table = name
        if not inplace:
            return self

    def __getitem__(self, indexer):
        if isinstance(indexer, str):
            indexer = (indexer,)
        if isinstance(indexer, slice):
            device = self.device
            if device is None:
                indexer = np.arange(indexer.start or 0, indexer.stop, indexer.step or 1)
            else:
                indexer = torch.arange(indexer.start or 0, indexer.stop, indexer.step or 1, device=device)
        if isinstance(indexer, tuple):
            if indexer[0] not in self.tables:
                indexer = (self.main_table, *indexer)
            current = self.tables[indexer[0]]
            if len(indexer) == 1:
                return self.slice_tables({indexer[0]: slice(None)})
            if len(indexer) > 1:
                if isinstance(indexer[1], list):
                    current = [current[name] for name in indexer[1]]
                else:
                    current = current[indexer[1]]
            if len(indexer) > 2:
                if isinstance(current, pd.DataFrame):
                    return current[indexer[2]]
                else:
                    current_column_names = self.subcolumn_names.get(indexer[0], {}).get(indexer[1], {})
                    if isinstance(indexer[2], str):
                        if indexer[2] in current_column_names:
                            current = current[:, current_column_names.index(indexer[2])]
                        else:
                            raise KeyError(indexer)
                    elif isinstance(indexer[2], int):
                        current = current[:, indexer[2]]
                    elif isinstance(indexer[2], (list, tuple)):
                        parts_idx = []
                        for part in indexer[2]:
                            if part in current_column_names:
                                parts_idx.append(current_column_names.index(part))
                            else:
                                raise KeyError((indexer[0], indexer[1], part))
                        current = current[:, parts_idx]
                    else:
                        raise KeyError(indexer)
            return current
        elif isinstance(indexer, list):
            if isinstance(indexer[0], str):
                return self.slice_tables({table_name: slice(None) for table_name in indexer})
            else:
                return self.query_ids(indexer)
        elif isinstance(indexer, dict):
            return self.slice_tables(indexer)
        else:
            if not (isinstance(indexer, np.ndarray) or torch.is_tensor(indexer)):
                if hasattr(indexer, 'toarray'):
                    indexer = indexer.toarray()
                else:
                    indexer = np.asarray(indexer)
            if len(indexer.shape) == 1 and indexer.dtype == np.bool:
                return self.query_ids(np.flatnonzero(indexer))
            elif len(indexer.shape) == 1 and torch.is_tensor(indexer) and indexer.dtype == torch.bool:
                return self.query_ids(torch.nonzero(indexer, as_tuple=True)[0])
            else:
                return self.query_ids(indexer)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if isinstance(value, (Batcher, self.__class__)) or str(type(value)) == "<class 'nlstruct.core.batcher.Batcher'>":
                foreign_ids = self.foreign_ids
                self.switch_foreign_ids_mode("absolute", inplace=True)
                value = value.switch_foreign_ids_mode("absolute")
                self.tables[key] = value.tables[key]
            else:
                raise Exception(f"TODO {type(value)}")
            return self
        elif isinstance(key, tuple):
            table = self.tables[key[0]]
            if isinstance(key[1], str):
                table[key[1]] = value
            elif isinstance(key[1], list):
                assert len(key[1]) == len(value)
                for name, val in zip(key[1], value):
                    table[name] = val
            else:
                raise Exception("Can only assign array or list of arrays to either (table_name:str, col_name:str) or (table_name:str, col_names: list of str)")
        else:
            raise Exception("Key should be either a str or a tuple of (table_name:str, col_name:str) or (table_name:str, col_names: list of str)")

    def __delitem__(self, key):
        if isinstance(key, str):
            del self.tables[key]
            if key == self.main_table:
                self.main_table = next(iter(self.tables.keys()))
            if key in self.masks:
                del self.masks[key]
            if key in self.foreign_ids:
                del self.foreign_ids[key]
            if key in self.subcolumn_names:
                del self.subcolumn_names[key]
            if key in self.primary_ids:
                del self.primary_ids[key]
            for name, foreigns in self.foreign_ids.items():
                for col_name, (table, mode) in list(foreigns.items()):
                    if table == key:
                        del foreigns[col_name]
        elif isinstance(key, tuple):
            table = self.tables[key[0]]
            if isinstance(key[1], str):
                del table[key[1]]
            elif isinstance(key[1], (list, tuple)):
                for name in key[1]:
                    del table[name]
        else:
            raise Exception("Can only delete columns: (table_name:str, col_name:str) or (table_name:str, col_names: list of str)")

    def __repr__(self):
        return BatcherPrinter(indent=2, depth=2).pformat(self)

    def dataloader(self,
                   batch_size=32,
                   sparse_sort_on=None,
                   shuffle=False,
                   device=None,
                   dtypes=None,
                   **kwargs):
        batch_sampler = kwargs.pop("batch_sampler", None)
        if sparse_sort_on is not None:
            batch_sampler = SparseBatchSampler(self, on=sparse_sort_on, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        else:
            kwargs['batch_size'] = batch_size
            kwargs['shuffle'] = shuffle
        self = self.switch_foreign_ids_mode("relative")
        return DataLoader(range(len(self)),  # if self._idx is None else self._idx,
                          collate_fn=lambda ids: self.query_ids(ids, device=device, dtypes=dtypes),
                          batch_sampler=batch_sampler,
                          **kwargs)

    @classmethod
    def query_table(cls, table, ids):
        new_table = {}
        for col_name, col in list(table.items()):
            new_table[col_name] = col[ids]
        return new_table

    def prepare_for_indexing(self, inplace=False):
        if not inplace:
            self = self.copy()

        self.fill_primary_ids(inplace=True)
        self.switch_foreign_ids_mode("relative", inplace=True)

        if not inplace:
            return self

    def fill_primary_ids(self, subset=None, offsets=None, inplace=False):
        if not inplace:
            self = self.copy()

        if isinstance(subset, str):
            subset = [subset]

        if offsets is None:
            offsets = {}

        device = self.device
        for sample_table_name, sample_columns in list(self.tables.items()):
            sample_id_name = self.primary_ids.get(sample_table_name, f"{sample_table_name}_id")
            if (subset is None or sample_table_name in subset) and sample_id_name not in sample_columns:
                self.primary_ids[sample_table_name] = sample_id_name
                offset = offsets.get(sample_table_name, 0)
                if device is None:
                    self.tables[sample_table_name][sample_id_name] = np.arange(offset, offset + next(iter(self.tables[sample_table_name].values())).shape[0])
                else:
                    self.tables[sample_table_name][sample_id_name] = torch.arange(offset, offset + next(iter(self.tables[sample_table_name].values())).shape[0], device=device)

        if not inplace:
            return self

    def switch_foreign_ids_mode(self, mode="relative", inplace=False):
        """
        Sort all tables against their main id if given, and put reversed ids in each table if it is
        accessible from another table
        """
        if not inplace:
            self = self.copy()
        # for sample_table_name, sample_columns in list(self.tables.items()):
        #     sample_id_name = self.get_id_of_table(sample_table_name)
        #     if sample_id_name in sample_columns:
        #         sample_fragment_row_numbers = sample_columns[sample_id_name].argsort()
        #         self.tables[sample_table_name] = self.query_table(sample_columns, sample_fragment_row_numbers)

        # Given a batcher
        # >>> Batcher(
        #   [sample]:
        #     (sample_id): ...
        #     (fragment_id): ...
        #     (fragment_mask): ...
        #   [fragment]:
        #     (fragment_id): ...
        #     (label): ...
        # We want to add the columns "sample_id", and "idx" to each fragment, so we know
        # for each fragment what is its position in the sample and which sample it belongs to

        # The variables in the following code have been named in order to improve readability as to their role
        if mode == "relative":
            table_name_to_others_referencing_it = {}
            for table_name, table_foreign_ids in self.foreign_ids.items():
                for table_foreign_id_col, (foreign_table_name, old_mode) in table_foreign_ids.items():
                    if old_mode == "absolute":
                        # If the table has a primary id
                        if foreign_table_name in self.primary_ids:
                            logging.debug(f"Switching {table_name}/{table_foreign_id_col} from absolute to relative mode")
                            table_name_to_others_referencing_it.setdefault(foreign_table_name, []).append((table_name, table_foreign_id_col))
                        table_foreign_ids[table_foreign_id_col] = (foreign_table_name, "relative")  # inplace modification
            for table_name, others in table_name_to_others_referencing_it.items():
                new_relative_ids, new_masks = factorize(
                    values=[self.tables[other_table_referencing_it][other_table_foreign_id]
                            for other_table_referencing_it, other_table_foreign_id in others],
                    mask=[self.tables[other_table_referencing_it].get(self.masks.get(other_table_referencing_it, {}).get(other_table_foreign_id, None), None)
                          for other_table_referencing_it, other_table_foreign_id in others],
                    reference_values=self.tables[table_name][self.primary_ids[table_name]],
                    freeze_reference=True
                )[:2]
                for (other_table_name, table_col_name), table_foreign_ids, table_foreign_mask in zip(others, new_relative_ids, new_masks):
                    self.tables[other_table_name][table_col_name] = table_foreign_ids
                    if table_foreign_mask is not None and self.masks.get(other_table_name, {}).get(table_col_name, None):
                        self.tables[other_table_name][self.masks[other_table_name][table_col_name]] = table_foreign_mask
        elif mode == "absolute":
            for table_name, table_foreign_ids in self.foreign_ids.items():
                for table_foreign_id_col, (foreign_table_name, old_mode) in table_foreign_ids.items():
                    if old_mode == "relative":
                        logging.debug(f"Switching {table_name}/{table_foreign_id_col} from relative to absolute mode")
                        table_foreign_ids[table_foreign_id_col] = (foreign_table_name, "absolute")  # inplace modification
                        array_to_change = self.tables[table_name][table_foreign_id_col]
                        if foreign_table_name not in self.primary_ids:
                            raise Exception(f"Table {foreign_table_name} is missing its primary id in order to switch {table_name}/{table_foreign_id_col} to absolute mode")
                        if issparse(array_to_change):
                            array_to_change = self.tables[table_name][table_foreign_id_col] = array_to_change.tocsr()
                            array_to_change.data = self.tables[foreign_table_name][self.primary_ids[foreign_table_name]][array_to_change.data]
                        else:
                            self.tables[table_name][table_foreign_id_col] = self.tables[foreign_table_name][self.primary_ids[foreign_table_name]][array_to_change]

        if not inplace:
            return self

    def old(self):
        for sample_table_name, sample_columns in list(self.tables.items()):
            # Ex: sample_table_name: doc
            #     sample_columns: [doc_id, mention_id, mention_mask]
            sample_id_name = self.primary_ids[sample_table_name]
            for sample_col_name, sample_column in sample_columns.items():
                # Ex: sample_col_name: mention_id
                #     fragment_table_name: mention
                #     sample_fragment_mask_name: mention_mask
                fragment_table_name = self.foreign_ids[sample_table_name][sample_col_name][0]
                sample_fragment_mask_name = self.masks.get(sample_table_name, {}).get(sample_col_name, None)
                if fragment_table_name and fragment_table_name != sample_table_name:
                    fragment_id_name = self.primary_ids[fragment_table_name]
                    try:
                        fragment_columns = self.tables[fragment_table_name]

                        if sample_id_name not in fragment_columns:
                            continue

                        if sample_fragment_mask_name:
                            sample_fragment_mask = sample_columns[sample_fragment_mask_name]  # = batcher["sample"]["fragment_mask"]
                            # Transform batcher["sample"]["fragment_id"] into batcher["fragment"] row numbers
                            # If batcher["fragment_id"] is sorted with no missing id (=arange(...)), then
                            sample_fragment_row_numbers = np.argsort(factorize(flatten_array(sample_column, sample_fragment_mask), None, fragment_columns[fragment_id_name])[0])
                            if issparse(sample_fragment_mask):
                                indices = sample_fragment_mask.nonzero()
                                fragment_columns[sample_id_name], fragment_columns['idx'] = indices[0][sample_fragment_row_numbers], indices[1][sample_fragment_row_numbers]
                            elif isinstance(sample_fragment_mask, np.ndarray):
                                for axis_name, idx in zip(
                                      [sample_id_name, 'idx', (f"idx_{i+2}" for i in range(len(sample_column.shape[2:])))],
                                      sample_fragment_mask.nonzero()):
                                    fragment_columns[axis_name] = idx[sample_fragment_row_numbers]
                            elif torch.is_tensor(sample_fragment_mask):
                                for axis_name, idx in zip(
                                      [sample_id_name, 'idx', (f"idx_{i+2}" for i in range(len(sample_column.shape[2:])))],
                                      sample_fragment_mask.nonzero()[sample_fragment_row_numbers].t()):
                                    fragment_columns[axis_name] = idx
                            else:
                                raise Exception(f"Unrecognized mask format for {sample_fragment_mask_name}: {sample_fragment_mask.__class__}")
                        else:
                            sample_fragment_row_numbers = factorize(flatten_array(sample_column), None, fragment_columns[fragment_id_name])[0]
                            sample_fragment_row_numbers = np.argsort(sample_fragment_row_numbers)
                            for axis_name, axis_i in zip([sample_id_name, 'idx', (f"idx_{i+2}" for i in range(len(sample_column.shape[2:])))], range(len(sample_column.shape))):
                                fragment_columns[axis_name] = (
                                    np.tile(
                                        np.arange(sample_column.shape[axis_i]).reshape(
                                            tuple(1 for _ in sample_column.shape[:axis_i]) + (-1,) + tuple(1 for _ in sample_column.shape[axis_i + 1:])
                                        ),
                                        tuple(n for n in sample_column.shape[:axis_i]) + (1,) + tuple(n for n in sample_column.shape[axis_i + 1:])).reshape(-1)
                                )[sample_fragment_row_numbers]
                    except:
                        raise Exception(f"Error occurred during ids completion from table {repr(sample_table_name)} to table {repr(fragment_table_name)}")

        self.sorted = True

        if not inplace:
            return self

    set_main = set_main_table

    def densify(self, device=None, dtypes=None):
        new_tables = dict(self.tables)
        new_column_names = self.subcolumn_names
        dtypes = dtypes or {}

        for table_name, table in self.tables.items():
            new_table = {}
            for col_name, col in table.items():
                if issparse(col):
                    col = col.toarray()
                if isinstance(col, pd.DataFrame):
                    new_column_names.setdefault(table_name, {}).setdefault(col_name, list(col.columns))
                    if device is not None:
                        col = col.values
                elif isinstance(col, pd.Series):
                    if device is not None:
                        col = col.values
                torch_dtype = dtypes.get(table_name, {}).get(col_name, torch.long if not torch.is_tensor(col) and np.issubdtype(col.dtype, np.integer) else None)
                if device is not None and (not torch.is_tensor(col) or col.device != device or (torch_dtype is not None and col.dtype != torch_dtype)):
                    col = torch.as_tensor(col, device=device, dtype=torch_dtype)
                new_table[col_name] = col
            new_tables[table_name] = new_table

        res = self.copy()
        res.tables = new_tables
        return res

    def sparsify(self):
        """
        Converts the columns into numpy arrays and scipy sparse arrays

        Returns
        -------
        Batcher
        """

        new_tables = {}
        densified_masks = {}
        sparsified_masks = {}

        # First sparsify column that have masks
        for table_name, table_masks in self.masks.items():
            for col_name, mask_name in table_masks.items():
                mask = self.tables[table_name][mask_name]
                col = self.tables[table_name][col_name]
                if issparse(mask) and not issparse(col):
                    if mask_name not in densified_masks:
                        densified_masks[mask_name] = mask.toarray()
                    col = as_numpy_array(col)
                    data = col[densified_masks[mask_name]]
                    col = mask.copy()
                    col.data = data
                elif issparse(col) and not issparse(mask):
                    if mask_name not in sparsified_masks:
                        sparsified_masks[mask_name] = mask = csr_matrix(as_numpy_array(mask))
                    else:
                        mask = sparsified_masks[mask_name]
                elif not issparse(col) and not issparse(mask):
                    data = as_numpy_array(col)[as_numpy_array(mask)]
                    if mask_name not in sparsified_masks:
                        sparsified_masks[mask_name] = mask = csr_matrix(as_numpy_array(mask))
                    else:
                        mask = sparsified_masks[mask_name]
                    col = mask.copy()
                    col.data = data
                new_tables.setdefault(table_name, {}).setdefault(mask_name, mask.tolil())
                new_tables.setdefault(table_name, {}).setdefault(col_name, col.tolil())

        # Then convert the other columns as numpy arrays
        for table_name, table in self.tables.items():
            for col_name, col in table.items():
                if new_tables.get(table_name, {}).get(col_name, None) is None:
                    new_tables.setdefault(table_name, {})[col_name] = as_numpy_array(col) if not issparse(col) else col

        res = self.copy()
        res.tables = new_tables
        return res

    def slice_tables(self, tables_and_columns):
        tables_and_columns = {
            table_name: ([col_name for col_name in self.tables[table_name]] if col_names == slice(None) else col_names)
            for table_name, col_names in tables_and_columns.items()
        }
        res = Batcher({
            table_name: {
                col_name: self.tables[table_name][col_name]
                for col_name in col_names
            }
            for table_name, col_names in tables_and_columns.items()},
            subcolumn_names={table_name: {col_name: sub_col_name
                                          for col_name, sub_col_name in col_name_to_subcol_name.items() if col_name in tables_and_columns[table_name]}
                             for table_name, col_name_to_subcol_name in self.subcolumn_names.items() if table_name in tables_and_columns},
            masks={table_name: {col_name: mask_name for col_name, mask_name in col_name_to_mask_name.items() if col_name in tables_and_columns[table_name]}
                   for table_name, col_name_to_mask_name in self.masks.items() if table_name in tables_and_columns},
            foreign_ids={table_name: {col_name: (foreign_table, mode) for col_name, (foreign_table, mode) in col_name_to_foreign_table.items() if
                                      col_name in tables_and_columns[table_name] and foreign_table in tables_and_columns}
                         for table_name, col_name_to_foreign_table in self.foreign_ids.items() if table_name in tables_and_columns},
            primary_ids={table_name: primary_id for table_name, primary_id in self.primary_ids.items() if table_name in tables_and_columns},
            check=False, )
        return res

    def query_ids(self, ids, **densify_kwargs):
        if not all(mode == "relative"
                   for table_name, col_to_modes in self.foreign_ids.items()
                   for col_name, (_, mode) in col_to_modes.items()):
            self = self.switch_foreign_ids_mode("relative")
        device = self.device
        if device is not None:
            ids = torch.as_tensor(ids, device=device)
        selected_ids = {self.main_table: ids}
        queried_tables = {}
        queue = {self.main_table}
        while len(queue):
            table_name = queue.pop()
            # Ex: table_name = relations
            table = self.tables[table_name]
            queried_table = self.query_table(table, selected_ids[table_name])
            logging.debug(f"Queried table {repr(table_name)}. Previously queried tables are {repr(tuple(queried_tables.keys()))}")
            for col_name, col in queried_table.items():
                # Ex: col_name = from_mention_id
                #     foreign_table_name = mention
                #     foreign_table_id = mention_id
                foreign_table_name, mode = self.foreign_ids.get(table_name, {}).get(col_name, (None, None))
                # We don't want to reindex the token_id column in the token table: it's useless and we will
                # moreover need it intact for when we rebuild the original data
                if foreign_table_name and foreign_table_name != table_name:
                    # assert mode == "relative", "You must first switch the mode of the {}/{} column to 'relative' using the method switch_foreign_ids_mode"
                    mask_name = self.masks.get(table_name, {}).get(col_name, None)
                    new_col, new_mask, unique_ids = factorize(
                        values=col,
                        mask=queried_table.get(mask_name, None),
                        reference_values=selected_ids.get(foreign_table_name, None),
                        # If querying was done against the main axis primary ids (main_table)
                        # then we don't want to any more ids than those that were given
                        # ex: batcher.set_main("relation")[:10] => only returns relations 0, 1, ... 9
                        # If a table refers to other relations through foreign keys, then those pointers will be masked
                        # For non main ids (ex: mentions), we allow different tables to disagree on the mentions to query
                        # and retrieve all of the needed mentions
                        freeze_reference=foreign_table_name in queried_tables,
                    )
                    # new_col, new_mask, unique_ids = col, queried_table.get(mask_name, None), col.tocsr().data if hasattr(col, 'tocsr') else col#selected_ids.get(foreign_table_name, None)
                    if mask_name is not None:
                        queried_table[mask_name] = new_mask
                    selected_ids[foreign_table_name] = unique_ids
                    queried_table[col_name] = new_col
                    if foreign_table_name not in queried_tables:
                        queue.add(foreign_table_name)
            queried_tables[table_name] = queried_table

        masks_length = {}
        for table_name, table_masks in self.masks.items():
            for col_name, mask_name in table_masks.items():
                if mask_name not in masks_length:
                    if issparse(queried_tables[table_name][mask_name]):
                        if hasattr(queried_tables[table_name][mask_name], 'indices'):
                            if len(queried_tables[table_name][mask_name].indices):
                                max_length = queried_tables[table_name][mask_name].indices.max() + 1
                            else:
                                max_length = 0
                        elif hasattr(queried_tables[table_name][mask_name], 'rows'):
                            max_length = max((max(r, default=-1) + 1 for r in queried_tables[table_name][mask_name].rows), default=0)
                        else:
                            raise Exception(f"Unrecognized mask format for {mask_name}: {queried_tables[table_name][mask_name].__class__}")
                        masks_length[mask_name] = max_length
                        queried_tables[table_name][mask_name].resize(queried_tables[table_name][mask_name].shape[0], masks_length[mask_name])
                    else:
                        max_length = queried_tables[table_name][mask_name].sum(-1).max()
                        masks_length[mask_name] = max_length
                        queried_tables[table_name][mask_name] = queried_tables[table_name][mask_name][:, :masks_length[mask_name]]
                if issparse(queried_tables[table_name][col_name]):
                    queried_tables[table_name][col_name].resize(queried_tables[table_name][col_name].shape[0], masks_length[mask_name])
                else:
                    queried_tables[table_name][col_name] = queried_tables[table_name][col_name][:, :masks_length[mask_name]]

        new_tables = dict(self.tables)
        new_tables.update(queried_tables)

        res = self.copy()
        res.tables = new_tables
        if densify_kwargs:
            res = res.densify(**densify_kwargs)
        return res

    @classmethod
    def concat(cls, batches, sparsify=True, allow_non_unique_primary_ids=False):
        """

        Parameters
        ----------
        batches: list of Batcher
            All those batches must have the same structure

        Returns
        -------
        Batcher
        """

        struct = batches[0].copy()
        new_tables = {}

        new_batches = []
        offsets = {table_name: 0 for table_name in struct.tables}
        for i, batch in enumerate(batches):
            new_batches.append(batch.fill_primary_ids(offsets=offsets).switch_foreign_ids_mode("absolute"))
            for table_name, table in batch.tables.items():
                offsets[table_name] += next(iter(table.values())).shape[0]
            if sparsify:
                new_batches[-1] = new_batches[-1].sparsify()
        batches = new_batches
        del new_batches

        struct = batches[0]

        batches = [b.tables for b in batches]

        for sample_table_name, sample_table in struct.tables.items():
            new_tables[sample_table_name] = {}
            for sample_col_name, sample_column in sample_table.items():
                if sample_col_name in new_tables[sample_table_name]:
                    continue
                mask_name = struct.masks.get(sample_table_name, {}).get(sample_col_name, None)

                # Convert to sparse if needed
                if issparse(sample_column):
                    if mask_name is None:
                        shape1 = max([dic[sample_table_name][sample_col_name].shape[1] for dic in batches])
                        for batch in batches:
                            batch[sample_table_name][sample_col_name].resize(batch[sample_table_name][sample_col_name].shape[0], shape1)
                        new_tables[sample_table_name][sample_col_name] = [dic[sample_table_name][sample_col_name] for dic in batches]
                    else:
                        shape1 = max([dic[sample_table_name][mask_name].shape[1] for dic in batches])
                        for batch in batches:
                            batch[sample_table_name][sample_col_name].resize(batch[sample_table_name][sample_col_name].shape[0], shape1)
                        new_tables[sample_table_name][sample_col_name] = [dic[sample_table_name][sample_col_name] for dic in batches]
                elif isinstance(sample_column, np.ndarray):
                    new_shape = np.asarray([dic[sample_table_name][sample_col_name].shape for dic in batches]).max(0)[1:]
                    new_tables[sample_table_name][sample_col_name] = [
                        np.pad(dic[sample_table_name][sample_col_name],
                               [(0, 0), *((0, p) for p in new_shape - np.asarray(dic[sample_table_name][sample_col_name].shape[1:]))])
                        for dic in batches]
                elif torch.is_tensor(sample_column):
                    new_shape = torch.as_tensor([dic[sample_table_name][sample_col_name].shape for dic in batches]).max(0)[0][1:]
                    new_tables[sample_table_name][sample_col_name] = [
                        torch.nn.functional.pad(dic[sample_table_name][sample_col_name],
                                                [a for p in reversed(new_shape - torch.tensor(dic[sample_table_name][sample_col_name].shape[1:])) for a in (0, p)],
                                                mode="constant", value=0)
                        for dic in batches]

                # Check if we need to align ids between two or more matrices
                # fragment_table_name = struct.foreign_ids[sample_col_name][0]
                # if fragment_table_name and fragment_table_name != sample_table_name:
                #     # If fragment_id doesn't exist in fragment table, (for ex: new fragments), create it
                #     fragment_table_id_name = struct.primary_ids[fragment_table_name]
                #     if fragment_table_id_name not in struct.tables[fragment_table_name]:
                #         i = 0
                #         for batch in batches:
                #             dic_size = next(iter(batch[fragment_table_name].values())).shape[0]
                #             batch[fragment_table_name][fragment_table_id_name] = np.arange(i, i + dic_size)
                #             i += dic_size
                #     # Until now new_tables["sample"]["fragment_id"] was row numbers to each batch in the given batches
                #     # Here we transform those row numbers into the actual ids batches["fragment"]["fragment_id"]
                #     if issparse(new_tables[sample_table_name][sample_col_name][0]):
                #         for sps, batch in zip(new_tables[sample_table_name][sample_col_name], batches):
                #             sps.data = batch[fragment_table_name][fragment_table_id_name][sps.data]
                #     else:
                #         new_tables[sample_table_name][sample_col_name] = [
                #             dic[fragment_table_name][fragment_table_id_name][x]
                #             for x, dic in zip(new_tables[sample_table_name][sample_col_name], batches)]

                # Finally concatenate the matrices, sparse or dense
                if issparse(new_tables[sample_table_name][sample_col_name][0]):
                    new_tables[sample_table_name][sample_col_name] = vstack(new_tables[sample_table_name][sample_col_name]).tolil()
                elif isinstance(new_tables[sample_table_name][sample_col_name][0], np.ndarray):
                    new_tables[sample_table_name][sample_col_name] = np.concatenate(new_tables[sample_table_name][sample_col_name])
                else:
                    new_tables[sample_table_name][sample_col_name] = torch.cat(new_tables[sample_table_name][sample_col_name])

        new_tables = Batcher(new_tables,
                             main_table=struct.main_table,
                             masks=struct.masks,
                             subcolumn_names=struct.subcolumn_names,
                             foreign_ids=struct.foreign_ids,
                             primary_ids=struct.primary_ids,
                             check=False, )
        for sample_table_name, table in list(new_tables.tables.items()):
            id_name = struct.primary_ids[sample_table_name]
            if id_name in table:
                if not allow_non_unique_primary_ids:
                    message = f"Primary id {sample_table_name}/{id_name} is not unique.\n" \
                              f"You can either set Batcher.concat `allow_non_unique_primary_ids` parameter to True or check that each concatenated" \
                              f" batch has no redundant {sample_table_name}/{id_name}.\n" \
                              f"You maybe forgot to set foreign_ids='relative' when you created these batches."

                if torch.is_tensor(table[id_name]):
                    uniq, inverse = torch.unique(table[id_name], sorted=False, return_inverse=True)
                    if not allow_non_unique_primary_ids:
                        assert len(uniq) == len(table[id_name]), message
                    uniquifier = torch.unique(torch.cat([table[id_name], uniq]), sorted=False, return_inverse=True)[1][-len(uniq):]
                elif isinstance(table[id_name], np.ndarray):
                    inverse, uniq = pd.factorize(table[id_name])
                    if not allow_non_unique_primary_ids:
                        assert len(uniq) == len(table[id_name]), message
                    uniquifier = pd.factorize(np.concatenate([table[id_name], uniq]))[0][-len(uniq):]
                else:
                    raise Exception(f"Primary id {id_name} of table {sample_table_name} should be a torch tensor or a numpy ndarray")
                new_tables.tables[sample_table_name] = new_tables.query_table(table, uniquifier)
        return new_tables
