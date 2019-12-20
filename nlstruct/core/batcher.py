import pprint

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse, vstack, csr_matrix
from torch.utils.data import DataLoader, BatchSampler


class Batcher:
    def __init__(self, tables, main_table=None, masks=None, column_names=None):
        self.tables = {table_name: {col_name: col for col_name, col in table.items() if col is not None}
                       for table_name, table in tables.items() if table is not None}
        self.main_table = main_table or next(iter(tables.keys()))
        self.masks = masks or {}
        self.column_names = column_names or {}

    def __len__(self):
        return next(iter(self.tables[self.main_table].values())).shape[0]

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
            masks=self.masks,
            column_names=self.column_names,
        )

    def complete_ids(self, inplace=False):
        """
        Sort all tables against their main id if given, and put reversed ids in each table if it is
        accessible from another table
        """
        if not inplace:
            self = self.copy()
        for table_name, table in list(self.tables.items()):
            id_name = self.get_id_name(table_name)
            if id_name in table:
                sorter = np.argsort(table[id_name])
                self.tables[table_name] = self.query_table(table, sorter)
            else:
                self.tables[table_name][id_name] = np.arange(next(iter(self.tables[table_name].values())).shape[0])
        for table_name, table in list(self.tables.items()):
            id_name = self.get_id_name(table_name)
            for col_name, col in table.items():
                other_table_name = self.get_table_of_id(col_name)
                mask_name = self.masks.get(table_name, {}).get(col_name, None)
                if other_table_name:
                    other_table = self.tables[other_table_name]
                    other_id_col = other_table.get(id_name, None)
                    if other_id_col is not None:
                        continue

                    if mask_name:
                        sorter = factorize_ids(flatten_ids(col, table[mask_name]), None, other_table[col_name])[0]
                        sorter = np.argsort(sorter)
                        if issparse(table[mask_name]):
                            indices = table[mask_name].nonzero()
                            other_table[id_name], other_table['idx'] = indices[0][sorter], indices[1][sorter]
                        elif isinstance(table[mask_name], np.ndarray):
                            for axis_name, idx in zip(
                                  [id_name, 'idx', (f"idx_{i+2}" for i in range(len(col.shape[2:])))], range(len(col.shape)),
                                  table[mask_name].nonzero()
                            ):
                                other_table[axis_name] = idx[sorter]
                        else:
                            raise Exception(f"Unrecognized mask format for {mask_name}: {table[mask_name].__class__}")
                    else:
                        sorter = factorize_ids(flatten_ids(col), None, other_table[col_name])[0]
                        sorter = np.argsort(sorter)
                        for axis_name, axis_i in zip([id_name, 'idx', (f"idx_{i+2}" for i in range(len(col.shape[2:])))], range(len(col.shape))):
                            other_table[axis_name] = (
                                np.tile(
                                    np.arange(col.shape[axis_i]).reshape(
                                        tuple(1 for _ in col.shape[:axis_i]) + (-1,) + tuple(1 for _ in col.shape[axis_i + 1:])
                                    ),
                                    tuple(n for n in col.shape[:axis_i]) + (1,) + tuple(n for n in col.shape[axis_i + 1:])).reshape(-1)
                            )[sorter]
        if not inplace:
            return self

    @classmethod
    def query_table(cls, table, ids):
        new_table = {}
        for col_name, col in list(table.items()):
            new_table[col_name] = col.iloc[ids] if hasattr(col, 'iloc') else col[ids]
        return new_table

    def get_id_name(self, name):
        """Maybe override this if the user needs it"""
        return f"{name}_id"

    def get_table_of_id(self, name):
        table_name = name[:-3] if name.endswith('_id') else None
        if table_name in self.tables:
            return table_name

    def set_main_table(self, name, inplace=False):
        if not inplace:
            return Batcher(
                self.tables,
                main_table=name,
                masks=self.masks,
                column_names=self.column_names,
            )
        else:
            self.main_table = name

    def densify(self, device=None, dtypes=None):
        new_tables = dict(self.tables)
        new_column_names = {}
        dtypes = dtypes or {}

        col_to_resize = {}
        masks_length = {}
        for table_name, table_masks in self.masks.items():
            for col_name, mask_name in table_masks.items():
                if mask_name not in masks_length:
                    if issparse(new_tables[table_name][mask_name]):
                        if hasattr(new_tables[table_name][mask_name], 'indices'):
                            if len(new_tables[table_name][mask_name].indices):
                                max_length = new_tables[table_name][mask_name].indices.max() + 1
                            else:
                                max_length = 0
                        elif hasattr(new_tables[table_name][mask_name], 'rows'):
                            max_length = max((max(r, default=-1) + 1 for r in new_tables[table_name][mask_name].rows), default=0)
                        else:
                            raise Exception(f"Unrecognized mask format for {mask_name}: {new_tables[table_name][mask_name].__class__}")
                        masks_length[mask_name] = max_length
                        new_tables[table_name][mask_name].resize(new_tables[table_name][mask_name].shape[0], masks_length[mask_name])
                    else:
                        max_length = new_tables[table_name][mask_name].sum(-1).max()
                        masks_length[mask_name] = max_length
                        new_tables[table_name][mask_name] = new_tables[table_name][mask_name][:, masks_length[mask_name]]
                if issparse(new_tables[table_name][col_name]):
                    new_tables[table_name][col_name].resize(new_tables[table_name][col_name].shape[0], masks_length[mask_name])
                else:
                    new_tables[table_name][col_name] = new_tables[table_name][col_name][:, masks_length[mask_name]]

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
                if device is not None:
                    col = torch.as_tensor(col, device=device, dtype=dtypes.get(table_name, {}).get(col_name, torch.long if col_name.endswith('_id') else None))
                new_table[col_name] = col
            new_tables[table_name] = new_table
        return Batcher(new_tables, main_table=self.main_table, masks=self.masks, column_names=new_column_names)

    def query_ids(self, ids, densify=False, **densify_kwargs):
        selected_ids = {self.get_id_name(self.main_table): ids}
        queried_tables = {}
        queue = {self.main_table}
        while len(queue):
            table_name = queue.pop()
            table = self.tables[table_name]
            queried_table = self.query_table(table, selected_ids[self.get_id_name(table_name)])
            for col_name, col in queried_table.items():
                new_table_name = self.get_table_of_id(col_name)
                # We don't want to reindex the token_id column in the token table: it's useless and we will
                # moreover need it intact for when we rebuild the original data
                if new_table_name and new_table_name != table_name:
                    mask_name = self.masks.get(table_name, {}).get(col_name, None)
                    new_col, new_mask, unique_ids = factorize_ids(
                        col=col,
                        mask=queried_table.get(mask_name, None),
                        prefered_unique_ids=selected_ids.get(col_name, None),
                    )
                    if mask_name is not None:
                        queried_table[mask_name] = new_mask
                    selected_ids[col_name] = unique_ids
                    queried_table[col_name] = new_col
                    if new_table_name not in queried_tables:
                        queue.add(new_table_name)
            queried_tables[table_name] = queried_table

        # for id_name in selected_ids:
        #     table_name = self.get_table_of_id(id_name)
        #     queried_tables[table_name][id_name] = np.asarray(selected_ids[id_name])
        new_tables = dict(self.tables)
        new_tables.update(queried_tables)

        res = Batcher(new_tables, main_table=self.main_table, masks=self.masks, column_names=self.column_names)
        if densify:
            res = res.densify(**densify_kwargs)
        return res

    def __getitem__(self, name):
        if isinstance(name, str):
            name = (name,)
        if isinstance(name, slice):
            name = np.arange(name.start or 0, name.stop, name.step or 1)
        if isinstance(name, tuple):
            if name[0] not in self.tables:
                name = (self.main_table, *name)
            current = self.tables[name[0]]
            if len(name) == 1:
                return self.set_main_table(name[0])
            if len(name) > 1:
                current = current[name[1]]
            if len(name) > 2:
                if isinstance(current, pd.DataFrame):
                    return current[name[2]]
                else:
                    current_column_names = self.column_names.get(name[0], {}).get(name[1], {})
                    if isinstance(name[2], str):
                        if name[2] in current_column_names:
                            current = current[:, current_column_names.index(name[2])]
                        else:
                            raise KeyError(name)
                    elif isinstance(name[2], int):
                        current = current[:, name[2]]
                    elif isinstance(name[2], (list, tuple)):
                        parts_idx = []
                        for part in name[2]:
                            if part in current_column_names:
                                parts_idx.append(current_column_names.index(part))
                            else:
                                raise KeyError((name[0], name[1], part))
                        current = current[:, parts_idx]
                    else:
                        raise KeyError(name)
            return current
        elif isinstance(name, (list, np.ndarray)):
            return self.query_ids(name)
        else:
            if hasattr(name, 'toarray'):
                name = name.toarray()
            else:
                name = np.asarray(name)
            if name.dtype == np.bool:
                return self.query_ids(np.flatnonzero(name))
            else:
                return self.query_ids(name)

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
        return DataLoader(range(len(self)),  # if self._idx is None else self._idx,
                          collate_fn=lambda ids: self.query_ids(ids).densify(device, dtypes),
                          batch_sampler=batch_sampler,
                          **kwargs)

    @classmethod
    def concat(cls, batches):
        struct = batches[0]
        batches = [b.tables for b in batches]
        res = {}

        for table_name, table0 in struct.tables.items():
            res[table_name] = {}
            for col_name, col0 in table0.items():
                if col_name in res[table_name]:
                    continue
                mask_name = struct.masks.get(table_name, {}).get(col_name, None)

                # Convert to sparse if needed
                if issparse(col0):
                    shape1 = max([dic[table_name][col_name].shape[1] for dic in batches])
                    for dic in batches:
                        dic[table_name][col_name].resize(dic[table_name][col_name].shape[0], shape1)
                    res[table_name][col_name] = [dic[table_name][col_name].tocsr() for dic in batches]
                elif mask_name is not None:
                    res[table_name][col_name] = [csr_matrix(
                        (dic[table_name][col_name][dic[table_name][mask_name]], dic[table_name][mask_name].nonzero()),
                        shape=dic[table_name][mask_name].shape,
                    ) for dic in batches]
                    shape1 = max([sps.shape[1] for sps in res[table_name][col_name]])
                    for sps in res[table_name][col_name]:
                        sps.resize(sps.shape[0], shape1)
                # If col_name IS a mask
                elif col_name in struct.masks.get(table_name, {}).values():
                    res[table_name][col_name] = [csr_matrix(
                        (np.ones(dic[table_name][col_name].sum(), dtype=bool), dic[table_name][col_name].nonzero()),
                        shape=dic[table_name][col_name].shape,
                    ) for dic in batches]
                    shape1 = max([sps.shape[1] for sps in res[table_name][col_name]])
                    for sps in res[table_name][col_name]:
                        sps.resize(sps.shape[0], shape1)
                else:
                    res[table_name][col_name] = [dic[table_name][col_name] for dic in batches]
                # Check if we need to align ids between two or more matrices
                other_table_name = struct.get_table_of_id(col_name)
                if other_table_name and other_table_name != table_name:
                    # If fragment_id doesn't exist in fragment table, (for ex: new fragments), create it
                    if col_name not in struct.tables[other_table_name]:
                        i = 0
                        for dic in batches:
                            dic_size = next(iter(dic[other_table_name].values())).shape[0]
                            dic[other_table_name][col_name] = np.arange(i, i + dic_size)
                            i += dic_size
                    # print(table_name, col_name, "lookup absolute ids: ", other_table_name, col_name)
                    if issparse(res[table_name][col_name][0]):
                        for sps, dic in zip(res[table_name][col_name], batches):
                            # print("rel", sps.data, "abs", dic[other_table_name][col_name])
                            sps.data = dic[other_table_name][col_name][sps.data]
                    else:
                        res[table_name][col_name] = [
                            dic[other_table_name][col_name][x]
                            for x, dic in zip(res[table_name][col_name], batches)]

                # Finally concatenate the matrices, sparse or dense
                if issparse(res[table_name][col_name][0]):
                    res[table_name][col_name] = vstack(res[table_name][col_name]).tolil()
                else:
                    res[table_name][col_name] = np.concatenate(res[table_name][col_name])

        res = Batcher(res, main_table=struct.main_table, masks=struct.masks, column_names=struct.column_names)
        for table_name, table in list(res.tables.items()):
            id_name = res.get_id_name(table_name)
            if id_name in table:
                sorter = np.unique(table[id_name], return_index=True)[1]  # sort and unifies
                res.tables[table_name] = res.query_table(table, sorter)
        return res

    def __repr__(self):
        # return "Batcher(\n  {}\n)".format("\n  ".join(f"({k}): {v.dtype}{v.shape}" for k, v in self.arrays.items() if v is not None))
        return BatcherPrinter(indent=2, depth=2).pformat(self)


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
        dtype_str = (str(obj.dtype) if hasattr(obj, 'dtype') else str(obj.dtypes.values[0]) if len(set(obj.dtypes.values)) == 1 else 'multiple')
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
                (self.batcher[self.on].getnnz(1) + np.random.poisson(1, size=length))[init_permut])
            for i in np.random.permutation(len(block_begins)):
                yield init_permut[sorter[block_begins[i]:block_ends[i]]]
        else:
            sorter = np.argsort(self.batcher[self.on].getnnz(1))
            for i in range(len(block_begins)):
                yield sorter[block_begins[i]:block_ends[i]]

    def __len__(self):
        if self.drop_last:
            return len(self.batcher) // self.batch_size
        else:
            return (len(self.batcher) + self.batch_size - 1) // self.batch_size


def flatten_ids(col, mask=None):
    if issparse(col):
        col = col.tocsr()
        col_bis = col.copy()
        col_bis.data = np.ones(len(col.data), dtype=bool)
        if mask is not None and issparse(mask):
            res = col[mask]
            # If empty mask, scipy returns a sparse matrix: we use toarray to densify
            if hasattr(res, 'toarray'):
                return res.toarray().reshape(-1)
            # else, scipy returns a 2dmatrix, we use asarray to densify
            return np.asarray(res).reshape(-1)
        col = col.toarray()
        if mask is not None:
            mask = densify(mask)
    if isinstance(col, np.ndarray):
        if mask is not None:
            return col[mask]
        else:
            return col.reshape(-1)
    else:
        raise Exception()


def factorize_ids(col, mask, prefered_unique_ids=None):
    if prefered_unique_ids is None:
        prefered_unique_ids = np.zeros(0, dtype=col.dtype)
    if mask is not None:
        mask = mask.tocsr()
    flat_ids = flatten_ids(col, mask)
    relative_ids, unique_ids = pd.factorize(np.concatenate((prefered_unique_ids, flat_ids)))
    unique_ids = prefered_unique_ids if len(prefered_unique_ids) else unique_ids
    relative_ids = relative_ids[len(prefered_unique_ids):]
    if mask is None and (relative_ids > len(unique_ids)).any():
        raise Exception("Cannot handle out of vocabulary labels/ids for array if not given a mask")
    if issparse(col):
        col = col.tocsr()
        if mask is not None:
            mask = mask.copy()
            col = mask.copy()
            col.data = relative_ids
            mask.data = relative_ids < len(unique_ids)
            mask.eliminate_zeros()
            new_col = mask.copy()
            new_col.data = flatten_ids(col, mask)
            return new_col.tolil(), mask.tolil(), unique_ids
        else:
            col.data = relative_ids
            return col.tolil(), None, unique_ids
    if isinstance(col, np.ndarray):
        if mask is not None:
            col = col.copy()
            col[mask] = relative_ids
            unique_ids = prefered_unique_ids if len(prefered_unique_ids) else unique_ids
            mask = mask & (col < len(unique_ids))
            col[~mask] = 0
            return col, mask, unique_ids
        else:
            col = relative_ids.reshape(col.shape)
            unique_ids = prefered_unique_ids if len(prefered_unique_ids) else unique_ids
            return col, None, unique_ids
    else:
        raise Exception(f"Data format {col.__class__} was not recognized during factorization")


def densify(array):
    if isinstance(array, np.ndarray):
        return array
    elif hasattr(array, 'toarray'):
        return array.toarray()
    else:
        return np.asarray(array)
