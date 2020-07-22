import pprint
from collections import defaultdict
from math import ceil

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse, csr_matrix
from torch.utils.data import DataLoader, BatchSampler

from nlstruct.utils.arrays import as_numpy_array, get_deduplicator, concat, factorize, as_array, as_same, index_slice


class BatcherPrinter(pprint.PrettyPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dispatch[list.__repr__] = BatcherPrinter.format_array
        self._dispatch[tuple.__repr__] = BatcherPrinter.format_array

    def format_batcher(self, obj, stream, indent, allowance, context, level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write("Batcher(")
        object_dict = obj.tables
        length = len(object_dict)
        items = [*((name, obj.tables[name], True) for name in obj.join_order),
                 *((name, table, False) for name, table in object_dict.items() if name not in obj.join_order)]
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            self.format_table(items, stream, indent, allowance + 1, context, level, inline=False)
        write('\n' + ' ' * indent + ')')

    def format_array(self, obj, stream, indent, allowance, context, level):
        dtype_str = (
              ("ndarray" if isinstance(obj, np.ndarray) else "tensor" if torch.is_tensor(obj) else str(obj.__class__.__name__)) +
              "[{}]".format(str(obj.dtype) if hasattr(obj, 'dtype') else str(obj.dtypes.values[0]) if hasattr(obj, 'dtypes') and len(set(obj.dtypes.values)) == 1 else 'any')
        )
        stream.write(dtype_str + str(tuple(obj.shape) if hasattr(obj, 'shape') else (len(obj),)))

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

    def format_table(self, table, stream, indent, allowance, context, level, inline=False):
        # Code almost equal to _format_dict_table, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(table) - 1
        if inline:
            delimnl = ' '
        else:
            delimnl = '\n' + ' ' * indent
            write('\n' + ' ' * indent)

        for i, (key, ent, joinable) in enumerate(table):
            last = i == last_index
            write("[{}]".format(key) + ':' + (' frozen' if not joinable else ''))
            self.format_columns(ent.data.items(), stream, indent,  # + len(key) + 4,
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


class SortedBatchSampler(BatchSampler):
    def __init__(self,
                 batcher,
                 keys_name,
                 sort_keys="ascending",
                 batch_size=None,
                 shuffle=False,
                 drop_last=False,
                 keys_noise=1.):
        assert not drop_last
        if isinstance(keys_name, str):
            keys_name = [keys_name]
        keys = []
        for key_name in keys_name:
            key = batcher[key_name]
            if hasattr(key, 'getnnz'):
                key = key.getnnz(1)
            elif hasattr(key, 'shape') and len(key.shape) > 1:
                key = key.reshape(key.shape[0], -1).sum(1)
            keys.append(key)
        self.keys = np.stack(keys, axis=1)
        self.sort_keys = sort_keys
        self.shuffle = shuffle
        if not shuffle:
            self.keys_noise = np.asarray([0 for name in keys_name])
        elif isinstance(keys_noise, (int, float)):
            self.keys_noise = np.asarray([keys_noise for name in keys_name])
        elif isinstance(keys_noise, dict):
            self.keys_noise = np.asarray([keys_noise.get(name, 0) for name in keys_name])
        else:
            self.keys_noise = np.asarray(keys_noise)
        self.length = len(batcher)
        self.compute_blocks(batch_size)

    def compute_blocks(self, batch_size):
        self.block_begins = np.arange(ceil(len(self.keys) / batch_size)) * batch_size
        self.block_ends = np.roll(self.block_begins, -1)
        self.block_ends[-1] = self.block_begins[-1] + batch_size

    def __iter__(self):
        init_permut = None
        sorter = None

        if self.shuffle:
            init_permut = np.random.permutation(self.length)
            if self.sort_keys:
                sorter = np.lexsort((self.keys[init_permut] + np.random.poisson(self.keys_noise, size=self.keys.shape)).T, axis=0)
                if self.sort_keys == "descending":
                    sorter = np.flip(sorter)
        else:
            if self.sort_keys:
                sorter = np.lexsort(self.keys.T, axis=0)
                if self.sort_keys == "descending":
                    sorter = np.flip(sorter)

        if self.shuffle:
            perm = np.random.permutation(len(self.block_begins))
            block_begins = self.block_begins[perm]
            block_ends = self.block_ends[perm]
        else:
            block_begins = self.block_begins
            block_ends = self.block_ends

        for i in range(len(block_begins)):
            if init_permut is not None:
                if sorter is not None:
                    yield init_permut[sorter[block_begins[i]:block_ends[i]]]
                else:
                    yield init_permut[block_begins[i]:block_ends[i]]
            else:
                if sorter is not None:
                    yield sorter[block_begins[i]:block_ends[i]]
                else:
                    yield np.arange(block_begins[i], block_ends[i])

    def __len__(self):
        return len(self.block_begins)


class Table:
    def __init__(self, data, primary_id=None, masks=None, subcolumn_names=None, foreign_ids=None, check=True, batcher=None):
        """

        Parameters
        ----------
        data: dict of (torch.Tensor or numpy.ndarray or scipy.sparse.spmatrix)
        primary_id: str
        masks: dict[str, str]
        subcolumn_names: dict[str, list of str]
        foreign_ids: dict[str, Table]
        check: bool
        """
        self.data = data
        self.primary_id = primary_id
        self.masks = masks or {}
        self.subcolumn_names = subcolumn_names or {}
        self.foreign_ids = foreign_ids or {}
        self.batcher = batcher
        for col_name in self.foreign_ids:
            mask_name = self.masks.get(col_name, None)
            if mask_name is not None:
                self.masks.setdefault('@' + col_name, mask_name)

    def __iter__(self):
        return iter(self.data.values())

    def __len__(self):
        return next(iter(self)).shape[0]

    @property
    def device(self):
        return getattr(next(iter(self)), 'device', None)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def copy(self):
        table = Table(
            data=dict(self.data),
            masks=dict(self.masks),
            subcolumn_names=dict(self.subcolumn_names),
            foreign_ids=dict(self.foreign_ids),
            primary_id=self.primary_id,
            batcher=self.batcher,
            check=False)
        return table

    @property
    def primary_ids(self):
        """

        Returns
        -------
        torch.Tensor or numpy.ndarray
        """
        return self.data[self.primary_id]

    def compute_foreign_relative_(self, name):
        if '@' + name in self.data:
            return self.data['@' + name]
        if name in self.foreign_ids:
            referenced_table = self.batcher.tables[self.foreign_ids[name]]
            mask_name = self.masks.get(name, None)
            relative_ids, new_mask = factorize(
                values=self.data[name],
                mask=self.data.get(mask_name, None),
                reference_values=referenced_table.primary_ids,
                freeze_reference=True,
            )[:2]

            # assert (new_mask is None or new_mask.sum() == np.prod(new_mask.shape)) or not (mask_name is None), \
            #    f"Unkown ids were found in {name} and no existing mask could be found to mask these values"
            self.data['@' + name] = relative_ids
            if new_mask is not None and mask_name is not None:
                self.data['@' + mask_name] = new_mask
            elif new_mask is not None:
                relative_ids[~new_mask] = -1
            return relative_ids
        elif name in self.masks:
            mask_name = name
            foreign_ids = [key for key, key_mask in self.foreign_ids.items() if key_mask == mask_name]
            assert len(foreign_ids) > 0
            mask = self.data.get(mask_name, None)
            for name in foreign_ids:
                referenced_table = self.batcher.tables[self.foreign_ids[name]]
                relative_ids, mask = factorize(
                    values=self.data[name],
                    mask=mask,
                    reference_values=referenced_table.primary_ids,
                    freeze_reference=True,
                )[:2]
                self.data['@' + name] = relative_ids
                assert (mask is None or mask.sum() == np.prod(mask.shape)) or not (mask_name is None), \
                    f"Unkown ids were found in {name} and no existing mask could be found to mask these values"
            if mask is not None and mask_name is not None:
                self.data['@' + mask_name] = mask
            return mask
        else:
            raise Exception("No way to compute @{} could be found".format(name))

    def compute_foreign_absolute_(self, name):
        if name in self.data:
            return self.data[name]
        referenced_table = self.batcher.tables[self.foreign_ids[name]]
        relative_name = '@' + name

        if issparse(self.data[relative_name]):
            array_to_change = self.data[name] = self.data[relative_name].tocsr(copy=True)
            array_to_change.data = as_numpy_array(referenced_table.primary_ids)[array_to_change.data]
        else:
            self.data[name] = as_same(
                referenced_table.primary_ids,
                t=type(self.data[relative_name]),
                device=getattr(self.data[relative_name], 'device', None))[self.data[relative_name]]
        return self.data[name]

    def get_col(self, name):
        if name in self.data:
            return self.data[name]
        else:
            if name.startswith('@'):
                return self.compute_foreign_relative_(name[1:])
            else:
                return self.compute_foreign_absolute_(name)

    def __getitem__(self, key):
        # table["mention_id"]
        if key is None:
            return None
        if isinstance(key, str):
            return self.get_col(key)
        # table["features", ["feat0", "feat1"]]
        elif isinstance(key, tuple):
            (top, *rest) = key
            assert isinstance(top, str)
            data = self.get_col(top)
            if len(key) == 1:
                return data
            if isinstance(rest, str):
                return data[self.subcolumn_names[key].index(rest)]
            elif hasattr(rest, "__iter__"):
                return data[[self.subcolumn_names[key].index(name) for name in rest]]
            else:
                raise Exception()
        # table[["mention_id", ("features", ["feat0", "feat1"]), "label"]]
        elif isinstance(key, list):
            if len(key) > 0 and isinstance(key[0], str):
                new_self = self.copy()
                new_self.data = {
                    col_name: new_self[col_name]
                    for col_name in key
                }
                new_self.foreign_ids = {
                    foreign_id: reference_table
                    for foreign_id, reference_table in self.foreign_ids.items()
                    if foreign_id in key or '@' + foreign_id in key
                }
                new_self.masks = {
                    col_name: mask_name
                    for col_name, mask_name in self.masks.items()
                    if col_name in key
                }
                new_self.subcolumn_names = {
                    col_name: subcolumn_names
                    for col_name, subcolumn_names in self.subcolumn_names.items()
                    if col_name in key
                }
                return new_self
        new_self = self.copy()
        new_self.data = {
            col_name: index_slice(mat, key)
            for col_name, mat in new_self.data.items()
        }
        return new_self

    def __setitem__(self, key, value):
        # TODO checks
        # table["mention_id"] = ...
        if isinstance(key, str):
            self.data[key] = value
        # table["features", ["feat0", "feat1"]] = ...
        elif isinstance(key, tuple):
            if len(key) == 1:
                self.data[key[0]] = value
            else:
                top, rest = key
                assert isinstance(top, str)
                data = self.data[top]
                if isinstance(rest, str):
                    data[self.subcolumn_names[key].index(rest)] = value
                elif hasattr(rest, "__iter__"):
                    data[[self.subcolumn_names[key].index(name) for name in rest]] = value
                else:
                    raise Exception()
        # table[["mention_id", ("features", ["feat0", "feat1"]), "label"]] = ...
        elif isinstance(key, list):
            for part in zip(key, value):
                self[part] = value
        else:
            raise Exception()

    def __delitem__(self, key):
        # TODO checks
        # del table["mention_id"]
        if isinstance(key, str):
            if key.startswith('@'):
                del self.data[key]
                self.masks = {col_name: mask_name for col_name, mask_name in self.masks.items() if mask_name != key and col_name != key}
            else:
                del self.data[key]
                if '@' + key in self.data:
                    del self.data['@' + key]
                if key in self.foreign_ids:
                    del self.foreign_ids[key]
                self.masks = {col_name: mask_name for col_name, mask_name in self.masks.items() if mask_name not in (key, '@' + key) and col_name not in (key, '@' + key)}
        # del table[["mention_id", "features", "label"]]
        elif isinstance(key, tuple):
            assert len(key) == 1
            del self.data[key[0]]
        elif isinstance(key, list):
            for part in key:
                del self[part]
        else:
            raise Exception()

    def prune_(self):

        masks_length = {}

        for col_name, mask_name in self.masks.items():
            if col_name not in self.data:
                continue
            if mask_name not in masks_length:
                mask = self.data[mask_name]
                if issparse(mask):
                    if hasattr(mask, 'indices'):
                        if len(mask.indices):
                            max_length = mask.indices.max() + 1
                        else:
                            max_length = 0
                    elif hasattr(mask, 'rows'):
                        max_length = max((max(r, default=-1) + 1 for r in mask.rows), default=0)
                    else:
                        raise Exception(f"Unrecognized mask format for {mask_name}: {mask.__class__}")
                    masks_length[mask_name] = max_length
                    mask.resize(mask.shape[0], masks_length[mask_name])
                else:
                    if 0 in mask.shape:
                        max_length = 0
                    else:
                        max_length = mask.sum(-1).max()
                    masks_length[mask_name] = max_length
                    self.data[mask_name] = mask[:, :masks_length[mask_name]]
            col = self.data[col_name]
            if issparse(col):
                col.resize(col.shape[0], masks_length[mask_name])
            else:
                self.data[col_name] = col[:, :masks_length[mask_name]]

    def densify_(self, device=None, dtypes=None):
        dtypes = dtypes or {}

        new_data = {}
        for col_name, col in self.data.items():
            torch_dtype = dtypes.get(col_name, torch.long if (device is not None and not torch.is_tensor(col) and np.issubdtype(col.dtype, np.integer)) else None)
            col = as_array(col,
                           t=torch.Tensor if (device is not None or col_name in dtypes) else np.ndarray,
                           device=device,
                           dtype=torch_dtype)
            new_data[col_name] = col
        self.data = new_data

    def sparsify_(self, device=None):
        new_data = {}
        densified_masks = {}
        sparsified_masks = {}
        for col_name, mask_name in self.masks.items():
            mask = self[mask_name]
            col = self[col_name]
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
            new_data.setdefault(mask_name, mask.tolil())
            new_data.setdefault(col_name, col.tolil())

        # Then convert the other columns as numpy arrays
        for col_name, col in self.data.items():
            if col_name not in new_data:
                col = as_numpy_array(col)
                new_data[col_name] = col
        self.data = new_data

    def densify(self, device=None, dtypes=None):
        new_self = self.copy()
        new_self.densify_(device, dtypes)
        return new_self

    def sparsify(self):
        new_self = self.copy()
        new_self.sparsify_()
        return new_self

    @property
    def non_relative_data(self):
        keys = list(dict.fromkeys(key[1:] if key.startswith('@') else key for key in self.data.keys()))
        return {name: self[name] for name in keys}

    def fill_absolute_data_(self):
        for name in self.foreign_ids:
            self.compute_foreign_absolute_(name)

    def fill_absolute_data(self):
        new_self = self.copy()
        new_self.fill_absolute_data_()
        return new_self

    @classmethod
    def concat(cls, tables, sparsify=True):
        data = defaultdict(lambda: [])
        for table in tables:
            if sparsify:
                table = table.sparsify()
            for name, col in table.non_relative_data.items():
                data[name].append(col)
        new_data = {name: concat(cols) for name, cols in data.items()}
        new_table = tables[0].copy()
        new_table.data = new_data
        new_table.batcher = None
        return new_table

    def drop_relative_data_(self):
        for name in self.foreign_ids:
            if '@' + name in self.data:
                del self.data['@' + name]

    def __getstate__(self):
        new_self = self.copy()
        for name in self.foreign_ids:
            new_self.compute_foreign_absolute_(name)
        new_self.drop_relative_data_()
        return new_self.__dict__


class QueryFunction:
    def __init__(self, batcher, **kwargs):
        self.batcher = batcher
        self.kwargs = kwargs

    def __call__(self, ids):
        return self.batcher.query_ids(ids, **self.kwargs)


class Batcher:
    def set_join_order_(self, partial_order=(), complete=False):
        if not complete:
            self.join_order = list(partial_order)
            return
        queried_tables = set(partial_order)
        self.join_order = []
        queue = list(partial_order)
        for name in queue:
            self.join_order.append(name)
            for table_id, referenced_table in self.tables[name].foreign_ids.items():
                if referenced_table not in queue and referenced_table not in self.join_order:
                    queue.append(referenced_table)

    def set_join_order(self, partial_order=(), complete=False):
        self = self.copy()
        self.set_join_order_(partial_order=partial_order, complete=complete)
        return self

    def __init__(self, tables, join_order=None, masks=None, subcolumn_names=None, foreign_ids=None, primary_ids=None, check=True):
        """

        Parameters
        ----------

        tables: dict[str, dict]
        """
        if isinstance(next(iter(tables.values())), Table):
            self.tables = tables
            for table in tables.values():
                table.batcher = self
            if join_order is not None:
                self.join_order = join_order
            else:
                self.set_join_order_([next(iter(tables.keys()))], complete=True)
            return
        subcolumn_names = {table_name: dict(cols) for table_name, cols in subcolumn_names.items()} if subcolumn_names is not None else {}
        if check:
            tables = {table_name: dict(cols) for table_name, cols in tables.items()}
            for table_name, table in tables.items():
                for col_name, col in table.items():
                    if isinstance(col, pd.DataFrame):
                        tables[table_name][col_name] = col.values
                        subcolumn_names.setdefault(table_name, {}).setdefault(col_name, list(col.columns))
                    elif isinstance(col, pd.Series):
                        tables[table_name][col_name] = col.values
                    elif isinstance(col, pd.Categorical):
                        tables[table_name][col_name] = col.codes
                    else:
                        assert col is not None, f"Column {repr(table_name)}{repr(col_name)} cannot be None"
                        tables[table_name][col_name] = col
        if primary_ids is not None:
            primary_ids = primary_ids
        else:
            primary_ids = {table_name: f"{table_name}_id" for table_name in tables if f"{table_name}_id" in tables[table_name]}
            if check:
                for table_name, primary_id in primary_ids.items():
                    uniques, counts = np.unique(tables[table_name][primary_id], return_counts=True)
                    duplicated = uniques[counts > 1]
                    assert len(duplicated) == 0, f"Primary id {repr(primary_id)} of {repr(table_name)} has {len(duplicated)} duplicate{'s' if len(duplicated) > 0 else ''}, " \
                                                 f"when it should be unique: {repr(list(duplicated[:5]))} (first 5 shown)"
        if foreign_ids is None:
            foreign_ids = {}
            for table_name, table_columns in tables.items():
                # ex: table_name = "mention", table_columns = ["sample_id", "begin", "end", "idx_in_sample"]
                for col_name in table_columns:
                    col_name = col_name.strip('@')
                    if col_name.endswith('_id') and col_name != primary_ids.get(table_name, None):
                        prefix = col_name[:-3]
                        foreign_table_name = next((table_name for table_name in tables if prefix.endswith(table_name)), None)
                        # foreign_table_id = f"{table_name}_id"
                        if foreign_table_name is not None:
                            foreign_ids.setdefault(table_name, {})[col_name] = foreign_table_name
        if masks is None:
            masks = {}
            for table_name, table_columns in tables.items():
                # ex: table_name = "mention", table_columns = ["sample_id", "begin", "end", "idx_in_sample"]
                for col_name in table_columns:
                    if col_name.endswith('_mask') and col_name != primary_ids.get(table_name, None):
                        id_name = col_name[:-5] + '_id'
                        # foreign_table_id = f"{table_name}_id"
                        if id_name in table_columns or '@' + id_name in table_columns:
                            masks.setdefault(table_name, {})[id_name] = col_name
        if check:
            # Check that all tables / columns exist in subcolumn names
            for table_name, cols in subcolumn_names.items():
                assert table_name in tables, f"Unknown table {repr(table_name)} in `subcolumn_names`"
                for col_name in cols:
                    assert col_name in tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `subcolumn_names`"
            # Check that all tables / columns exist in masks
            for table_name, cols in masks.items():
                assert table_name in tables, f"Unknown table {repr(table_name)} in `masks`"
                for col_name, mask_name in cols.items():
                    assert col_name in tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `masks`"
                    assert mask_name in tables[table_name], f"Unknown mask {repr(mask_name)} for column {table_name}/{col_name} in `masks`"
            # Check that all tables / columns exist in foreign_ids
            for table_name, cols in foreign_ids.items():
                assert table_name in tables, f"Unknown table {repr(table_name)} in `foreign_ids`"
                for col_name, foreign_table_name in cols.items():
                    assert col_name in tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `foreign_ids`"
                    assert foreign_table_name in tables, f"Unknown foreign table {repr(foreign_table_name)} for column {table_name}/{col_name} in `foreign_ids`"
            # Check that all tables / columns exist in primary_ids
            for table_name, col_name in primary_ids.items():
                assert table_name in tables, f"Unknown table {repr(table_name)} in `primary_ids`"
                assert col_name in tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `primary_ids`"
        self.tables = {
            key: Table(table_data,
                       primary_id=primary_ids.get(key, None),
                       masks=masks.get(key, None),
                       subcolumn_names=subcolumn_names.get(key, None),
                       foreign_ids=foreign_ids.get(key, None),
                       batcher=self)
            for key, table_data in tables.items()
        }  # type: dict[str, Table]
        if join_order is None:
            join_order = next(iter(tables.keys()))
        if isinstance(join_order, str):
            self.set_join_order_([join_order], complete=True)
        else:
            assert isinstance(join_order, (tuple, list))
            self.join_order = join_order

    @property
    def primary_ids(self):
        return self.tables[self.join_order[0]].primary_ids

    @property
    def device(self):
        return getattr(next(iter(self.tables[self.join_order[0]].values())), 'device', None)

    def __len__(self):
        return len(self.tables[self.join_order[0]])

    def keys(self):
        return self.tables[self.join_order[0]].keys()

    def values(self):
        return self.tables[self.join_order[0]].values()

    def items(self):
        return self.tables[self.join_order[0]].items()

    def __iter__(self):
        return iter(self.values())

    def __getitem__(self, key):
        table = None
        if isinstance(key, str):
            key = (key,)
        if isinstance(key, tuple):
            if key[0] not in self.tables:
                key = (self.join_order[0], *key)
            if len(key) == 1:
                self = self.copy()
                self.set_join_order_(key, complete=True)
                return self
            elif isinstance(key[1], str):
                if len(key) == 2:
                    return self.tables[key[0]][key[1]]
                return self.tables[key[0]][key[1:]]
            elif isinstance(key[1], list) and isinstance(key[1][0], str):
                assert len(key) == 2
                return Batcher({key[0]: self.tables[key[0]][key[1]]}, join_order=[key[0]])
            else:
                assert len(key) == 2
                table, indexer = key
        elif isinstance(key, list):
            if isinstance(key[0], str):
                assert set(type(k) for k in key) == {str}
                return self.slice_tables(key)
            else:
                indexer = key
        else:
            indexer = key

        if isinstance(indexer, slice):
            device = self.device
            if device is None:
                indexer = np.arange(indexer.start or 0, indexer.stop, indexer.step or 1)
            else:
                indexer = torch.arange(indexer.start or 0, indexer.stop, indexer.step or 1, device=device)
        else:
            dtype = getattr(indexer, 'dtype', None)
            if dtype is torch.bool:
                indexer = torch.nonzero(indexer, as_tuple=True)[0]
            elif dtype == np.bool:
                indexer = np.nonzero(indexer)[0]
        if table is not None:
            self = self[table]
        return self.query_ids(indexer)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if isinstance(value, Batcher) and len(value.tables.keys()) == 1:
                self.tables[key] = next(iter(value.tables.values()))
                self.tables[key].batcher = self
            elif isinstance(value, Table):
                self.tables[key] = value
                self.tables[key].batcher = self
            else:
                raise Exception()
        elif isinstance(key, tuple):
            if len(key) == 2:
                self.tables[key[0]][key[1]] = value
            self.tables[key[0]][key[1:]] = value
        elif isinstance(key, list):
            if isinstance(value, Batcher):
                assert len(value.tables) == len(key)
                for name, table in zip(key, value.tables.values()):
                    self.tables[name] = table.copy()
                    self.tables[name].batcher = self
            elif not len(key) or isinstance(key[0], (str, tuple, list)):
                for part, val in zip(key, value):
                    self[part] = val
        else:
            raise Exception()

    def __delitem__(self, key):
        if isinstance(key, str):
            self.slice_tables_([name for name in self.tables if name != key])
            return
        elif isinstance(key, tuple):
            if isinstance(key[1], (str, list)):
                assert len(key) == 2
                del self.tables[key[0]][key[1]]
                return
        elif isinstance(key, list):
            if isinstance(key[0], str):
                assert set(type(k) for k in key) == {str}
                self.slice_tables_([name for name in self.tables if name not in key])
                return
        raise Exception()

    def copy(self):
        return Batcher({key: table.copy() for key, table in self.tables.items()}, join_order=self.join_order)

    def query_ids(self, ids, inplace=False, **densify_kwargs):
        if not inplace:
            self = self.copy()
        selected_ids = {self.join_order[0]: ids}
        queried_tables = set()
        for table_name in self.join_order:
            # Ex: table_name = relations
            table = self.tables[table_name][selected_ids[table_name]]
            table.prune_()
            self.tables[table_name] = table
            queried_tables.add(table_name)
            for foreign_id, reference_table in table.foreign_ids.items():
                # Ex: col_name = from_mention_id
                #     foreign_table_name = mention
                #     foreign_table_id = mention_id

                # We don't want to reindex the token_id column in the token table: it's useless and we will
                # moreover need it intact for when we rebuild the original data
                mask_name = table.masks.get(foreign_id, None)
                table_iloc = selected_ids.get(reference_table, None)
                if reference_table in queried_tables:
                    relative_ids, new_mask, unique_ids = factorize(
                        values=table[foreign_id],
                        mask=table[mask_name],
                        reference_values=self.tables[reference_table].primary_ids,
                        freeze_reference=True,
                    )
                else:
                    relative_ids, new_mask, selected_ids[reference_table] = factorize(
                        values=table['@' + foreign_id],
                        mask=table[mask_name] if mask_name is not None else (table['@' + foreign_id] != -1),
                        reference_values=selected_ids.get(reference_table, None),
                        freeze_reference=False,
                    )
                table['@' + foreign_id] = relative_ids
                if mask_name is not None and new_mask is not None:
                    table['@' + mask_name] = new_mask
                elif mask_name is None and new_mask is not None:
                    relative_ids[~new_mask] = -1

        if len(densify_kwargs):
            self.densify_(**densify_kwargs)
        return self

    def __repr__(self):
        return BatcherPrinter(indent=2, depth=2).pformat(self)

    def densify_(self, device, dtypes=None):
        dtypes = dtypes or {}
        for table in self.tables.values():
            table.densify_(device, dtypes)

    def sparsify_(self):
        for table in self.tables.values():
            table.sparsify_()

    def densify(self, device=None, dtypes=None):
        new_self = self.copy()
        new_self.densify_(device, dtypes)
        return new_self

    def sparsify(self):
        new_self = self.copy()
        new_self.sparsify_()
        return new_self

    def slice_tables_(self, names):
        self.join_order = names
        for name in names:
            table = self.tables[name]
            new_foreign_ids = {}
            for foreign_id, referenced_table_name in table.foreign_ids.items():
                if referenced_table_name not in names:
                    table.compute_foreign_absolute_(foreign_id)
                    if '@' + foreign_id in table.keys():
                        del table['@' + foreign_id]
                else:
                    new_foreign_ids[foreign_id] = referenced_table_name
            table.foreign_ids = new_foreign_ids
        for name in list(self.tables.keys()):
            if name not in names:
                del self.tables[name]
        self.tables = {key: self.tables[key] for key in names}

    def slice_tables(self, names):
        new_self = self.copy()
        new_self.slice_tables_(names)
        return new_self

    @classmethod
    def concat(cls, batches, sparsify=True):
        tables = defaultdict(lambda: [])
        for batch in batches:
            for key, table in batch.tables.items():
                tables[key].append(table)
        new_tables = {key: Table.concat(tables, sparsify=sparsify) for key, tables in tables.items()}
        new_batcher = batches[0].copy()
        new_batcher.tables = new_tables
        for table in new_tables.values():
            table.batcher = new_batcher
        return new_batcher

    def drop_duplicates(self, names=None):
        if names is None:
            names = list(self.tables)
        elif isinstance(names, str):
            names = [names]
        self = self.copy()
        for name, table in self.tables.items():
            table.fill_absolute_data_()
            table.drop_relative_data_()
        for name in names:
            table = self.tables[name].fill_absolute_data()
            index = get_deduplicator(table.primary_ids)
            self.tables[name] = table[index]
        return self

    def merge(self, other):
        names = list(set(self.tables.keys()) | set(other.tables.keys()))
        self = self.drop_duplicates(names)
        other = other.drop_duplicates(names)
        self_other_ids, mask = factorize(self.primary_ids, reference_values=other.primary_ids)[:2]
        skip_in_self = not mask.all()
        if skip_in_self:
            self = self[mask]
        other = other[self_other_ids[mask] if skip_in_self else self_other_ids]
        for name in names:
            self.tables[name].data.update({k: v for k, v in other.tables[name].data.items() if k not in self.tables[name].data})
            self.tables[name].masks.update({k: v for k, v in other.tables[name].masks.items() if k not in self.tables[name].masks})
            self.tables[name].subcolumn_names.update({k: v for k, v in other.tables[name].subcolumn_names.items() if k not in self.tables[name].subcolumn_names})
            self.tables[name].foreign_ids.update({k: v for k, v in other.tables[name].foreign_ids.items() if k not in self.tables[name].foreign_ids})
        for name in other.tables:
            if name not in names:
                self.tables[name] = other.tables[name]
                self.tables[name].batcher = self
        return self

    def dataloader(self,
                   batch_size=32,
                   sort_on=None,
                   shuffle=False,
                   device=None,
                   dtypes=None,
                   **kwargs):
        batch_sampler = kwargs.pop("batch_sampler", None)
        sparse_sort_on = kwargs.pop("sparse_sort_on", None)
        assert sort_on is None or sparse_sort_on is None
        sort_on = sparse_sort_on or sort_on
        if sort_on is not None:
            sort_keys = kwargs.pop("sort_keys", "ascending")
            keys_noise = kwargs.pop("keys_noise", 1.)
            batch_sampler = SortedBatchSampler(self, keys_name=sort_on, sort_keys=sort_keys, batch_size=batch_size, shuffle=shuffle, keys_noise=keys_noise, drop_last=False)
        else:
            kwargs['batch_size'] = batch_size
            kwargs['shuffle'] = shuffle
        if batch_sampler is not None:
            kwargs.pop("batch_size", None)
            kwargs.pop("shuffle", None)
            kwargs.pop("sampler", None)
            kwargs.pop("drop_last", None)
        return DataLoader(range(len(self)),  # if self._idx is None else self._idx,
                          collate_fn=QueryFunction(self, device=device),
                          batch_sampler=batch_sampler,
                          **kwargs)


class DataloaderMixer(object):
    def __init__(self, dataloaders):
        if not isinstance(dataloaders, dict):
            dataloaders = dict(enumerate(dataloaders))
        self.dataloaders = list(dataloaders.values())
        self.tasks = np.concatenate([
            np.full(len(dataloader), fill_value=i)
            for i, dataloader in enumerate(self.dataloaders)
        ])
        self.task_names = list(dataloaders.keys())

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        tasks = self.tasks.copy()
        np.random.shuffle(tasks)
        iterators = [iter(dataloader) for dataloader in self.dataloaders]
        for task_id in tasks:
            yield self.task_names[task_id], next(iterators[task_id])


if __name__ == "__main__":
    batcher = Batcher({
        "doc": {
            "doc_id": np.asarray([10000, 20000, 30000]),
            "token_id": csr_matrix(np.asarray([
                [10001, 10002, 10003, 0],
                [20001, 20002, 20003, 20004],
                [30001, 30002, 0, 0],
            ])),
            "token_mask": csr_matrix(np.asarray([
                [True, True, True, False],
                [True, True, True, True],
                [True, True, False, False],
            ])).astype(bool),
        },
        "token": {
            "token_id": np.asarray([10001, 10002, 10003, 20001, 20002, 20003, 20004, 30001, 30002]),
            "doc_id": np.asarray([10000, 10000, 10000, 20000, 20000, 20000, 20000, 30000, 30000]),
            "word": np.asarray([0, 1, 0, 2, 3, 4, 2, 5, 6]),
        },
    })
    print(Batcher.concat([
        batcher["doc", [0, 0, 2]],
        batcher["doc", [1, 0, 2]]
    ])["doc"].densify(torch.device('cpu')).drop_duplicates().sparsify().densify(torch.device('cpu'))["doc", ["doc_id", "token_id"]])
    batcher2 = batcher[[1, 0, 0, 2, 0, 0, 1]]
    print(batcher2["token", "@doc_id"])
    print(batcher2["doc", "doc_id"])
    batcher = batcher["token", [0, 2]]
    print(batcher)
    batcher2 = batcher[["doc"]].densify(torch.device('cpu'))
    print("A", batcher2["doc", "token_id"])
    print(batcher2)
    batcher2 = batcher[["token"]].densify(torch.device('cpu'))
    print("B", batcher2["token", "doc_id"], type(batcher2["token", "doc_id"]))
    print(batcher2)
    print("-----------")
    # batcher["doc", :2]  # get first two docs
    # batcher["doc"].loc[[10000, 30000]]  # get those docs
    # batcher.loc["doc", [10000, 30000]]  # get those docs
    # batcher.to(torch.device('cuda'))  # densify and send to cuda device
    # batcher.sparsify()  # load from any device and sparsify as numpy / csr_matrices
    # batcher["doc"].relative()  # get those docs
    # batcher["doc", "token_id"]  # get the docs token_id
    # batcher["doc", "word"][batcher["doc", "token_id"]]  # get the docs token_id
    # batcher["doc", "word"]  # get the docs words
    # batcher["doc"]["token_id"]  # get those docs
