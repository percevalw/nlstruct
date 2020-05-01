from collections import Iterable, Sequence, Mapping
from logging import warning

import pandas as pd
from pandas.core.computation.ops import UndefinedVariableError

from nlstruct.utils.pandas import merge_with_spans


class Dataset(Mapping):
    def __init__(self, **dfs):
        self.dfs = {k: v for k, v in dfs.items()}
        self.main = next(iter(self.dfs.keys())) if len(self.dfs) else None

    def keys(self):
        return self.dfs.keys()

    def values(self):
        return self.dfs.values()

    def items(self):
        return self.dfs.items()

    def update(self, other):
        self.dfs.update(other)

    def groupby(self, cols):
        if isinstance(cols, str):
            cols = [cols]
        tables_to_groupby = [df for key, df in self.dfs.items() if all(col in df.columns for col in cols)]
        for col_ids, group in tables_to_groupby[0].groupby(cols):
            for other_table in tables_to_groupby[1:]:
                querier = None
                for col, col_id in zip(cols, col_ids):
                    querier = querier & (other_table[col] == col_id)
                other_group = other_table[querier]
        pass

    def query(self, query, names=None, propagate=False, keyerror='ignore'):
        if names is None:
            names = list(self.dfs.keys())
        elif isinstance(names, str):
            names = [names]
        dfs = dict(self.dfs)
        seen = set()
        for name1 in names:
            if name1 not in seen:
                try:
                    dfs[name1] = dfs[name1].query(query)
                    seen.add(name1)
                except UndefinedVariableError:
                    if keyerror == "ignore":
                        continue
                    else:
                        raise
                if propagate:
                    for name2, df2 in dfs.items():
                        if name2 not in seen:
                            dfs[name2] = merge_with_spans(dfs[name1][[c for c in dfs[name1].columns if c in df2.columns and c.endswith("_id")]].drop_duplicates(), df2, how="inner")
                            seen.add(name2)
        # main_df = self.dfs[self.main].query(query).reset_index(drop=True)
        # dfs = [main_df, *((merge_with_spans(main_df[[c for c in main_df.columns if c in df.columns and c.endswith("_id")]], df, how="inner")
        #                    if df is not None else None)
        #                   for df in list(self.dfs.values())[1:])]
        return Dataset(**{key: df.copy() for key, df in dfs.items()})

    def __len__(self):
        return len(self.dfs[self.main])

    def __contains__(self, item):
        return ((isinstance(item, int) and item < len(self.dfs)) or item in self.dfs) and self.dfs[item] is not None

    def prune_ids(self, dfs=None, inplace=False):
        if dfs is None:
            dfs = self.dfs
        if not inplace:
            dfs = {k: df.copy() if df is not None else None for k, df in dfs.items()}
        dtypes = {}
        for df in dfs.values():
            if df is None:
                continue
            for c in df.columns:
                if c in dtypes:
                    df[c] = df[c].astype(dtypes[c])
                elif c.endswith('_id') and hasattr(df[c], 'cat'):
                    df[c] = df[c].cat.remove_unused_categories()
                    dtypes[c] = df[c].dtype
        if not inplace:
            return Dataset(**dfs)

    def take(self, ids):
        main_df = self.dfs[self.main].iloc[ids].reset_index(drop=True)
        dfs = [main_df, *((merge_with_spans(main_df[[c for c in main_df.columns if c in df.columns and c.endswith('_id')]], df, how="inner")
                           if df is not None else None)
                          for df in list(self.dfs.values())[1:])]
        return Dataset(**dict(zip(self.dfs.keys(), dfs)))

    def merge(self, merged):
        dfs = [(pd.merge(df, merged[[c for c in merged.columns if c in df.columns]], how="inner")
                if df is not None else None)
               for df in self.dfs.values()]
        return Dataset(**dict(zip(self.dfs.keys(), dfs)))

    def __iter__(self):
        return iter(self.dfs.values())

    def __getitem__(self, item):
        if isinstance(item, (Iterable, Sequence)) and not isinstance(item, str):
            return Dataset(**{
                (list(self.dfs.keys())[k] if isinstance(k, int) else k):
                    list(self.dfs.values())[k] if isinstance(k, int) else self.dfs.get(k, None)
                for k in item})
        return list(self.dfs.values())[item] if isinstance(item, int) else self.dfs.get(item, None)

    def __setitem__(self, item, val):
        if isinstance(item, (tuple, list)):
            if isinstance(val, Dataset):
                assert len(val.keys()) == len(item), "Passed keys and dataset frames must have the same number of elements"
                if not all(name in val for name in item):
                    warning(f"Some of the keys in could not be found in the passed dataset during assignment (passed keys are {item}), no alignment will be done.")
                    for key, v in zip(item, val.values()):
                        self.dfs[key] = v
                else:
                    for key in item:
                        self.dfs[key] = val[key]
            else:
                for key, v in zip(item, val):
                    self.dfs[key] = v
            if len(self.dfs) == 0:
                self.main = item[0]
        else:
            assert isinstance(val, pd.DataFrame)
            if len(self.dfs) == 0:
                self.main = item
            self.dfs[item] = val

    @classmethod
    def concat(cls, batches):
        """
        Make a Dataset by concatenating multiple (Dataset) batches

        Parameters
        ----------
        batches: Sequence of Dataset

        Returns
        -------
        Dataset
        """
        batches = list(filter(lambda x: len(x) > 0, batches))
        zips = {k: [dic[k] for dic in batches] for k in batches[0].dfs.keys()}
        dic = {}
        for k, z in zips.items():
            non_null = next((b for b in z if b is not None), None)
            if non_null is not None:
                dic[k] = pd.concat(z, ignore_index=True, sort=False)
                for c in non_null.columns:
                    if hasattr(non_null[c], 'cat'):
                        dic[k][c] = dic[k][c].astype('category')
            else:
                dic[k] = None
        return Dataset(**dic)

    def copy(self):
        return Dataset(**{k: (df.copy() if df is not None else None) for k, df in self.dfs.items()})

    def __repr__(self):
        max_key_len = max(map(len, self.dfs.keys()))
        max_nb_len = max(len(str(len(df))) for df in self.dfs.values())
        return "Dataset(\n  {}\n)".format("\n  ".join("({}):{}{} * {}".format(
            k,
            " " * (max_key_len - len(k) + max_nb_len - len(str(len(v))) + 1),
            len(v),
            tuple(v.columns)
        ) for k, v in self.dfs.items() if v is not None))
