import logging
from collections import defaultdict, Sized
from functools import reduce

import numpy as np
import pandas as pd
from pandas._libs.lib import fast_zip
from pandas._libs.parsers import union_categoricals
from pandas.core.dtypes.common import is_numeric_dtype
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._traversal import connected_components

logger = logging.getLogger("nlstruct")


def join_cols(df, sep="/"):
    df = df.astype(str)
    return reduce(lambda x, y: x + sep + y, (df.iloc[:, i] for i in range(1, len(df.columns))), df.iloc[:, 0])


def get_sequence_length(obj):
    if isinstance(obj, str) or not isinstance(obj, Sized):
        return -1
    elif isinstance(obj, Sized) and all(not isinstance(i, Sized) and pd.isnull(i) for i in obj):
        return -2
    else:
        return len(obj)


def flatten(frame,
            index_name=None,
            as_index=False,
            keep_na=False,
            columns=None,
            tile_index=False):
    """
    Flatten the input before the transformation
    Parameters
    ----------
    frame: pandas.DataFrame
    index_name: str
        Name of the index to append to indentify each item uniquely
    keep_na: bool or str
        Should non-sequences elements (or sequences full of None) be kept in the dataframe
        as an empty row (value given is None and new index value is None also)
    columns: tuple of str
        Flatten only sequence in these columns if not None
    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame, callable)
        flattened input:
            Flattened input to transform
        length:
            Lengths of the sequences. We will actually only want to know if it was a sequence
            or not (see get_sequence_length(...)), during either unflattening if regroup is
            True or during rationale backpropagation
        sequence_constructor:
            Returns the "type" of the sequences contained in the frame, or more specifically
            the function used to build an instance of these sequences. Will be used during
            unflattening if self.regroup is True and during rationale backpropagation
    """
    if isinstance(as_index, bool):
        as_column = not as_index
    elif isinstance(as_index, str) and index_name is None:
        index_name = as_index
        as_column = False
    else:
        raise Exception("as_index must be str or bool, and if str, index_name must be None")

    if isinstance(frame, pd.Series):
        res = flatten(pd.DataFrame({"X": frame}), index_name, as_column, keep_na, columns, tile_index)
        new_frame = res["X"]
        new_frame.name = frame.name
        return new_frame

    if keep_na is True:
        keep_na = 'null_index'
    elif keep_na is False:
        keep_na = 'remove'
    assert keep_na in ('null_index', 'as_single_item', 'remove')

    assert isinstance(frame, pd.DataFrame), "Can only flatten DataFrame"
    if columns is None:
        columns = frame.columns
    elif not isinstance(columns, (tuple, list)):
        columns = [columns]
    else:
        columns = list(columns)
    lengths = frame[columns].applymap(lambda seq: get_sequence_length(seq))
    for col in frame.columns:
        if col not in columns:
            lengths[col] = -1
    result_lengths = lengths.max(axis=1)

    # Each column element will be expanded on multiple rows,
    # even if it is a non-iterable object
    # We must know before how many rows will the expansion take
    # and we take this length from the maximum sequence size
    if keep_na == 'remove':
        bool_row_selector = result_lengths > 0
        result_lengths = result_lengths[bool_row_selector]
        selected_lengths = lengths[bool_row_selector]
        frame = frame[bool_row_selector]
        nulls = None
    else:
        nulls = result_lengths < 0
        # Non sequence or sequence full of None will give rise to 1 row
        result_lengths[nulls] = 1
        selected_lengths = lengths
        nulls = result_lengths.cumsum()[nulls] - 1

    categoricals = {}
    frame = frame.copy()
    for col in frame.columns:
        if hasattr(frame[col], 'cat'):
            categoricals[col] = frame[col].cat.categories
            frame[col] = frame[col].cat.codes

    flattened = {col: [] for col in frame.columns}
    for col_name, col in frame.iteritems():
        for obj, res_length, length in zip(col.values, result_lengths, selected_lengths[col_name]):
            if length >= 0:  # we have a normal sequence
                flattened[col_name].append(obj if isinstance(obj, pd.Series) else pd.Series(obj))

            # Otherwise it a non sequence, create as many rows as needed for it
            else:
                # -2 means sequence full of None, we put a None instead here
                if length == -2:
                    obj = None
                if res_length == 1:
                    flattened[col_name].append(pd.Series([obj]))
                else:
                    flattened[col_name].append(pd.Series([obj] * res_length))

    index = frame.index.repeat(result_lengths) if index_name is not None else None
    for col_name in flattened:
        flattened[col_name] = pd.concat(flattened[col_name], ignore_index=True)
        if index is not None:
            flattened[col_name].index = index

    flattened = pd.DataFrame(flattened)

    # flattened = pd.DataFrame(
    #     data={col_name: pd.concat(flattened[col_name], ignore_index=True) for col_name in flattened},
    #     index=frame.index.repeat(result_lengths) if index_name is not None else None)

    for name, categories in categoricals.items():
        flattened[name] = pd.Categorical.from_codes(flattened[name], categories=categories)
    # Adds an index under the name `self.index_name` to identify uniquely every row
    # of the frame
    if index_name is not None:
        if index_name in flattened.columns:
            flattened.set_index(index_name, append=True, inplace=True)
        else:
            if tile_index:
                new_index_values = np.concatenate([np.arange(s) for s in result_lengths])
                flattened[index_name] = new_index_values
            else:
                new_index_values = np.arange(len(flattened))
                flattened[index_name] = new_index_values
                flattened[index_name] = flattened[index_name]

            flattened.set_index(index_name, append=True, inplace=True)
            if keep_na == 'null_index' and nulls is not None:
                new_labels = np.arange(len(flattened))
                # noinspection PyUnresolvedReferences
                new_labels[nulls.values] = -1
                flattened.index.set_codes(
                    new_labels, level=index_name, inplace=True)
        if as_column:
            flattened.reset_index(index_name, inplace=True)
            flattened.reset_index(inplace=True, drop=True)
    # flattened.index = flattened.index.remove_unused_levels()

    return flattened


def make_merged_names(left_span_names, right_span_names, left_on, right_on, left_columns, right_columns,
                      suffixes=('_x', '_y')):
    right_columns = set(right_columns) - set(right_on)
    left_columns = set(left_columns) - set(left_on)
    left_merged = [name + (suffixes[0] if name in right_columns else '') for name in left_span_names]
    right_merged = [name + (suffixes[1] if name in left_columns else '') for name in right_span_names]
    return left_merged, right_merged


def make_merged_names_map(left_columns, right_columns, left_on, right_on, suffixes=('_x', '_y')):
    right_columns = set(right_columns) - set(right_on)
    left_columns = set(left_columns) - set(left_on)
    left_merged = [name + (suffixes[0] if name in right_columns else '') for name in left_columns]
    right_merged = [name + (suffixes[1] if name in left_columns else '') for name in right_columns]
    return dict(zip(left_columns, left_merged)), dict(zip(right_columns, right_merged))


def merge_with_spans(
      left, right=None,
      how='inner',
      on=None,
      left_on=None,
      right_on=None,
      suffixes=('_x', '_y'),
      span_policy='partial_strict',
      placeholder_columns=(),
      **kwargs):
    """
    Just like pandas.merge, but handles the merging of spans
    Any tuple in the "on" column will be considered a (begin, end) span
    How to merge those span

    Parameters
    ----------
    left: pd.DataFrame
    right: pd.DataFrame
    how: str
        "inner", "outer", "left", "right"
    on: list of (str or tuple of str)
    left_on: list of (str or tuple of str)
    right_on: list of (str or tuple of str)
    suffixes: list of str
    span_policy: str
        How to merge spans ?
        One of: "partial", "exact", "partial_strict"
    placeholder_columns:
        Zero will be put as a value instead of nan for any empty cell in those columns after the merge
    kwargs: any
        Any kwargs for the pd.merge function

    Returns
    -------
    pd.DataFrame
    """
    if right is None:
        right = left
    left = left.copy()
    right = right.copy()
    if isinstance(on, str):
        on = [on]
    if left_on is None:
        left_on = on
    if right_on is None:
        right_on = on
    left_columns = left.columns if hasattr(left, 'columns') else [left.name]
    right_columns = right.columns if hasattr(right, 'columns') else [right.name]
    if left_on is None and right_on is None:
        left_on = right_on = list(set(left_columns) & set(right_columns))
    left_on_spans = [o for o in left_on if isinstance(o, tuple)]
    right_on_spans = [o for o in right_on if isinstance(o, tuple)]

    left_on = [c for c in left_on if not isinstance(c, tuple)]  # flatten_sequence(left_on)
    right_on = [c for c in right_on if not isinstance(c, tuple)]  # flatten_sequence(right_on)
    left_names, right_names = make_merged_names(
        left_columns, right.columns,
        left_on=left_on,
        right_on=right_on,
        left_columns=left_columns, right_columns=right_columns, suffixes=suffixes)
    left_names_map = dict(zip(left_columns, left_names))
    right_names_map = dict(zip(right_columns, right_names))

    categoricals = {}
    for left_col, right_col in zip(left_on, right_on):
        left_cat = getattr(left[left_col] if hasattr(left, 'columns') else left, 'cat', None)
        right_cat = getattr(right[right_col] if hasattr(right, 'columns') else right, 'cat', None)
        if left_cat is not None or right_cat is not None:
            if (left_cat and right_cat and not (left_cat.categories is right_cat.categories)) or (
                  (left_cat is None) != (right_cat is None)):
                left[left_col] = left[left_col].astype('category')
                right[right_col] = right[right_col].astype('category')
                cat_merge = union_categoricals([left[left_col], right[right_col]])
                if hasattr(left, 'columns'):
                    left[left_col] = cat_merge[:len(left)]
                else:
                    left = cat_merge[:len(left)]
                if hasattr(right, 'columns'):
                    right[right_col] = cat_merge[len(left):]
                else:
                    right = cat_merge[len(left):]

            categoricals[left_names_map[left_col]] = left[left_col].cat.categories
            categoricals[right_names_map[right_col]] = right[right_col].cat.categories
            if hasattr(left, 'columns'):
                left[left_col] = left[left_col].cat.codes
            else:
                left = left.cat.codes
            if hasattr(right, 'columns'):
                right[right_col] = right[right_col].cat.codes
            else:
                right = right.cat.codes

    if len(left_on_spans) == 0:
        merged = pd.merge(left, right, left_on=left_on, right_on=right_on, suffixes=suffixes, how=how, **kwargs)
    else:
        if how != 'inner':
            left['_left_index'] = np.arange(len(left))
            right['_right_index'] = np.arange(len(right))

        merged = pd.merge(left, right, left_on=left_on, right_on=right_on, suffixes=suffixes, how='inner', **kwargs)
        for i, (left_span_names, right_span_names) in enumerate(zip(left_on_spans, right_on_spans)):
            (left_begin, left_end), (right_begin, right_end) = make_merged_names(
                left_span_names, right_span_names, left_on=left_on, right_on=right_on,
                left_columns=left.columns, right_columns=right_columns, suffixes=suffixes)
            merged[f'overlap_size_{i}'] = np.minimum(merged[left_end], merged[right_end]) - np.maximum(merged[left_begin], merged[right_begin])
            if span_policy != "none":
                results = []
                chunk_size = 1000000
                for chunk_i in range(0, len(merged), chunk_size):
                    if span_policy == "partial_strict":
                        results.append(merged.iloc[chunk_i:chunk_i + chunk_size].query(f'({right_end} > {left_begin} and {left_end} > {right_begin})'))
                    elif span_policy == "partial":
                        results.append(merged.iloc[chunk_i:chunk_i + chunk_size].query(f'({right_end} >= {left_begin} and {left_end} >= {right_begin})'))
                    elif span_policy == "exact":
                        results.append(merged.iloc[chunk_i:chunk_i + chunk_size].query(f'({left_begin} == {right_begin} and {left_end} == {right_end})'))
                    else:
                        results.append(merged.iloc[chunk_i:chunk_i + chunk_size].query(span_policy))
                if len(results):
                    merged = pd.concat(results, sort=False, ignore_index=True)
                else:
                    merged = merged.iloc[:0]
            elif span_policy == "none":
                pass
            else:
                raise Exception(f"Unrecognized policy {span_policy}")

        if how != 'inner':
            if how in ('left', 'outer'):
                missing = left[~left['_left_index'].isin(merged['_left_index'])].copy()
                missing = missing.rename(left_names_map, axis=1)

                for col in right.columns:
                    if hasattr(right[col], 'cat') and right_names_map[col] not in missing.columns:
                        missing[right_names_map[col]] = pd.Categorical([None] * len(missing),
                                                                       categories=right[col].cat.categories)
                for col in placeholder_columns:
                    if col not in left_on and right_names_map.get(col, col) not in left.columns:
                        missing[right_names_map.get(col, col)] = 0  # -np.arange(len(missing)) - 1
                merged = pd.concat([merged, missing.rename(dict(zip(left.columns, left_names)), axis=1)], sort=False,
                                   ignore_index=True)
            if how in ('right', 'outer'):
                missing = right[~right['_right_index'].isin(merged['_right_index'])].copy()
                missing = missing.rename(right_names_map, axis=1)

                for col in left.columns:
                    if hasattr(left[col], 'cat') and left_names_map[col] not in missing.columns:
                        missing[left_names_map[col]] = pd.Categorical([None] * len(missing),
                                                                      categories=left[col].cat.categories)
                for col in placeholder_columns:
                    if col not in right_on and left_names_map.get(col, col) not in right.columns:
                        missing[left_names_map.get(col, col)] = 0  # -np.arange(len(missing)) - 1
                merged = pd.concat([merged, missing.rename(dict(zip(right.columns, right_names)), axis=1)], sort=False,
                                   ignore_index=True)
            merged = merged.sort_values(['_left_index', '_right_index'])
            del merged['_left_index']
            del merged['_right_index']
        merged = merged.reset_index(drop=True)

    for col, categories in categoricals.items():
        merged[col] = pd.Categorical.from_codes(merged[col].fillna(-1).astype(int), categories=categories)

    return merged


def make_id_from_merged(*indices_arrays, same_ids=False, apply_on=None):
    """
    Compute new ids from connected components by looking at `indices_arrays`
    Parameters
    ----------
    indices_arrays: collections.Sequence
        1d array of positive integers
    same_ids: bool
        Do the multiple arrays represent the same ids ? (a 3 in one column should therefore be
        connected to a 3 in another, event if they are not on the same row)
    apply_on: list of (int, any)
        Return the new ids matching old ids
        for each (index, vector) in apply_on:
            return new_ids matching those in vector that should be considered the same
            of those of the vector number `index` in the `indices_arrays`

    Returns
    -------
    list of np.ndarray
    """
    if not same_ids:
        indices_arrays, unique_objects = zip(*(factorize_rows(array, return_categories=True) for array in indices_arrays))
    else:
        indices_arrays, unique_objects = factorize_rows(indices_arrays, return_categories=True)
        unique_objects = [unique_objects] * len(indices_arrays)

    offset = max(indices_array.max() for indices_array in indices_arrays) + 1
    N = offset * (len(indices_arrays) + 1)
    if same_ids:
        N = offset
        offset = 0
    offseted_ids = [s + i * offset for i, s in enumerate(indices_arrays)]

    left_ids, right_ids = zip(*[(offseted_ids[i], offseted_ids[j])
                                for i in range(0, len(indices_arrays) - 1)
                                for j in range(i + 1, len(indices_arrays))])
    left_ids = np.concatenate(left_ids)
    right_ids = np.concatenate(right_ids)
    _, matches = connected_components(csr_matrix((np.ones(len(left_ids)), (left_ids, right_ids)), shape=(N, N)))
    matches = pd.factorize(matches)[0]
    if apply_on is None:
        return [
            matches[s]
            for s in offseted_ids
        ]
    else:
        return [
            matches[factorize_rows(s, categories=unique_objects[i], return_categories=False) + i * offset]
            for i, s in apply_on
        ]


def df_to_csr(rows, cols, data=None, n_rows=None, n_cols=None):
    """
    Transforms a dataframe into a csr_matrix

    Parameters
    ----------
    data: pd.Series
        Data column (column full one True will be used if None)
    rows: pd.Series
        Column containing row indices (can be Categorical and then codes will be used)
    cols: pd.Series
        Column containing column indices (can be Categorical and then codes will be used)
    n_rows: int
    n_cols: int
    Returns
    -------
    csr_matrix
    """
    if data is None:
        data = np.ones(len(rows), dtype=bool)
    if hasattr(rows, 'cat'):
        n_rows = len(rows.cat.categories)
        rows, rows_cat = rows.cat.codes, rows.cat.categories
    else:
        n_rows = n_rows or (rows.max() + 1 if len(rows) > 0 else 0)
    if hasattr(cols, 'cat'):
        n_cols = len(cols.cat.categories)
        cols, cols_cat = cols.cat.codes, cols.cat.categories
    else:
        n_cols = n_cols or (cols.max() + 1 if len(cols) > 0 else 0)
    return csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(n_rows, n_cols))


def df_to_flatarray(rows, data, n_rows=None):
    """
    Transforms a dataframe into a flat array

    Parameters
    ----------
    data: pd.Series
        Data column (column full one True will be used if None)
    rows: pd.Series
        Column containing row indices (can be Categorical and then codes will be used)
    n_rows: int
    Returns
    -------
    np.ndarray
    """
    if hasattr(rows, 'cat'):
        n_rows = len(rows.cat.categories)
        rows, rows_cat = rows.cat.codes, rows.cat.categories
    else:
        n_rows = n_rows or (rows.max() + 1)
    res = np.zeros(n_rows, dtype=data.dtype)
    res[rows] = np.asarray(data)
    return res


def csr_to_df(csr, row_categories=None, col_categories=None, row_name=None, col_name=None, value_name=None):
    """
    Convert a csr_matrix to a dataframe

    Parameters
    ----------
    csr: csr_matrix
    row_categories: any
        Categories to rebuild the real object from their row indices
    col_categories: any
        Categories to rebuild the real object from their col indices
    row_name: str
        What name to give to the column built from the row indices
    col_name: str
        What name to give to the column built from the col indices
    value_name:
        What name to give to the column built from the values
        If None, no value column will be built

    Returns
    -------
    pd.DataFrame
    """
    csr = csr.tocoo()
    rows, cols, values = csr.row, csr.col, csr.data
    if isinstance(row_categories, pd.DataFrame):
        rows_df = row_categories.iloc[rows]
    elif isinstance(row_categories, pd.Series):
        rows_df = pd.DataFrame({row_categories.name: row_categories.iloc[rows]})
    elif isinstance(row_categories, pd.CategoricalDtype):
        rows_df = pd.DataFrame({row_name: pd.Categorical.from_codes(rows, dtype=row_categories)})
    else:
        rows_df = pd.DataFrame({row_name: rows})
    if isinstance(col_categories, pd.DataFrame):
        cols_df = col_categories.iloc[cols]
    elif isinstance(col_categories, pd.Series):
        cols_df = pd.DataFrame({col_categories.name: col_categories.iloc[cols]})
    elif isinstance(col_categories, pd.CategoricalDtype):
        cols_df = pd.DataFrame({col_name: pd.Categorical.from_codes(cols, dtype=col_categories)})
    else:
        cols_df = pd.DataFrame({col_name: cols})
    res = (rows_df.reset_index(drop=True), cols_df.reset_index(drop=True))
    if value_name is not None:
        res = res + (pd.DataFrame({value_name: values}),)
    return pd.concat(res, axis=1)


def factorize_rows(rows, categories=None, group_nans=True, subset=None, freeze_categories=True, return_categories=True):
    if not isinstance(rows, list):
        was_list = False
        all_rows = [rows]
    else:
        all_rows = rows
        was_list = True
    del rows
    not_null_subset = (subset if subset is not None else all_rows[0].columns if hasattr(all_rows[0], 'columns') else [all_rows[0].name])
    cat_arrays = [[] for _ in not_null_subset]
    for rows in (categories, *all_rows) if categories is not None else all_rows:
        for (col_name, col), dest in zip(([(0, rows)] if len(rows.shape) == 1 else rows[subset].items() if subset is not None else rows.items()), cat_arrays):
            dest.append(np.asarray(col))
    cat_arrays = [np.concatenate(arrays) for arrays in cat_arrays]
    is_not_nan = None
    if not group_nans:
        is_not_nan = ~pd.isna(np.stack(cat_arrays, axis=1)).any(1)
        cat_arrays = [arrays[is_not_nan] for arrays in cat_arrays]
    if len(cat_arrays) > 1:
        relative_values, unique_values = pd.factorize(fast_zip(cat_arrays))
    else:
        relative_values, unique_values = pd.factorize(cat_arrays[0])
    if freeze_categories and categories is not None:
        relative_values[relative_values >= len(categories)] = -1
    if not group_nans:
        new_relative_values = np.full(is_not_nan.shape, fill_value=-1, dtype=relative_values.dtype)
        new_relative_values[is_not_nan] = relative_values
        new_relative_values[~is_not_nan] = len(unique_values) + np.arange((~is_not_nan).sum())
        relative_values = new_relative_values

    offset = len(categories) if categories is not None else 0
    res = []
    for rows in all_rows:
        new_rows = relative_values[offset:offset + len(rows)]
        if isinstance(rows, (pd.DataFrame, pd.Series)):
            new_rows = pd.Series(new_rows)
            new_rows.index = rows.index
            new_rows.name = "+".join(not_null_subset)

        res.append(new_rows)
        offset += len(rows)
    if categories is None and return_categories:
        if isinstance(all_rows[0], pd.DataFrame):
            if len(cat_arrays) > 1:
                categories = pd.DataFrame(dict(zip(not_null_subset, [np.asarray(l) for l in zip(*unique_values)])))
            else:
                categories = pd.DataFrame({not_null_subset[0]: unique_values})
            categories = categories.astype({k: dtype for k, dtype in next(rows for rows in all_rows if len(rows)).dtypes.items() if k in not_null_subset})
        elif isinstance(all_rows[0], pd.Series):
            categories = pd.Series(unique_values)
            categories.name = all_rows[0].name
            categories = categories.astype(next(rows.dtype for rows in all_rows if len(rows)))
        else:
            categories = np.asarray([l for l in zip(*unique_values)])
    if not was_list:
        res = res[0]
    if not return_categories:
        return res
    return res, categories


def normalize_vocabularies(dfs, vocabularies=None, train_vocabularies=True, unk=None, verbose=0):
    """
    Categorize the columns of the dataframes so that they share the same
    categories if they share the same columns
    If a column's name ends up with '_id', do not categorize it since it is no something we want to train on

    Parameters
    ----------
    dfs: list of pd.DataFrame
        DataFrame whose columns will be categorized
    vocabularies: dict or None
        Existing vocabulary to use if any
    train_vocabularies: bool or dict of (str, bool)
        Which category to extend/create in the voc ?
    unk: dict of (str, any)
        Which filler should we put for an unknown object if we cannot train the corresponding voc ?
    verbose: int

    Returns
    -------
    list of pd.DataFrame, dict
    """
    # Define label vocabulary
    if unk is None:
        unk = {}
    if vocabularies is None:
        vocabularies = {}
    elif isinstance(vocabularies, dict):
        vocabularies = dict(vocabularies)
    voc_order = list(vocabularies.keys())

    if train_vocabularies is False:
        train_vocabularies = defaultdict(lambda: False)
    else:
        train_vocabularies_ = defaultdict(lambda: True)
        if isinstance(train_vocabularies, dict):
            train_vocabularies_.update(train_vocabularies)
        train_vocabularies = train_vocabularies_
        del train_vocabularies_

    for col_name in vocabularies:
        if col_name not in train_vocabularies:
            train_vocabularies[col_name] = False

    for df in dfs:
        for col_name in df:
            if not col_name.endswith('_id') and not is_numeric_dtype(df[col_name].dtype):
                if train_vocabularies[col_name]:
                    train_vocabularies[col_name] = True
                else:
                    train_vocabularies[col_name] = False
    for col_name, will_train in train_vocabularies.items():
        if will_train and verbose:
            logger.info(f"Will train vocabulary for {col_name}")
    for df in dfs:
        for col_name in df:
            if hasattr(df[col_name], 'cat') and col_name not in vocabularies and not col_name.endswith('_id'):
                if verbose:
                    logger.info(f"Discovered existing vocabulary ({len(df[col_name].cat.categories)} entities) for {col_name}")
                vocabularies[col_name] = list(df[col_name].dtype.categories)
    for voc_name, train_voc in train_vocabularies.items():
        if train_voc:
            voc = list(vocabularies.get(voc_name, []))
            if voc_name in unk and unk[voc_name] not in voc:
                voc.append(unk[voc_name])
            if hasattr(voc, 'categories'):
                voc = list(voc.categories)
            for df in dfs:
                if voc_name in df:
                    voc.extend(df[voc_name].astype("category").cat.categories)
            voc = pd.factorize(voc)[1]
            dtype = pd.CategoricalDtype(pd.factorize(voc)[1])
            for df in dfs:
                if voc_name in df:
                    df[voc_name] = df[voc_name].astype(dtype)
                    vocabularies[voc_name] = voc
                    if voc_name in unk:
                        df[voc_name].fillna(unk[voc_name], inplace=True)
        else:
            voc = vocabularies.get(voc_name)
            if voc is None:
                continue
            if not hasattr(voc, 'categories'):
                voc = pd.CategoricalDtype(voc)
            for df in dfs:
                if voc_name in df:
                    df[voc_name] = df[voc_name].astype(voc)
                    if verbose:
                        unk_msg = f"unk {unk[voc_name]}" if voc_name in unk else "no unk"
                        logger.info(f"Normalized {voc_name}, with given vocabulary and {unk_msg}")
                    if voc_name in unk:
                        df[voc_name].fillna(unk[voc_name], inplace=True)

    # Reorder vocabularies to keep same order as the vocabulary passed in parameters
    vocabularies = dict((*((c, vocabularies[c]) for c in voc_order if c in vocabularies),
                         *((c, vocabularies[c]) for c in vocabularies if c not in voc_order)))

    # Reorder dataframes according to vocabulary order
    dfs = [
        df[[*(c for c in vocabularies if c in df.columns), *(c for c in df.columns if c not in vocabularies)]]
        for df in dfs
    ]
    return dfs, vocabularies


def assign_sorted_id(df, name, groupby, sort_on):
    """
    Assign a sorted id grouped by the values in `groupby`

    Parameters
    ----------
    df:  pd.DataFrame
    name: str
        Name of the new id
    groupby: (list of str) or str
    sort_on: (list of str) or str

    Returns
    -------
    pd.DataFrame
    """
    df = df.sort_values(sort_on)
    df[name] = np.arange(len(df))
    return df.nlstruct.groupby_assign(groupby, {name: lambda x: tuple(np.argsort(np.argsort(x)))})


def encode_ids(dfs, names=None, inplace=True):
    """
    Encode multiple columns of several dataframes into a unique shared id

    Parameters
    ----------
    dfs: list of pd.DataFrame or pd.DataFrame
        DataFrame on which we want to replace ids by a unique code
    names: str or tuple or list
        Tuple of columns to transform into a unique ids, or list to specify names for each dataframe
    inplace: bool
        Inplace
    Returns
    -------
    pd.DataFrame or (pd.DataFrame, pd.DataFrame)
    """
    if isinstance(names, str):
        names = (names,)
    if isinstance(names, tuple):
        names = [names for _ in dfs]
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    assert len(names) == len(dfs)
    all_encoded_ids, unique_ids = factorize_rows([frame[list(n)] for n, frame in zip(names, dfs)])
    res = []
    if not inplace:
        id_to_frame = {}
        for i, frame in enumerate(list(dfs)):
            if id(frame) not in id_to_frame:
                id_to_frame[id(frame)] = frame.copy()
            dfs[i] = id_to_frame[id(frame)]
    for frame, n, frame_encoded_id in zip(dfs, names, all_encoded_ids):
        frame[n[-1]] = frame_encoded_id
        if not inplace:
            res.append(frame)
    if not inplace:
        return res, unique_ids
    return unique_ids


class FasterGroupBy:
    def __init__(self, groupby_object, dtypes, name=None):
        self.groupby_object = groupby_object
        self.dtypes = dtypes
        self.name = name

    def _retype(self, res):
        if self.name is None:
            return res.astype(self.dtypes)
        return (res.astype(self.dtypes) if self.dtypes is not None else res).reset_index().rename({0: self.name}, axis=1)

    def agg(self, *args, **kwargs):
        return self._retype(self.groupby_object.agg(*args, **kwargs))

    def apply(self, *args, **kwargs):
        return self._retype(self.groupby_object.apply(*args, **kwargs))

    def __getitem__(self, item):
        return FasterGroupBy(self.groupby_object[item], self.dtypes.get(item, None), item if not isinstance(item, (list, tuple)) else None)


class NLStructAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def factorize(self, subset=None, categories=None, group_nans=False,
                  return_categories=False, freeze_categories=True):
        return factorize_rows(self._obj,
                              subset=subset,
                              categories=categories,
                              group_nans=group_nans,
                              return_categories=return_categories, freeze_categories=freeze_categories)

    def flatten(self, *args, **kwargs):
        return flatten(self._obj, *args, **kwargs)

    def to_flatarray(self, row_column, data_column, n_rows=None):
        return df_to_flatarray(self._obj[row_column], self._obj[data_column], n_rows=n_rows)

    def to_csr(self, row_column, col_column, data_column=None, n_rows=None, n_cols=None):
        return df_to_csr(self._obj[row_column], self._obj[col_column], self._obj[data_column] if data_column is not None else None,
                         n_rows=n_rows, n_cols=n_cols)

    def groupby(self, by, *args, decategorize=None, as_index=False, observed=True, **kwargs):
        if not as_index:
            if decategorize is None:
                decategorize = by
            new_dtypes = {k: v if not hasattr(v, 'categories') else v.categories.dtype for k, v in self._obj.dtypes[decategorize].items()}
            return FasterGroupBy(self._obj.astype(new_dtypes).groupby(by=by, *args, as_index=as_index, observed=observed, **kwargs), self._obj.dtypes[decategorize])
        else:
            return self._obj.groupby(by=by, *args, as_index=as_index, **kwargs)

    def groupby_assign(self, by, agg, as_index=False, observed=True, **kwargs):
        res = self._obj.assign(_index=np.arange(len(self._obj)))
        res = res.drop(columns=list(agg.keys())).merge(
            # .astype({key: "category" for key in mentions_cluster_ids})
            res.groupby(by, observed=observed, **kwargs)
                .agg({**agg, "_index": tuple}).reset_index(drop=True)
                .nlstruct.flatten("_index"),
            how='left',
            on='_index',
        ).drop(columns=["_index"])
        if as_index:
            res = res.set_index(by)
        return res


pd.api.extensions.register_dataframe_accessor("nlstruct")(NLStructAccessor)
pd.api.extensions.register_series_accessor("nlstruct")(NLStructAccessor)
