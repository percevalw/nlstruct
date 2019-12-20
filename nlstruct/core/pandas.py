from collections import defaultdict, Sized

import numpy as np
import pandas as pd
from pandas._libs.lib import fast_zip
from pandas._libs.parsers import union_categoricals
from pandas.core.dtypes.common import is_numeric_dtype
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._traversal import connected_components


def get_constructor(X):
    """When applying StackSequences(estimator=...), and not aggregating/grouping in the given
    estimator, we automatically return a frame with sequences and we want to have the same
    collection type in the output as in the input
    (tuple->tuple, list->list, np.ndarray->np.ndarray)
    With numpy arrays, we cannot construct an instance using the object type ndarray, but instead
    using np.array. This function handles this special case
    """
    if isinstance(X, np.ndarray):
        return np.array
    else:
        return type(X)


def get_sequence_length(obj):
    """
    Get the "sequence length", even for object that are not sequences !
    If the object is a scalar, it returns -1 (a code we will use later to compute rationales)
    Also, if the object is a list of None, it counts as None, count it as so we mark it as -1
    Else, give the object length
    Parameters
    ----------
    obj: typing.Iterable or typing.Sized or Any
    Returns
    -------
    int
        - the length if it is a normal non-null sequence not full of None
        - -1 if it is not a sequence
        - -2 if it is a sequence full of None
    """
    # noinspection PyTypeChecker
    if isinstance(obj, str) or not isinstance(obj, Sized):
        return -1
    elif isinstance(obj, Sized) and all(not isinstance(i, Sized) and pd.isnull(i) for i in obj):
        return -2
    else:
        return len(obj)


def stack_sequences_frame(frame,
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
        res = stack_sequences_frame(pd.DataFrame({"X": frame}), index_name, as_column, keep_na, columns, tile_index)
        new_frame = res[0]["X"]
        new_frame.name = frame.name
        return (new_frame, *res[1:])

    if keep_na is True:
        keep_na = 'null_index'
    elif keep_na is False:
        keep_na = 'remove'
    assert keep_na in ('null_index', 'as_single_item', 'remove')

    assert isinstance(frame, pd.DataFrame), "Can only flatten DataFrame"
    sequence_constructor = None
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
                if sequence_constructor is None:
                    sequence_constructor = get_constructor(obj)
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
                flattened[index_name] = flattened[index_name].astype("category")

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

    return (
        flattened,
        lengths,
        sequence_constructor,
    )


def flatten(X, *args, **kwargs):
    return stack_sequences_frame(X, *args, **kwargs)[0]


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
      span_policy='partial',
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
    indices_arrays: any
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
            matches[s + i * offset]
            for i, s in apply_on
        ]


def aggregate_on(frame, on, **kwargs):
    """
    Same as groupby but selected levels for grouping are all levels except `on`
    Parameters
    ----------
    frame: pd.DataFrame or pd.Series
    on: list of (str or int)
        Level not to group on
    Returns
    -------
    pd.DataFrame or pd.Series
    """
    if not isinstance(on, (tuple, list)):
        on = [on]
    levels = [i if n is None else n for i, n in enumerate(frame.index.names) if n not in on]
    return frame.groupby(level=levels, **kwargs)


def get_explicit_index_names(index, prefix="_level_"):
    if isinstance(index, pd.MultiIndex):
        return np.array([prefix + str(i) if n is None else n for i, n in enumerate(index.names)])
    elif index.name is None:
        return np.array([prefix + '0'])
    else:
        return index.names


def align_frames(frame1, frame2, how="left", keep_order=True):
    """
    Aligns a pandas frame, (Sparse)Series/DataFrame on a new index
    /!\ You must not have duplicated index values, each row must be uniquely identifiable
        otherwise the result is undefined
    Think of this operation like frame being a corrupted version of the data with missing rows
    or two many rows for example, an index being the clean reference we want to align on
    Multiple cases can happen:
    - if an index level is in `frame` but not `index`, it is kept
    - if an index level is not in `frame` but in `index`, it is added
    - if an index level exists in `frame` and `index`, the kept values depend on the join policy
    Pandas already has a method .align() to perform this kind of operation but
    1. it does not work with MultiIndex
    2. it index values where lost (ex a sample) in frame, align does not recover it
    Parameters
    ----------
    frame1: pandas.DataFrame or pandas.Series
    frame2: pandas.DataFrame or pandas.Series
    how: str
        "inner", "outer", "left", "right"
        How to perform merge between the two frames' indexes
    keep_order: bool
        Should we ensure to keep the row order during merge
        For example, if you merge a dataframe containing all the samples and an index with a type
        like sample_type, with a dataframe containing whose index contains only the sample_type
        You should lose the order of the sample when performing "inner"/"outer" joins since
        sample_id is not part of the merged index.
        You do not have to worry about this when the index you don't want to mess up is part of
        the merged indexes
    Returns
    -------
    pandas.DataFrame or pandas.Series
    """

    # We are going to perform inplace ops on `frame1` and `frame2` so copy it now
    new_frame1 = frame1.copy()
    new_frame2 = frame2.copy()

    # We will need to change names used in `index` because
    # performing reset_index with names like None have unpredictable results
    # ex: None -> "0"
    # So we decide now a new explicit name for each index level
    index_name_map = {}

    frame1_index_names = get_explicit_index_names(new_frame1.index)
    index_name_map.update(dict(zip(frame1_index_names, new_frame1.index.names)))
    if isinstance(new_frame1.index, pd.MultiIndex):
        new_frame1.index.names = frame1_index_names
    else:
        new_frame1.index.name = frame1_index_names[0]
    if isinstance(new_frame1, pd.DataFrame):
        new_frame1.columns = ["_col1_" + c for c in new_frame1.columns]
    else:
        new_frame1 = pd.DataFrame({"_col1_": new_frame1})

    # Idem from levels of frame2's index
    frame2_index_names = get_explicit_index_names(new_frame2.index)
    index_name_map.update(dict(zip(frame2_index_names, new_frame2.index.names)))
    if isinstance(new_frame2.index, pd.MultiIndex):
        new_frame2.index.names = frame2_index_names
    elif new_frame2.index.name is None:
        new_frame2.index.name = frame2_index_names[0]
    if isinstance(new_frame2, pd.DataFrame):
        new_frame2.columns = ["_col2_" + c for c in new_frame2.columns]
    else:
        new_frame2 = pd.DataFrame({"_col2_": new_frame2})

    # Now we pass every index of the `index` and `frame` in the columns with reset_index
    new_frame2 = new_frame2.reset_index()
    new_frame1 = new_frame1.reset_index()

    # We will merge our new two dataframe, on overlapping index names
    # Default behavior of merge but it is better to explicit it
    merge_on = list(np.intersect1d(frame1_index_names, frame2_index_names))

    # + we reorder levels to have the ref index level first, and the frame new levels after
    index_by = list(pd.unique([*frame1_index_names, *frame2_index_names]))

    # When the first index level is not in the merged indexes, how = "inner" / "outer" lose the
    # rows order: ie our samples can be completely shuffled after aligning the two frames together
    # If however the first index level is in the merged index, then order will be kept and
    # no further computation is needed around pd.merge
    if merge_on[0] == frame1_index_names[0]:
        # It will be kept
        keep_order = False

    # If we want inner merge between indexes, then we perform left merge and remove
    # rows of the second frame that did not exist before (_original_position2 is NaN)
    # If we want outer merge between indexes, then let pandas perform outer merge and
    # reorder rows
    # https://stackoverflow.com/a/28334396/6067181
    merge_how = how
    if keep_order and how == "outer":
        new_frame1["_original_position1"] = np.arange(len(new_frame1))
        new_frame2["_original_position2"] = np.arange(len(new_frame2))
    elif keep_order and how == "inner":
        new_frame2["_original_position2"] = np.arange(len(new_frame2))
        merge_how = "left"

    # Perform the merge between indexes
    new_frame = new_frame1.merge(
        new_frame2,
        how=merge_how,
        on=merge_on)

    if keep_order and how == "inner":
        # as planned, remove rows that had no match in frame2
        new_frame = new_frame[~pd.isnull(new_frame["_original_position2"])]
        del new_frame["_original_position2"]
    elif keep_order and how == "outer":
        # as planned, remove rows that had no match in frame2
        new_frame.sort_values(by=["_original_position1", "_original_position2"])
        del new_frame["_original_position1"]
        del new_frame["_original_position2"]

    # Move the index values from the columns to the index and rename name like before
    new_frame.set_index(index_by, inplace=True)
    new_frame.index.names = [index_name_map[n] for n in index_by]

    # Separate the two aligned dataframe and recover their indexes
    new_frame1 = new_frame[[c for c in new_frame.columns if c.startswith("_col1_")]]
    new_frame2 = new_frame[[c for c in new_frame.columns if c.startswith("_col2_")]]
    new_frame1.columns = [c[6:] for c in new_frame1.columns]
    new_frame2.columns = [c[6:] for c in new_frame2.columns]

    if isinstance(frame1, pd.Series):
        new_frame1 = new_frame1.iloc[:, 0]
        new_frame1.name = frame1.name
    if isinstance(frame2, pd.Series):
        new_frame2 = new_frame2.iloc[:, 0]
        new_frame2.name = frame2.name
    return new_frame1, new_frame2


def align_frame(frame, ref):
    if isinstance(ref, pd.Index):
        ref = pd.DataFrame({}, index=ref)
    return align_frames(ref, frame)[1]


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
        n_rows = n_rows or (rows.max() + 1)
    if hasattr(cols, 'cat'):
        n_cols = len(cols.cat.categories)
        cols, cols_cat = cols.cat.codes, cols.cat.categories
    else:
        n_cols = n_cols or (cols.max() + 1)
    return csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(n_rows, n_cols))


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


def group_as_sequences(data, by=None, on=None, sequence_constructor=None, as_index=False):
    if isinstance(data, pd.Series):
        res = group_as_sequences(pd.DataFrame({"data": data}), by, sequence_constructor)
        return res["data"]
    if on is not None and not isinstance(on, (list, tuple)):
        on = [on]
    if by is None:
        by = list(set(data.columns) - set(on))
    data = data.set_index(by)
    groups = pd.Series(np.arange(len(data)), index=data.index)
    if by is None:
        sizes = aggregate_on(groups, on, observed=True).size()
        nulls = aggregate_on(groups, on, observed=True).first().isnull()
    else:
        sizes = groups.groupby(level=by, observed=True).size()
        nulls = groups.groupby(level=by, observed=True).first().isnull()

    # Gather the rows of the same group (ie all their index labels are the same except the
    # one of index_name, to be able to construct a partition using numpy split
    data = align_frame(data, sizes.index)

    # Compute the indices at which we should split to make our partition
    split_indices = np.cumsum(sizes.values)[:-1]

    # Split the sequences according to the previously computed partition indices
    result = {
        col: (
            [pd.Categorical.from_codes(subset, dtype=data[col].dtype) for subset in np.split(data[col].cat.codes.values, split_indices)]
            if hasattr(data[col], 'cat') else
            [pd.Series(subset, dtype=data[col].dtype) for subset in np.split(data[col].values, split_indices)])
        for col in data.columns
    }

    # Restore the sequences type used in the input
    if sequence_constructor is None:
        sequence_constructor = lambda x: x  # list

    result = {
        col_name: [(None if np.isscalar(a[0]) and pd.isnull(a[0])
                    else sequence_constructor([a[0]])) if isnull
                   else sequence_constructor(a)
                   for a, isnull in zip(col, nulls)]
        for col_name, col in result.items()
    }
    result = pd.DataFrame(result, index=sizes.index)
    if not as_index:
        result = result.reset_index()
    return result


unflatten = group_as_sequences


class InvisibleCounter:
    __slots__ = ['value']

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash(0)


def sizes_to_slices(sizes):
    cumsum = np.cumsum(sizes)
    return [slice(a, b) for a, b in zip((0, *cumsum[:-1]), cumsum)]


def replace_index_inplace(df, index):
    df.index = index
    return df


def factorize(self, group_nans=False, subset=slice(None), categories=None,
              return_categories=False,
              return_rows=False):
    """

    Parameters
    ----------
    self: pd.DataFrame or list of pd.DataFrame
    group_nans: ??
    subset: list of columns to factorize on
    categories: pd.Series of (tuple of any or any)
        Unique combinations of the column values allowed, (-1 will be given for the row otherwise)
    return_categories:
        Return the new categories if `categories` is None
    return_rows: pd.DataFrame
        Return the rows

    Returns
    -------
    any
    """

    if isinstance(subset, str):
        subset = [subset]
    if isinstance(self, list):
        concat = pd.concat([df[subset] for df in self], axis=0, ignore_index=True)
        sizes = [len(part) for part in self]
        res = factorize(concat, group_nans, subset=subset, categories=categories, return_categories=return_categories, return_rows=return_rows)
        slices = sizes_to_slices(sizes)
        if return_rows and return_categories:
            codes, rows, out_categories = res
            return (
                [replace_index_inplace(codes.iloc[s], part.index) for s, part in zip(slices, self)],
                [replace_index_inplace(rows.iloc[s], part.index) for s, part in zip(slices, self)],
                out_categories)
        elif return_rows:
            codes, rows = res
            return (
                [replace_index_inplace(codes.iloc[s], part.index) for s, part in zip(slices, self)],
                [replace_index_inplace(rows.iloc[s], part.index) for s, part in zip(slices, self)])
        elif return_categories:
            codes, out_categories = res
            return ([replace_index_inplace(codes.iloc[s], part.index) for s, part in zip(slices, self)],
                    out_categories)
        else:
            return [replace_index_inplace(res.iloc[s], part.index) for s, part in zip(slices, self)]
    if isinstance(self, pd.DataFrame):
        dedup_df = self[subset].copy()
        was_cat = {name: isinstance(dtype, pd.CategoricalDtype) for name, dtype in dedup_df.dtypes.items()}
        for name in dedup_df.columns:
            if hasattr(dedup_df[name], 'cat'):
                dedup_df[name] = dedup_df[name].cat.codes
        if not group_nans:
            for c_i, c_name in enumerate(dedup_df.columns):
                if was_cat:
                    col = dedup_df.iloc[:, c_i]
                    nulls = np.flatnonzero(col == -1)
                    if len(nulls):
                        dedup_df.iloc[nulls, c_i] = (-np.arange(len(nulls)) - 1).astype(int)
        if return_rows:
            items = pd.Series(fast_zip([*[dedup_df[c].values for c in dedup_df.columns],
                                        np.asarray([InvisibleCounter(i) for i in range(len(dedup_df))])]))
        else:
            items = pd.Series(fast_zip([dedup_df[c].values for c in dedup_df.columns]))
        items = replace_index_inplace(items.astype("category" if categories is None else pd.CategoricalDtype(categories=categories)), self.index)
        rows = out_categories = None
        if return_categories:
            if return_rows:
                out_categories = pd.Index(np.asarray([cat[:-1] for cat in items.cat.categories]))
            else:
                out_categories = items.cat.categories
        if return_rows:
            rows = self.iloc[[zipped[-1].value for zipped in items.cat.categories]]
        if return_rows and return_categories:
            return items.cat.codes, rows, out_categories
        elif return_rows:
            return items.cat.codes, rows
        elif return_categories:
            return items.cat.codes, out_categories
        else:
            return items.cat.codes
    elif isinstance(self, pd.Series):
        assert subset == slice(None)
        if not group_nans:
            dedup_series = self.copy().astype("category").cat.codes
            nulls = np.flatnonzero(dedup_series == -1)
            dedup_series.iloc[nulls] = np.arange(len(nulls)) + dedup_series.max() + 1
            indices, uniques = dedup_series.factorize()
            uniques = self.iloc[uniques]
        else:
            indices, uniques = self.factorize()
        return indices, pd.Series(uniques, name=self.name)
    else:
        raise NotImplementedError()


def normalize_vocabularies(dfs, vocabularies=None, train_vocabularies=True, unk=None, verbose=0):
    """
    Categorize the columns of the dataframes so that they share the same
    categories if they share the same columns
    If a column's name ends up with '_id', do not categorize it since it is no something we want to train on

    Parameters
    ----------
    dfs: list of pd.DataFrame
        DataFrame whose columns will be categorized
    vocabularies: dict
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
            print(f"Will train vocabulary for {col_name}")
    for df in dfs:
        for col_name in df:
            if hasattr(df[col_name], 'cat') and col_name not in vocabularies and not col_name.endswith('_id'):
                if verbose:
                    print(f"Discovered existing vocabulary ({len(df[col_name].cat.categories)} entities) for {col_name}")
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
            if not hasattr(voc, 'categories'):
                voc = pd.CategoricalDtype(voc)
            for df in dfs:
                if voc_name in df:
                    df[voc_name] = df[voc_name].astype(voc)
                    if verbose:
                        print(f"Normalized {voc_name}, with given vocabulary and unk {unk.get(voc_name, 'None')}")
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


class NLPAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def as_factorized(self, fn):
        # return the geographic center point of this DataFrame
        index, subset = factorize(self._obj)
        res = fn(subset)
        res = res.iloc[index]
        res.index = self._obj.index
        return res

    def factorize(self, group_nans=False, subset=slice(None), categories=None,
                  return_categories=False,
                  return_rows=False):
        return factorize(self._obj,
                         group_nans=group_nans,
                         subset=subset,
                         categories=categories,
                         return_categories=return_categories,
                         return_rows=return_rows)

    def unflatten(self, *args, **kwargs):
        return unflatten(self._obj, *args, **kwargs)

    def flatten(self, *args, **kwargs):
        return flatten(self._obj, *args, **kwargs)

    def aggregate_on(self, on, **kwargs):
        """
        Same as groupby but selected levels for grouping are all levels except `on`
        Parameters
        ----------
        frame: pd.DataFrame or pd.Series
        on: list of (str or int)
            Level not to group on
        Returns
        -------
        pd.DataFrame or pd.Series
        """
        if not isinstance(on, (tuple, list)):
            on = [on]
        levels = [i if n is None else n for i, n in enumerate(self._obj.columns) if n not in on]
        return self._obj.groupby(levels, **kwargs)

    def partition(self, large, overlap_policy="none", new_id_name="sample_id", span_policy="partial_strict"):
        from nlstruct.core.text import partition_spans

        return partition_spans([self._obj], large, overlap_policy=overlap_policy, new_id_name=new_id_name, span_policy=span_policy)[0][0]


pd.api.extensions.register_dataframe_accessor("nlp")(NLPAccessor)
pd.api.extensions.register_series_accessor("nlp")(NLPAccessor)
