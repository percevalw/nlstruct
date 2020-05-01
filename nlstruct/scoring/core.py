import numpy as np
import pandas as pd

from nlstruct.utils.pandas import merge_with_spans, make_merged_names


def merge_pred_and_gold(
      pred, gold,
      on=('doc_id', ('begin', 'end'), 'label'), span_policy='partial_strict',
      atom_pred_level=None, atom_gold_level=None, suffixes=('_pred', '_gold')):
    """
    Performs an outer merge between pred and gold that can be in 3 configurations:
    - (pred == nan, gold != nan) => pred_count = 0, gold_count = 1, tp = 0
    - (pred != nan, gold == nan) => pred_count = 1, gold_count = 0, tp = 0
    - (pred != nan, gold != nan) => pred_count = 1, gold_count = 1, tp = 1
    How the merge is done is by trying to merge on the columns given in "on" and using the span policy "policy"
    to merge spans

    Parameters
    ----------
    suffixes: tuple of str
    pred: pd.DataFrame
    gold: pd.DataFrame
    on: typing.Sequence of (str or tuple)
    span_policy: str
    atom_pred_level: (typing.Sequence of str) or str
    atom_gold_level: (typing.Sequence of str) or str

    Returns
    -------
    pd.DataFrame
    """
    delete_atom_pred_level = delete_atom_gold_level = False
    if isinstance(atom_pred_level, (list, tuple)):
        pred = pred.assign(_pred_id=pred[atom_pred_level].nlstruct.factorize())
        atom_pred_level = '_pred_id'
        delete_atom_pred_level = True
    elif atom_pred_level is None:
        pred = pred.assign(_pred_id=np.arange(len(pred)))
        atom_pred_level = '_pred_id'
        delete_atom_pred_level = True
    if isinstance(atom_gold_level, (list, tuple)):
        gold = gold.assign(_gold_id=gold[atom_gold_level].nlstruct.factorize())
        atom_gold_level = '_gold_id'
        delete_atom_gold_level = True
    elif atom_gold_level is None:
        gold = gold.assign(_gold_id=np.arange(len(gold)))
        atom_gold_level = '_gold_id'
        delete_atom_gold_level = True

    # pred_names, gold_names = make_merged_names(pred.columns, gold.columns, left_on=on, right_on=on,
    #                                           left_columns=pred.columns, right_columns=gold.columns)
    # pred_names_map = dict(zip(pred.columns, pred_names))
    # gold_names_map = dict(zip(gold.columns, gold_names))
    # categoricals = {}
    # for col in pred.columns:
    #     if hasattr(pred[col], 'cat'):
    #         categoricals[pred_names_map[col]] = pred[col].cat.categories
    #         pred[col] = pred[col].cat.codes
    # for col in gold.columns:
    #     if hasattr(gold[col], 'cat'):
    #         categoricals[gold_names_map[col]] = gold[col].cat.categories
    #         gold[col] = gold[col].cat.codes
    merged = merge_with_spans(pred, gold, on=on, how='inner', span_policy=span_policy, suffixes=suffixes)
    pred_new_names, gold_new_names = make_merged_names(
        pred.columns, gold.columns,
        left_on=on,
        right_on=on,
        left_columns=pred.columns, right_columns=gold.columns, suffixes=suffixes)
    pred_names_map = list(zip(pred.columns, pred_new_names))
    gold_names_map = list(zip(gold.columns, gold_new_names))

    overlap_size_names = [c for c in merged.columns if c.startswith("overlap_size_")]
    merged = merged.groupby([atom_pred_level, atom_gold_level], as_index=False, observed=True).agg({
        **{n: 'sum' for n in overlap_size_names},
        **{n: 'first' for n in merged.columns if n not in (*overlap_size_names, atom_pred_level, atom_gold_level)}})
    if overlap_size_names:
        merged = merged.sort_values(overlap_size_names)
    res = None
    if not len(merged):
        res = merged.iloc[:0]
    while len(merged):
        tmp = merged
        tmp = tmp.groupby(atom_gold_level, as_index=False, observed=True).last()
        tmp = tmp.groupby(atom_pred_level, as_index=False, observed=True).last()
        res = res.append(tmp) if res is not None else tmp
        merged = merged[np.logical_and(~merged[atom_pred_level].isin(res[atom_pred_level]),
                                       ~merged[atom_gold_level].isin(res[atom_gold_level]))]

    pred = pred.groupby([atom_pred_level], as_index=False, observed=True).last()
    gold = gold.groupby([atom_gold_level], as_index=False, observed=True).last()
    res = pd.concat((res,
                     pred[~pred[atom_pred_level].isin(res[atom_pred_level])][[old_col for old_col, new_col in pred_names_map if old_col in res or new_col in res]].rename(dict(pred_names_map), axis=1),
                     gold[~gold[atom_gold_level].isin(res[atom_gold_level])][[old_col for old_col, new_col in gold_names_map if old_col in res or new_col in res]].rename(dict(gold_names_map), axis=1)),
                     sort=False)
    # for col, categories in categoricals.items():
    #     res[col] = pd.Categorical.from_codes(res[col].fillna(-1).astype(int), categories=categories)

    res['pred_count'] = (~res[atom_pred_level].isnull()).astype(int)
    res['gold_count'] = (~res[atom_gold_level].isnull()).astype(int)
    res['tp'] = res['pred_count'] * res['gold_count']
    res['root'] = 0

    res = res.drop(columns=(
          ([atom_pred_level] if delete_atom_pred_level else []) +
          ([atom_gold_level] if delete_atom_gold_level else [])))

    return res


def compute_metrics(merged_results, level='root', prefix='', aggregate=True):
    """

    Parameters
    ----------
    merged_results: pd.DataFrame
        - pred_count
        - gold_count
        - tp
        - root
        Result of the merge_pred_and_gold function
    level: str
        Column describing the intermediate level to aggregate results if any

    Returns
    -------
    dict
    """
    res = merged_results.groupby([level], as_index=False)[['pred_count', 'gold_count', 'tp']].sum()
    res['recall'] = res['tp'] / res['gold_count']
    res['precision'] = res['tp'] / res['pred_count']
    res['f1'] = 2 / (1 / res['precision'] + 1 / res['recall'])

    # Recall is 0 where expected count = 0 and predicted count != 0
    null_r = res['gold_count'] == 0
    res.loc[null_r, ('recall', 'f1')] = 0
    # Precision is 0 where expected count = 0 and expected count != 0
    null_p = res['pred_count'] == 0
    res.loc[null_p, ('precision', 'f1')] = 0
    # Precision, recall & f1 is 1 where expected count = 0 and predicted count =  0
    null_tp = np.logical_and(null_p, null_r)
    res.loc[null_tp, ('precision', 'recall', 'f1')] = 1
    if aggregate:
        res = res.agg({'recall': 'mean', 'precision': 'mean', 'f1': 'mean',
                       'pred_count': 'sum', 'gold_count': 'sum', 'tp': 'sum'})
        res.index = [prefix+c for c in res.index]
    return res


def compute_confusion_matrix(pred, gold, n_labels=None, normalize="f1"):
    if hasattr(pred, 'cat'):
        if n_labels is None:
            n_labels = len(pred.cat.categories)
        pred = pred.cat.codes
    if hasattr(gold, 'cat'):
        if n_labels is None:
            n_labels = len(gold.cat.categories)
        gold = gold.cat.codes
    confusion = np.zeros((n_labels, n_labels), dtype=int)
    np.add.at(confusion, (gold, pred), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        total_pred = confusion.sum(1).reshape(-1, 1)
        total_gold = confusion.sum(0).reshape(1, -1)
        if normalize == "f1":
            recall = confusion / total_gold
            precision = confusion / total_pred
            confusion = 2 / (1 / recall + 1 / precision)
        elif normalize == "recall":
            confusion = confusion / total_gold
        elif normalize == "precision":
            confusion = confusion / total_pred
        else:
            assert normalize is None
    confusion[np.isnan(confusion)] = 0
    return confusion
