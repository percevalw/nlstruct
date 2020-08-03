import re
import warnings

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from nlstruct.utils.pandas import join_cols

MIMIC_SENTENCE_REGEX = r"(\s*\n\s*\n\s*)"
NEWLINE_SENTENCE_REGEX = r"((?:\s*\n)+\s*)"
TOKEN_REGEX = r"[\w*]+|[^\w\s\n*]"


def regex_sentencize(docs,
                     max_sentence_length=None,
                     min_sentence_length=None,
                     n_threads=1,
                     reg_split=NEWLINE_SENTENCE_REGEX,
                     reg_token=TOKEN_REGEX,
                     text_col="text",
                     balance_parentheses=True,
                     with_tqdm=False, **kwargs):
    """
    Simple split docs into sentences using regexes

    Parameters
    ----------
    docs: pd.DataFrame
    reg_split: str
    reg_token: str
    max_sentence_length: int
    min_sentence_length: int
    with_tqdm: bool
    balance_parentheses: bool
    text_col: str
    n_threads: int

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
    """
    n_threads = min(n_threads, len(docs))
    if n_threads > 1:
        text_chunks = np.array_split(np.arange(len(docs)), n_threads)
        pool = ProcessPool(nodes=n_threads)
        pool.restart(force=True)
        results = [pool.apipe(regex_sentencize, docs.iloc[chunk], max_sentence_length, 1, with_tqdm=False) for chunk in text_chunks]
        results = [r.get() for r in results]
        pool.close()
        return pd.concat(results, ignore_index=True)

    if "doc_id_col" in kwargs:
        warnings.warn("doc_id_col is not used anymore in the regex_sentencize function", FutureWarning)
        kwargs.pop("doc_id_col")
    doc_id_col = "id"
    while doc_id_col in docs.columns:
        doc_id_col += "_"
    docs = docs.assign(**{doc_id_col: np.arange(len(docs))})

    reg_split = re.compile(reg_split)
    reg_token = re.compile(reg_token)
    doc_ids = []
    sentence_idx_list = []
    begins = []
    ends = []
    sentences = []
    max_size = 0
    min_size = 10000000
    for doc_id, txt in zip(docs[doc_id_col], (tqdm(docs[text_col], desc="Splitting docs into sentences") if with_tqdm else docs[text_col])):
        idx = 0
        queued_spans = []
        sentence_idx = 0
        for i, part in enumerate(reg_split.split(txt)):
            if i % 2 == 0:  # we're in a sentence
                queued_spans.extend([(m.start() + idx, m.end() + idx) for m in reg_token.finditer(part)])
                if max_sentence_length is None:
                    max_sentence_length_ = len(queued_spans)
                else:
                    max_sentence_length_ = max_sentence_length
                while len(queued_spans) > max_sentence_length_:
                    b = queued_spans[0][0]
                    e = queued_spans[max_sentence_length_ - 1][1]
                    doc_ids.append(doc_id)
                    sentence_idx_list.append(sentence_idx)
                    begins.append(b)
                    ends.append(e)

                    max_size, min_size = max(max_size, max_sentence_length_), min(min_size, max_sentence_length_)
                    queued_spans = queued_spans[max_sentence_length_:]
                    sentences.append(txt[b:e])
                    sentence_idx += 1
                if min_sentence_length is not None and len(queued_spans) < min_sentence_length or (balance_parentheses and part.count("(") > part.count(")")):
                    idx += len(part)
                    continue
                if len(queued_spans):
                    b = queued_spans[0][0]
                    e = queued_spans[-1][1]
                    doc_ids.append(doc_id)
                    sentence_idx_list.append(sentence_idx)
                    begins.append(b)
                    ends.append(e)
                    max_size, min_size = max(max_size, len(queued_spans)), min(min_size, len(queued_spans))
                    queued_spans = []
                    sentences.append(txt[b:e])
                    sentence_idx += 1
            else:
                queued_spans.extend([(m.start() + idx, m.end() + idx) for m in reg_token.finditer(part)])
            if part is not None:
                idx += len(part)
    df = pd.DataFrame({
        doc_id_col: doc_ids,
        "sentence_idx": sentence_idx_list,
        "begin": begins,
        "end": ends,
        "text": sentences,
    }).astype({doc_id_col: docs[doc_id_col].dtype})
    df = df.merge(docs[[doc_id_col] + [col for col in docs.columns if col not in df.columns and col != "text"]])
    df["sentence_id"] = join_cols(df[[doc_id_col, "sentence_idx"]], "/")
    del df[doc_id_col]
    return df
