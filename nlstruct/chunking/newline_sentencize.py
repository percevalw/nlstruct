import re

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from nlstruct.core.cache import cached


def newline_sentencize_(docs, max_sentence_length=None, n_threads=1,
                        reg_split=r"((?:\s*\n)+\s*)",
                        reg_token=r"[\w*]+|[^\w\s\n*]",
                        with_tqdm=True):
    """
    Simple split MIMIC docs into sentences:
    - sentences bounds are found when multiple newline occurs
    - sentences too long are cut into `max_sentence_length` length sentences
      by splitting each sentence into the tokens using a dumb regexp.
    Parameters
    ----------
    docs: pd.DataFrame
    max_sentence_length: int
    with_tqdm: bool

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
    """
    n_threads = min(n_threads, len(docs))
    if n_threads > 1:
        text_chunks = np.array_split(np.arange(len(docs)), n_threads)
        pool = ProcessPool(nodes=n_threads)
        pool.restart(force=True)
        results = [pool.apipe(newline_sentencize_, docs.iloc[chunk], max_sentence_length, 1, with_tqdm=False) for chunk in text_chunks]
        results = [r.get() for r in results]
        pool.close()
        return pd.concat(results, ignore_index=True)

    reg_split = re.compile(reg_split)
    reg_token = re.compile(reg_token)
    doc_ids = []
    sentence_ids = []
    begins = []
    ends = []
    sentences = []
    for doc_id, txt in zip(docs["doc_id"], (tqdm(docs["text"], desc="Splitting docs into sentences") if with_tqdm else docs["text"])):
        idx = 0
        sentence_id = 0
        for i, part in enumerate(reg_split.split(txt)):
            if i % 2 == 0:  # we're in a sentence
                spans = [(m.start() + idx, m.end() + idx) for m in reg_token.finditer(part)]
                last = 0
                if max_sentence_length is None:
                    max_sentence_length_ = len(spans)
                else:
                    max_sentence_length_ = max_sentence_length
                for j in range(last + max_sentence_length_, len(spans) - max_sentence_length_, max(max_sentence_length_, 1)):
                    b = spans[last][0]
                    e = spans[j - 1][1]
                    doc_ids.append(doc_id)
                    sentence_ids.append(sentence_id)
                    begins.append(b)
                    ends.append(e)
                    sentences.append(txt[b:e])
                    last = j
                    sentence_id += 1
                if last < len(spans):
                    b = spans[last][0]
                    e = spans[-1][1]
                    doc_ids.append(doc_id)
                    sentence_ids.append(sentence_id)
                    begins.append(b)
                    ends.append(e)
                    sentences.append(txt[b:e])
                    sentence_id += 1
            idx += len(part)
    return pd.DataFrame({
        "doc_id": doc_ids,
        "sentence_id": sentence_ids,
        "begin": begins,
        "end": ends,
        "sentence": sentences,
    }).astype({"doc_id": docs["doc_id"].dtype})


@cached.will_ignore(("n_threads", "with_tqdm"))
def mimic_sentencize(texts, max_sentence_length=None, n_threads=1, with_tqdm=True):
    return newline_sentencize_(texts, max_sentence_length, reg_split=r"(\s*\n\s*\n\s*)", n_threads=n_threads, with_tqdm=with_tqdm)


@cached.will_ignore(("n_threads", "with_tqdm"))
def newline_sentencize(texts, max_sentence_length=None, n_threads=1, with_tqdm=True):
    return newline_sentencize_(texts, max_sentence_length, reg_split=r"((?:\s*\n){1,}\s*)", n_threads=n_threads, with_tqdm=with_tqdm)
