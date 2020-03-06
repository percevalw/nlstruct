import re

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from nlstruct.core.cache import cached


def newline_sentencize_(docs, max_sentence_length=None, min_sentence_length=None, n_threads=1,
                        reg_split=r"((?:\s*\n)+\s*)",
                        reg_token=r"[\w*]+|[^\w\s\n*]",
                        with_tqdm=False, verbose=0):
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
    verbose: int

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
    max_size = 0
    min_size = 10000000
    for doc_id, txt in zip(docs["doc_id"], (tqdm(docs["text"], desc="Splitting docs into sentences") if with_tqdm else docs["text"])):
        idx = 0
        queued_spans = []
        sentence_id = 0
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
                    sentence_ids.append(sentence_id)
                    begins.append(b)
                    ends.append(e)
                    max_size, min_size = max(max_size, max_sentence_length_), min(min_size, max_sentence_length_)
                    queued_spans = queued_spans[max_sentence_length_-1:]
                    sentences.append(txt[b:e])
                    sentence_id += 1
                if min_sentence_length is not None and len(queued_spans) < min_sentence_length:
                    idx += len(part)
                    continue
                if len(queued_spans):
                    b = queued_spans[0][0]
                    e = queued_spans[-1][1]
                    doc_ids.append(doc_id)
                    sentence_ids.append(sentence_id)
                    begins.append(b)
                    ends.append(e)
                    max_size, min_size = max(max_size, len(queued_spans)), min(min_size, len(queued_spans))
                    queued_spans = []
                    sentences.append(txt[b:e])
                    sentence_id += 1
            idx += len(part)
    if verbose:
        print("Sentence size: max = {}, min = {}".format(max_size, min_size))
    return pd.DataFrame({
        "doc_id": doc_ids,
        "sentence_id": sentence_ids,
        "begin": begins,
        "end": ends,
        "sentence": sentences,
    }).astype({"doc_id": docs["doc_id"].dtype})


@cached.will_ignore(("n_threads", "with_tqdm"))
def mimic_sentencize(texts, max_sentence_length=None, min_sentence_length=None, n_threads=1, with_tqdm=False, verbose=0):
    return newline_sentencize_(texts, max_sentence_length, min_sentence_length, reg_split=r"(\s*\n\s*\n\s*)", n_threads=n_threads, with_tqdm=with_tqdm, verbose=verbose)


@cached.will_ignore(("n_threads", "with_tqdm"))
def newline_sentencize(texts, max_sentence_length=None, min_sentence_length=None, n_threads=1, with_tqdm=False, verbose=0):
    return newline_sentencize_(texts, max_sentence_length, min_sentence_length, reg_split=r"((?:\s*\n){1,}\s*)", n_threads=n_threads, with_tqdm=with_tqdm, verbose=verbose)
