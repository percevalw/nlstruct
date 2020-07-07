import re
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm


def huggingface_tokenize(docs, tokenizer, with_tqdm=False, with_token_spans=True, text_col="text", **kwargs):
    if "doc_id_col" in kwargs:
        warnings.warn("doc_id_col is not used anymore in the huggingface_tokenize function", FutureWarning)
        kwargs.pop("doc_id_col")
    doc_id_col = "id"
    while doc_id_col in docs.columns:
        doc_id_col += "_"

    doc_ids = []
    tokens = []
    begins = []
    ends = []
    token_idx = []
    docs = docs.copy()
    docs[doc_id_col] = np.arange(len(docs))

    if with_token_spans:
        special_tokens = [t for token in tokenizer.special_tokens_map.values() for t in ((token,) if isinstance(token, str) else token)]
        special_tokens += ["▁", "##", "</w>"]
        for doc_id, text in tqdm(zip(docs[doc_id_col], docs[text_col]), disable=not with_tqdm, total=len(docs), leave=True, desc="Tokenizing"):
            i = 0
            token_id = 0

            sentence_pieces = tokenizer.tokenize(text)
            tokenizer_output = tokenizer.encode_plus(tokenizer.convert_tokens_to_ids(sentence_pieces), return_special_tokens_mask=True, **kwargs)
            encoded_pieces = tokenizer.convert_ids_to_tokens(tokenizer_output["input_ids"])
            pieces = np.asarray(encoded_pieces)
            pieces[~np.asarray(tokenizer_output["special_tokens_mask"], dtype=bool)] = sentence_pieces
            for piece, encoded_piece in zip(pieces, encoded_pieces):
                doc_ids.append(doc_id)
                tokens.append(encoded_piece)
                striped_piece = piece
                for special in special_tokens:
                    striped_piece = striped_piece.replace(special, "")
                piece_size = len(striped_piece)
                delta = len(re.search(r"^\s*", text[i:]).group(0))
                if striped_piece.lower() != text[i+delta:i+delta + piece_size].lower():
                    raise Exception(f"During processing of doc {doc_id}, wordpiece tokenizer replaced {repr(text[i+delta:i+delta + piece_size])} (in {repr(text[i:i+delta + piece_size + 5])}) "
                                    f"with {repr(striped_piece)} (or multiple pieces). "
                                    f"You must perform substitutions before to ensure that this does not happen, otherwise wordpieces characters cannot be computed.")
                i += delta
                begins.append(i)
                i += piece_size
                ends.append(i)
                token_idx.append(token_id)
                token_id += 1
        tokens = pd.DataFrame({doc_id_col: doc_ids, "token_id": range(len(token_idx)), "token_idx": token_idx, "token": tokens, "begin": begins, "end": ends})
    else:
        for doc_id, text in tqdm(zip(docs[doc_id_col], docs[text_col]), disable=not with_tqdm, total=len(docs), leave=False, desc="Tokenizing"):
            token_id = 0
            for encoded_piece in tokenizer.convert_ids_to_tokens(tokenizer.encode(text, **kwargs)):
                doc_ids.append(doc_id)
                tokens.append(encoded_piece)
                token_idx.append(token_id)
                token_id += 1
        tokens = pd.DataFrame({doc_id_col: doc_ids, "token_id": range(len(token_idx)), "token_idx": token_idx, "token": tokens})

    voc = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
    counts = {}
    for i, token in enumerate(list(voc)):
        counts[token] = counts.get(token, 0) + 1
        if counts[token] > 1:
            voc[i] = token + "-{}".format(i)
    token_voc = pd.CategoricalDtype(voc)
    tokens = tokens.astype({"token": token_voc})
    tokens = tokens.merge(docs[[doc_id_col] + [col for col in docs.columns if col not in tokens.columns and col != "text"]])
    del docs[doc_id_col]
    return tokens
