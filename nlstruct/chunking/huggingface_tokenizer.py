import pandas as pd
import unidecode
from tqdm import tqdm


def huggingface_tokenize(docs, tokenizer, with_tqdm=False, **kwargs):
    doc_ids = []
    tokens = []
    begins = []
    ends = []
    token_idx = []
    special_tokens = [t for token in tokenizer.special_tokens_map.values() for t in ((token,) if isinstance(token, str) else token)]
    special_tokens += ["▁", "##", "</w>"]
    for doc_id, text in tqdm(zip(docs["doc_id"], docs["text"]), disable=not with_tqdm, total=len(docs), leave=False, desc="Tokenizing"):
        i = 0
        token_id = 0

        lookuptext = unidecode.unidecode(text.lower())
        for piece in tokenizer.convert_ids_to_tokens(tokenizer.encode(text, **kwargs)):
            doc_ids.append(doc_id)
            tokens.append(piece)
            striped_piece = piece
            for special in special_tokens:
                striped_piece = striped_piece.replace(special, "")
            piece_size = len(striped_piece)
            delta = lookuptext[i:].find(unidecode.unidecode(striped_piece.lower()))
            assert 0 <= (delta - lookuptext[i:i + delta].count(' ') - lookuptext[i:i + delta].count('\n')) < 5, (lookuptext[i:i + 50], striped_piece.lower())
            i += delta
            begins.append(i)
            i += piece_size
            ends.append(i)
            token_idx.append(token_id)
            token_id += 1
    tokens = pd.DataFrame({"doc_id": doc_ids, "token_id": range(len(token_idx)), "token_idx": token_idx, "token": tokens, "begin": begins, "end": ends})
    voc = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
    counts = {}
    for i, token in enumerate(list(voc)):
        counts[token] = counts.get(token, 0) + 1
        if counts[token] > 1:
            voc[i] = token + "-{}".format(i)
    token_voc = pd.CategoricalDtype(voc)
    tokens = tokens.astype({"doc_id": docs["doc_id"].dtype, "token": token_voc})
    return tokens
