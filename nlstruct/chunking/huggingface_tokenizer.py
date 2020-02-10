import pandas as pd

from nlstruct.core.cache import cached


@cached
def huggingface_tokenize(docs, tokenizer):
    doc_ids = []
    tokens = []
    begins = []
    ends = []
    token_idx = []

    for doc_id, text in zip(docs["doc_id"], docs["text"]):
        token_id = 0
        i = 0

        # Sentence
        lookuptext = text.lower()
        for piece in tokenizer.convert_ids_to_tokens(tokenizer.encode(text)):
            doc_ids.append(doc_id)
            tokens.append(piece)
            piece_size = len(piece) - (4 if (piece.endswith("</s>") or piece.endswith("</w>")) else 0)
            delta = lookuptext[i:].find(piece[:min(piece_size, 1)].lower())
            assert delta < 50, (lookuptext[i:i + 50], piece[:min(piece_size, 1)].lower())
            i += delta
            begins.append(i)
            i += piece_size
            ends.append(i)
            token_idx.append(token_id)
            token_id += 1
    tokens = pd.DataFrame({"doc_id": doc_ids, "token_id": range(len(token_idx)), "token_idx": token_idx, "token": tokens, "begin": begins, "end": ends})
    voc = [None] * len(tokenizer.encoder)
    for token, token_i in tokenizer.encoder.items():
        voc[token_i] = token
    token_voc = pd.CategoricalDtype(voc)
    tokens = tokens.astype({"doc_id": docs["doc_id"].dtype, "token": token_voc})
    return tokens
