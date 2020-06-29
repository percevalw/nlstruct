import pandas as pd

from nlstruct.text.sub import apply_substitutions, reverse_deltas


def sentencepiece_tokenize(docs, path):
    subs = [
        (r"\s+", " "),
        ("^([^\s])", r" \1"),
    ]
    docs, deltas = apply_substitutions(docs, *zip(*subs), return_deltas=True)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    doc_ids = []
    tokens = []
    begins = []
    ends = []
    token_idx = []
    for doc_id, text in zip(docs["doc_id"], docs["text"]):
        token_id = 0
        i = 0

        # Begin of sentence
        doc_ids.append(doc_id)
        token_idx.append(token_id)
        begins.append(0)
        ends.append(0)
        tokens.append("<s>")
        token_id += 1

        # Sentence
        for piece in sp.EncodeAsPieces(text):
            doc_ids.append(doc_id)
            tokens.append(piece)
            begins.append(i + 1)
            i += len(piece)
            ends.append(i)
            token_idx.append(token_id)
            token_id += 1

        # End of sentence
        doc_ids.append(doc_id)
        begins.append(i)
        ends.append(i)
        tokens.append("</s>")
        token_idx.append(token_id)
        token_id += 1
    tokens = pd.DataFrame({"doc_id": doc_ids, "token_id": range(len(token_idx)), "token_idx": token_idx, "token": tokens, "begin": begins, "end": ends})
    token_voc = pd.CategoricalDtype(["<pad>", *(sp.IdToPiece(piece_id) for piece_id in range(sp.GetPieceSize()))])
    tokens = tokens.astype({"doc_id": docs["doc_id"].dtype, "token": token_voc})
    tokens = reverse_deltas(tokens, deltas, "doc_id")
    return tokens
