import pandas as pd


def make_spacy_model(need_pipelines=(), lang="en_core_web_sm"):
    import spacy
    nlp = spacy.load(lang, pipeline=need_pipelines)
    return nlp


SPACY_ATTRIBUTES = (
    'dep_',
    'ent_id_',
    'ent_iob_',
    'ent_kb_id_',
    'ent_type_',
    'is_alpha',
    'is_ascii',
    'is_bracket',
    'is_currency',
    'is_digit',
    'is_left_punct',
    'is_lower',
    'is_oov',
    'is_punct',
    'is_quote',
    'is_right_punct',
    'is_sent_start',
    'is_space',
    'is_stop',
    'is_title',
    'is_upper',
    'lang_',
    # 'lemma_',
    # 'lower_',
    'norm_',
    # 'orth_',
    'pos_',
    'prefix_',
    'shape_',
    'suffix_',
    'tag_',
    'whitespace_')


def spacy_tokenize(docs, spacy_model=None, spacy_attributes=SPACY_ATTRIBUTES, **spacy_args):
    if spacy_model is None:
        spacy_model = make_spacy_model(**spacy_args)
    doc_ids = []
    token_idx = []
    # tokens = []
    begins = []
    ends = []
    attributes = {("token_" + name).strip("_id_").strip("_"): [] for name in sorted(spacy_attributes)}
    for doc_id, doc in zip(docs["doc_id"], spacy_model.pipe(map(str, docs["text"]))):
        token_id = 0
        for token in doc:
            striped = str(token).strip(" ")
            if striped:
                doc_ids.append(doc_id)
                token_idx.append(token_id)
                begins.append(token.idx)
                ends.append(token.idx + len(token))
                # tokens.append(striped)
                for attr in spacy_attributes:
                    attributes[("token_" + attr).strip("_id_").strip("_")].append(getattr(token, attr))
                token_id += 1
    tokens = pd.DataFrame({"doc_id": doc_ids,
                           "token_id": range(len(token_idx)),
                           "begin": begins,
                           "end": ends,
                           "token_idx": token_idx,
                           **attributes})
    tokens = tokens.astype({"doc_id": docs["doc_id"].dtype})
    return tokens


def spacy_sentencize(docs, need_pipelines=('sentencizer',), max_token_length=200, **spacy_args):
    spacy_model = make_spacy_model(need_pipelines, **spacy_args)
    doc_ids = []
    sentence_idx = []
    sentences = []
    begins = []
    ends = []
    for doc_id, doc in zip(docs["doc_id"], spacy_model.pipe(map(str, docs["text"]))):
        sentence_id = 0
        doc_str = str(doc)
        for sent in doc.sents:
            if len(sent) < max_token_length:
                doc_ids.append(doc_id)
                sentence_idx.append(sentence_id)
                begins.append(sent[0].idx)
                ends.append(sent[-1].idx + len(sent[-1]))
                sentences.append(doc_str[begins[-1]:ends[-1]])
                sentence_id += 1
            else:
                last = 0
                for i in range(max_token_length, len(sent), max_token_length):
                    doc_ids.append(doc_id)
                    sentence_idx.append(sentence_id)
                    begins.append(sent[last].idx)
                    ends.append(sent[i - 1].idx + len(sent[i - 1]))
                    sentences.append(doc_str[begins[-1]:ends[-1]])
                    last = i
                    sentence_id += 1
                if last < max_token_length:
                    doc_ids.append(doc_id)
                    sentence_idx.append(sentence_id)
                    begins.append(sent[last].idx)
                    ends.append(sent[-1].idx + len(sent[-1]))
                    sentences.append(doc_str[begins[-1]:ends[-1]])
                    sentence_id += 1
    return pd.DataFrame({
        "doc_id": doc_ids,
        "sentence_idx": sentence_idx,
        "sentence_id": range(len(sentence_idx)),
        "begin": begins,
        "end": ends,
        "sentence": sentences,
    }).astype({"doc_id": docs["doc_id"].dtype})
