import torch
from unittest import TestCase

from transformers import AutoTokenizer

from nlstruct.collections import Dataset, Batcher
from nlstruct.dataloaders import (
    load_ncbi_disease,
    load_alt_medic_mapping
)
from nlstruct.environment import hash_object
from nlstruct.text import apply_substitutions, apply_deltas, regex_sentencize, partition_spans, huggingface_tokenize, split_into_spans
from nlstruct.train import seed_all
from nlstruct.utils import assign_sorted_id, encode_ids, df_to_csr, normalize_vocabularies


def preprocess_corpus(dataset, max_sentence_length, tokenizer, vocabularies=None, do_lowercase=True):
    if vocabularies is None:
        vocabularies = {}
    # Check that there is no mention overlap
    docs, mentions, fragments, labels = dataset[["docs", "mentions", "fragments", "labels"]].copy()
    mentions = mentions.merge(fragments.groupby(["doc_id", "mention_id"], as_index=False, observed=True).agg({"begin": "min", "end": "max"}))

    # Extract text
    docs, deltas = apply_substitutions(
        docs, *zip(
            ("(?<=[a-zA-Z])(?=[0-9])", r" "),
            ("(?<=[0-9])(?=[A-Za-z])", r" "),
            ("\s*(?:\n\s*)+", "\n"),
        ), apply_unidecode=True)
    mentions = apply_deltas(mentions, deltas, on=['doc_id'])

    mentions = assign_sorted_id(mentions, "mention_idx_in_doc", "doc_id", sort_on="begin")
    sentences = regex_sentencize(docs,
                                 reg_split=r"((?<=[.])\s*\n+|(?<=[a-z0-9][.]) (?=[A-Z]))",
                                 min_sentence_length=0, max_sentence_length=max_sentence_length)
    [mentions], sentences, sentence_to_docs = partition_spans([mentions], sentences, new_id_name="sentence_id", overlap_policy=False)
    sentences = assign_sorted_id(sentences, "sentence_idx_in_doc", "doc_id", "begin").sort_values(["doc_id", "sentence_idx"])
    mentions = assign_sorted_id(mentions, "mention_idx_in_sentence", ["doc_id", "sentence_id"], "begin")
    labels = assign_sorted_id(labels, "label_idx_in_mention", ["doc_id", "mention_id"], "label")

    if do_lowercase:
        sentences["text"] = sentences["text"].str.lower()
    tokens = huggingface_tokenize(sentences, tokenizer)
    mentions = split_into_spans(mentions, tokens, pos_col="token_idx", overlap_policy=False)

    mention_tokens = huggingface_tokenize(mentions, tokenizer)

    [tokens, mentions, sentences, docs, mention_tokens, labels], vocs = normalize_vocabularies(
        [tokens, mentions, sentences, docs, mention_tokens, labels],
        train_vocabularies={**{"text": False, }, **({"label": True} if "label" not in vocabularies else {})},
        vocabularies={**(vocabularies or {}), "split": ["train", "dev", "test"], "label": vocabularies.get("label", ["END"])},
        verbose=True)
    mentions = mentions.sort_values(["doc_id", "mention_idx_in_doc"])
    sentences = sentences.sort_values(["doc_id", "sentence_idx"])

    prep = Dataset(
        docs=docs,
        sentences=sentences,
        tokens=tokens,
        labels=labels,
        mentions=mentions,
        mention_tokens=mention_tokens,
        deltas=deltas,
    ).copy()

    mention_ids = encode_ids([mentions, mention_tokens, labels], ("doc_id", "mention_id"))
    sentence_ids = encode_ids([sentences, mentions, tokens, mention_tokens], ("doc_id", "sentence_id"))
    doc_ids = encode_ids([docs, sentences, mentions, tokens, mention_tokens, labels], ("doc_id"))

    batcher = Batcher({
        "doc": {
            "doc_id": docs["doc_id"],
            "split": docs["split"].cat.codes,
            "sentence_id": df_to_csr(sentences["doc_id"], sentences["sentence_idx"], sentences["sentence_id"], n_rows=len(doc_ids)),
            "sentence_mask": df_to_csr(sentences["doc_id"], sentences["sentence_idx"], n_rows=len(doc_ids)),
            "mention_id": df_to_csr(mentions["doc_id"], mentions["mention_idx_in_doc"], mentions["mention_id"], n_rows=len(doc_ids)),
            "mention_mask": df_to_csr(mentions["doc_id"], mentions["mention_idx_in_doc"], n_rows=len(doc_ids)),
        },
        "sentence": {
            "sentence_id": sentences["sentence_id"],
            "mention_id": df_to_csr(mentions["sentence_id"], mentions["mention_idx_in_sentence"], mentions["mention_id"], n_rows=len(sentence_ids)),
            "mention_mask": df_to_csr(mentions["sentence_id"], mentions["mention_idx_in_sentence"], n_rows=len(sentence_ids)),
            "doc_id": sentences["doc_id"],
            "token": df_to_csr(tokens["sentence_id"], tokens["token_idx"], tokens["token"].cat.codes, n_rows=len(sentence_ids)),
            "token_mask": df_to_csr(tokens["sentence_id"], tokens["token_idx"], n_rows=len(sentence_ids)),
            # "token_tag": df_to_csr(tokens["sentence_id"], tokens["token_idx"], tokens["tag"].cat.codes, n_rows=len(sentence_ids)),
        },
        "mention": {
            "mention_id": mentions["mention_id"],
            "mention_idx_in_sentence": mentions["mention_idx_in_sentence"],
            "mention_idx_in_doc": mentions["mention_idx_in_doc"],
            "sentence_id": mentions["sentence_id"],
            "doc_id": mentions["doc_id"],
            "begin": mentions["begin"],
            "end": mentions["end"],
            # "label": mentions["label"].cat.codes,

            "label": df_to_csr(labels["mention_id"], labels["label_idx_in_mention"], labels["label"].cat.codes, n_rows=len(mention_ids)),
            "label_mask": df_to_csr(labels["mention_id"], labels["label_idx_in_mention"], n_rows=len(mention_ids)),

            "token": df_to_csr(mention_tokens["mention_id"], mention_tokens["token_idx"], mention_tokens["token"].cat.codes, n_rows=len(mention_ids)),
            "token_mask": df_to_csr(mention_tokens["mention_id"], mention_tokens["token_idx"], n_rows=len(mention_ids)),
        },
    },
        masks={
            "sentence": {
                "token": "token_mask",
                "mention_id": "mention_mask",
            },
            "doc": {
                "sentence_id": "sentence_mask",
                "mention_id": "mention_mask",
            },
            "mention": {
                "token": "token_mask",
                "label": "label_mask",
            }
        },
        join_order=["doc", "sentence", "mention"])

    encoded_prep = {
        "docs": docs,
        "sentences": sentences,
        "tokens": tokens,
        "mentions": mentions,
        "labels": labels,
    }
    ids = {
        "mention_ids": mention_ids,
        "sentence_ids": sentence_ids,
        "doc_ids": doc_ids,
    }
    return batcher, prep, encoded_prep, ids, vocs


class DataloadersTest(TestCase):
    def test_preprocessing(self):
        bert_name = "allenai/scibert_scivocab_uncased"
        dataset = load_ncbi_disease()
        label_mapping = load_alt_medic_mapping("06-07-12")
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        batcher, prep, encoded_prep, ids, vocs = preprocess_corpus(
            dataset,
            max_sentence_length=140,
            tokenizer=tokenizer,
            do_lowercase=True,
        )
        self.assertEqual(str(batcher), """Batcher(
  [doc]:
    (doc_id): ndarray[int64](792,)
    (split): ndarray[int8](792,)
    (sentence_id): csr_matrix[int64](792, 17)
    (sentence_mask): csr_matrix[bool](792, 17)
    (mention_id): csr_matrix[int64](792, 30)
    (mention_mask): csr_matrix[bool](792, 30)
  [sentence]:
    (sentence_id): ndarray[int64](6091,)
    (mention_id): csr_matrix[int64](6091, 14)
    (mention_mask): csr_matrix[bool](6091, 14)
    (doc_id): ndarray[int64](6091,)
    (token): csr_matrix[int16](6091, 166)
    (token_mask): csr_matrix[bool](6091, 166)
  [mention]:
    (mention_id): ndarray[int64](6881,)
    (mention_idx_in_sentence): ndarray[int64](6881,)
    (mention_idx_in_doc): ndarray[int64](6881,)
    (sentence_id): ndarray[int64](6881,)
    (doc_id): ndarray[int64](6881,)
    (begin): ndarray[int64](6881,)
    (end): ndarray[int64](6881,)
    (label): csr_matrix[int16](6881, 5)
    (label_mask): csr_matrix[bool](6881, 5)
    (token): csr_matrix[int16](6881, 26)
    (token_mask): csr_matrix[bool](6881, 26)
)""")
        self.assertEqual(hash_object(batcher), "ceb7f44e99ab7ac0")
        self.assertEqual(hash_object(prep), "971717677b37cb84")

        seed_all(42)
        batches = list(batcher["sentence"].dataloader(batch_size=32, shuffle=True, device=torch.device('cpu'), sort_on="token_mask", keys_noise=2.))
        self.assertEqual(len(batches), 191)
        self.assertEqual(str(batches[0]), """Batcher(
  [sentence]:
    (sentence_id): tensor[torch.int64](32,)
    (mention_id): tensor[torch.int64](32, 3)
    (mention_mask): tensor[torch.bool](32, 3)
    (doc_id): tensor[torch.int64](32,)
    (token): tensor[torch.int64](32, 23)
    (token_mask): tensor[torch.bool](32, 23)
    (@mention_id): tensor[torch.int64](32, 3)
    (@mention_mask): tensor[torch.bool](32, 3)
    (@doc_id): tensor[torch.int64](32,)
  [mention]:
    (mention_id): tensor[torch.int64](17,)
    (mention_idx_in_sentence): tensor[torch.int64](17,)
    (mention_idx_in_doc): tensor[torch.int64](17,)
    (sentence_id): tensor[torch.int64](17,)
    (doc_id): tensor[torch.int64](17,)
    (begin): tensor[torch.int64](17,)
    (end): tensor[torch.int64](17,)
    (label): tensor[torch.int64](17, 1)
    (label_mask): tensor[torch.bool](17, 1)
    (token): tensor[torch.int64](17, 13)
    (token_mask): tensor[torch.bool](17, 13)
    (@sentence_id): tensor[torch.int64](17,)
    (@doc_id): tensor[torch.int64](17,)
  [doc]:
    (doc_id): tensor[torch.int64](32,)
    (split): tensor[torch.int64](32,)
    (sentence_id): tensor[torch.int64](32, 13)
    (sentence_mask): tensor[torch.bool](32, 13)
    (mention_id): tensor[torch.int64](32, 24)
    (mention_mask): tensor[torch.bool](32, 24)
    (@sentence_id): tensor[torch.int64](32, 13)
    (@sentence_mask): tensor[torch.bool](32, 13)
    (@mention_id): tensor[torch.int64](32, 24)
    (@mention_mask): tensor[torch.bool](32, 24)
)""")
        self.assertEqual(str(batches[-1]), """Batcher(
  [sentence]:
    (sentence_id): tensor[torch.int64](32,)
    (mention_id): tensor[torch.int64](32, 4)
    (mention_mask): tensor[torch.bool](32, 4)
    (doc_id): tensor[torch.int64](32,)
    (token): tensor[torch.int64](32, 29)
    (token_mask): tensor[torch.bool](32, 29)
    (@mention_id): tensor[torch.int64](32, 4)
    (@mention_mask): tensor[torch.bool](32, 4)
    (@doc_id): tensor[torch.int64](32,)
  [mention]:
    (mention_id): tensor[torch.int64](25,)
    (mention_idx_in_sentence): tensor[torch.int64](25,)
    (mention_idx_in_doc): tensor[torch.int64](25,)
    (sentence_id): tensor[torch.int64](25,)
    (doc_id): tensor[torch.int64](25,)
    (begin): tensor[torch.int64](25,)
    (end): tensor[torch.int64](25,)
    (label): tensor[torch.int64](25, 2)
    (label_mask): tensor[torch.bool](25, 2)
    (token): tensor[torch.int64](25, 13)
    (token_mask): tensor[torch.bool](25, 13)
    (@sentence_id): tensor[torch.int64](25,)
    (@doc_id): tensor[torch.int64](25,)
  [doc]:
    (doc_id): tensor[torch.int64](32,)
    (split): tensor[torch.int64](32,)
    (sentence_id): tensor[torch.int64](32, 14)
    (sentence_mask): tensor[torch.bool](32, 14)
    (mention_id): tensor[torch.int64](32, 25)
    (mention_mask): tensor[torch.bool](32, 25)
    (@sentence_id): tensor[torch.int64](32, 14)
    (@sentence_mask): tensor[torch.bool](32, 14)
    (@mention_id): tensor[torch.int64](32, 25)
    (@mention_mask): tensor[torch.bool](32, 25)
)""")
