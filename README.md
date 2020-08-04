## Main features
#### Pandas-based preprocessing
Most of the input or infered data can be expressed as a DataFrame of features, ids and span indices.

This library therefore takes advantage of pandas advanced frame indexing and combining features to make the preprocessing fast and explicit.

#### Easy nested/relational batching
In structuring problems (and for text data all the more), features can be highly relational. 

This library introduces a flexible, yet performant batching structure that allows to switch between numpy, scipy and torch matrices and easily split relational data.

#### Caching
Every procedure can be easily cached using and explicit, flexible and performant caching mecanism. Smart parameter hashing functions have been written to handle numpy, pandas and torch (cuda/cpu) data structures and models seamlessly, and return unique hashes across multiple machines.

This caching mecanism is useful for checkpointing models, restarting trainings from a given epoch, and instantly preprocessing often used data. Log saving and replay during cache loading is also possible.

#### Other features
- relative shareable paths
- split and transform texts, record the transformations and apply them to spans or reverse them on predictions
- "many" dataloaders to facilitate access to nlp datasets and improve reproducibility
- training helpers to seamlessly restart trainings from the last checkpoint if any
- random seed helpers to ensure reproducibility (handles builtin, numpy and torch random functions)
- colored pretty tables to monitor training
- multiple tokenizers (spacy, transformers, regex)
- brat/conll exporters
- linear tagging CRFs that automatically handle mention spans <-> token tags conversion

## Example

The following shows an example of a custom preprocessing of the NCBI dataset. 

We split documents into sentences, apply substitutions to the text and transform
mentions spans accordingly. Finally, we create a batching structure to iterate through documents and automatically query relevant sentences, tokens and mentions.

```python
>>> from nlstruct.dataloaders import load_ncbi_disease
>>> from nlstruct.utils import normalize_vocabularies, encode_ids, df_to_csr, assign_sorted_id
>>> from nlstruct.text import huggingface_tokenize, split_into_spans, regex_sentencize, apply_substitutions, apply_deltas, partition_spans
>>> from nlstruct.collections import Batcher
>>> from transformers import AutoTokenizer

>>> ncbi = load_ncbi_disease()
>>> docs, mentions, fragments = ncbi[["docs", "mentions", "fragments"]]
>>> mentions = mentions.merge(fragments)
>>> print(ncbi)
Dataset(
  (docs):       792 * ('doc_id', 'text', 'split')
  (mentions):  6881 * ('doc_id', 'mention_id', 'category')
  (labels):    7059 * ('label_id', 'doc_id', 'mention_id', 'label')
  (fragments): 6881 * ('doc_id', 'mention_id', 'begin', 'end', 'fragment_id')
)

>>> sentences = regex_sentencize(docs, reg_split='((?<=\.)[\n ](?:[A-Z]))')
>>> [mentions] = partition_spans([mentions], sentences, overlap_policy=False)[0]
>>> print(sentences.head(5))
   sentence_idx  begin  end                           text    doc_id  split sentence_id
0             0      0   77  A common human skin tumour...  10192393  train         0/0
1             1     78  141  WNT signalling orchestrate...  10192393  train         0/1
2             2    142  312  In response to this stimul...  10192393  train         0/2
3             3    313  477  One of the target genes fo...  10192393  train         0/3
4             4    478  742  Most colon cancers arise f...  10192393  train         0/4
>>> print(mentions.head(5))
     doc_id sentence_id  mention_id fragment_id         category  begin  end
0  10192393         0/0  10192393-0  10192393-0     DiseaseClass     15   26
1  10192393         0/3  10192393-1  10192393-1     DiseaseClass    130  136
2  10192393         0/4  10192393-2  10192393-2     DiseaseClass      5   18
3  10192393         0/4  10192393-3  10192393-3  SpecificDisease     61   87
4  10192393         0/4  10192393-4  10192393-4  SpecificDisease     89   92

>>> # Get substitutions `deltas` to shift train mentions and restored true char positions on predictions
>>> sentences, deltas = apply_substitutions(sentences, [r"sk(.)n"], [r"sk\1\1\1\1\1\1n"], doc_cols=("doc_id", "sentence_id"), apply_unidecode=True)
>>> mentions = apply_deltas(mentions, deltas, on="sentence_id")
>>> print(sentences.head(5))
                            text  sentence_idx    doc_id  split sentence_id
0  A common human skiiiiiin t...             0  10192393  train         0/0
1  WNT signalling orchestrate...             1  10192393  train         0/1
2  In response to this stimul...             2  10192393  train         0/2
3  One of the target genes fo...             3  10192393  train         0/3
4  Most colon cancers arise f...             4  10192393  train         0/4
>>> print(mentions.head(5))
     doc_id sentence_id  mention_id fragment_id         category  begin    end
0  10192393         0/0  10192393-0  10192393-0     DiseaseClass   15.0   31.0  <-- notice that the end has moved due to the substitution
1  10192393         0/3  10192393-1  10192393-1     DiseaseClass  130.0  136.0
2  10192393         0/4  10192393-2  10192393-2     DiseaseClass    5.0   18.0
3  10192393         0/4  10192393-3  10192393-3  SpecificDisease   61.0   87.0
4  10192393         0/4  10192393-4  10192393-4  SpecificDisease   89.0   92.0

>>> tokens = huggingface_tokenize(sentences, AutoTokenizer.from_pretrained('camembert-base'))
>>> # Express mentions as token spans instead of char spans
>>> mentions = split_into_spans(mentions, tokens, pos_col="token_idx")
>>> mentions = assign_sorted_id(mentions, "mention_idx", groupby=["doc_id", "sentence_id"], sort_on="begin")
>>> print(tokens.head(5))
   id  token_id  token_idx  token  begin  end  sentence_idx    doc_id  split sentence_id
0   0         0          0    <s>      0    0             0  10192393  train         0/0
1   0         1          1     ▁A      0    1             0  10192393  train         0/0
2   0         2          2  ▁comm      2    6             0  10192393  train         0/0
3   0         3          3     on      6    8             0  10192393  train         0/0
4   0         4          4      ▁      9    9             0  10192393  train         0/0
>>> print(mentions.head(5))
     doc_id sentence_id  mention_id fragment_id         category  begin  end  mention_idx
0  10094559       234/1  10094559-1  10094559-1  SpecificDisease      1    7            0
1   3258663       548/1   3258663-2   3258663-2  SpecificDisease      1   10            0
2  10633128       465/1  10633128-1  10633128-1  SpecificDisease      1    7            0
3   8252631       268/7  8252631-11  8252631-11         Modifier      1    4            0
4   7437512       417/0   7437512-0   7437512-0  SpecificDisease      1    9            0

>>> # Encode object / strings etc that is not an id as a pandas categories
>>> [sentences, tokens, mentions], vocabularies = normalize_vocabularies([sentences, tokens, mentions], train_vocabularies={"text": False})

>>> # Encode doc/sentence/mention ids as integers
>>> unique_mention_ids = encode_ids([mentions], ("doc_id", "mention_id"), inplace=True)
>>> unique_sentence_ids = encode_ids([sentences, mentions, tokens], ("doc_id", "sentence_id"), inplace=True)
>>> unique_doc_ids = encode_ids([docs, sentences, mentions, tokens], "doc_id", inplace=True)

>>> # Create the batcher collection
>>> batcher = Batcher({
>>>     "doc": {
>>>         "doc_id": docs["doc_id"],
>>>         "sentence_id": df_to_csr(sentences["doc_id"], sentences["sentence_idx"], sentences["sentence_id"]),
>>>         "sentence_mask": df_to_csr(sentences["doc_id"], sentences["sentence_idx"]),
>>>     },
>>>     "sentence": {
>>>         "sentence_id": sentences["sentence_id"],
>>>         "token": df_to_csr(tokens["sentence_id"], tokens["token_idx"], tokens["token"].cat.codes),
>>>         "token_mask": df_to_csr(tokens["sentence_id"], tokens["token_idx"]),
>>>         "mention_id": df_to_csr(mentions["sentence_id"], mentions["mention_idx"], mentions["mention_id"]),
>>>         "mention_mask": df_to_csr(mentions["sentence_id"], mentions["mention_idx"]),
>>>     },
>>>     "mention": {
>>>         "mention_id": mentions["mention_id"],
>>>         "begin": mentions["begin"],
>>>         "end": mentions["end"],
>>>         "category": mentions["category"].cat.codes,
>>>     },
>>> }, masks={"sentence": {"mention_id": "mention_mask", "token": "token_mask"}, "doc": {"sentence_id": "sentence_mask"}})
>>> print(batcher)
Batcher(
  [doc]:
    (doc_id): ndarray[int64](792,)
    (sentence_id): csr_matrix[int64](792, 44)
    (sentence_mask): csr_matrix[bool](792, 44)
  [sentence]:
    (sentence_id): ndarray[int64](6957,)
    (token): csr_matrix[int16](6957, 211)
    (token_mask): csr_matrix[bool](6957, 211)
    (mention_id): csr_matrix[int64](6957, 13)
    (mention_mask): csr_matrix[bool](6957, 13)
  [mention]:
    (mention_id): ndarray[int64](6881,)
    (begin): ndarray[int64](6881,)
    (end): ndarray[int64](6881,)
    (category): ndarray[int8](6881,)
)

>>> # Query some documents and convert them to torch
>>> batch = batcher["doc"][[3,4,5]].densify(torch.device('cpu'))
>>> print(batch)
Batcher(
  [doc]:
    (doc_id): tensor[torch.int64](3,)
    (sentence_id): tensor[torch.int64](3, 9)
    (sentence_mask): tensor[torch.bool](3, 9)
    (@sentence_id): tensor[torch.int64](3, 9) <-- indices relative to the batch have been created
    (@sentence_mask): tensor[torch.bool](3, 9)
  [sentence]:
    (sentence_id): tensor[torch.int64](22,)
    (token): tensor[torch.int64](22, 74)
    (token_mask): tensor[torch.bool](22, 74) <-- token tensor has been resized to remove excess pad tokens
    (mention_id): tensor[torch.int64](22, 3)
    (mention_mask): tensor[torch.bool](22, 3)
    (@mention_id): tensor[torch.int64](22, 3)
    (@mention_mask): tensor[torch.bool](22, 3)
  [mention]:
    (mention_id): tensor[torch.int64](22,)
    (begin): tensor[torch.int64](22,)
    (end): tensor[torch.int64](22,)
    (category): tensor[torch.int64](22,)
)

>>> # Easily access tensors in the batch
>>> print(batch["sentence", "token"].shape)
torch.Size([22, 74])
```
## Install

This project is still under development and subject to changes.

```bash
pip install nlstruct
```
