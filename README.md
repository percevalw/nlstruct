# NLStruct

Natural language struturing library.
Currently, it implements only a NER model, but other algorithms will follow.

### Features

- processes large documents seamlessly: it automatically handles tokenization and sentence splitting.
- do not train twice: an automatic caching mechanism detects when an experiment has already been run
- stop & resume with checkpoints
- easy import and export of data
- handles nested or overlapping entities
- pretty logging with [rich_logger](https://github.com/percevalw/rich_logger)
- heavily customizable, without config files (see [train_ner.py](https://github.com/percevalw/nlstruct/blob/nlstruct/recipes/train_ner.py))
- built on top of [transformers](https://github.com/huggingface/transformers) and [pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning)

### How to train a NER model

```python
from nlstruct.recipes import train_ner

model = train_ner(
    dataset={
        "train": "path to your train brat/standoff data",
        "val": 0.05,  # or path to your validation data
        # "test": # and optional path to your test data
    },
    finetune_bert=False,
    seed=42,
    bert_name="camembert/camembert-base",
    fasttext_file="",
    gpus=0,
    xp_name="my-xp",
)
model.save_pretrained("ner.pt")
```

### How to use it

```python
from nlstruct import load_pretrained
from nlstruct.datasets import load_from_brat, export_to_brat

ner = load_pretrained("ner.pt")
export_to_brat(ner.predict(load_from_brat("path/to/brat/test")), filename_prefix="path/to/exported_brat")
```

## Install

This project is still under development and subject to changes.

```bash
pip install nlstruct
```
