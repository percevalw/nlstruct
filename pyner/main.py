import glob

import pytorch_lightning as pl
import torch
import transformers

from rich_logger import RichTableLogger
from .data_utils import *
from .datasets import load_from_brat
from .modules import Vocabulary, NER

if __name__ == "__main__":
    val_ids = ["filepdf-277-cas", "filepdf-176-cas", "filepdf-830-cas", "filepdf-509-2-cas", "filepdf-57-cas", "filepdf-533-1-cas", "filepdf-32-2-cas", "filepdf-728-cas", "filepdf-781-cas",
               "filepdf-119-cas"]
    task_data = list(load_from_brat(glob.glob("data/resources/deft_2020/t3-appr/*.txt")))
    train_data = [sample for sample in task_data if sample["doc_id"] not in val_ids]
    val_data = [sample for sample in task_data if sample["doc_id"] in val_ids]

    pl.utilities.seed.seed_everything(42)
    bert_name = "data/resources/huggingface/pretrained_models/camembert-large/"
    vocabularies = torch.nn.ModuleDict({
        "char": Vocabulary(string.punctuation + string.ascii_letters + string.digits, with_unk=True, with_pad=True),
        "label": Vocabulary(sorted(set([mention["label"] for doc in task_data for mention in doc["mentions"]])), with_unk=False, with_pad=False),
    }).eval()
    ner = NER(
        sentence_split_regex=r"((?:\s*\n)+\s*|(?:(?<=[a-z0-9)]\.)\s+))(?=[A-Z])",
        sentence_balance_chars=("()",),
        preprocessor=dict(
            bert_name=bert_name,
            vocabularies=vocabularies,
            word_regex='[\\w\']+|[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',
            substitutions=(
                (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "),
                (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "),
                # ("(?<=[a-zA-Z])(?=[0-9])", r" "),
                # ("(?<=[0-9])(?=[A-Za-z])", r" "),
            )
        ),

        use_embedding_batch_norm=True,
        word_encoders=[
            dict(
                name="char_cnn",
                n_chars=len(vocabularies["char"].values),
                in_channels=8,
                out_channels=50,
                kernel_sizes=(3, 4, 5),
            ),
            dict(
                name="bert",
                path=bert_name,
                n_layers=4,
                freeze_n_layers=-1,  # freeze all
                dropout=0.20,
            )
        ],
        decoder=dict(
            name="exhaustive_biaffine",
            dim=200,
            label_dim=100,
            n_labels=len(vocabularies["label"].values),
            dropout=0.2,
            use_batch_norm=False,
            contextualizer=dict(
                name="lstm",
                gate=False,
                input_size=1024 + 150,
                hidden_size=200,
                num_layers=4,
                dropout=0.2,
            ),
        ),

        init_labels_bias=True,

        batch_size=24,
        use_lr_schedules=True,
        top_lr=5e-3,
        main_lr=5e-3,
        bert_lr=4e-5,
        warmup_rate=0.1,
        optimizer=transformers.AdamW,
    )

    trainer = pl.Trainer(gpus=[0], progress_bar_refresh_rate=False, logger=RichTableLogger(key="epoch", fields={
        "epoch": {},
        "step": {},
        "train_loss": {"goal": "lower_is_better", "format": "{:.4f}"},
        "train_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_f1"},
        "train_precision": {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_p"},
        "train_recall": {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_r"},

        "val_loss": {"goal": "lower_is_better", "format": "{:.4f}"},
        "val_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_f1"},
        "val_precision": {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_p"},
        "val_recall": {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_r"},

        "main_lr": {"format": "{:.2e}"},
        "top_lr": {"format": "{:.2e}"},
        "bert_lr": {"format": "{:.2e}"},
    }), max_epochs=50)
    trainer.fit(ner, train_data, val_data)
