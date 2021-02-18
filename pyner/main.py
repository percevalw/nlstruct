import glob

import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
import torch
import transformers
from optuna.integration import PyTorchLightningPruningCallback

from .data_utils import *
from .datasets import load_from_brat
from .modules import Vocabulary, NER


def objective(trial):
    val_ids = ["filepdf-277-cas", "filepdf-176-cas", "filepdf-830-cas", "filepdf-509-2-cas", "filepdf-57-cas", "filepdf-533-1-cas", "filepdf-32-2-cas", "filepdf-728-cas", "filepdf-781-cas",
               "filepdf-119-cas"]
    task_data = list(load_from_brat(glob.glob("/export/home/cse190022/data/resources/deft_2020/t3-appr/*.txt")))
    train_data = [sample for sample in task_data if sample["doc_id"] not in val_ids]
    val_data = [sample for sample in task_data if sample["doc_id"] in val_ids]
    train_data = [
        {**doc, "entities": [entity for entity in doc["entities"] if entity["label"] not in ("duree", "frequence")]}
        for doc in train_data
    ]
    val_data = [
        {**doc, "entities": [entity for entity in doc["entities"] if entity["label"] not in ("duree", "frequence")]}
        for doc in val_data
    ]

    bert_name = "camembert/camembert-large"
    vocabularies = torch.nn.ModuleDict({
        "char": Vocabulary(string.punctuation + string.ascii_letters + string.digits, with_unk=True, with_pad=True),
        "label": Vocabulary(sorted(set([entity["label"] for doc in train_data for entity in doc["entities"]])), with_unk=False, with_pad=False),
    }).eval()
    ner = NER(
        seed=random.randint(1, 1000),
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
                module="char_cnn",
                n_chars=len(vocabularies["char"].values),
                in_channels=8,
                out_channels=50,
                kernel_sizes=(3, 4, 5),
            ),
            dict(
                module="bert",
                path=bert_name,
                n_layers=4,
                freeze_n_layers=0,  # freeze all
                dropout_p=0.10,
            )
        ],
        decoder=dict(
            module="exhaustive_biaffine_ner",
            dim=trial.suggest_int("decoder/dim", 64, 384, 32),
            label_dim=trial.suggest_int("decoder/label_dim", 32, 256, 16),
            n_labels=len(vocabularies["label"].values),
            dropout_p=trial.suggest_float("decoder/dropout_p", 0., 0.4, step=0.05),
            use_batch_norm=False,
            contextualizer=dict(
                module="lstm",
                gate=dict(
                    module="sigmoid_gate",
                    ln_mode=trial.suggest_categorical("decoder/contextualizer/gate/ln_mode", [False, "pre", "post"]),
                    init_value=0,
                    proj=False,
                    dim=trial.params["decoder/dim"],
                ),
                input_size=1024 + 150,
                hidden_size=trial.params["decoder/dim"],
                num_layers=trial.suggest_int("decoder/contextualizer/num_layers", 1, 6),
                dropout_p=trial.suggest_float("decoder/contextualizer/dropout_p", 0., 0.4, step=0.05),
            )
        ),

        init_labels_bias=True,
        batch_size=24,
        use_lr_schedules=True,
        gradient_clip_val=5.,
        main_lr=trial.suggest_float("main_lr", 1e-4, 1e-2, log=True),
        top_lr=trial.params["main_lr"],
        bert_lr=4e-5,
        warmup_rate=0.1,
        optimizer=transformers.AdamW,
    )
    trainer = pl.Trainer(
        gpus=1,
        progress_bar_refresh_rate=False,
        logger=[
            pl.loggers.NeptuneLogger(api_key="MY KEY", name="better-ner-hp-deft-v2"),
        ],
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_f1")],
        max_epochs=50)
    trainer.fit(ner, train_data, val_data)

    return trainer.callback_metrics["val_f1"].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    pl.seed_everything(42)
    study.optimize(objective, n_trials=100, gc_after_trial=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
