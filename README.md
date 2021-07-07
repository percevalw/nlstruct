# pyner

Named entity recognition (nested or not) in Python

### How to train

```python
from pyner import InformationExtractor
from pyner.datasets import BRATDataset
import string
import pytorch_lightning as pl
from rich_logger import RichTableLogger

bert_name = "camembert/camembert-base"
dataset = BRATDataset(
    train="path/to/brat/train",
    test="path/to/brat/test",    # None for training only, test directory otherwise
    val=0.2,  # first 20% doc will be for validation
    seed=False,  # don't shuffle before splitting
)
model = InformationExtractor(
    seed=42,
    preprocessor=dict(
        module="ner_preprocessor",
        bert_name=bert_name, # transformer name
        bert_lower=False,
        split_into_multiple_samples=True,
        sentence_split_regex=r"\n", # regex to use to split sentences (must not contain consuming patterns)
        sentence_balance_chars=('()',), # try to avoid splitting between parentheses
        sentence_entity_overlap="split", # raise when an entity spans more than one sentence, or use "split" to split entities in 2 when this happens
        word_regex=r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]', # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
        substitutions=( # Apply these regex substitutions on sentences before tokenizing
            (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "), # insert a space before punctuations
            (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "), # insert a space after punctuations
            #("(?<=[a-zA-Z])(?=[0-9])", r" "), # insert a space between letters and numbers
            #("(?<=[0-9])(?=[A-Za-z])", r" "), # insert a space between numbers and letters
        ),
        max_tokens=200,         # Maximum number of bert tokens in a sentence (will split if more than this number)
                                # Must be equal to or lower than the max number of tokens in the Bert model
        min_tokens=50,  # Minimum number of tokens in a sentence
        join_small_sentence_rate=0.9,  # How frequently do we join two sentences that are shorter than the max number of tokens
        large_sentences="equal-split", # for these large sentences, split them in equal sub sentences < max_tokens tokens 
        empty_entities="raise", # when an entity cannot be mapped to any word, "raise" or "drop"
        multi_label=False,
        filter_entities=None,
        vocabularies={ # vocabularies to use, call .train() before initializing to fill/complete them automatically from training data
            "entity_label": dict(module="vocabulary", values=sorted(dataset.labels()), with_unk=False, with_pad=False),
            #"char": dict(module="vocabulary", values=string.ascii_letters+string.digits+string.punctuation, with_unk=True, with_pad=False),
        },
    ),
    dynamic_preprocessing=False,

    # Text encoders
    encoder=dict(
        module="concat",
        encoders=[
            dict(
                module="bert",
                path=bert_name,
                n_layers=4,
                freeze_n_layers=0, # freeze 0 layer (including the first embedding layer)
                dropout_p=0.1,
                token_dropout_p=0.,
                output_lm_embeds=False,
                combine_mode="softmax",
                word_pooler=dict(module="pooler", mode="mean"),
            ),
            #dict(
            #    module="char_cnn",
            #    in_channels=8,
            #    out_channels=50,
            #    kernel_sizes=(3, 4, 5),
            #),
        ]
    ),
    decoder=dict(
        module="contiguous_entity_decoder",
        contextualizer=dict(
            module="lstm",
            num_layers=3,
            hidden_size=256,
            gate=dict(module="residual_gate", init_value=1., ln_mode="post"),
            bidirectional=True,
            dropout_p=0.4,
            keep_cell_state=False,
            gate_reference="input",
        ),
        span_scorer=dict(
            module="bitag",
            max_fragments_count=200,
            max_length=40,
            hidden_size=64,
            do_biaffine=True,
            do_viterbi_filtering=True,
            threshold=0.5,
            do_tagging=True,
            do_length=False,
            do_norm=False,
            do_density=False,
            allow_overlap=True,
            learn_bounds=True,
            do_tag_bounds=True,
            share_bounds=False,
            learnable_transitions=False,
            positive_tag_only_loss=False,
            tag_loss_weight=0.2,
            eps=1e-10,
        ),
        intermediate_loss_slice=slice(-1, None),
    ),

    batch_size=32,

    # Use learning rate schedules (linearly decay with warmup)
    use_lr_schedules=True,
    warmup_rate=0.1,

    gradient_clip_val=50.,

    # Learning rates
    main_lr=8e-5,
    fast_lr=1e-3,
    bert_lr=5e-5,

    # Optimizer, can be class or str
    optimizer_cls="transformers.AdamW",
    metrics={
        "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1.),
        # Example of two metrics with a subset of all the entities
        "3_1": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., filter_entities=['anatomie', 'date', 'dose', 'examen', 'mode', 'moment', 'substance', 'traitement', 'valeur']),
        "3_2": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., filter_entities=['pathologie', 'sosy']),
        "span": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=0.),
        "label": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1.),
    },
).train()

trainer = pl.Trainer(
    gpus=1,
    progress_bar_refresh_rate=False,
    checkpoint_callback=False, # do not make checkpoints since it slows down the training a lot
    logger=[
        #        pl.loggers.TestTubeLogger("path/to/logs", name="my_experiment"),
        RichTableLogger(key="epoch", fields={
            "epoch": {},
            "step": {},
            
            "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.4f}"},
            "(.*)_precision": False,#{"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_p"},
            "(.*)_recall": False,#{"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_r"},
            "(.*)_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_f1"},
        
            ".*_lr": {"format": "{:.2e}"},
            "duration": {"format": "{:.0f}", "name": "dur(s)"},
        }),
    ],
    val_check_interval=200,
    max_steps=1000)
trainer.fit(model, dataset)
model.save_pretrained("ner.pt")
```

### How to use

```python
from pyner import load_pretrained
from pyner.datasets import load_from_brat, export_to_brat

ner = load_pretrained("ner.pt")
export_to_brat(ner.predict(load_from_brat("path/to/brat/test")), filename_prefix="path/to/exported_brat")
```

### How to search hyperparameters

```python
from pyner import InformationExtractor
from pyner.datasets import BRATDataset
import string
import torch
import pytorch_lightning as pl
import gc
import optuna

dataset = BRATDataset(
    train="/path/to/brat/train/",
    test=None,
    val=0.2,
    seed=False,  # do not shuffle for val split, just take the first 20% docs
)

def objective(trial):
    bert_name = "camembert/camembert-base"
    
    model = InformationExtractor(
        seed=42,
        preprocessor=dict(
            module="ner_preprocessor",
            bert_name=bert_name, # transformer name
            bert_lower=False,
            split_into_multiple_samples=True,
            sentence_split_regex=r"\n", # regex to use to split sentences (must not contain consuming patterns)
            sentence_balance_chars=('()',), # try to avoid splitting between parentheses
            sentence_entity_overlap="raise", # raise when an entity spans more than one sentence, or use "split" to split entities in 2 when this happens
            word_regex=r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]', # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
            substitutions=( # Apply these regex substitutions on sentences before tokenizing
                (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "), # insert a space before punctuations
                (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "), # insert a space after punctuations
                #("(?<=[a-zA-Z])(?=[0-9])", r" "), # insert a space between letters and numbers
                #("(?<=[0-9])(?=[A-Za-z])", r" "), # insert a space between numbers and letters
            ),
            max_tokens=512,         # Maximum number of tokens in a sentence (will split if more than this number)
                                    # Must be equal to or lower than the max number of tokens in the Bert model
            min_tokens=50,  # Minimum number of tokens in a sentence
            join_small_sentence_rate=0.9,  # How frequently do we join two sentences that are shorter than the max number of tokens
            large_sentences="equal-split", # for these large sentences, split them in equal sub sentences < max_tokens tokens 
            empty_entities="raise", # when an entity cannot be mapped to any word, "raise" or "drop"
            multi_label=False,
            filter_entities=None,
            vocabularies={ # vocabularies to use, call .train() before initializing to fill/complete them automatically from training data
                "entity_label": dict(module="vocabulary", values=sorted(dataset.labels()), with_unk=False, with_pad=False),
                #"char": dict(module="vocabulary", values=string.ascii_letters+string.digits+string.punctuation, with_unk=True, with_pad=False),
            },
        ),
        dynamic_preprocessing=True,
    
        # Text encoders
        encoder=dict(
            module="concat",
            encoders=[
                dict(
                    module="bert",
                    path=bert_name,
                    n_layers=4,
                    freeze_n_layers=0, # freeze 0 layer (including the first embedding layer)
                    dropout_p=0.1,
                    token_dropout_p=0.,
                    output_lm_embeds=False,
                    combine_mode="softmax",
                    word_pooler=dict(module="pooler", mode="mean"),
                ),
                #dict(
                #    module="char_cnn",
                #    in_channels=8,
                #    out_channels=50,
                #    kernel_sizes=(3, 4, 5),
                #),
            ]
        ),
        decoder=dict(
            module="contiguous_entity_decoder",
            contextualizer=dict(
                module="lstm",
                #  v ------------- Try a range of hyperparameters ------------- v 
                num_layers=trial.suggest_int("decoder/contextualizer/num_layers", 1, 6),
                hidden_size=256,
                gate=dict(module="residual_gate", init_value=1., ln_mode="post"),
                bidirectional=True,
                dropout_p=0.4,
                keep_cell_state=False,
                gate_reference="input",
            ),
            span_scorer=dict(
                module="bitag",
                keep_fragments_threshold=False,
                max_fragments_count=200,
                max_length=40,
                hidden_size=64,
                do_biaffine=True,
                do_viterbi_filtering=True,
                threshold=0.5,
                do_tagging=True,
                do_length=False,
                do_norm=False,
                do_density=False,
                allow_overlap=True,
                learn_bounds=True,
                do_tag_bounds=True,
                share_bounds=False,
                learnable_transitions=False,
                positive_tag_only_loss=False,
                tag_loss_weight=0.2,
                eps=1e-10,
            ),
            intermediate_loss_slice=slice(-1, None),
        ),
    
        batch_size=32,
    
        # Use learning rate schedules (linearly decay with warmup)
        use_lr_schedules=True,
        warmup_rate=0.1,
    
        gradient_clip_val=50.,
    
        # Learning rates
        main_lr=8e-5,
        fast_lr=1e-3,
        bert_lr=5e-5,
    
        # Optimizer, can be class or str
        optimizer_cls="transformers.AdamW",
        metrics={
            "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1.),
        },
    )

    trainer = pl.Trainer(
        gpus=1,
        progress_bar_refresh_rate=False,
        move_metrics_to_cpu=True,
        logger=[
            pl.loggers.TestTubeLogger("tensorboard_data", name="pyner-brat"),
        ],
        max_epochs=50)

    print(trial.params)
    trainer.fit(model, dataset)

    res = trainer.callback_metrics["val_f1"].item()
    print("=>", res)

    # Clean cuda memory after trial because optuna does not
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return res


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.PercentilePruner(75, n_startup_trials=5, n_warmup_steps=10))
pl.seed_everything(42)
study.optimize(objective, n_trials=100, gc_after_trial=True)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```
