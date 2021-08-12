"""
from main import main

seeds = (42, 1558, 555, 123, 456, 789)

for dataset in ("deft",): # or deft:dev
    for seed in seeds:
        main(dataset, seed=seed, do_tagging="full", do_biaffine=True, finetune_bert=True)
        main(dataset, seed=seed, do_tagging="full", do_biaffine=True, finetune_bert=False)

"""

import argparse
import gc
import json
import string

import pandas as pd

from pyner.base import *
from pyner.checkpoint import *
from pyner.datasets import *
from rich_logger import RichTableLogger

if "display" not in globals():
    display = print

shared_cache = {}

BASE_WORD_REGEX = r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]'
BASE_SENTENCE_REGEX = r"((?:\s*\n)+\s*|(?:(?<=[\w0-9]{2,}\.|[)]\.)\s+))(?=[[:upper:]]|•|\n)"


def main(
      dataset_name,
      seed,
      do_char=True,
      do_biaffine=True,
      do_tagging="full",
      doc_context=True,
      finetune_bert=False,
      bert_lower=False,
      n_bert_layers=4,
      biaffine_size=None,
      bert_proj_size=None,
      biaffine_loss_weight=1.,
      hidden_size=400,
      max_steps=None,
      resources="",
      bert_name=None,
      fasttext_file=None,  # set to "" to disable
      unique_label=False,
      norm_bert=False,
      dropout_p=0.1,
      batch_size=32,
      lr=1e-3,
      use_lr_schedules=True,
      word_pooler_mode="mean",
      bert_size=None,
      hf_resources="",
      predict_kwargs={},
      gpus=1,
):
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    if max_steps is None:
        if "dev" in dataset_name:
            if finetune_bert:
                max_steps = 2000
            else:
                max_steps = 10000
        else:
            if finetune_bert:
                max_steps = 4000
            else:
                max_steps = 20000

    if biaffine_size is None:
        if dataset_name.split(":")[0] in ("genia_easy", "genia_hard", "conll"):
            biaffine_size = 150
        else:
            biaffine_size = 64

    if bert_size is None:
        if "dev" in dataset_name:
            bert_size = "base"
        else:
            bert_size = "large"

    if bert_name is None:
        if dataset_name.split(":")[0] in ("deft", "ezmammo"):
            if bert_size == "large":
                bert_name = os.path.join(hf_resources, "camembert/camembert-large")
            else:
                bert_name = os.path.join(hf_resources, "camembert/camembert-base")
        elif "genia" in dataset_name:
            if bert_size == "large":
                bert_name = os.path.join(hf_resources, "dmis-lab/biobert-large-cased-v1.1")
            else:
                bert_name = os.path.join(hf_resources, "dmis-lab/biobert-base-cased-v1.1")
        else:
            if bert_size == "large":
                bert_name = os.path.join(hf_resources, "bert-large-cased")
            else:
                bert_name = os.path.join(hf_resources, "bert-base-cased")

    if fasttext_file is None:
        if dataset_name.split(":")[0] in ("conll", "genia_easy", "genia_hard"):
            fasttext_file = os.path.join(resources, "cc.en.300.vec.filtered")
        elif dataset_name.split(":")[0] in ("deft", "ezmammo"):
            fasttext_file = os.path.join(resources, "cc.fr.300.vec.filtered")
    for name, value in locals().items():
        print(name.ljust(40), value)
    # bert_name = "/export/home/opt/data/camembert/v0/camembert-base/"

    filter_predictions = False
    if dataset_name.split(":")[0] == "genia_hard":
        filter_predictions = "no_crossing_same_label"
        dataset = GENIA(os.path.join(resources, "genia_ner"), test_split=0.10004, val_split=0.1, merge_composite_types=True)
        word_regex = BASE_WORD_REGEX
        sentence_split_regex = r"\s*\n\s*"
        metrics = {
            "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., word_regex=word_regex),
            "half_word": dict(module="dem", binarize_tag_threshold=0.5, binarize_label_threshold=1., word_regex=word_regex),
            "any_word": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., word_regex=word_regex),
        }
    elif dataset_name.split(":")[0] == "conll":
        dataset = BRATDataset(
            train=os.path.join(resources, "brat/conll-2003/train"),
            val=os.path.join(resources, "brat/conll-2003/val"),
            test=os.path.join(resources, "brat/conll-2003/test"),
        )
        word_regex = r'[^\s]+'
        sentence_split_regex = r"\s*\n\s*"
        filter_predictions = "no_overlapping"
        metrics = {
            "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., word_regex=word_regex),
            "half_word": dict(module="dem", binarize_tag_threshold=0.5, binarize_label_threshold=1., word_regex=word_regex),
            "any_word": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., word_regex=word_regex),
        }
    elif dataset_name.split(":")[0] == "deft":
        dataset = DEFT(
            os.path.join(resources, "deft_2020/"),
            val=0.2,
            dropped_entity_label=('duree', 'frequence', 'date'),
            seed=seed if ":dev" in dataset_name else False,
        )
        word_regex = BASE_WORD_REGEX
        sentence_split_regex = BASE_SENTENCE_REGEX

        metrics = {
            "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., word_regex=word_regex),
            "3_1": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., filter_entities=['anatomie', 'dose', 'examen', 'mode', 'moment', 'substance', 'traitement', 'valeur'],
                        word_regex=word_regex),
            "3_2": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., filter_entities=['pathologie', 'sosy'], word_regex=word_regex),
            "half_word": dict(module="dem", binarize_tag_threshold=0.5, binarize_label_threshold=1., word_regex=word_regex),
            "any_word": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., word_regex=word_regex),
        }
    elif dataset_name.split(":")[0] == "ezmammo":
        dataset = BRATDataset(
            train=os.path.join(resources, "brat/ezmammo-v3/train"),
            val=0.2,
            test=os.path.join(resources, "brat/ezmammo-v3//test"),
            seed=seed if ":dev" in dataset_name else False,
        )
        word_regex = BASE_WORD_REGEX
        sentence_split_regex = BASE_SENTENCE_REGEX
        metrics = {
            "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., word_regex=word_regex),
            "half_word": dict(module="dem", binarize_tag_threshold=0.5, binarize_label_threshold=1., word_regex=word_regex),
            "any_word": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., word_regex=word_regex),
        }

    else:
        raise Exception("Unrecognized dataset {}".format(dataset_name))

    if "filter_predictions" not in predict_kwargs and filter_predictions is not False:
        predict_kwargs["filter_predictions"] = filter_predictions

    if ":dev" not in dataset_name:
        dataset_name = dataset_name.split(":")[0]
        dataset.train_data = dataset.train_data + dataset.val_data
        dataset.val_data = dataset.test_data

    if unique_label:
        for split in (dataset.train_data, dataset.val_data, dataset.test_data):
            if split:
                for doc in split:
                    for e in doc["entities"]:
                        e["label"] = "main"

    display(dataset.describe())

    model = InformationExtractor(
        seed=seed,
        preprocessor=dict(
            module="ner_preprocessor",
            bert_name=bert_name,  # transformer name
            bert_lower=bert_lower,
            split_into_multiple_samples=True,
            sentence_split_regex=sentence_split_regex,  # regex to use to split sentences (must not contain consuming patterns)
            sentence_balance_chars=(),  # try to avoid splitting between parentheses
            sentence_entity_overlap="split",  # raise when an entity spans more than one sentence
            word_regex=word_regex,  # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
            substitutions=(),  # Apply these regex substitutions on sentences before tokenizing
            keep_bert_special_tokens=False,
            min_tokens=0,
            doc_context=doc_context,
            join_small_sentence_rate=0.,
            max_tokens=256,  # split when sentences contain more than 512 tokens
            large_sentences="equal-split",  # for these large sentences, split them in equal sub sentences < 512 tokens
            empty_entities="raise",  # when an entity cannot be mapped to any word, raise
            vocabularies={
                **{  # vocabularies to use, call .train() before initializing to fill/complete them automatically from training data
                    "entity_label": dict(module="vocabulary", values=sorted(dataset.labels()), with_unk=False, with_pad=False),
                },
                **({
                       "char": dict(module="vocabulary", values=string.ascii_letters + string.digits + string.punctuation, with_unk=True, with_pad=False),
                   } if do_char else {})
            },
            fragment_label_is_entity_label=True,
            multi_label=False,
            filter_entities=None,  # "entity_type_score_density", "entity_type_score_lesion"),
        ),
        dynamic_preprocessing=False,

        # Text encoders
        encoder=dict(
            module="concat",
            dropout_p=0.5,
            encoders=[
                dict(
                    module="bert",
                    path=bert_name,
                    n_layers=n_bert_layers,
                    freeze_n_layers=0 if finetune_bert is not False else -1,  # freeze 0 layer (including the first embedding layer)
                    bert_dropout_p=None if finetune_bert else 0.,
                    token_dropout_p=0.,
                    proj_size=bert_proj_size,
                    output_lm_embeds=False,
                    combine_mode="scaled_softmax" if not norm_bert else "softmax",
                    do_norm=norm_bert,
                    do_cache=not finetune_bert,
                    word_pooler=dict(module="pooler", mode=word_pooler_mode),
                ),
                *([dict(
                    module="char_cnn",
                    in_channels=8,
                    out_channels=50,
                    kernel_sizes=(3, 4, 5),
                )] if do_char else []),
                *([dict(
                    module="word_embeddings",
                    filename=fasttext_file,
                )] if fasttext_file else [])
            ],
        ),
        decoder=dict(
            module="contiguous_entity_decoder",
            contextualizer=dict(
                module="lstm",
                num_layers=3,
                gate=dict(module="sigmoid_gate", init_value=0., proj=True),
                bidirectional=True,
                hidden_size=hidden_size,
                dropout_p=0.4,
                gate_reference="last",
            ),
            span_scorer=dict(
                module="bitag",
                do_biaffine=do_biaffine,
                do_tagging=do_tagging,
                do_length=False,

                threshold=0.5,
                max_fragments_count=200,
                max_length=40,
                hidden_size=biaffine_size,
                allow_overlap=True,
                dropout_p=dropout_p,
                tag_loss_weight=1.,
                biaffine_loss_weight=biaffine_loss_weight,
                eps=1e-14,
            ),
            intermediate_loss_slice=slice(-1, None),
        ),

        _predict_kwargs=predict_kwargs,
        batch_size=batch_size,

        # Use learning rate schedules (linearly decay with warmup)
        use_lr_schedules=use_lr_schedules,
        warmup_rate=0.1,

        gradient_clip_val=10.,
        _size_factor=0.001,

        # Learning rates
        main_lr=lr,
        fast_lr=lr,
        bert_lr=5e-5,

        # Optimizer, can be class or str
        optimizer_cls="transformers.AdamW",
        metrics=metrics,
    ).train()

    model.encoder.encoders[0].cache = shared_cache

    try:
        trainer = pl.Trainer(
            gpus=gpus,
            progress_bar_refresh_rate=False,
            checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
            callbacks=[ModelCheckpoint(path='checkpoints/' + dataset_name + '-{hashkey}-{global_step:05d}')],
            logger=[
                #        pl.loggers.TestTubeLogger("path/to/logs", name="my_experiment"),
                RichTableLogger(key="epoch", fields={
                    "epoch": {},
                    "step": {},

                    "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.4f}"},
                    "(.*)_precision": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_p"},
                    "(.*)_recall": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_r"},
                    "(.*)_tp": False,
                    "(.*)_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_f1"},

                    ".*_lr|max_grad": {"format": "{:.2e}"},
                    "duration": {"format": "{:.0f}", "name": "dur(s)"},
                }),
            ],
            val_check_interval=max_steps // 10,
            max_steps=max_steps)
        trainer.fit(model, dataset)
        trainer.logger[0].finalize(True)

        result_output_filename = "checkpoints/" + dataset_name + "-{}.json".format(trainer.callbacks[0].hashkey)
        if not os.path.exists(result_output_filename):
            model.cuda();
            results = model.metrics(list(model.predict(dataset.val_data)), dataset.val_data)
            display(pd.DataFrame(results).T)

            def json_default(o):
                if isinstance(o, slice):
                    return str(o)
                raise

            with open(result_output_filename, 'w') as json_file:
                json.dump({
                    "config": {**get_config(model), "max_steps": max_steps},
                    "results": results,
                }, json_file, default=json_default)
    except AlreadyRunningException as e:
        model = None
        print("Experiment was already running")
        print(e)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name', help='dataset_name', choices=["conll", "deft", "conll", "ezmammo"], default="conll")
    parser.add_argument('--seed', help='seed', type=int, default=42)

    parser.add_argument('--do_char', help='do_char', action="store_true", default=False)
    parser.add_argument('--do_biaffine', help='do_biaffine', action="store_true", default=False)
    parser.add_argument('--do_tagging', help='do_tagging', default=False, const="full", nargs="?")
    parser.add_argument('--doc_context', help='doc_context', action="store_true", default=False)

    parser.add_argument('--finetune_bert', help='finetune_bert', action="store_true", default=False)
    parser.add_argument('--bert_lower', help='bert_lower', action="store_true", default=False)

    parser.add_argument('--n_bert_layers', help='n_bert_layers', type=int, default=4)
    parser.add_argument('--bert_proj_size', help='bert_proj_size', default=None)
    parser.add_argument('--unique_label', help='unique_label', action="store_true", default=False)
    parser.add_argument('--hidden_size', help='hidden_size', type=int, default=400)

    parser.add_argument('--biaffine_size', help='biaffine_size', type=int, default=None)
    parser.add_argument('--biaffine_loss_weight', help='biaffine_loss_weight', type=float, default=1.)
    parser.add_argument('--max_steps', help='max_steps', type=int, default=None)
    parser.add_argument('--bert_name', help='bert_name', default=None)
    parser.add_argument('--fasttext_file', help='fasttext_file', default=None)
    parser.add_argument('--norm_bert', help='norm_bert', default=False)

    parser.add_argument('--resources', help='resources', default="")

    args = parser.parse_args()

    main(**vars(args))
