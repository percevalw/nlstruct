import glob

import pytorch_lightning as pl
import torch
import transformers

from data_utils import *
from datasets import load_from_brat
from metrics import PrecisionRecallF1Metric
from modules import CharCNNWordEncoder, BERTEncoder, ExhaustiveBiaffineNERDecoder, FlatBatchNorm, Vocabulary
from optimization import ScheduledOptimizer, LinearSchedule
from rich_pl_logger import RichTableLogger
from torch_utils import batch_to_tensors


class Preprocessor(torch.nn.Module):
    def __init__(self, bert_name, word_regex=None, bert_lower=False, do_unidecode=True, vocabularies={}):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(bert_name)
        self.do_unidecode = do_unidecode
        self.bert_lower = bert_lower
        self.word_regex = word_regex
        self.vocabularies = vocabularies

    def forward(self, sample):
        if not isinstance(sample, dict):
            return map(self, sample)
        bert_tokens = huggingface_tokenize(sample["text"].lower() if self.bert_lower else sample["text"], tokenizer=self.tokenizer, do_unidecode=self.do_unidecode)
        if self.word_regex is not None:
            words = regex_tokenize(sample["text"], reg=self.word_regex, do_unidecode=self.do_unidecode)
        else:
            words = bert_tokens
        tokens_indice = self.tokenizer.convert_tokens_to_ids(bert_tokens["word"])
        words_bert_begin, words_bert_end = split_spans(words["begin"], words["end"], bert_tokens["begin"], bert_tokens["end"])
        words_bert_begin, words_bert_end = words_bert_begin.tolist(), words_bert_end.tolist()
        words_chars = [[self.vocabularies["char"][char] for char in word] for word in words["word"]]
        if len(sample["mentions"]):
            mentions_begin, mentions_end, mentions_label, mention_ids = map(list, zip(*[[fragment["begin"], fragment["end"], mention["label"], mention["mention_id"] + "/" + str(i)]
                                                                                        for mention in sample["mentions"] for i, fragment in enumerate(mention["fragments"])]))
            mentions_begin, mentions_end = split_spans(mentions_begin, mentions_end, words["begin"], words["end"])
            mentions_end -= 1  # end now means the index of the last word
            mentions_label = [self.vocabularies["label"][label] for label in mentions_label]
            mentions_begin, mentions_end = mentions_begin.tolist(), mentions_end.tolist()
        else:
            mentions_begin, mentions_end, mentions_label, mention_ids = [], [], [], []
        return {
            "tokens": tokens_indice,
            "tokens_mask": [True] * len(tokens_indice),
            "words_mask": [True] * len(words_chars),
            "words": words["word"],
            "words_id": [sample["doc_id"] + "-" + str(i) for i in range(len(words_chars))],
            "words_chars": words_chars,
            "words_chars_mask": [[True] * len(word_chars) for word_chars in words_chars],
            "words_bert_begin": words_bert_begin,
            "words_bert_end": words_bert_end,
            "words_begin": words["begin"],
            "words_end": words["end"],
            "mentions_begin": mentions_begin,
            "mentions_end": mentions_end,
            "mentions_label": mentions_label,
            "mentions_id": mention_ids,
            "mentions_doc_id": [sample["doc_id"]] * len(mention_ids),
            "mentions_mask": [True] * len(mention_ids),
            "doc_id": sample["doc_id"],
        }

    def tensorize(self, batch, device=None):
        return batch_to_tensors(batch, ids_mapping={"mentions_doc_id": "doc_id"}, device=device)

    def decode(self, spans, sample):
        doc_mentions = [[] for _ in range(len(embeds))]
        for doc_id, begin, end, label in zip(results["doc_ids"].tolist(), results["begin"].tolist(), results["end"].tolist(), results["label"].tolist()):
            doc_mentions[doc_id].append((batch["words_begin"][doc_id, begin], batch["words_end"][doc_id, begin], batch["words_begin"][doc_id, begin]))


class NER(pl.LightningModule):
    NER_DECODERS = {"exhaustive_biaffine": ExhaustiveBiaffineNERDecoder}
    WORD_ENCODERS = {"char_cnn": CharCNNWordEncoder, "bert": BERTEncoder}

    def __init__(
          self,
          preprocessor,
          word_encoders,
          decoder,
          use_embedding_batch_norm=True,
          sentence_split_regex=r"((?:\s*\n)+\s*|(?:(?<=[a-z0-9)]\.)\s+))(?=[A-Z])",
          sentence_balance_chars=('()', '[]'),

          init_labels_bias=True,
          batch_size=16,
          top_lr=1e-4,
          main_lr=1e-4,
          bert_lr=4e-5,
          warmup_rate=0.1,
          use_lr_schedules=True,
          optimizer=torch.optim.Adam
    ):
        super().__init__()

        self.preprocessor = Preprocessor(**preprocessor)
        self.sentence_balance_chars = sentence_balance_chars
        self.sentence_split_regex = sentence_split_regex
        self.word_encoders = torch.nn.ModuleList([
            self.WORD_ENCODERS[word_encoder["name"]](**{k: v for k, v in word_encoder.items() if k != "name"})
            for word_encoder in word_encoders
        ])
        self.embedding_batch_norm = FlatBatchNorm(decoder["contextualizer"]["input_size"]) if use_embedding_batch_norm else None
        self.ner_decoder = self.NER_DECODERS[decoder["name"]](**{k: v for k, v in decoder.items() if k != "name"})
        self.train_metric = PrecisionRecallF1Metric(prefix="train_")
        self.val_metric = PrecisionRecallF1Metric(prefix="val_")

        self.init_labels_bias = init_labels_bias

        self.top_lr = top_lr
        self.main_lr = main_lr
        self.bert_lr = bert_lr
        self.use_lr_schedules = use_lr_schedules
        self.warmup_rate = warmup_rate
        self.batch_size = batch_size
        self.optimizer_cls = optimizer

    def forward(self, inputs, return_loss=False):
        self.last_inputs = inputs
        device = next(self.parameters()).device
        input_tensors = self.preprocessor.tensorize(inputs, device=device)
        embeds = torch.cat([word_encoder(input_tensors).rename("sample", "word", "dim") for word_encoder in self.word_encoders], dim="dim")

        if self.embedding_batch_norm is not None:
            embeds = self.embedding_batch_norm(embeds, input_tensors["words_mask"].rename("sample", "word"))
        results = self.ner_decoder(embeds, input_tensors, return_loss=return_loss)
        preds = [[] for _ in range(len(embeds))]
        for doc_id, begin, end, label in zip(results["doc_ids"].tolist(), results["begins"].tolist(), results["ends"].tolist(), results["labels"].tolist()):
            preds[doc_id].append((begin, end, label))
        gold = [list(zip(sample["mentions_begin"], sample["mentions_end"], sample["mentions_label"])) for sample in inputs]
        return {
            "preds": preds,
            "gold": gold,
            **results,
        }

    def training_step(self, inputs, batch_idx):
        outputs = self(inputs, return_loss=True)
        self.train_metric(outputs['preds'], outputs['gold'])
        return {'loss': outputs["loss"], 'preds': outputs["preds"], "inputs": inputs}

    def validation_step(self, inputs, batch_idx):
        outputs = self(inputs, return_loss=True)
        self.val_metric(outputs['preds'], outputs['gold'])
        return {'loss': outputs["loss"], 'preds': outputs["preds"], "inputs": inputs}

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metric.compute())
        loss = sum(output["loss"] * len(output["inputs"]) for output in outputs) / sum(len(output["inputs"]) for output in outputs)
        self.log("train_loss", loss)
        self.log("main_lr", self.optimizers().param_groups[0]["lr"])
        self.log("top_lr", self.optimizers().param_groups[1]["lr"])
        self.log("bert_lr", self.optimizers().param_groups[2]["lr"])

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metric.compute())
        loss = sum(output["loss"] * len(output["inputs"]) for output in outputs) / sum(len(output["inputs"]) for output in outputs)
        self.log("val_loss", loss)

    def prepare_data(self):
        for dl, shuffle in [(self.train_dataloader, True), (self.val_dataloader, False), (self.test_dataloader, False)]:
            if dl() is None:
                continue
            data = dl()
            if self.sentence_split_regex is not None:
                data = sentencize(data, self.sentence_split_regex, balance_chars=self.sentence_balance_chars)
            data = list(self.preprocessor(data))
            data = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=self.batch_size, collate_fn=lambda x: x)
            dl.dataloader = data

    def on_pretrain_routine_start(self):
        # Setup label bias
        # Should be a good place to learn vocabularies ?
        labels_count = torch.zeros(len(self.preprocessor.vocabularies["label"].values))
        candidates_count = 0
        for batch in self.train_dataloader():
            for sample in batch:
                for label in sample["mentions_label"]:
                    labels_count[label] += 1
                candidates_count += (len(sample["words_mask"]) * (len(sample["words_mask"]) + 1)) // 2
        frequencies = labels_count / candidates_count
        self.ner_decoder.bias.data = (torch.log(frequencies) - torch.log1p(frequencies)).to(self.ner_decoder.bias.data.device)

    def postprocess(self, result):
        pass

    def configure_optimizers(self):
        bert_params = list(self.word_encoders[1].parameters())
        top_params = self.ner_decoder.top_params()
        main_params = [p for p in self.parameters() if not any(p is q for q in bert_params) and not any(p is q for q in top_params)]
        max_steps = self.trainer.max_epochs * len(self.train_dataloader.dataloader)
        optimizer = ScheduledOptimizer(self.optimizer_cls([
            {"params": main_params,
             "lr": self.main_lr,
             "schedules": LinearSchedule(path="lr", warmup_rate=0, total_steps=max_steps) if self.use_lr_schedules else []},
            {"params": top_params,
             "lr": self.top_lr,
             "schedules": LinearSchedule(path="lr", warmup_rate=0, total_steps=max_steps) if self.use_lr_schedules else []},
            {"params": bert_params,
             "lr": self.bert_lr,
             "schedules": LinearSchedule(path="lr", warmup_rate=self.warmup_rate, total_steps=max_steps) if self.use_lr_schedules else []},
        ]))
        return optimizer


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
