import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers

from .data_utils import *
from .metrics import PrecisionRecallF1Metric
from .optimization import ScheduledOptimizer, LinearSchedule
from .torch_utils import batch_to_tensors
from .torch_utils import einsum, bce_with_logits, get_module, register


@register("vocabulary")
class Vocabulary(torch.nn.Module):
    def __init__(self, values=(), with_pad=True, with_unk=False):
        super().__init__()
        self.with_pad = with_pad
        self.with_unk = with_unk
        values = (["__pad__"] if with_pad else []) + (["__unk__"] if with_unk else []) + list(values)
        self.inversed = {v: i for i, v in enumerate(values)}

    @property
    def values(self):
        return list(self.inversed.keys())

    def __getitem__(self, obj):
        if self.training:
            return self.inversed.setdefault(obj, len(self.inversed))
        res = self.inversed.get(obj, None)
        if res is None:
            return self.inversed["__unk__"]
        return res

    def __repr__(self):
        return f"Vocabulary(count={len(self.inversed)}, with_pad={self.with_pad}, with_unk={self.with_unk})"


@register("flat_batch_norm")
class FlatBatchNorm(torch.nn.BatchNorm1d):
    def forward(self, inputs, mask):
        flat = inputs.rename(None)[mask.rename(None)]
        flat = super().forward(flat)
        res = torch.zeros_like(inputs)
        res[mask] = flat
        return res.rename(*inputs.names)


@register("char_cnn")
class CharCNNWordEncoder(torch.nn.Module):
    def __init__(self, n_chars, in_channels=8, out_channels=50, kernel_sizes=(3, 4, 5)):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_chars, in_channels)
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0)
            for kernel_size in kernel_sizes
        )

    def forward(self, batch):
        chars = batch["words_chars"][batch["words_mask"]]
        chars_mask = batch["words_chars_mask"][batch["words_mask"]]
        embeds = self.embedding(chars).rearrange("word char dim -> word dim char")
        embeds = torch.cat([
            conv(embeds.pad(char=(conv.kernel_size[0] // 2, (conv.kernel_size[0] - 1) // 2)).rename(None)).rearrange("word dim char -> word char dim").masked_fill(~chars_mask.unsqueeze(-1), -100000)
            for conv in self.convs
        ], dim="dim").max("char").values
        return embeds[batch["@words_id"]].rename(None)


@register("rezero")
class ReZeroGate(torch.nn.Module):
    def __init__(self, init_value=1e-3, dim=None, ln_mode="post"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1) * init_value)
        self.ln_mode = ln_mode
        if ln_mode is not False:
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, after, before):
        if self.ln_mode == "post":
            return self.norm(before + after * self.weight)
        elif self.ln_mode == "pre":
            return before + self.norm(after) * self.weight
        else:
            return before + after * self.weight


@register("sigmoid_gate")
class SigmoidGate(torch.nn.Module):
    def __init__(self, init_value=1e-3, dim=None, ln_mode="post"):
        super().__init__()
        if dim is None:
            self.weight = torch.nn.Parameter(torch.ones(1) * init_value)
        else:
            self.linear = torch.nn.Linear(dim, 1)

        self.ln_mode = ln_mode
        if ln_mode is not False:
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, after, before):
        gate = F.sigmoid(self.weight if hasattr(self, 'weight') else self.linear(after))
        if self.ln_mode == "post":
            return self.norm(before * gate + after * (1 - gate))
        elif self.ln_mode == "pre":
            return before * gate + self.norm(after) * (1 - gate)
        else:
            return before * gate + after * (1 - gate)


@register("bert")
class BERTEncoder(torch.nn.Module):
    def __init__(self, bert=None, path=None, n_layers=4, dropout=0.1, freeze_n_layers=-1):
        super().__init__()
        self.bert = bert if bert is not None else transformers.AutoModel.from_pretrained(path)
        self.n_layers = n_layers
        self.weight = torch.nn.Parameter(torch.randn(n_layers))

        if freeze_n_layers == -1:
            freeze_n_layers = len(self.bert.encoder.layer) + 1
        for module in (self.bert.embeddings, *self.bert.encoder.layer)[:freeze_n_layers]:
            for param in module.parameters():
                param.requires_grad = False
        for module in self.bert.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout

    def forward(self, batch):
        token_features = self.bert.forward(batch["tokens"], batch["tokens_mask"], output_hidden_states=True)[2]
        token_features = einsum(torch.stack(token_features[-self.n_layers:], dim=2), self.weight, "sample token layer dim, layer -> sample token dim")

        word_bert_begin = batch["words_bert_begin"].rename("sample", "word")
        word_bert_end = batch["words_bert_end"].rename("sample", "word")
        bert_features_cumsum = token_features.rename("sample", "token", "dim").cumsum("token")
        bert_features_cumsum = torch.cat([torch.zeros_like(bert_features_cumsum[:, :1]), bert_features_cumsum], dim="token")
        word_bert_features = (
                                   bert_features_cumsum.smart_gather(word_bert_begin, dim="token") -
                                   bert_features_cumsum.smart_gather(word_bert_end, dim="token")
                             ) / (word_bert_end - word_bert_begin).clamp_min(1)
        return word_bert_features


@register("lstm")
class LSTMContextualizer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, gate=False, dropout=0.1, bidirectional=True):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
            self.initial_linear = None
        else:
            self.initial_linear = torch.nn.Linear(input_size, hidden_size)

        self.dropout = torch.nn.Dropout(dropout)
        if gate is False:
            self.gate_modules = [None] * num_layers
        else:
            self.gate_modules = torch.nn.ModuleList([get_module(gate["module"])(**{"dim": hidden_size, **{k: v for k, v in gate.items() if k != "module"}}) for _ in range(num_layers)])
        self.lstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=1, bidirectional=bidirectional, batch_first=True)
            for dim in [hidden_size] * num_layers
        ])

    def forward(self, features, mask):
        sentence_lengths = mask.long().sum(1)
        sorter = (-sentence_lengths).argsort()
        sentence_lengths = sentence_lengths[sorter]
        names = features.names
        features = features.rename(None)[sorter]
        if self.initial_linear is not None:
            features = self.initial_linear(features)  # sample * token * hidden_size
        for lstm, gate_module in zip(self.lstm_layers, self.gate_modules):
            out = lstm(torch.nn.utils.rnn.pack_padded_sequence(features, sentence_lengths.cpu(), batch_first=True))[0]
            rnn_output = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
            rnn_output = self.dropout(rnn_output)
            if gate_module is None:
                features = rnn_output
            else:
                features = gate_module(features, rnn_output)

        return features[sorter.argsort()].rename(*names)


@register("exhaustive_biaffine_ner")
class ExhaustiveBiaffineNERDecoder(torch.nn.Module):
    def __init__(self, dim, n_labels, label_dim, use_batch_norm=True, dropout=0.2, contextualizer=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n_labels, label_dim, label_dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.ff = torch.nn.Linear(dim, label_dim * n_labels * 2)
        self.bias = torch.nn.Parameter(torch.zeros(n_labels))
        if use_batch_norm:
            self.batch_norm = FlatBatchNorm(contextualizer["hidden_size"])
        else:
            self.batch_norm = None
        self.n_labels = n_labels
        if contextualizer is not None:
            self.contextualizer = get_module(contextualizer["module"])(**{k: v for k, v in contextualizer.items() if k != "module"})
        else:
            self.contextualizer = None

    def top_params(self):
        return [
            self.weight,
            self.bias,
            *self.ff.parameters(),
            *(self.batch_norm.parameters() if self.batch_norm is not None else ()),
        ]

    def forward(self, features, batch, return_loss=False):
        device = features.device

        mask = batch["words_mask"]
        if self.contextualizer is not None:
            features = self.contextualizer(features, mask)

        if self.batch_norm is not None:
            features = self.batch_norm(features, mask)
        features = F.relu(self.ff(self.dropout(features)))
        start_features, end_features = features.rearrange("... (n_labels label_dim bounds) -> ... n_labels label_dim bounds", n_labels=self.n_labels, bounds=2).unbind("bounds")

        spans_labels_score = einsum(start_features, end_features, "sample start label dim, sample end label dim -> sample label start end") + self.bias.rename("label")
        spans_mask = (
              torch.triu(torch.ones(1, mask.shape[1], mask.shape[1], dtype=torch.bool, device=device)).rename("sample", "start", "end")
              & mask.rename("sample", "start")
              & mask.rename("sample", "end")
        ).repeat("sample label start end", label=spans_labels_score.size("label"))
        loss = None
        targets = None
        if return_loss:
            targets = torch.zeros_like(spans_mask)
            if torch.is_tensor(batch["mentions_mask"]) and batch["mentions_mask"].any():
                targets[batch["@mentions_doc_id"][batch["mentions_mask"]], batch["mentions_label"][batch["mentions_mask"]], batch["mentions_begin"][batch["mentions_mask"]], batch["mentions_end"][
                    batch["mentions_mask"]]] = True
            loss = bce_with_logits(spans_labels_score[spans_mask], targets[spans_mask])

        pred_doc_ids, pred_labels, pred_begins, pred_ends = (spans_labels_score.masked_fill(~spans_mask, -10000) > 0).nonzero(as_tuple=True)

        return {
            "scores": spans_labels_score,
            "doc_ids": pred_doc_ids,
            "labels": pred_labels,
            "begins": pred_begins,
            "ends": pred_ends,
            "loss": loss,
            "targets": targets,
            "spans_labels_score": spans_labels_score,
            "spans_mask": spans_mask,
        }


@register("preprocessor")
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


@register("ner")
class NER(pl.LightningModule):
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
            get_module(word_encoder["module"])(**{k: v for k, v in word_encoder.items() if k != "module"})
            for word_encoder in word_encoders
        ])
        self.embedding_batch_norm = FlatBatchNorm(decoder["contextualizer"]["input_size"]) if use_embedding_batch_norm else None
        self.ner_decoder = get_module(decoder["module"])(**{k: v for k, v in decoder.items() if k != "module"})
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
