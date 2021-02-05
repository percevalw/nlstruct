import torch
import torch.nn.functional as F
import transformers

from torch_utils import einsum, bce_with_logits


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


class FlatBatchNorm(torch.nn.BatchNorm1d):
    def forward(self, inputs, mask):
        flat = inputs.rename(None)[mask.rename(None)]
        flat = super().forward(flat)
        res = torch.zeros_like(inputs)
        res[mask] = flat
        return res.rename(*inputs.names)


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


class ReZeroSigmoidGate(torch.nn.Module):
    def __init__(self, init_value=1e-3, dim=None, ln_mode="post"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1) * init_value)
        self.ln_mode = ln_mode
        if ln_mode is not False:
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, after, before):
        gate = F.sigmoid(self.weight)
        if self.ln_mode == "post":
            return self.norm(before * gate + after * (1 - gate))
        elif self.ln_mode == "pre":
            return before * gate + self.norm(after) * (1 - gate)
        else:
            return before * gate + after * (1 - gate)


class SigmoidGate(torch.nn.Module):
    def __init__(self, dim=None, ln_mode="post"):
        super().__init__()
        self.linear = torch.nn.Linear(dim, 1)
        self.ln_mode = ln_mode
        if ln_mode is not False:
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, before, after):
        gate = F.sigmoid(self.linear(after))
        if self.ln_mode == "post":
            return self.norm(before * gate + after * (1 - gate))
        elif self.ln_mode == "pre":
            return before * gate + self.norm(after) * (1 - gate)
        else:
            return before * gate + after * (1 - gate)


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


class LSTMContextualizer(torch.nn.Module):
    GATES = {
        "sigmoid": SigmoidGate,
        "rezero": ReZeroGate,
    }

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
            self.gate_modules = torch.nn.ModuleList([self.GATES[gate["name"]](**{"dim": hidden_size, **{k: v for k, v in gate.items() if k != "name"}}) for _ in range(num_layers)])
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


class ExhaustiveBiaffineNERDecoder(torch.nn.Module):
    CONTEXTUALIZERS = {"lstm": LSTMContextualizer}

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
            self.contextualizer = self.CONTEXTUALIZERS[contextualizer["name"]](**{k: v for k, v in contextualizer.items() if k != "name"})
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
