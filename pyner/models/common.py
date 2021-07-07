import math
from collections import namedtuple

import torch
import torch.nn.functional as F
import transformers
from transformers.models.roberta.modeling_roberta import RobertaLMHead, gelu
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from pyner.base import register
from pyner.torch_utils import fork_rng

RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


def identity(x):
    return x


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def has_len(x):
    try:
        len(x)
        return True
    except:
        return False


@register("vocabulary")
class Vocabulary(torch.nn.Module):
    def __init__(self, values=(), with_pad=True, with_unk=False):
        super().__init__()
        self.with_pad = with_pad
        if with_unk is True:
            with_unk = '__unk__'
        self.with_unk = with_unk
        values = (["__pad__"] if with_pad and "__pad__" not in values else []) + ([with_unk] if with_unk and with_unk not in values else []) + list(values)
        self.inversed = {v: i for i, v in enumerate(values)}
        self.eval()

    @property
    def values(self):
        return list(self.inversed.keys())

    def get(self, obj):
        if self.training:
            return self.inversed.setdefault(obj, len(self.inversed))
        res = self.inversed.get(obj, None)
        if res is None:
            try:
                return self.inversed[self.with_unk]
            except KeyError:
                raise KeyError(f"Could not find indice in vocabulary for {repr(obj)}")
        return res

    def __repr__(self):
        return f"Vocabulary(count={len(self.inversed)}, with_pad={self.with_pad}, with_unk={self.with_unk})"


@register("flat_batch_norm")
class FlatBatchNorm(torch.nn.BatchNorm1d):
    def forward(self, inputs, mask):
        flat = inputs[mask]
        flat = super().forward(flat)
        res = torch.zeros_like(inputs)
        res[mask] = flat
        return res


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
        embeds = self.embedding(chars).transpose(0, 2, 1)  # word char dim -> word dim char
        embeds = torch.cat([
            conv(embeds.pad(char=(conv.kernel_size[0] // 2, (conv.kernel_size[0] - 1) // 2))).transpose(0, 2, 1).masked_fill(~chars_mask.unsqueeze(-1), -100000)
            for conv in self.convs
        ], dim=2).max(1).values
        return embeds[batch["@words_id"]]


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == None:
        return identity
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


@register("gate")
class Gate(torch.nn.Module):
    def forward(self, before, after):
        raise NotImplementedError()


@register("residual_gate")
class ResidualGate(Gate):
    def __init__(self, init_value=1e-3, input_size=None, ln_mode="post"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1) * init_value)
        self.ln_mode = ln_mode
        if ln_mode is not False:
            self.norm = torch.nn.LayerNorm(input_size)

    def forward(self, before, after):
        if self.ln_mode == "post":
            return self.norm(before + after * self.weight)
        elif self.ln_mode == "pre":
            return before + self.norm(after) * self.weight
        else:
            return before + after * self.weight


@register("sigmoid_gate")
class SigmoidGate(Gate):
    def __init__(self, init_value=1e-3, input_size=None, ln_mode="post", proj=False):
        super().__init__()
        if proj:
            self.linear = torch.nn.Linear(input_size, 1)
        else:
            self.weight = torch.nn.Parameter(torch.ones(1) * init_value)

        self.ln_mode = ln_mode
        if ln_mode is not False:
            self.norm = torch.nn.LayerNorm(input_size)

    def forward(self, before, after):
        gate = torch.sigmoid(self.weight if hasattr(self, 'weight') else self.linear(after))
        if self.ln_mode == "post":
            return self.norm(before * (1 - gate) + after * gate)
        elif self.ln_mode == "pre":
            return before * (1 - gate) + self.norm(after) * gate
        else:
            return before * (1 - gate) + after * gate


class RobertaLMHeadWithLastHidden(RobertaLMHead):
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        logits = self.decoder(x)

        return logits, x


class BertLMPredictionHeadWithLastHidden(torch.nn.Module):
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        logits = self.decoder(hidden_states)
        return logits, hidden_states


LM_HEAD_CLS_MAPPING = {
    RobertaLMHead: RobertaLMHeadWithLastHidden,
    BertLMPredictionHead: BertLMPredictionHeadWithLastHidden,
}


def rearrange_and_prune(tensors, mask):
    assert mask.ndim == 3
    device = mask.device
    if not isinstance(tensors, (list, tuple)):
        [new_tensor], new_mask = rearrange_and_prune([tensors], mask)
        return new_tensor, new_mask
    lengths = mask.sum(-1).sum(-1)
    new_max_length = int(lengths.max())
    new_mask = torch.arange(new_max_length, device=device)[None, :] < lengths[:, None]
    new_tensors = []
    flat_mask = mask[mask.any(-1)]
    for flat_tensor in tensors:
        if flat_tensor is not None:
            new_tensor = torch.zeros(mask.shape[0], new_max_length, flat_tensor.shape[-1], device=device)
            new_tensor[new_mask] = flat_tensor[flat_mask]
        else:
            new_tensor = None
        new_tensors.append(new_tensor)
    return new_tensors, new_mask


@register("text_encoder")
class TextEncoder(torch.nn.Module):
    @property
    def output_size(self):
        return self._output_size

    def forward(self, batch):
        raise NotImplementedError()


@register("bert")
class BERTEncoder(TextEncoder):
    def __init__(self,
                 _bert=None,
                 bert_config=None,
                 path=None,
                 n_layers=4,
                 combine_mode="softmax",
                 dropout_p=0.1,
                 output_lm_embeds=False,
                 token_dropout_p=0.1,
                 word_pooler={"module": "pooler", "mode": "mean"},
                 freeze_n_layers=-1,
                 _preprocessor=None, ):
        super().__init__()
        with fork_rng(True):
            if output_lm_embeds:
                self.bert = _bert if _bert is not None else transformers.AutoModelForMaskedLM.from_pretrained(path, config=bert_config)
                if hasattr(self.bert, 'lm_head'):
                    self.bert.lm_head.__class__ = LM_HEAD_CLS_MAPPING[self.bert.lm_head.__class__]
                else:
                    self.bert.cls.predictions.__class__ = LM_HEAD_CLS_MAPPING[self.bert.cls.predictions.__class__]
            else:
                self.bert = _bert if _bert is not None else transformers.AutoModel.from_pretrained(path, config=bert_config)
        self.output_lm_embeds = output_lm_embeds
        self.n_layers = n_layers
        if n_layers > 1:
            with fork_rng(True):
                self.weight = torch.nn.Parameter(torch.zeros(n_layers)) if combine_mode == "softmax" else torch.nn.Parameter(torch.ones(n_layers) / n_layers)
        with fork_rng(True):
            self.word_pooler = Pooler(**word_pooler) if word_pooler is not None else None
        self.combine_mode = combine_mode

        bert_model = self.bert.bert if hasattr(self.bert, 'bert') else self.bert.roberta if hasattr(self.bert, 'roberta') else self.bert
        self._output_size = bert_model.embeddings.word_embeddings.weight.shape[1]
        if freeze_n_layers < 0:
            freeze_n_layers = len(bert_model.encoder.layer) + 2 + freeze_n_layers
        for module in (bert_model.embeddings, *bert_model.encoder.layer)[:freeze_n_layers]:
            for param in module.parameters():
                param.requires_grad = False
        for module in bert_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_p
        self.token_dropout_p = token_dropout_p

    @property
    def output_embeddings(self):
        bert = self.bert  # .bert if hasattr(self.bert, 'bert') else self.bert.roberta
        return bert.lm_head.decoder.weight if hasattr(bert, 'lm_head') else bert.cls.predictions.weight

    @property
    def output_bias(self):
        bert = self.bert  # .bert if hasattr(self.bert, 'bert') else self.bert.roberta
        return bert.lm_head.decoder.bias if hasattr(bert, 'lm_head') else bert.cls.predictions.bias

    @property
    def bert_config(self):
        return self.bert.config

    def forward(self, batch):
        tokens, mask = batch["tokens"], batch["tokens_mask"]
        if self.training & (self.token_dropout_p > 0):
            tokens[mask & (torch.rand_like(mask, dtype=torch.float) < self.token_dropout_p)] = 32004  # self.bert.config.mask_token_id
        if tokens.ndim == 3:
            needs_concat = True
            flat_tokens = tokens[mask.any(-1)]
            flat_mask = mask[mask.any(-1)]
        else:
            needs_concat = False
            flat_tokens = tokens
            flat_mask = mask
        if self.output_lm_embeds:
            token_features = self.bert.forward(flat_tokens, flat_mask, output_hidden_states=True)
            lm_embeds = list(token_features.logits)
            token_features = token_features.hidden_states
        else:
            lm_embeds = None
            token_features = self.bert.forward(flat_tokens, flat_mask, output_hidden_states=True)[2]
        if self.n_layers == 1:
            token_features = token_features[0]
        else:
            token_features = torch.einsum("stld,l->std", torch.stack(token_features[-self.n_layers:], dim=2), self.weight.softmax(-1) if self.combine_mode == "softmax" else self.weight)
        if lm_embeds is not None:
            token_features = lm_embeds[1]
        if needs_concat and lm_embeds is not None:
            token_features, lm_embeds[0], lm_embeds[1] = rearrange_and_prune([token_features, lm_embeds[0], lm_embeds[1]], mask)[0]
        elif needs_concat and lm_embeds is None:
            token_features, = rearrange_and_prune([token_features], mask)[0]
        # assert (self.word_pooler(token_features, (batch["words_bert_begin"], batch["words_bert_end"]))[mask.any(1)] == token_features[mask.any(1)]).all()
        # assert (self.word_pooler(lm_embeds, (batch["words_bert_begin"], batch["words_bert_end"]))[mask.any(1)] == lm_embeds[mask.any(1)]).all()
        if self.word_pooler is not None:
            if token_features is not None:
                token_features = self.word_pooler(token_features, (batch["words_bert_begin"], batch["words_bert_end"]))
            if lm_embeds is not None:
                lm_embeds = (
                    self.word_pooler(lm_embeds[0], (batch["words_bert_begin"], batch["words_bert_end"])),
                    self.word_pooler(lm_embeds[1], (batch["words_bert_begin"], batch["words_bert_end"]))
                )

        if self.output_lm_embeds:
            return token_features, lm_embeds
        return token_features


@register("char_cnn")
class CharCNNWordEncoder(TextEncoder):
    def __init__(self, n_chars=None, in_channels=8, out_channels=50, kernel_sizes=(3, 4, 5), _preprocessor=None):
        super().__init__()
        if n_chars is None:
            n_chars = len(_preprocessor.vocabularies['char'].values)
        self.embedding = torch.nn.Embedding(n_chars, in_channels)
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0)
            for kernel_size in kernel_sizes
        )
        self._output_size = out_channels * len(kernel_sizes)

    def forward(self, batch):
        chars = batch["words_chars"][batch["words_mask"]]
        chars_mask = batch["words_chars_mask"][batch["words_mask"]]
        embeds = self.embedding(chars).permute(0, 2, 1)  # word char dim -> word dim char
        embeds = torch.cat([
            conv(F.pad(embeds, pad=[conv.kernel_size[0] // 2, (conv.kernel_size[0] - 1) // 2])).permute(0, 2, 1).masked_fill(~chars_mask.unsqueeze(-1), -100000)
            for conv in self.convs
        ], dim=2).max(1).values
        res = torch.zeros(*batch["words_mask"].shape, embeds.shape[-1], device=embeds.device)
        res[batch["words_mask"]] = embeds
        return res  # embeds[batch["@words_id"]]


@register("concat")
class Concat(torch.nn.Module):
    def __init__(self, encoders, **kwargs):
        super().__init__()
        self.encoders = torch.nn.ModuleList([TextEncoder(**{**e, **kwargs}) for e in encoders])
        self.output_size = sum(e.output_size for e in self.encoders)
        self.bert = next((encoder.bert for encoder in self.encoders if hasattr(encoder, 'bert')), None)

    def forward(self, *args, **kwargs):
        res = []
        additional_res = []
        has_tuple = False
        for encoder in self.encoders:
            encoder_res = encoder(*args, **kwargs)
            if isinstance(encoder_res, tuple):
                additional_res.extend(encoder_res[1:])
                res.append(encoder_res[0])
                has_tuple = True
            else:
                res.append(encoder_res)
        res = torch.cat(res, dim=-1)
        return (res, *additional_res) if has_tuple else res


@register("contextualizer")
class Contextualizer(torch.nn.Module):
    def forward(self, features, mask, return_global_state=False):
        raise NotImplementedError()


@register("lstm")
class LSTMContextualizer(Contextualizer):
    def __init__(self, input_size, hidden_size, num_layers=1, keep_cell_state=False, rand_init=False, gate=False, dropout_p=0.1, bidirectional=True, gate_reference='input'):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
            self.initial_linear = None
        else:
            self.initial_linear = torch.nn.Linear(input_size, hidden_size)
        self.rand_init = rand_init
        self.keep_cell_state = keep_cell_state
        if keep_cell_state:
            self.cell_state_proj = torch.nn.Linear(hidden_size, hidden_size)
        assert gate_reference in ("input", "last")
        self.gate_reference = gate_reference

        self.dropout = torch.nn.Dropout(dropout_p)
        if gate is False:
            self.gate_modules = [None] * num_layers
        else:
            self.gate_modules = torch.nn.ModuleList([
                Gate(**{**gate, "input_size": hidden_size})
                for _ in range(num_layers)])
        self.lstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=1, bidirectional=bidirectional, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output_size = hidden_size

    @property
    def gate(self):
        return self.gate_modules[0] if len(self.gate_modules) else False

    def forward(self, features, mask, return_all_layers=False, return_global_state=False):
        sentence_lengths = mask.long().sum(1)
        sorter = (-sentence_lengths).argsort()
        inv_sorter = sorter.argsort()
        sentence_lengths = sentence_lengths[sorter]
        names = features.names
        features = features[sorter]
        if self.initial_linear is not None:
            features = self.initial_linear(features)  # sample * token * hidden_size
        updates = torch.zeros_like(features)
        cell_states = []
        all_outputs = []

        num_directions = 2 if self.lstm_layers[0].bidirectional else 1
        real_hidden_size = self.lstm_layers[0].proj_size if getattr(self.lstm_layers[0], 'proj_size', 0) > 0 else self.lstm_layers[0].hidden_size
        initial_h_n = (torch.randn if self.rand_init and self.training else torch.zeros)(num_directions,
                                  len(features), real_hidden_size,
                                  dtype=features.dtype, device=features.device) * 0.1
        initial_c_n = (torch.randn if self.rand_init and self.training else torch.zeros)(num_directions,
                                  len(features), real_hidden_size,
                                  dtype=features.dtype, device=features.device) * 0.1
        for lstm, gate_module in zip(self.lstm_layers, self.gate_modules):
            out, (_, c_n) = lstm(
                torch.nn.utils.rnn.pack_padded_sequence(features + updates, sentence_lengths.cpu(), batch_first=True),
                (initial_h_n, initial_c_n),
            )
            if self.keep_cell_state:
                initial_c_n = c_n
                initial_c_n = initial_c_n.transpose(0, 1)
                initial_c_n = self.cell_state_proj(c_n.transpose(0, 1).reshape(mask.shape[0], -1)).reshape(initial_c_n.shape).transpose(0, 1).contiguous()
            cell_states.append(c_n.transpose(0, 1).reshape(mask.shape[0], -1)[inv_sorter])
            rnn_output = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
            rnn_output = self.dropout(rnn_output)
            if self.gate_reference == "input":
                updates = rnn_output if gate_module is None else gate_module(updates, rnn_output)
                all_outputs.append((features + updates)[inv_sorter])
            else:
                features = rnn_output if gate_module is None else gate_module(updates, rnn_output)
                all_outputs.append(features[inv_sorter])
        if return_all_layers:
            all_outputs = torch.stack(all_outputs, dim=0)
            cell_states = torch.stack(cell_states, dim=0)
        else:
            all_outputs = all_outputs[-1]
            cell_states = cell_states[-1]

        if return_global_state:
            return all_outputs, cell_states
        return all_outputs


@register("pooler")
class Pooler(torch.nn.Module):
    def __init__(self, mode="mean", dropout_p=0., input_size=None, n_heads=None, do_value_proj=False):
        super().__init__()
        self.mode = mode
        assert mode in ("max", "sum", "mean", "attention")
        self.dropout = torch.nn.Dropout(dropout_p)
        if mode == "attention":
            self.key_proj = torch.nn.Linear(input_size, n_heads)
            self.value_proj = torch.nn.Linear(input_size, input_size) if do_value_proj else None
        self.output_size = input_size

    def forward(self, features, mask):
        device = features.device
        if self.mode == "attention" and isinstance(mask, tuple):
            position = torch.arange(features.shape[-2], device=device).reshape([1] * (features.ndim - 2) + [features.shape[-2]])
            mask = (mask[0].unsqueeze(-1) <= position) & (position < mask[1].unsqueeze(-1))
            features = features.unsqueeze(-3)
        if isinstance(mask, tuple):
            begins, ends = mask
            begins = begins.expand(*features.shape[:begins.ndim-1], begins.shape[-1]).clamp_min(0)
            ends = ends.expand(*features.shape[:begins.ndim-1], ends.shape[-1]).clamp_min(0)
            final_shape = (*begins.shape, *features.shape[begins.ndim:])
            features = features.view(-1, features.shape[-2], features.shape[-1])
            begins = begins.reshape(features.shape[0], begins.numel() // features.shape[0] if len(features) else 0)
            ends = ends.reshape(features.shape[0], ends.numel() // features.shape[0] if len(features) else 0)

            max_window_size = max(0, int((ends - begins).max())) if 0 not in ends.shape else 0
            flat_indices = torch.arange(max_window_size, device=device)[None, None, :] + begins[..., None]
            flat_indices_mask = flat_indices < ends[..., None]
            flat_indices += torch.arange(len(flat_indices), device=device)[:, None, None] * features.shape[1]

            flat_indices = flat_indices[flat_indices_mask]
            return F.embedding_bag(
                input=flat_indices,
                weight=self.dropout(features.reshape(-1, features.shape[-1])),
                offsets=torch.cat([torch.tensor([0], device=device), flat_indices_mask.sum(-1).reshape(-1)]).cumsum(0)[:-1].clamp_max(flat_indices.shape[0]),
                mode=self.mode,
            ).reshape(final_shape)
        elif torch.is_tensor(mask):
            mask = mask
            features = features
            features = self.dropout(features)
            if mask.ndim <= features.ndim - 1:
                mask = mask.unsqueeze(-1)
            if 0 in mask.shape:
                return features.sum(-2)
            if self.mode == "attention":
                weights = self.key_proj(features).masked_fill(~mask, -100000).softmax(-2)  # ... tokens heads
                values = self.value_proj(features) if self.value_proj is not None else features
                values = values.view(*values.shape[:-1], weights.shape[-1], -1)  # ... tokens heads dim
                res = torch.einsum('...nhd,...nh->...hd', values, weights)
                return res.view(*res.shape[:-2], -1)
            elif self.mode == "max":
                features = features.masked_fill(~mask, -100000).max(-2).values.masked_fill(~(mask.any(-2)), 0)
            elif self.mode == "abs-max":
                values, indices = features.abs().masked_fill(~mask, -100000).max(-2)
                features = features.gather(dim=-2, index=indices.unsqueeze(1)).squeeze(1)
            elif self.mode in ("sum", "mean"):
                features = features.masked_fill(~mask, 0).sum(-2)
                if self.mode == "mean":
                    features = features / mask.float().sum(-2).clamp_min(1.)
            elif self.mode == "softmax":
                weights = (features.detach() * self.alpha).masked_fill(~mask, -100000).softmax(-2)
                features = torch.einsum('...nd,...nd->...d', weights, features.masked_fill(~mask, 0))
            elif self.mode == "softmax-abs":
                weights = (features.detach().abs() * self.alpha).masked_fill(~mask, -100000).softmax(-2)
                features = torch.einsum('...nd,...nd->...d', weights, features.masked_fill(~mask, 0))
            return features


@register("scaler")
class Scaler(torch.nn.Module):
    def __init__(self, dim, scale=1., affine=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim) * scale)
        if affine:
            self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        res = x * self.weight
        if hasattr(self, 'bias'):
            res = res + self.bias
        return res


@register("pos")
class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=512, temperature=10000.0, mode="sin", seed=None):
        super().__init__()
        self.dim = dim
        if mode.endswith("-proj"):
            self.proj = torch.nn.Linear(dim, dim)
            mode = mode[:-5]
        elif mode.endswith("-scale1d-init0"):
            self.proj = Scaler(dim, 0)
            mode = mode[:-14]
        elif mode.endswith("-scale1d-init0-affine"):
            self.proj = Scaler(dim, 0, affine=True)
            mode = mode[:-21]
        elif mode.endswith("-scale1d-init1"):
            self.proj = Scaler(dim, 1)
            mode = mode[:-14]
        elif mode.endswith("-scale1d-init1-affine"):
            self.proj = Scaler(dim, 1, affine=True)
            mode = mode[:-21]
        elif mode.endswith("-scale0d-init0"):
            self.proj = Scaler(1, 0)
            mode = mode[:-14]
        elif mode.endswith("-scale0d-init0-affine"):
            self.proj = Scaler(1, 0, affine=True)
            mode = mode[:-21]
        elif mode.endswith("-scale0d-init1"):
            self.proj = Scaler(1, 1)
            mode = mode[:-14]
        elif mode.endswith("-scale0d-init1-affine"):
            self.proj = Scaler(1, 1, affine=True)
            mode = mode[:-21]
        self.mode = mode
        if mode == "sin" or mode == "sym-sin" or mode == "inv-sin" or mode == "shift-sin":
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(temperature) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        elif mode == "learned":
            if seed is not None:
                with fork_rng(seed):
                    self.pe = torch.nn.Embedding(max_len, dim).weight
            else:
                self.pe = torch.nn.Embedding(max_len, dim).weight
        elif mode == "random":
            self.pe = None
        elif mode == "zeros":
            self.pe = None
        else:
            raise Exception()

    def forward(self, mask, device=None):
        if torch.is_tensor(mask):
            shape = mask.shape
            mask = mask
        elif isinstance(mask, tuple):
            shape = mask
            mask = None
        elif isinstance(mask, int):
            shape = (mask, self.pe.shape[0])
            mask = None
        else:
            raise Exception("PositionalEncoding input should be size or bool tensor, but is {}".format(type(mask)))
        if device is None:
            device = mask.device
        n = shape[-1]
        if self.mode == "learned" and shape[-1] != self.pe.shape[0] and self.training:
            slot_indices = torch.rand((*shape[:-1], self.pe.shape[0]), dtype=torch.float, device=device).argsort(-1)[..., :n]
            slot_indices = slot_indices.sort(-1)[0]
            res = F.embedding(slot_indices, self.pe)
        elif self.mode == "learned" or self.mode == "sin" or (self.mode == "shift-sin" and not self.training):
            view_shape = [1] * (len(shape) - 1) + [n, self.pe.shape[1]]
            repeat_shape = [s for s in shape[:-1]] + [1, 1]
            res = self.pe[:n].view(*view_shape).repeat(*repeat_shape)
        elif self.mode == "shift-sin" and self.training:
            margins = self.pe.shape[0] - mask.long().sum(1)
            p = (margins.unsqueeze(1) - torch.arange(margins.max(), device=device).unsqueeze(0)).float().clamp_min(0)
            p = p / p.sum(1).unsqueeze(1)
            res = (torch.distributions.Categorical(probs=p).sample().unsqueeze(1) + torch.arange(n, device=device).unsqueeze(0)).masked_fill(~mask.bool(), 0)
            res = F.embedding(res, self.pe)
        elif self.mode == "random":
            res = (F.normalize(torch.randn(*shape, self.dim, device=device), dim=-1) * math.sqrt(self.dim))
        elif self.mode == "zeros":
            res = torch.zeros(*shape, self.dim, device=device)
        if hasattr(self, 'proj'):
            res = res + self.proj(res)
        return res


class FeedForwardNetwork(torch.nn.Sequential):
    def __init__(self, input_size, sizes, dropout_p=0.1, activation='gelu'):
        super().__init__()
        layers = []
        self.dropout = torch.nn.Dropout(dropout_p)
        for i, (s_in, s_out) in enumerate(zip((input_size, *sizes[:-1]), sizes)):
            layers.append(torch.nn.Linear(s_in, s_out))
        self.layers = torch.nn.ModuleList(layers)
        self.activation_fn = get_activation_fn(activation)

    def forward(self, input):
        for i, layer in self.layers:
            if i > 0:
                input = self.activation_fn(input)
            input = self.dropout(input)
            input = layer(input)
        return input