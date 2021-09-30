import math
from collections import namedtuple

import torch
import torch.nn.functional as F
import transformers
from transformers.models.roberta.modeling_roberta import RobertaLMHead, gelu
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from nlstruct.registry import register, get_instance
from nlstruct.torch_utils import fork_rng, shift

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
    ENSEMBLE = "ensemble_text_encoder"

    @property
    def output_size(self):
        return self._output_size

    def forward(self, batch):
        raise NotImplementedError()


def multi_dim_slice(tensors, slices_begin, slices_end):
    source_max_length = tensors[0].shape[slices_begin.ndim]
    device = tensors[0].device
    positions = torch.arange(source_max_length, device=device)
    slice_source_mask = (positions >= slices_begin.unsqueeze(-1)) & (positions < slices_end.unsqueeze(-1))
    dest_max_length = (slices_end.unsqueeze(-1) - slices_begin.unsqueeze(-1)).max()
    slice_dest_mask = torch.arange(dest_max_length, device=device) < slices_end.unsqueeze(-1) - slices_begin.unsqueeze(-1)

    results = []
    for tensor in tensors:
        sliced_output = torch.zeros(*slice_dest_mask.shape, *tensor.shape[slice_source_mask.ndim:], device=device, dtype=tensor.dtype)
        sliced_output[slice_dest_mask] = tensor[slice_source_mask]
        results.append(sliced_output)
    return results


@register("ensemble_text_encoder")
class EnsembleEncoder(TextEncoder):
    def __init__(self, models):
        super().__init__()

        self.models = torch.nn.ModuleList([get_instance(e) for e in models])
        self._output_size = models[0].output_size

    def forward(self, batch):
        results = []
        for model in self.models:
            results.append(model(batch))
        return results


@register("bert")
class BERTEncoder(TextEncoder):
    ENSEMBLE = "ensemble_text_encoder"

    def __init__(self,
                 _bert=None,
                 bert_config=None,
                 path=None,
                 n_layers=4,
                 combine_mode="softmax",
                 bert_dropout_p=None,
                 output_lm_embeds=False,
                 token_dropout_p=0.,
                 dropout_p=0.,
                 word_pooler={"module": "pooler", "mode": "mean"},
                 proj_size=None,
                 freeze_n_layers=-1,
                 do_norm=True,
                 do_cache=False,
                 _preprocessor=None, ):
        super().__init__()
        assert not ("scaled" in combine_mode and do_norm)
        if do_cache:
            assert freeze_n_layers == -1, "Must freeze bert to enable caching: set freeze_n_layers=-1"

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
                self.weight = torch.nn.Parameter(torch.zeros(n_layers)) if "softmax" in combine_mode else torch.nn.Parameter(torch.ones(n_layers) / n_layers) if combine_mode == "linear" else None
        with fork_rng(True):
            self.word_pooler = Pooler(**word_pooler) if word_pooler is not None else None
        if "scaled" in combine_mode:
            self.bert_scaling = torch.nn.Parameter(torch.ones(()))
        self.combine_mode = combine_mode

        bert_model = self.bert.bert if hasattr(self.bert, 'bert') else self.bert.roberta if hasattr(self.bert, 'roberta') else self.bert
        bert_output_size = bert_model.embeddings.word_embeddings.weight.shape[1] * (1 if combine_mode != "concat" else n_layers)
        self.bert_output_size = bert_output_size
        if proj_size is not None:
            self.proj = torch.nn.Linear(bert_output_size, proj_size)
            self._output_size = proj_size
        else:
            self.proj = None
            self._output_size = bert_output_size
        self.norm = torch.nn.LayerNorm(self._output_size) if do_norm else Identity()

        if freeze_n_layers < 0:
            freeze_n_layers = len(bert_model.encoder.layer) + 2 + freeze_n_layers
        for module in (bert_model.embeddings, *bert_model.encoder.layer)[:freeze_n_layers]:
            for param in module.parameters():
                param.requires_grad = False
        if bert_dropout_p is not None:
            for module in bert_model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = bert_dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
        self.token_dropout_p = token_dropout_p
        self.cache = {}
        self.do_cache = do_cache

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Do not save bert if it was frozen. We assume that the freeze_n_layers == -1 was not
        # changed during training, and therefore that the weights are identical to those at init
        state = super().state_dict(destination, prefix, keep_vars)
        if self.freeze_n_layers == -1:
            for name in list(state.keys()):
                if ".bert." in name:
                    del state[name]
        return state

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

    def exec_bert(self, tokens, mask, slices_begin=None, slices_end=None):
        res = self.bert.forward(tokens, mask, output_hidden_states=True)
        if self.output_lm_embeds:
            lm_embeds = list(res.logits)
            token_features = res.hidden_states
        else:
            lm_embeds = ()
            token_features = res[2]
        if self.n_layers == 1:
            token_features = token_features[-1].unsqueeze(-2)
        else:
            token_features = torch.stack(token_features[-self.n_layers:], dim=2)

        results = (token_features, *lm_embeds)
        if slices_begin is not None:
            results = multi_dim_slice(results, slices_begin, slices_end)
        return results

    def forward(self, batch):
        tokens, mask = batch["tokens"], batch["tokens_mask"]
        device = mask.device
        if self.training & (self.token_dropout_p > 0):
            tokens[mask & (torch.rand_like(mask, dtype=torch.float) < self.token_dropout_p)] = 32004  # self.bert.config.mask_token_id
        if tokens.ndim == 3 and tokens.shape[1] == 1:
            flat_tokens = tokens.squeeze(1)
            flat_mask = mask.squeeze(1)
            flat_slices_begin = batch['slice_begin'].squeeze(1) if 'slice_begin' in batch else torch.zeros(len(flat_mask), device=device, dtype=torch.long)
            flat_slices_end = batch['slice_end'].squeeze(1) if 'slice_end' in batch else flat_mask.long().sum(1)
            needs_concat = False
        elif tokens.ndim == 3:
            needs_concat = True
            flat_tokens = tokens[mask.any(-1)]
            flat_mask = mask[mask.any(-1)]
            flat_slices_begin = batch['slice_begin'][mask.any(-1)] if 'slice_begin' in batch else torch.zeros(len(flat_mask), device=device, dtype=torch.long)
            flat_slices_end = batch['slice_end'][mask.any(-1)] if 'slice_end' in batch else flat_mask.long().sum(1)
        else:
            needs_concat = False
            flat_tokens = tokens
            flat_mask = mask
            flat_slices_begin = batch['slice_begin'] if 'slice_begin' in batch else torch.zeros(len(flat_mask), device=device, dtype=torch.long)
            flat_slices_end = batch['slice_end'] if 'slice_end' in batch else flat_mask.long().sum(1)
        if self.do_cache:
            keys = [hash((tuple(row[:length]), begin, end)) for row, length, begin, end in zip(flat_tokens.tolist(), flat_mask.sum(1).tolist(), flat_slices_begin.tolist(), flat_slices_end.tolist())]

            missing_keys = [key for key in keys if key not in self.cache]
            missing_keys_mask = [key in missing_keys for key in keys]
            if sum(missing_keys_mask) > 0:
                missing_embeds = self.exec_bert(
                    flat_tokens[missing_keys_mask],
                    flat_mask[missing_keys_mask],
                    slices_begin=flat_slices_begin[missing_keys_mask] if flat_slices_begin is not None else None,
                    slices_end=flat_slices_end[missing_keys_mask] if flat_slices_end is not None else None)
                cache_entries = [tuple(t[:length] for t in tensor_set)
                                 for tensor_set, length in zip(zip(*(t.cpu().unbind(0) for t in missing_embeds)), flat_mask[missing_keys_mask].sum(-1).tolist())]
                for key, cache_entry in zip(missing_keys, cache_entries):
                    self.cache[key] = cache_entry
                    if (len(self.cache) % 1000) == 0:
                        print("cache size:", len(self.cache))

            if sum(missing_keys_mask) == len(missing_keys_mask):
                (token_features, *lm_embeds) = missing_embeds
            else:
                (token_features, *lm_embeds) = (pad_embeds(embeds_list).to(device) for embeds_list in zip(*[self.cache[key] for key in keys]))
        else:
            (token_features, *lm_embeds) = self.exec_bert(
                flat_tokens,
                flat_mask,
                slices_begin=flat_slices_begin,
                slices_end=flat_slices_end)

        if self.n_layers == 1:
            token_features = token_features.squeeze(-2)
        elif self.combine_mode != "concat":
            token_features = torch.einsum("stld,l->std", token_features, self.weight.softmax(-1) if "softmax" in self.combine_mode else self.weight)
        else:
            token_features = token_features.view(*token_features.shape[:-2], -1)

        if hasattr(self, 'bert_scaling'):
            token_features = token_features * self.bert_scaling

        token_features = self.dropout(token_features)

        if self.proj is not None:
            token_features = F.gelu(self.proj(token_features))

        token_features = self.norm(token_features)

        if needs_concat and len(lm_embeds) > 0:
            token_features, lm_embeds[0], lm_embeds[1] = rearrange_and_prune([token_features, lm_embeds[0], lm_embeds[1]], mask)[0]
        elif needs_concat and len(lm_embeds) == 0:
            token_features, = rearrange_and_prune([token_features], mask)[0]
        # assert (self.word_pooler(token_features, (batch["words_bert_begin"], batch["words_bert_end"]))[mask.any(1)] == token_features[mask.any(1)]).all()
        # assert (self.word_pooler(lm_embeds, (batch["words_bert_begin"], batch["words_bert_end"]))[mask.any(1)] == lm_embeds[mask.any(1)]).all()
        if self.word_pooler is not None:
            if token_features is not None:
                token_features = self.word_pooler(token_features, (batch["words_bert_begin"], batch["words_bert_end"]))
            if len(lm_embeds) > 0:
                lm_embeds = tuple((
                    self.word_pooler(part, (batch["words_bert_begin"], batch["words_bert_end"]))
                    for part in lm_embeds
                ))

        if self.output_lm_embeds:
            return token_features, lm_embeds
        return token_features


def pad_embeds(embeds_list):
    lengths = [t.shape[0] for t in embeds_list]
    max_length = max(lengths)
    res = torch.zeros(len(embeds_list), max_length, *embeds_list[0].shape[1:])
    for i, (embeds, length) in enumerate(zip(embeds_list, lengths)):
        res[i, :length] = embeds
    return res


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
            F.relu(conv(F.pad(embeds, pad=[conv.kernel_size[0] // 2, (conv.kernel_size[0] - 1) // 2])).permute(0, 2, 1).masked_fill(~chars_mask.unsqueeze(-1), 0))
            for conv in self.convs
        ], dim=2).max(1).values
        res = torch.zeros(*batch["words_mask"].shape, embeds.shape[-1], device=embeds.device)
        res[batch["words_mask"]] = embeds
        return res  # embeds[batch["@words_id"]]


@register("word_embeddings")
class WordEmbeddings(TextEncoder):
    def __init__(self, filename, _preprocessor=None):
        super().__init__()

        print(f"Loading {filename} embeddings... ", end="")
        with open(filename) as f:
            lines = f.readlines()
        words = ["__unk__"]
        weight = []
        for line in lines:
            line = line.rstrip().split(" ")
            words.append(line[0])
            weight.append(list(map(float, line[1:])))
        weight = torch.tensor([[0.] * len(weight[0])] + weight)
        self.embedding = torch.nn.Embedding(weight.shape[0], weight.shape[1])
        self.embedding.weight.data = weight
        self.embedding.weight.requires_grad = False
        self._output_size = weight.shape[1]
        print(f"done")

        if _preprocessor is not None:
            _preprocessor.vocabularies['word'] = Vocabulary(values=words, with_unk='__unk__', with_pad=False)

    def forward(self, batch):
        return self.embedding(batch['words'])


@register("concat")
class Concat(torch.nn.Module):
    def __init__(self, encoders, dropout_p=0., **kwargs):
        super().__init__()
        self.encoders = torch.nn.ModuleList([TextEncoder(**{**e, **kwargs}) for e in (encoders.values() if isinstance(encoders, dict) else encoders)])
        self.output_size = sum(e.output_size for e in self.encoders)
        self.bert = next((encoder.bert for encoder in self.encoders if hasattr(encoder, 'bert')), None)
        self.dropout = torch.nn.Dropout(dropout_p)

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
        res = self.dropout(torch.cat(res, dim=-1))
        return (res, *additional_res) if has_tuple else res


@register("contextualizer")
class Contextualizer(torch.nn.Module):
    def forward(self, features, mask, return_global_state=False):
        raise NotImplementedError()


@register("lstm")
class LSTMContextualizer(Contextualizer):
    def __init__(self, input_size, hidden_size=None, num_layers=1, keep_cell_state=False, rand_init=False, do_better_init=False, gate=False, dropout_p=0.1, bidirectional=True, gate_reference='input'):
        super().__init__()

        if hidden_size is None:
            hidden_size = input_size
            same_size = True
        else:
            if gate_reference == "input":
                self.initial_linear = torch.nn.Linear(input_size, hidden_size)
                same_size = True
            else:
                same_size = False

        self.same_size = same_size
        self.rand_init = rand_init
        self.keep_cell_state = keep_cell_state
        if keep_cell_state:
            self.cell_state_proj = torch.nn.Linear(hidden_size, hidden_size)
        assert gate_reference in ("input", "last")
        self.gate_reference = gate_reference

        self.dropout = torch.nn.Dropout(dropout_p)

        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "lstm": torch.nn.LSTM(
                    input_size=input_size if (gate_reference == "last" and i == 0) else hidden_size,
                    hidden_size=hidden_size // 2,
                    num_layers=1,
                    bidirectional=bidirectional,
                    batch_first=True),
                "gate": Gate(**{**gate, "input_size": hidden_size}) if gate and (i > 0 or same_size) else None,
            })
            for i in range(num_layers)
        ])
        self.output_size = hidden_size
        if do_better_init:
            self._better_init_weights()

    def _better_init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for layer in self.layers:
            for name, param in layer["lstm"].named_parameters():
                if "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    torch.nn.init.zeros_(param.data)
                    param.data[layer["lstm"].hidden_size:2 * layer["lstm"].hidden_size] = 1

    @property
    def gate(self):
        return self.layers[-1]["gate"] if len(self.layers) else False

    def fast_params(self):
        return [
            # *((self.main_gate.linear.parameters() if self.main_gate.linear is not None else [self.main_gate.weight]) if self.main_gate is not None else []),
            # *(param for layer in self.layers for param in ((layer.gate.linear.parameters() if layer.gate.linear is not None else [layer.gate.weight]) if layer.gate is not None else [])),
        ]

    def forward(self, features, mask, return_all_layers=False, return_global_state=False):
        if hasattr(self, 'initial_linear'):
            features = F.gelu(self.initial_linear(features))
        sentence_lengths = mask.long().sum(1)
        sorter = (-sentence_lengths).argsort()
        inv_sorter = sorter.argsort()
        sentence_lengths = sentence_lengths[sorter]
        max_length = (sentence_lengths.max() if 0 not in sentence_lengths.shape else None)
        features = features[sorter, :max_length]
        # mask = mask[sorter, :max_length]
        # if self.initial_linear is not None:
        #    features = self.initial_linear(features)  # sample * token * hidden_size
        updates = torch.zeros(*features.shape[:-1], self.output_size, device=features.device)
        cell_states = []
        all_outputs = []

        first_lstm = self.layers[0]["lstm"]
        num_directions = 2 if first_lstm.bidirectional else 1
        real_hidden_size = first_lstm.proj_size if getattr(first_lstm, 'proj_size', 0) > 0 else first_lstm.hidden_size
        initial_h_n = (torch.randn if self.rand_init and self.training else torch.zeros)(num_directions,
                                                                                         len(features), real_hidden_size,
                                                                                         dtype=features.dtype, device=features.device) * 0.1
        initial_c_n = (torch.randn if self.rand_init and self.training else torch.zeros)(num_directions,
                                                                                         len(features), real_hidden_size,
                                                                                         dtype=features.dtype, device=features.device) * 0.1
        for i, layer in enumerate(self.layers):
            lstm, gate = layer["lstm"], layer["gate"]
            out, (_, c_n) = lstm(
                torch.nn.utils.rnn.pack_padded_sequence(
                    features + updates if self.gate_reference == "input"
                    else features,
                    sentence_lengths.cpu(), batch_first=True),
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
                updates = rnn_output if gate is None else gate(updates, rnn_output)
                all_outputs.append((features + updates)[inv_sorter])
            else:
                features = rnn_output if gate is None else gate(updates, rnn_output)
                all_outputs.append(features[inv_sorter])
        if return_all_layers:
            all_outputs = torch.stack(all_outputs, dim=0)
            cell_states = torch.stack(cell_states, dim=0)
        else:
            all_outputs = all_outputs[-1]
            cell_states = cell_states[-1]
        if all_outputs.shape[-2] != mask.shape[-1]:
            all_outputs = F.pad(all_outputs, (0, 0, 0, mask.shape[-1] - all_outputs.shape[-2]))

        if return_global_state:
            return all_outputs, cell_states
        return all_outputs


@register("pooler")
class Pooler(torch.nn.Module):
    def __init__(self, mode="mean", dropout_p=0., input_size=None, n_heads=None, do_value_proj=False):
        super().__init__()
        self.mode = mode
        assert mode in ("max", "sum", "mean", "attention", "first", "last")
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
            original_dtype = features.dtype
            if features.dtype == torch.int or features.dtype == torch.long:
                features = features.float()
            begins, ends = mask
            if self.mode == "first":
                ends = torch.minimum(begins + 1, ends)
            if self.mode == "last":
                begins = torch.maximum(ends - 1, begins)
            begins = begins.expand(*features.shape[:begins.ndim - 1], begins.shape[-1]).clamp_min(0)
            ends = ends.expand(*features.shape[:begins.ndim - 1], ends.shape[-1]).clamp_min(0)
            final_shape = (*begins.shape, *features.shape[begins.ndim:])
            features = features.view(-1, features.shape[-2], features.shape[-1])
            begins = begins.reshape(features.shape[0], begins.numel() // features.shape[0] if len(features) else 0)
            ends = ends.reshape(features.shape[0], ends.numel() // features.shape[0] if len(features) else 0)

            max_window_size = max(0, int((ends - begins).max())) if 0 not in ends.shape else 0
            flat_indices = torch.arange(max_window_size, device=device)[None, None, :] + begins[..., None]
            flat_indices_mask = flat_indices < ends[..., None]
            flat_indices += torch.arange(len(flat_indices), device=device)[:, None, None] * features.shape[1]

            flat_indices = flat_indices[flat_indices_mask]
            res = F.embedding_bag(
                input=flat_indices,
                weight=self.dropout(features.reshape(-1, features.shape[-1])),
                offsets=torch.cat([torch.tensor([0], device=device), flat_indices_mask.sum(-1).reshape(-1)]).cumsum(0)[:-1].clamp_max(flat_indices.shape[0]),
                mode=self.mode if self.mode not in ("first", "last") else "max",
            ).reshape(final_shape)
            if res.dtype != original_dtype:
                res = res.type(original_dtype)
            return res
        elif torch.is_tensor(mask):
            features = features
            features = self.dropout(features)
            if self.mode == "first":
                mask = ~shift(mask.long(), n=1, dim=-1).cumsum(-1).bool() & mask
            elif self.mode == "last":
                mask = mask.flip(-1)
                mask = (~shift(mask.long(), n=1, dim=-1).cumsum(-1).bool() & mask).flip(-1)

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
            elif self.mode in ("sum", "mean", "first", "last"):
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

    @property
    def output_size(self):
        return self.layers[-1].weight.shape[0]

    def forward(self, input):
        for i, layer in enumerate(self.layers):
            if i > 0:
                input = self.activation_fn(input)
            input = self.dropout(input)
            input = layer(input)
        return input
