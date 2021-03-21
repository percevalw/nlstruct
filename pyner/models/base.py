import functools
import inspect
import textwrap
from collections import namedtuple, Mapping
from copy import deepcopy

import pytorch_lightning as pl
import torch
import transformers

from pyner.data_utils import loop
from pyner.torch_utils import einsum, fork_rng
from .registry import registry

RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


def register(name, do_not_serialize=()):
    def fn(cls):
        class new_cls(cls, Mapping):
            registry_name = name
            _do_not_serialize_ = do_not_serialize
            __doc__ = None

            def __init__(self, *args, **kwargs):
                kwargs.pop("module", None)
                super().__init__(*args, **kwargs)

            functools.update_wrapper(__init__, cls.__init__)
            if __init__.__doc__ is not None:
                __init__.__doc__ = textwrap.dedent(__init__.__doc__.strip("\n"))
            functools.update_wrapper(cls.__call__, cls.forward)

            def __new__(cls, *args, **kwargs):
                module = kwargs.pop("module", None)
                if module is not None:
                    arg_cls = get_module(module)
                    assert issubclass(arg_cls, cls), f"{arg_cls.__name__} is not a subclass of {cls.__name__}"
                    cls = arg_cls
                self = super().__new__(cls)
                args = inspect.getcallargs(cls.__init__, self, *args, **kwargs)
                for arg, value in args.items():
                    if arg != "self" and not arg.startswith('_') and not hasattr(self, arg):
                        self.__dict__[arg] = value
                return self

            def __len__(self):
                return len(get_config(self))

            def __iter__(self):
                return iter(get_config(self))

            def __hash__(self):
                return torch.nn.Module.__hash__(self)

            def __getitem__(self, item):
                return get_config(self)[item]

        new_cls.__name__ = cls.__name__
        registry[name] = new_cls
        return new_cls

    return fn


def get_module(name):
    return registry[name]


def get_instance(kwargs):
    if isinstance(kwargs, torch.nn.Module):
        return deepcopy(kwargs)
    if not isinstance(kwargs, dict):
        return kwargs
    kwargs = dict(kwargs)
    module = kwargs["module"]
    return get_module(module)(**kwargs)


def get_config(self, path=(), drop_unserialized_keys=False):
    config = {"module": getattr(self.__class__, "registry_name", self.__class__.__name__)}
    for key in inspect.getfullargspec(getattr(self.__init__, 'fn', self.__init__)).args[1:]:
        if key.startswith('_') or (drop_unserialized_keys and key in self.__class__._do_not_serialize_):
            continue
        value = getattr(self, key)
        if hasattr(value, 'to_diff_dict'):
            config[key] = value.to_diff_dict()
        elif hasattr(value, 'to_dict'):
            config[key] = value.to_dict()
        elif isinstance(value, torch.nn.ModuleList):
            config[key] = {i: get_config(item, drop_unserialized_keys=drop_unserialized_keys) for i, item in enumerate(value)}
        elif isinstance(value, torch.nn.ModuleDict):
            config[key] = {name: get_config(item, path=(*path, key), drop_unserialized_keys=drop_unserialized_keys) for name, item in value.items()}
        elif isinstance(value, torch.nn.Module):
            config[key] = get_config(value, drop_unserialized_keys=drop_unserialized_keys)
        elif isinstance(value, torch.Tensor):
            pass
        elif isinstance(value, (int, float, str, tuple, list, dict, slice, range)):
            config[key] = value
        elif value is None:
            config[key] = None
        elif isinstance(value, type):
            config[key] = f"{value.__module__}.{value.__name__}" if value.__module__ != "builtins" else value.__name__
        else:
            raise ValueError("Cannot get config from {}".format(str(value)[:40]))
    return config


def identity(x):
    return x


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, epoch_length=None):
        super().__init__()
        self.data = iter(data)
        self.epoch_length = epoch_length

    def __iter__(self):
        return self.data

    def __len__(self):
        if self.epoch_length is not None:
            return self.epoch_length
        raise AttributeError()


class PytorchLightningBase(pl.LightningModule):
    @property
    def train_dataloader(self):
        def fn():
            if getattr(self, 'train_data', None) is None:
                return None
            with fork_rng(self.data_seed):
                prep = self.preprocess(self.train_data, split="train")
                batch_size = getattr(self, 'step_batch_size', self.batch_size)
                non_default_epoch_length = (
                    self.trainer.val_check_interval * batch_size
                    if getattr(self, 'trainer', None) is not None and self.trainer.val_check_interval is not None
                    else None
                )
                if hasattr(prep, '__getitem__') and non_default_epoch_length is None:
                    return torch.utils.data.DataLoader(prep, shuffle=True, batch_size=batch_size, collate_fn=identity)
                elif non_default_epoch_length is not None and hasattr(prep, '__len__'):
                    prep = loop(prep, shuffle=True)
                    return torch.utils.data.DataLoader(
                        DummyIterableDataset(prep, epoch_length=non_default_epoch_length),
                        shuffle=False, batch_size=batch_size, collate_fn=identity)
                else:
                    return torch.utils.data.DataLoader(
                        DummyIterableDataset(prep, epoch_length=non_default_epoch_length),
                        shuffle=False, batch_size=batch_size, collate_fn=identity)

        return fn

    def transfer_batch_to_device(self, inputs, device):
        return inputs

    @property
    def val_dataloader(self):
        def fn():
            if getattr(self, 'val_data', None) is None:
                return None
            with fork_rng(self.data_seed):
                prep = self.preprocess(self.val_data, split="val")
                batch_size = self.batch_size
                if hasattr(prep, '__getitem__'):
                    return torch.utils.data.DataLoader(prep, shuffle=False, batch_size=batch_size, collate_fn=identity)
                else:
                    return torch.utils.data.DataLoader(
                        DummyIterableDataset(prep, None),
                        shuffle=False, batch_size=batch_size, collate_fn=identity)

        return fn

    @property
    def test_dataloader(self):
        def fn():
            if getattr(self, 'test_data', None) is None:
                return None
            with fork_rng(self.data_seed):
                prep = self.preprocess(self.test_data, split="test")
                batch_size = self.batch_size
                if hasattr(prep, '__getitem__'):
                    return torch.utils.data.DataLoader(prep, shuffle=False, batch_size=batch_size, collate_fn=identity)
                else:
                    return torch.utils.data.DataLoader(
                        DummyIterableDataset(prep, None),
                        shuffle=False, batch_size=batch_size, collate_fn=identity)

        return fn

    @train_dataloader.setter
    def train_dataloader(self, data):
        self.train_data = data()
        if hasattr(self.train_data, 'dataset'):
            self.train_data = self.train_data.dataset

    @val_dataloader.setter
    def val_dataloader(self, data):
        self.val_data = data()
        if hasattr(self.val_data, 'dataset'):
            self.val_data = self.val_data.dataset

    @test_dataloader.setter
    def test_dataloader(self, data):
        self.test_data = data()
        if hasattr(self.test_data, 'dataset'):
            self.test_data = self.test_data.dataset


def save_pretrained(self, filename):
    config = get_config(self)
    torch.save({"config": config, "state_dict": self.state_dict()}, filename)


def load_pretrained(path, map_location=None):
    loaded = torch.load(path, map_location=map_location)
    instance = get_instance(loaded["config"])
    instance.load_state_dict(loaded["state_dict"], strict=False)
    instance.eval()
    return instance


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
        self.with_unk = with_unk
        values = (["__pad__"] if with_pad and "__pad__" not in values else []) + (["__unk__"] if with_unk and "__unk__" not in values else []) + list(values)
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
                return self.inversed["__unk__"]
            except KeyError:
                raise KeyError(f"Could not find indice in vocabulary for {repr(obj)}")
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


@register("rezero_gate")
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
    def __init__(self, dim, init_value=1e-3, proj=False, ln_mode="post"):
        super().__init__()
        if proj:
            self.linear = torch.nn.Linear(dim, 1)
        else:
            self.weight = torch.nn.Parameter(torch.ones(1) * init_value)

        self.ln_mode = ln_mode
        if ln_mode is not False:
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, after, before):
        gate = torch.sigmoid(self.weight if hasattr(self, 'weight') else self.linear(after))
        if self.ln_mode == "post":
            return self.norm(before * (1 - gate) + after * gate)
        elif self.ln_mode == "pre":
            return before * (1 - gate) + self.norm(after) * gate
        else:
            return before * (1 - gate) + after * gate


@register("bert")
class BERTEncoder(torch.nn.Module):
    def __init__(self, _bert=None, bert_config=None, path=None, n_layers=4, dropout_p=0.1, freeze_n_layers=-1):
        super().__init__()
        self.bert = _bert if _bert is not None else transformers.AutoModel.from_pretrained(path, config=bert_config)
        self.n_layers = n_layers
        self.weight = torch.nn.Parameter(torch.randn(n_layers))

        if freeze_n_layers == -1:
            freeze_n_layers = len(self.bert.encoder.layer) + 1
        for module in (self.bert.embeddings, *self.bert.encoder.layer)[:freeze_n_layers]:
            for param in module.parameters():
                param.requires_grad = False
        for module in self.bert.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_p

    @property
    def bert_config(self):
        return self.bert.config

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
    def __init__(self, input_size, hidden_size, num_layers=1, gate=False, dropout_p=0.1, bidirectional=True):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
            self.initial_linear = None
        else:
            self.initial_linear = torch.nn.Linear(input_size, hidden_size)

        self.dropout = torch.nn.Dropout(dropout_p)
        if gate is False:
            self.gate_modules = [None] * num_layers
        else:
            self.gate_modules = torch.nn.ModuleList([
                get_instance(gate)
                for _ in range(num_layers)])
        self.lstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=1, bidirectional=bidirectional, batch_first=True)
            for dim in [hidden_size] * num_layers
        ])

    @property
    def gate(self):
        return self.gate_modules[0] if len(self.gate_modules) else False

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
