import inspect
import types
from collections import Mapping

import pytorch_lightning as pl
import torch
import transformers

from pyner.torch_utils import einsum, fork_rng, monkey_patch

monkey_patch()

if "registry" not in globals():
    registry = {}


def set_closure_variable(fn, name, new_obj):
    def dummy():
        return new_obj

    new_closure = list(fn.__closure__ or ())
    if name in fn.__code__.co_freevars:
        new_closure[fn.__code__.co_freevars.index(name)] = dummy.__closure__[0]
        code = fn.__code__
    else:
        c = fn.__code__
        code = types.CodeType(c.co_argcount, c.co_kwonlyargcount, c.co_nlocals,
                              c.co_stacksize, c.co_flags, c.co_code, c.co_consts, c.co_names,
                              c.co_varnames, c.co_filename, c.co_name, c.co_firstlineno,
                              c.co_lnotab, c.co_freevars + ('__class__',), c.co_cellvars)
        new_closure.append(dummy.__closure__[0])

    return types.FunctionType(
        code,
        fn.__globals__,
        fn.__name__,
        fn.__defaults__,
        tuple(new_closure),
    )


def register(name, do_not_serialize=()):
    def fn(cls):
        mro = list(cls.mro()[1:])
        torch_index = mro.index(torch.nn.Module)
        mro[torch_index + 1:torch_index + 1] = [SerializableModule]
        new_cls = type(cls.__name__, tuple(mro), dict(cls.__dict__))
        new_cls._do_not_serialize_ = do_not_serialize
        new_cls.__init__ = set_closure_variable(cls.__init__, '__class__', new_cls)
        new_cls.forward = set_closure_variable(cls.forward, '__class__', new_cls)
        new_cls.__hash__ = torch.nn.Module.__hash__
        new_cls.registry_name = name
        registry[name] = new_cls
        return new_cls

    return fn


class SerializableModule(Mapping):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        args = inspect.getcallargs(cls.__init__, self, *args, **kwargs)
        for arg, value in args.items():
            if arg != "self" and arg not in self._do_not_serialize_ and not arg.startswith('_') and not hasattr(self, arg):
                self.__dict__[arg] = value
        return self

    def __len__(self):
        return len(get_config(self))

    def __iter__(self):
        return iter(get_config(self))

    def __getitem__(self, item):
        return get_config(self)[item]


def get_module(name):
    return registry[name]


def get_instance(kwargs, defaults={}):
    if not isinstance(kwargs, dict):
        return kwargs
    kwargs = {**defaults, **kwargs}
    module = kwargs.pop("module")
    return get_module(module)(**kwargs)


def get_config(self, path=()):
    config = {"module": getattr(self.__class__, "registry_name", self.__class__.__name__)}
    for key in inspect.getfullargspec(getattr(self.__init__, 'fn', self.__init__)).args[1:]:
        if key.startswith('_'):
            continue
        value = getattr(self, key)
        if hasattr(value, 'to_diff_dict'):
            config[key] = value.to_diff_dict()
        elif hasattr(value, 'to_dict'):
            config[key] = value.to_dict()
        elif isinstance(value, torch.nn.ModuleList):
            config[key] = {i: get_config(item) for i, item in enumerate(value)}
        elif isinstance(value, torch.nn.ModuleDict):
            config[key] = {name: get_config(item, path=(*path, key)) for name, item in value.items()}
        elif isinstance(value, torch.nn.Module):
            config[key] = get_config(value)
        elif isinstance(value, torch.Tensor):
            pass
        elif isinstance(value, (int, float, str, tuple, list, dict)):
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
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __iter__(self):
        return iter(self.data)


class PytorchLightningBase(pl.LightningModule):
    @property
    def train_dataloader(self):
        def fn():
            if getattr(self, 'train_data', None) is None:
                return None
            prep = self.preprocess(self.train_data, split="train")
            with fork_rng(self.data_seed):
                return (
                    torch.utils.data.DataLoader(prep, shuffle=True, batch_size=self.batch_size, collate_fn=identity) if hasattr(prep, '__getitem__')
                    else torch.utils.data.DataLoader(DummyIterableDataset(prep), shuffle=False, batch_size=self.batch_size, collate_fn=identity))

        return fn

    @property
    def val_dataloader(self):
        def fn():
            if getattr(self, 'val_data', None) is None:
                return None
            prep = self.preprocess(self.val_data, split="val")
            return (
                torch.utils.data.DataLoader(prep, shuffle=False, batch_size=self.batch_size, collate_fn=identity) if hasattr(prep, '__getitem__')
                else torch.utils.data.DataLoader(DummyIterableDataset(prep), shuffle=False, batch_size=self.batch_size, collate_fn=identity))

        return fn

    @property
    def test_dataloader(self):
        def fn():
            if getattr(self, 'test_data', None) is None:
                return None
            prep = self.preprocess(self.test_data, split="test")
            return (
                torch.utils.data.DataLoader(prep, shuffle=False, batch_size=self.batch_size, collate_fn=identity) if hasattr(prep, '__getitem__')
                else torch.utils.data.DataLoader(DummyIterableDataset(prep), shuffle=False, batch_size=self.batch_size, collate_fn=identity))

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
    instance.load_state_dict(loaded["state_dict"])
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

    def __getitem__(self, obj):
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
