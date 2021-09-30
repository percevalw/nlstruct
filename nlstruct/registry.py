import inspect
import functools
import torch
from collections import Mapping
from copy import deepcopy
import abc

registry = {}


class RegistryMetaclass(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        module = kwargs.pop("module", None)
        if module is not None:
            arg_cls = get_module(module)
            if arg_cls is not cls:
                assert issubclass(arg_cls, cls), f"{arg_cls.__name__} is not a subclass of {cls.__name__}"
                cls = arg_cls
                return cls(*args, **kwargs)
        return super().__call__(*args, **kwargs)


def save_pretrained(self, filename):
    config = get_config(self)
    torch.save({"config": config, "state_dict": self.state_dict()}, filename)


def register(name, do_not_serialize=()):
    def fn(base_cls):
        class new_cls(base_cls, Mapping, metaclass=RegistryMetaclass):
            registry_name = name
            _do_not_serialize_ = do_not_serialize
            __doc__ = None

            def __init__(self, *args, **kwargs):
                kwargs.pop("module", None)

                super().__init__(*args, **kwargs)

                base_init = base_cls.__init__
                args = inspect.getcallargs(base_init, self, *args, **kwargs)
                for arg, value in args.items():
                    if arg != "self" and not arg.startswith('_') and not hasattr(self, arg):
                        self.__dict__[arg] = value

            functools.update_wrapper(__init__, base_cls.__init__)
            if hasattr(base_cls.__init__, '__doc__'):
                __init__.__doc__ = base_cls.__init__.__doc__
            functools.update_wrapper(base_cls.__call__, base_cls.forward)
            __init__.fn = getattr(base_cls.__init__, 'fn', base_cls.__init__)

            def __len__(self):
                return len(get_config(self))

            def __iter__(self):
                return iter(get_config(self))

            def __hash__(self):
                return torch.nn.Module.__hash__(self)

            def __getitem__(self, item):
                return get_config(self)[item]

            save_pretrained = save_pretrained

        new_cls.__name__ = base_cls.__name__
        registry[name] = new_cls
        return new_cls

    return fn


def get_module(name):
    if isinstance(name, str):
        return registry[name]
    else:
        return name


def get_instance(kwargs):
    if isinstance(kwargs, torch.nn.Module):
        return deepcopy(kwargs)
    if not isinstance(kwargs, dict):
        return kwargs
    kwargs = dict(kwargs)
    module = kwargs["module"]
    return get_module(module)(**kwargs)


def get_config(self, path=(), drop_unserialized_keys=True):
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


def merge_configs(*configs):
    if len(configs) == 1:
        return configs[0]
    a, b, rest = configs[-2], configs[-1], configs[:-2]
    if len(rest) > 0:
        return merge_configs(*rest, merge_configs(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        a = dict(a)
        for key in b.keys():
            if key in a:
                a[key] = merge_configs(a[key], b[key])
            else:
                a[key] = b[key]
        return a
    return b
