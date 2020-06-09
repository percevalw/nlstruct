import contextlib
import importlib
import inspect
import os
import tempfile
from os.path import expanduser
from pathlib import Path, PureWindowsPath, PurePosixPath

import yaml
from dotenv import load_dotenv

tmp_dir = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))


# noinspection PyUnresolvedReferences
class RelativePath(Path):
    if os.name == 'nt':
        _flavour = PureWindowsPath._flavour
    else:
        _flavour = PurePosixPath._flavour

    def __new__(cls, *args, **kwargs):
        self = cls._from_parts(args, init=False)
        if not self._flavour.is_supported:
            raise NotImplementedError("cannot instantiate %r on your system"
                                      % (cls.__name__,))
        self._init()
        return self

    def __init__(self, *args, **kwargs):
        super(RelativePath, self).__init__()

    def format(self, *args, **kwargs):
        return type(self)(str(self).format(*args, **kwargs))

    @staticmethod
    def rebuild_from_state(path, path_type):
        if path_type == "cache":
            return _cache_path / path
        if path_type == "resource":
            return _resource_path / path

    def __repr__(self):
        try:
            return f"[resource]/{str(super().relative_to(_resource_path))}"
        except ValueError:
            return f"[cache]/{str(super().relative_to(_cache_path))}"

    def __reduce__(self):
        try:
            return RelativePath.rebuild_from_state, (str(super().relative_to(_resource_path)), "resource")
        except ValueError:
            return RelativePath.rebuild_from_state, (str(super().relative_to(_cache_path)), "cache")

    @classmethod
    def from_yaml(cls, loader, tag_suffix, node):
        assert isinstance(cls.rebuild_from_state(node.value, tag_suffix), RelativePath)
        return cls.rebuild_from_state(node.value, tag_suffix)

    @classmethod
    def to_yaml(cls, dumper, data):
        try:
            return dumper.represent_scalar("!env/resource", str(data.relative_to(_resource_path)))
        except ValueError:
            return dumper.represent_scalar("!env/cache", str(data.relative_to(_cache_path)))

    def relative_to(self, *other):
        return Path(super(RelativePath, self).relative_to(*other))


# Required for safe_load
yaml.SafeLoader.add_multi_constructor('!env/', RelativePath.from_yaml)
yaml.Loader.add_multi_constructor('!env/', RelativePath.from_yaml)
# Required for safe_dump
yaml.SafeDumper.add_multi_representer(RelativePath, RelativePath.to_yaml)
yaml.Dumper.add_multi_representer(RelativePath, RelativePath.to_yaml)


def create_config_if_not_exist():
    home = expanduser("~")
    config_path = home + "/nlstruct.env"
    if os.path.exists(config_path):
        return config_path
    with open(config_path, "w") as file:
        default_config_file_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../example.env'))
        with open(default_config_file_path, "r") as default_file:
            file.write(default_file.read())
            print(f"Config file was created at")
            print("    ", config_path)
            print("please modify it and change values to your prefered paths.")
            print("You can change the config file location by setting the NLSTRUCT_CONFIG_PATH environment variable.")
            print("""Once you're done, execute the following lines:
from dotenv import load_dotenv
load_dotenv({}, override=True)""".format(repr(config_path)))
    return config_path


class DataRoot(object):
    def __init__(self, config_file_path=None):
        if 'NLSTRUCT_CONFIG_PATH' in os.environ:
            self.config_file_path = os.environ['NLSTRUCT_CONFIG_PATH']
        else:
            if config_file_path is None:
                self.config_file_path = create_config_if_not_exist()
            elif config_file_path.startswith('/'):
                self.config_file_path = config_file_path
            elif "/" in config_file_path:
                self.config_file_path = os.path.realpath(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file_path))
            else:
                raise Exception("Cannot process config file path '{}'".format(config_file_path))
            # else:
            #     self.config_file_path = os.path.realpath(
            #         os.path.join(os.path.dirname(os.path.realpath(__file__)),
            #                      '../../configs/{}.env'.format(config_file_path)))
        load_dotenv(self.config_file_path)

    def load_config(self):
        with open(self.config_file_path, 'r') as stream:
            local_config = yaml.load(stream)
        return local_config

    def __getitem__(self, item):
        return os.environ[item]

    @staticmethod
    def get(item, *args):
        return os.environ.get(item, *args)

    @classmethod
    def other(cls, config_file_path):
        return cls(config_file_path)

    def define_basedir(self):
        self.basedir = os.path.abspath(os.path.dirname(inspect.stack()[1][0].f_globals.get('__file__', '.')))

    def resource(self, item):
        return RelativePath("resource", self["RESOURCES_PATH"], item)

    def source(self, item):
        return RelativePath("source", self.basedir, item)

    @classmethod
    def tmp(cls, item=None, ext=None):
        assert (item is None) != (ext is None)
        assert ext is None or "." in ext
        os.makedirs(tmp_dir, exist_ok=True)
        if item is None:
            return os.path.join(tmp_dir, f"{next(tempfile._get_candidate_names())}{ext}")
        return os.path.join(tmp_dir, str(item))


@contextlib.contextmanager
def cwd(d):
    old = os.getcwd()
    os.chdir(d)
    yield
    os.chdir(old)


def import_obj(obj_str):
    parts = obj_str.split('.')
    return getattr(importlib.import_module(f".{parts[-2]}", '.'.join(parts[:-2])), parts[-1])


root = DataRoot()
root.define_basedir()
_resource_path = Path(root["RESOURCES_PATH"])
_cache_path = Path(root["CACHE_PATH"])
