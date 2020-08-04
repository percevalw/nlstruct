import glob
import inspect
import io
import logging
import os
import pickle
import sys
import types
from collections import Sequence, Mapping, defaultdict
from contextlib import contextmanager
from copy import copy
from copyreg import dispatch_table
from pathlib import Path
from pickle import PicklingError

import joblib
import numpy as np
from pandas.core.internals.managers import BlockManager
from pandas.core.base import PandasObject
from pandas import DataFrame
import torch
import xxhash
import yaml
from joblib.hashing import Hasher, _MyHash
from joblib.numpy_pickle import NumpyArrayWrapper, NumpyUnpickler
from joblib.numpy_pickle_compat import NDArrayWrapper
from joblib.numpy_pickle_utils import _read_fileobject
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator

from nlstruct.environment.path import RelativePath, root

logger = logging.getLogger("nlstruct")

PY3_OR_LATER = sys.version_info[0] >= 3
if PY3_OR_LATER:
    Pickler = pickle._Pickler
else:
    Pickler = pickle.Pickler


class HashWriter:
    def __init__(self):
        self._hash = xxhash.xxh64()

    def write(self, data):
        self._hash.update(data)

    def hexdigest(self):
        return self._hash.hexdigest()


class Dispatcher(dict):
    def __getitem__(self, item):
        try:
            return super(Dispatcher, self).__getitem__(item)
        except KeyError:
            res = next((f for v, f in self.items() if issubclass(item, v)), None)
            if res is None:
                res = Hasher.dispatch.get(item, None)
                if res is None:
                    raise
            return res

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default


class TruncatedNumpyHasher(Hasher):
    def __init__(self, max_length=2000, coerce_mmap=False):

        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        protocol = (pickle.DEFAULT_PROTOCOL if PY3_OR_LATER else pickle.HIGHEST_PROTOCOL)
        self.stream = HashWriter()
        Pickler.__init__(self, self.stream, protocol=protocol)

        self.coerce_mmap = coerce_mmap
        # delayed import of numpy, to avoid tight coupling
        import numpy as np
        self.np = np
        if hasattr(np, 'getbuffer'):
            self._getbuffer = np.getbuffer
        else:
            self._getbuffer = memoryview

        self.max_length = max_length

    def hash(self, obj, return_digest=True):
        try:
            self.dump(obj)
        except pickle.PicklingError as e:
            e.args += ('PicklingError while hashing %r: %r' % (obj, e),)
            raise
        if return_digest:
            hexdigest = self.stream.hexdigest()
            return hexdigest

    def save_memoryview(self, obj):
        self.write(b'BUFFER')
        self.write(obj)

    def save(self, obj, save_persistent_id=True, bypass_dispatch=False):
        if not bypass_dispatch:
            self.framer.commit_frame()

            # Check for persistent id (defined by a subclass)
            pid = self.persistent_id(obj)
            if pid is not None and save_persistent_id:
                self.save_pers(pid)
                return

            # Check the memo
            x = self.memo.get(id(obj))
            if x is not None:
                self.write(self.get(x[0]))
                return

            # Check the type dispatch table
            t = type(obj)
            f = self.dispatch.get(t)
            if f is not None:
                f(self, obj)  # Call unbound method with explicit self
                return
        else:
            t = type(obj)

        # Check private dispatch table if any, or else copyreg.dispatch_table
        reduce = getattr(self, 'dispatch_table', dispatch_table).get(t)
        if reduce is not None:
            rv = reduce(obj)
        else:
            # Check for a class with a custom metaclass; treat as regular class
            try:
                issc = issubclass(t, type)
            except TypeError:  # t is not a class (old Boost; see SF #502085)
                issc = False
            if issc:
                self.save_global(obj)
                return

            # Check for a __reduce_ex__ method, fall back to __reduce__
            reduce = getattr(obj, "__reduce_ex__", None)
            if reduce is not None:
                rv = reduce(self.proto)
            else:
                reduce = getattr(obj, "__reduce__", None)
                if reduce is not None:
                    rv = reduce()
                else:
                    raise PicklingError("Can't pickle %r object: %r" %
                                        (t.__name__, obj))

        # Check for string returned by reduce(), meaning "save as global"
        if isinstance(rv, str):
            self.save_global(obj, rv)
            return

        # Assert that reduce() returned a tuple
        if not isinstance(rv, tuple):
            raise PicklingError("%s must return string or tuple" % reduce)

        # Assert that it returned an appropriately sized tuple
        l = len(rv)
        if not (2 <= l <= 5):
            raise PicklingError("Tuple returned by %s must have "
                                "two to five elements" % reduce)

        # Save the reduce() output and finally memoize the object
        self.save_reduce(obj=obj, *rv)

    def save_dataframe(self, obj):
        self.save(hash_object(obj.__getstate__()))

    # noinspection PyProtectedMember
    def save_blockmanager(self, obj):
        obj = next(iter(obj.__getstate__()[3].values()))
        for block in obj['blocks']:
            block['values'] = block['values'].T
        self.save(obj)

    # noinspection PyProtectedMember
    def save_pandas_object(self, obj):
        if hasattr(obj, '__getstate__'):
            self.save(obj, bypass_dispatch=True)
            return
        reduce = getattr(obj, "__reduce_ex__", None)
        if reduce is not None:
            rv = reduce(self.proto)
        else:
            reduce = getattr(obj, "__reduce__", None)
            if reduce is not None:
                rv = reduce()
            else:
                raise PicklingError("Can't pickle %r object: %r" % (type(obj).__name__, obj))
        # Assert that reduce() returned a tuple
        if isinstance(rv, str):
            self.save_global(obj, rv)
            return
        if not isinstance(rv, tuple):
            raise PicklingError("%s must return string or tuple" % reduce)
        if rv[2] is not None and isinstance(rv, dict):
            if '_cache' in rv[2]:
                del rv[2]['_cache']
            if '_cacher' in rv[2]:
                del rv[2]['_cacher']
            if '_ordered' in rv[2]:
                del rv[2]['_ordered']
        self.save_reduce(obj=obj, *rv)

    def save_ndarray(self, obj):
        if not obj.dtype.hasobject:
            obj = obj[tuple(slice(min(size, self.max_length)) for size in obj.shape)]
            # Compute a hash of the object
            # The update function of the hash requires a c_contiguous buffer.
            if obj.shape == ():
                # 0d arrays need to be flattened because viewing them as bytes
                # raises a ValueError exception.
                obj_c_contiguous = obj.flatten()
            elif obj.flags.c_contiguous:
                obj_c_contiguous = obj
            elif obj.flags.f_contiguous:
                obj_c_contiguous = obj.T
            else:
                # Cater for non-single-segment arrays: this creates a
                # copy, and thus aleviates this issue.
                # XXX: There might be a more efficient way of doing this
                obj_c_contiguous = obj.flatten()

            # memoryview is not supported for some dtypes, e.g. datetime64, see
            # https://github.com/numpy/numpy/issues/4983. The
            # workaround is to view the array as bytes before
            # taking the memoryview.

            # We store the class, to be able to distinguish between
            # Objects with the same binary content, but different
            # classes.
            if self.coerce_mmap and isinstance(obj, self.np.memmap):
                # We don't make the difference between memmap and
                # normal ndarrays, to be able to reload previously
                # computed results with memmap.
                klass = self.np.ndarray
            else:
                klass = obj.__class__
            # We also return the dtype and the shape, to distinguish
            # different views on the same data with different dtypes.

            # The object will be pickled by the pickler hashed at the end.
            obj = (klass, 'HASHED', obj.dtype, obj.shape, obj.strides)
            self.save(obj)
            self.save_memoryview(self._getbuffer(obj_c_contiguous.view(self.np.uint8)))
        else:
            self.save(obj, bypass_dispatch=True)

    def save_ndtype(self, obj):
        klass = obj.__class__
        obj = (klass, ('HASHED', obj.descr))
        self.save(obj)

    def save_mapping(self, obj):
        self.save((type(obj), sorted(obj.items(), key=lambda x: hash_object(x[0]))))

    def save_tuple(self, obj):
        if not obj:  # tuple is empty
            if self.bin:
                self.write(pickle.EMPTY_TUPLE)
            else:
                self.write(pickle.MARK + pickle.TUPLE)
            return

        n = len(obj)
        save = self.save
        memo = self.memo
        if n <= 3 and self.proto >= 2:
            for element in obj:
                save(element)
            # Subtle.  Same as in the big comment below.
            if id(obj) in memo:
                get = self.get(memo[id(obj)][0])
                self.write(pickle.POP * n + get)
            else:
                self.write(pickle._tuplesize2code[n])
                self.memoize(obj)
            return

        # proto 0 or proto 1 and tuple isn't empty, or proto > 1 and tuple
        # has more than 3 elements.
        write = self.write
        write(pickle.MARK)
        for element in obj:
            save(element)

        if id(obj) in memo:
            # Subtle.  d was not in memo when we entered save_tuple(), so
            # the process of saving the tuple's elements must have saved
            # the tuple itself:  the tuple is recursive.  The proper action
            # now is to throw away everything we put on the stack, and
            # simply GET the tuple (it's already constructed).  This check
            # could have been done in the "for element" loop instead, but
            # recursive tuples are a rare thing.
            get = self.get(memo[id(obj)][0])
            if self.bin:
                write(pickle.POP_MARK + get)
            else:  # proto 0 -- POP_MARK not available
                write(pickle.POP * (n + 1) + get)
            return

        # No recursion.
        write(pickle.TUPLE)
        self.memoize(obj)

    def save_sequence(self, obj):
        if self.bin:
            self.write(pickle.EMPTY_LIST)
        else:  # proto 0 -- can't use EMPTY_LIST
            self.write(pickle.MARK + pickle.LIST)

        self.memoize(obj)
        if len(obj) > self.max_length:
            self._batch_appends(obj[:self.max_length // 2])
            self._batch_appends(obj[-self.max_length // 2:])
        else:
            self._batch_appends(obj)

    def save_optimizer(self, obj):
        new_optim = copy(obj)
        new_optim.param_groups = []
        for i, g in enumerate(obj.param_groups):
            new_g = dict(g)
            new_g['params'] = [p for p in new_g['params'] if p.requires_grad]
            if new_g['params']:
                new_optim.param_groups.append(new_g)
        new_optim.state = defaultdict(dict)
        for k, v in obj.state.items():
            if k.requires_grad:
                new_optim.state[k] = v
        self.save(new_optim, bypass_dispatch=True)

    def save_tensor(self, obj):
        self.save(obj.detach().cpu().numpy()[tuple(slice(min(size, self.max_length)) for size in obj.shape)])

    def save_spmatrix(self, obj):
        self.save(obj.tocsr(), bypass_dispatch=True)

    def save_parameter(self, obj):
        obj = (obj.detach().cpu().numpy()[tuple(slice(min(size, self.max_length)) for size in obj.shape)],
               obj.requires_grad)
        self.save(obj)

    def save_base_estimator(self, obj):
        self.save((obj.__class__, ("HASHER", obj.get_params())))

    def save_method(self, obj):
        # the Pickler cannot pickle instance methods; here we decompose
        # them into components that make them uniquely identifiable
        if hasattr(obj, '__func__'):
            func_name = obj.__func__.__name__
        else:
            func_name = obj.__name__
        inst = obj.__self__
        if type(inst) == type(pickle):
            obj = _MyHash(func_name, inst.__name__)
        elif inst is None:
            # type(None) or type(module) do not pickle
            obj = _MyHash(func_name, inst)
        else:
            cls = obj.__self__.__class__
            obj = _MyHash(func_name, inst, cls)
        self.save(obj)

    def save_type(self, obj):
        if hasattr(obj, '__name__'):
            self.save(("HASHED_TYPE", obj.__name__, obj.__mro__[1:]))
            self.memoize(obj)
        else:
            self.save(obj, bypass_dispatch=True)

    def no_save(self, obj):
        pass

    dispatch = Dispatcher()
    dispatch[tuple] = save_tuple
    dispatch[type] = save_type
    dispatch[str] = Pickler.save_str
    dispatch[memoryview] = save_memoryview
    dispatch[types.MethodType] = save_method
    dispatch[type({}.pop)] = save_method
    dispatch[DataFrame] = save_dataframe
    dispatch[BlockManager] = save_blockmanager
    dispatch[PandasObject] = save_pandas_object
    dispatch[np.ndarray] = save_ndarray
    dispatch[spmatrix] = save_spmatrix
    dispatch[np.dtype] = save_ndtype
    dispatch[torch.nn.Parameter] = save_parameter
    dispatch[torch.Tensor] = save_tensor
    dispatch[torch.device] = no_save
    dispatch[Mapping] = save_mapping
    dispatch[Sequence] = save_sequence
    dispatch[torch.optim.Optimizer] = save_optimizer
    dispatch[BaseEstimator] = save_base_estimator

    def _batch_setitems(self, items):
        # We assume that we are in Python 3.6+, where dict item order is consistent:
        # no need to sort the keys anymore like joblib's hasher used to do
        Pickler._batch_setitems(self, items)


def hash_object(obj, max_length=2000, mmap_mode=None):
    return TruncatedNumpyHasher(max_length=max_length, coerce_mmap=mmap_mode is not None).hash(obj)


class CacheHandle(RelativePath):
    def __init__(self, path, loader=None, dumper=None):
        super(CacheHandle, self).__init__(path)
        self.loader = loader if loader is not None else load
        self.dumper = dumper if dumper is not None else dump
        logging.info(f"Using cache {self}")

    def entry(self, name):
        os.makedirs(str(self), exist_ok=True)
        return RelativePath(os.path.join(str(self), name))

    def tmp(self, item):
        return root.tmp(os.path.join(item))

    def load(self, source="output.pkl", loader=None, verbose=True, **kwargs):
        path = str(self / source)
        if os.path.exists(str(path)):
            if verbose:
                logging.info(f"Loading {path}... ")
            if loader is None:
                res = self.loader(path, **kwargs)
            else:
                res = loader(path, **kwargs)
            return res
        return None

    def dump(self, obj, dest="output.pkl", dumper=None, **kwargs):
        assert obj is not None
        path = str(self / dest)
        if dumper is None:
            self.dumper(obj, path, **kwargs)
        else:
            dumper(obj, path, **kwargs)
        return path

    def listdir(self, glob_expr="*"):
        return sorted(glob.glob(os.path.join(str(self), glob_expr)))

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.as_posix())


class RAMCacheHandle(RelativePath):
    records = {}

    def __init__(self, path):
        super(RAMCacheHandle, self).__init__(path)

    def entry(self, name):
        return RelativePath(os.path.join(str(self), name))

    def tmp(self, item):
        raise NotImplementedError()

    def load(self, name="output.pkl", **kwargs):
        path = str(self / name)
        if str(path) in self.records:
            record, is_io = self.records[str(path)]
            if is_io:
                record = record.getvalue()
            return record
        return None

    def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None):
        f = io.BytesIO() if 'b' in mode else io.StringIO()
        self.records[str(self)] = (f, True)
        return f

    def dump(self, obj, name="output.pkl", **kwargs):
        assert obj is not None
        path = str(self / name)
        self.records[str(path)] = (obj, False)
        return path

    def listdir(self, glob_expr="*"):
        raise NotImplementedError()

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.as_posix())


class CustomUnpickler(NumpyUnpickler):
    def __init__(self, filename, file_handle, mmap_mode=None, current_cache=None):
        super(CustomUnpickler, self).__init__(filename, file_handle, mmap_mode)
        self.current_cache = current_cache

    dispatch = NumpyUnpickler.dispatch.copy()

    def load_build(self):
        stack = self.stack
        state = stack.pop()
        inst = stack[-1]
        if isinstance(inst, RelativePath):
            inst.suffix = state['suffix']
            inst.source = state['source']
            if state['source'] == "resource":
                inst.prefix = root['RESOURCES_PATH']
            elif state['source'] == "source":
                inst.prefix = root.basedir
            elif state['source'] == "cache":
                if self.current_cache is not None:
                    inst.prefix = self.current_cache
                else:
                    inst.prefix = state['prefix']
            else:
                raise Exception(f"Unrecognized file source {state['source']}")
            return

        ##########################################
        #      Default Unpickler behavior        #
        ##########################################
        setstate = getattr(inst, "__setstate__", None)
        if setstate is not None:
            setstate(state)
            return
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if state:
            inst_dict = inst.__dict__
            intern = sys.intern
            for k, v in state.items():
                if type(k) is str:
                    inst_dict[intern(k)] = v
                else:
                    inst_dict[k] = v
        if slotstate:
            for k, v in slotstate.items():
                setattr(inst, k, v)

        ##########################################
        #       Numpy Unpickler behavior         #
        ##########################################
        # For backward compatibility, we support NDArrayWrapper objects.
        if isinstance(self.stack[-1], (NDArrayWrapper, NumpyArrayWrapper)):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, "
                                  "but numpy didn't import correctly")
            array_wrapper = self.stack.pop()
            # If any NDArrayWrapper is found, we switch to compatibility mode,
            # this will be used to raise a DeprecationWarning to the user at
            # the end of the unpickling.
            if isinstance(array_wrapper, NDArrayWrapper):
                self.compat_mode = True
            self.stack.append(array_wrapper.read(self))

    dispatch[pickle.BUILD[0]] = load_build


def load(filename, mmap_mode=None, current_cache=None):
    """Unpickling function."""
    if hasattr(filename, 'read'):
        fobj = filename
        filename = getattr(fobj, 'name', '')
        with _read_fileobject(fobj, filename, mmap_mode) as fobj:
            unpickler = CustomUnpickler(filename, fobj, mmap_mode=mmap_mode, current_cache=current_cache)
            return unpickler.load()
    with open(filename, 'rb') as f:
        with _read_fileobject(f, filename, mmap_mode) as fobj:
            unpickler = CustomUnpickler(filename, fobj, mmap_mode=mmap_mode, current_cache=current_cache)
            return unpickler.load()


dump = joblib.dump


def yaml_dump(obj, dest=None):
    if isinstance(dest, str):
        with open(dest, "w") as f:
            return yaml.dump(obj, f, sort_keys=False)
    else:
        return yaml.dump(obj, dest, sort_keys=False)


def yaml_load(dest):
    if isinstance(dest, str):
        with open(dest, "r") as f:
            return yaml.unsafe_load(f)
    return yaml.unsafe_load(dest)


def text_dump(obj, dest=None):
    if isinstance(dest, str):
        with open(dest, "w") as f:
            return f.write(obj)
    dest.write(obj)


def text_load(dest):
    if isinstance(dest, str):
        with open(dest, "r") as f:
            return f.read()
    return dest.read()


def get_class_that_defined_method(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
    return None


class cached(object):
    MAP = {}

    @classmethod
    def will_ignore(cls, names):
        def apply_on_func(func):
            func._ignore_args = tuple(names)
            return func

        return apply_on_func

    def __init__(self, with_state=False, hash_only=None, ram=False, save_log=True, ignore=None, loader=None, dumper=None, default_cache_mode="rw"):
        self.ready = False
        self.with_state = with_state
        self.cls = None
        self.ignore = ignore
        self.hash_only = hash_only
        self.ram = ram
        self.save_log = save_log
        self.loader = loader
        self.dumper = dumper
        self.default_cache_mode = default_cache_mode
        self.func = None
        self.caller_self = None

        if hasattr(with_state, '__call__'):
            fn = with_state
            self.with_state = False
            self(fn)

    @property
    def nocache(self):
        return self.func.__get__(self.caller_self) if hasattr(self.func, '__get__') and self.caller_self is not None else self.func

    def __reduce__(self):
        if self.func is not None:
            name = self.func.__name__
        return object.__reduce__(self)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        self.caller_self = instance
        return self

    def __call__(self, *args, cache=None, **kwargs):
        if self.ready:
            if self.caller_self is not None:
                args = (self.caller_self, *args)
            if global_no_cache:
                return self.func(*args, **kwargs)
            sig = inspect.signature(self.func)
            bound_arguments = sig.bind_partial(*args, **kwargs)
            bound_arguments.apply_defaults()
            # Get an ordered dict
            for name in self.ignore:
                if name in bound_arguments.arguments:
                    del bound_arguments.arguments[name]
            expect_cache_handle = False
            if '_cache' in sig.parameters:
                expect_cache_handle = True
                if "_cache" in bound_arguments.arguments:
                    del bound_arguments.arguments['_cache']

            if self.cls is not None:
                keys = (self.cls.__name__, self.func.__name__)
            else:
                keys = (self.func.__name__,)

            if self.with_state:
                (caller_self, *args) = args
                cache_mode = self.default_cache_mode if cache is None else cache
                if not cache_mode:
                    return self.func(*args, **kwargs)
                if self.hash_only is not None:
                    handle = get_cache(keys,
                                       self.hash_only(caller_self, *bound_arguments.args, **bound_arguments.kwargs),
                                       on_ram=self.ram,
                                       loader=self.loader,
                                       dumper=self.dumper)
                else:
                    handle = get_cache(keys,
                                       (caller_self, bound_arguments.args, bound_arguments.kwargs),
                                       on_ram=self.ram,
                                       loader=self.loader,
                                       dumper=self.dumper, )
                cached_result = handle.load() if "r" in cache_mode else None

                if cached_result is None:
                    root_logger = log_file_handler = None
                    if "w" in cache_mode and self.save_log:
                        root_logger = logging.getLogger()
                        log_file_handler = logging.StreamHandler((handle/"info.log").open(mode="w"))
                        log_file_handler.setFormatter("")
                        root_logger.addHandler(log_file_handler)

                    if expect_cache_handle:
                        result = self.func(caller_self, *args, _cache=handle, **kwargs)
                    else:
                        result = self.func(caller_self, *args, **kwargs)

                    if "w" in cache_mode:
                        if log_file_handler is not None:
                            root_logger.removeHandler(log_file_handler)
                            log_file_handler.stream.close()
                        filename = handle.dump((caller_self, result))
                else:
                    old_log = handle.load("info.log", loader=text_load, verbose=False)
                    if old_log:
                        for line in old_log.split("\n"):
                            if line:
                                logger.info(line)
                    cached_self, result = cached_result
                    caller_self.__dict__ = cached_self.__dict__
                return result
            else:
                cache_mode = self.default_cache_mode if cache is None else cache
                if not cache_mode:
                    return self.func(*args, **kwargs)
                if self.hash_only is not None:
                    handle = get_cache(keys,
                                       self.hash_only(*bound_arguments.args, **bound_arguments.kwargs),
                                       on_ram=self.ram,
                                       loader=self.loader,
                                       dumper=self.dumper)
                else:
                    handle = get_cache(keys, (bound_arguments.args, bound_arguments.kwargs),
                                       on_ram=self.ram,
                                       loader=self.loader,
                                       dumper=self.dumper, )
                result = handle.load() if "r" in cache_mode else None

                if result is None:
                    root_logger = log_file_handler = None
                    if "w" in cache_mode and self.save_log:
                        root_logger = logging.getLogger()
                        log_file_handler = logging.StreamHandler((handle / "info.log").open(mode="w"))
                        log_file_handler.setFormatter("")
                        root_logger.addHandler(log_file_handler)

                    if expect_cache_handle:
                        result = self.func(*args, _cache=handle, **kwargs)
                    else:
                        result = self.func(*args, **kwargs)

                    if "w" in cache_mode:
                        if log_file_handler is not None:
                            root_logger.removeHandler(log_file_handler)
                            log_file_handler.stream.close()
                        filename = handle.dump(result)
                else:
                    old_log = handle.load("info.log", loader=text_load, verbose=False)
                    if old_log is not None:
                        for line in old_log.split("\n"):
                            if line:
                                logger.info(line)
                return result
        else:
            func = args[0]
            if self.ignore is None:
                self.ignore = getattr(func, '_ignore_args', ())
            cache_key = (func, self.with_state, self.ram, self.save_log, self.ignore, self.loader, self.dumper, self.default_cache_mode)
            if cache_key in cached.MAP:
                return cached.MAP[cache_key]
            self.cls = get_class_that_defined_method(func)
            self.func = func
            self.ready = True
            self.__name__ = func.__name__
            self.__signature__ = inspect.signature(func)
            self.__doc__ = func.__doc__
            self.__module__ = func.__module__
            cached.MAP[cache_key] = self
            return self


def get_cache(keys, args=None, loader=None, dumper=None, on_ram=False):
    """
    Get a unique cache object for given identifier and args

    Parameters
    ----------
    keys: (typing.Sequence of str) or str or int or RelativePath
    args: Any
    loader:
    dumper:

    Returns
    -------
    CacheHandle
    """
    if isinstance(keys, RelativePath):
        keys = str(keys.relative_to(Path(root["CACHE_PATH"])))
    if isinstance(keys, str):
        keys = [*str(keys).split("/")]
    elif not hasattr(keys, '__len__'):
        keys = [str(keys)]
    keys = list(keys) + [hash_object(args, mmap_mode=None)]
    if on_ram:
        cache_handle = RAMCacheHandle(os.path.join(root['CACHE_PATH'], *keys))
    else:
        cache_handle = CacheHandle(os.path.join(root['CACHE_PATH'], *keys), loader=loader, dumper=dumper)
        inputs_path = cache_handle.entry("inputs.txt")
        # TODO print input files top structures like dist / lists into an inputs.txt file
        if not inputs_path.exists() and args is not None:
            with open(str(cache_handle.entry("inputs.txt")), "w") as inputs_file:
                inputs_file.write("{}({})\n".format(
                    ".".join(keys),
                    ", ".join((  # *("[{}]{}".format(hash_object(a), repr(a)) for a in args[0]),
                        # FIXIT: repr(val) bellow may take a long long time for big collections
                        (f"[{hash_object(val)}]{name}={repr(val)}"
                         for name, val in (args.items() if hasattr(args, 'items') else zip(range(len(args)), args)))))))
    return cache_handle


global_no_cache = False


@contextmanager
def nocache():
    global global_no_cache
    last = global_no_cache
    global_no_cache = True
    yield
    global_no_cache = last
