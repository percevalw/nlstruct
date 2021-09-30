import warnings

import parse
import glob
import pytorch_lightning as pl
import xxhash
import re
from nlstruct.registry import get_config
import torch
import os
import traceback


def flat_config(d):
    if d is None:
        return d
    if isinstance(d, dict):
        return tuple(sorted(((k, flat_config(v)) for k, v in sorted(d.items())), key=lambda x: x[0]))
    elif isinstance(d, (list, tuple)) and len(d) > 0 and isinstance(d[0], dict):
        return tuple((flat_config(v) for v in d))
    elif isinstance(d, (list, tuple)):
        return tuple(d)
    elif isinstance(d, (int, bool, str, float)):
        return d
    else:
        return str(d)


def get_hashkey(model):
    xxhash.xxh64(str(flat_config(model.hparams_initial))).hexdigest()
    return xxhash.xxh64(str(flat_config(model.hparams_initial))).hexdigest()


class AlreadyRunningException(Exception):
    pass


class ModelCheckpoint(pl.callbacks.Callback):
    def __init__(self, path, keep_n=1, only_last=False):
        super().__init__()
        if not (path.endswith('.ckpt') or path.endswith('.pt')):
            path = path + ".ckpt"
        assert keep_n is False or keep_n > 0
        self.keep_n = keep_n
        self.path = path
        self._all_logged_metrics = []
        self.last_train_metrics = {}
        self._hashkey = None
        self.only_last = only_last

    @property
    def hashkey(self):
        return self._hashkey

    def list_paths(self, model):

        if self._hashkey is None:
            self._hashkey = get_hashkey(model)
        glob_search = re.sub('{.*?}', '*', self.path.replace("{hashkey}", self._hashkey))
        paths = glob.glob(glob_search, recursive=True)
        parsed_paths = [(path, parse.parse(self.path, path)) for path in paths]

        return parsed_paths

    def lock_file_path(self, model):

        if self._hashkey is None:
            self._hashkey = get_hashkey(model)
        lock_file = self.path.replace("{hashkey}", self._hashkey).format(global_step=0, current_epoch=0).replace(".ckpt", '.lock').replace(".pt", '.lock')

        return lock_file

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):

        parsed_paths = self.list_paths(pl_module)

        finished_path = next(((path, r) for path, r in parsed_paths if r.named.get('global_step', 0) == pl_module.max_steps - 1), (None, None))[0]
        if finished_path is not None:
            pl_module._is_resuming_finished_model = True
        else:
            print("Will save checkpoints under path {}".format(self.path.replace("{hashkey}", self._hashkey)))
            lock_file_path = self.lock_file_path(pl_module)
            if os.path.exists(lock_file_path):
                raise AlreadyRunningException("Found a lock file {} indicating that the experiment is already running.".format(lock_file_path))
            else:
                open(lock_file_path, 'a').close()

    def on_fit_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):
        pl_module._is_resuming_finished_model = False

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):
        lock_file_path = self.lock_file_path(pl_module)
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):
        parsed_paths = self.list_paths(pl_module)

        if not len(parsed_paths):
            return
        latest_path = max(parsed_paths, key=lambda r: r[1].named.get('global_step', 0))[0]
        print("Resuming from {}".format(latest_path))
        state = torch.load(latest_path)

        pl_module._load_state(state)
        if "all_logged_metrics" in state:
            self._all_logged_metrics = state["all_logged_metrics"]
            for log_dict in state["all_logged_metrics"]:
                trainer.logger.log_metrics(log_dict)

    def on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):
        if trainer.global_step == 0:
            return
        self._all_logged_metrics.append({**trainer.logged_metrics, "step": int(trainer.global_step) - 1, "epoch": int(trainer.current_epoch)})

        state = pl_module._save_state(increment_step=True)
        state["all_logged_metrics"] = self._all_logged_metrics

        if self._hashkey is None:
            self._hashkey = get_hashkey(pl_module)

        save_path = self.path.format(
            global_step=trainer.global_step - 1,
            epoch=trainer.current_epoch,
            hashkey=self._hashkey)
        if not self.only_last or trainer.global_step == pl_module.max_steps:
            torch.save(state, save_path + ".tmp")
            os.rename(save_path + ".tmp", save_path)

        parsed_paths = self.list_paths(pl_module)
        if self.keep_n is not False:
            for remove_path, _ in sorted(parsed_paths, key=lambda r: r[1].named.get('global_step', 0))[:-self.keep_n]:
                os.remove(remove_path)
