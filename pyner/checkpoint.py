import warnings

import parse
import glob
import pytorch_lightning as pl
import xxhash
import re
from pyner.registry import get_config
import torch
import os


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
    def __init__(self, path, keep_n=1, do_lock_experiment=True):
        super().__init__()
        if not (path.endswith('.ckpt') or path.endswith('.pt')):
            path = path + ".ckpt"
        self.do_lock_experiment = do_lock_experiment
        assert keep_n is False or keep_n > 0
        self.keep_n = keep_n
        self.path = path
        self._all_logged_metrics = []
        self._hashkey = None

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
            if self.do_lock_experiment:
                lock_file_path = self.lock_file_path(pl_module)
                if os.path.exists(lock_file_path):
                    raise AlreadyRunningException("Found a lock file {} indicating that the experiment is already running.".format(lock_file_path))
                else:
                    if lock_file_path.rsplit("/", 1)[0].strip():
                        os.makedirs(lock_file_path.rsplit("/", 1)[0], exist_ok=True)
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

        if "current_epoch" in state:
            trainer.current_epoch = state["current_epoch"]
        if "global_step" in state:
            trainer.global_step = state["global_step"]
        if "state_dict" in state:
            pl_module.load_state_dict(state["state_dict"], strict=False)
        if "optimizers" in state:
            for optim, optim_state in zip(trainer.optimizers, state["optimizers"]):
                optim.load_state_dict(optim_state)
        else:
            warnings.warn("Missing optimizers state in checkpoint")
        dataset = getattr(trainer.train_dataloader.dataset, 'datasets', trainer.train_dataloader.dataset)
        if "train_dataset" in state and hasattr(dataset, 'load_state_dict'):
            dataset.load_state_dict(state["train_dataset"])
        elif hasattr(trainer.train_dataloader.dataset, 'load_state_dict'):
            warnings.warn("Missing train dataset state in checkpoint")
        if "all_logged_metrics" in state:
            self._all_logged_metrics = state["all_logged_metrics"]
            for log_dict in state["all_logged_metrics"]:
                trainer.logger.log_metrics(log_dict)

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):
        if trainer.global_step == 0:
            return
        self._all_logged_metrics.append({**trainer.logged_metrics, "step": int(trainer.global_step), "epoch": int(trainer.current_epoch)})

        dataset = getattr(trainer.train_dataloader.dataset, 'datasets', trainer.train_dataloader.dataset)
        train_dataset_state = dataset.state_dict() if hasattr(dataset, 'state_dict') else None
        optimizers_state = [
            optim.state_dict()
            for optim in pl_module.trainer.optimizers
        ]
        model_state = pl_module.state_dict()

        state = {
            "config": get_config(pl_module),
            "current_epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step + 1,
            "state_dict": model_state,
            "optimizers": optimizers_state,
            "train_dataset": train_dataset_state,
            "all_logged_metrics": self._all_logged_metrics,
        }

        if self._hashkey is None:
            self._hashkey = get_hashkey(pl_module)

        save_path = self.path.format(
            global_step=trainer.global_step,
            epoch=trainer.current_epoch,
            hashkey=self._hashkey)
        if save_path.rsplit("/", 1)[0].strip():
            os.makedirs(save_path.rsplit("/", 1)[0], exist_ok=True)
        torch.save(state, save_path + ".tmp")
        os.rename(save_path + ".tmp", save_path)

        parsed_paths = self.list_paths(pl_module)
        if self.keep_n is not False:
            for remove_path, _ in sorted(parsed_paths, key=lambda r: r[1].named.get('global_step', 0))[:-self.keep_n]:
                os.remove(remove_path)
