import itertools
import logging
import threading
import time
from warnings import warn

import regex
import sh

from nlstruct.environment.cache import yaml_load, yaml_dump
from nlstruct.train.logging import TrainingLogger
from nlstruct.train.random import seed_all
from nlstruct.train.schedule import ConcatSchedule
from nlstruct.utils.deep_attributes import set_deep_attr
from nlstruct.utils.torch import torch_global as tg

logging.getLogger("sh").setLevel(logging.WARNING)


class TrainingState(object):
    def __init__(self,
                 goal,
                 patience_warmup=None,
                 patience=None,
                 patience_rate=None,
                 max_epoch=None,
                 exit_on_score=None):
        self.epoch = 0
        self.patience_counter = 0
        self.goal = goal
        self.last = None
        self.max_epoch = max_epoch

        self.best_loss = None
        self.last_patience_reset_best_loss = None
        self.best_epoch = None
        assert patience_rate is None or patience is not None, "Must set patience (int > 0) if patience_rate is not None"
        assert patience_warmup is None or patience is not None, "Must set patience (int > 0) if patience_warmup is not None"
        self.patience = patience
        self.patience_warmup = patience_warmup or 0
        self.patience_rate = patience_rate or 0
        self.exit_on_score = exit_on_score
        self.quick_exit = False

    def record(self, loss):
        self.epoch += 1
        if loss is None:
            self.best_loss = loss
            self.best_epoch = self.epoch
            return
        if self.best_loss is None or abs(self.goal - loss) < abs(self.goal - self.best_loss):
            self.best_loss = loss
            self.best_epoch = self.epoch
        if self.patience is not None and (self.last_patience_reset_best_loss is None or abs(self.goal - loss) < abs(
              self.goal - self.last_patience_reset_best_loss) * (
                                                1 - self.patience_rate)):
            self.patience_counter = 0
            self.last_patience_reset_best_loss = self.best_loss
        elif self.patience is not None and self.epoch >= self.patience_warmup:
            self.patience_counter += 1
            if self.exit_on_score is not None and abs(self.goal - self.exit_on_score) < abs(self.goal - loss):
                self.quick_exit = True
        elif self.exit_on_score is not None and abs(self.goal - self.exit_on_score) < abs(self.goal - loss):
            self.quick_exit = True

    @property
    def keep_going(self):
        if (self.max_epoch is None or self.epoch < self.max_epoch) and (self.patience is None or self.patience_counter <= self.patience or self.epoch < self.patience_warmup) and not self.quick_exit:
            return True
        return False


def make_optimizer_and_schedules(net, optim_factory, optim_params, names, num_iter_per_epoch):
    param_groups = []
    init_optim = {k: v[0] if hasattr(v, '__len__') else v for k, v in optim_params.items()}
    default_optim = optim_factory(net.parameters(), lr=0.1).defaults
    net_params = list(net.named_parameters())
    matched = [False for _ in net_params]
    for (*group_update, layer_name) in zip(*[(v.tolist() if hasattr(v, "tolist") else v) if hasattr(v, "__len__") and not isinstance(v, str) else itertools.repeat(v) for v in init_optim.values()],
                                           names):
        group_params = dict(default_optim)
        group_params["params"] = []
        for key, val in zip(init_optim.keys(), group_update):
            set_deep_attr(group_params, key, val)
            for i, ((param_name, param), param_matched) in enumerate(zip(net_params, matched)):
                if not param_matched and regex.match(layer_name, param_name):
                    group_params["params"].append(param)
                    matched[i] = True
        param_groups.append(group_params)
    del default_optim

    optim = optim_factory(param_groups)
    schedules = {k: v[1:] for k, v in optim_params.items() if hasattr(v, '__len__') and len(v) > 1}
    schedules = {
        name: ConcatSchedule([
            schedule_fn(name, optim, val.tolist() if hasattr(val, 'tolist') else val if isinstance(val, (list, tuple)) else [val for _ in names], int(num_iter_per_epoch * epochs))
            for schedule_fn, val, epochs in schedule_params])
        for name, schedule_params in schedules.items()
    }
    return optim, schedules


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class OptimizationIterator:
    def __init__(self,
                 metrics_info,
                 max_epoch,
                 main_score=None,
                 patience_warmup=None,
                 patience_rate=None,
                 patience=None,
                 state=None,
                 n_save_checkpoints=1,
                 exit_on_score=None,
                 cache=None,
                 cache_policy="all",
                 with_writer=False,
                 seed=42,
                 mutable_state_policy="warn",
                 ):
        self.state = {} if state is None else state

        # Check state
        for key, val in self.state.items():
            if not isinstance(val, (int, float, complex, str, tuple, frozenset, bytes)) and not hasattr(val, 'load_state_dict'):
                if mutable_state_policy == "warn":
                    warn(f"Entry '{key}' in the state seems to be mutable but has no load_state_dict/state_dict methods. This could lead to unpredictable behaviors.")
                elif mutable_state_policy == "raise":
                    raise Exception(f"Entry '{key}' in the state seems to be mutable but has no load_state_dict/state_dict methods. This could lead to unpredictable behaviors.")
                else:
                    assert mutable_state_policy is False, "mutable_state_policy must be 'warn', 'raise' or False"

        if cache is None:
            cache_policy = []
        elif cache_policy is None:
            cache_policy = []
        elif cache_policy == "all":
            cache_policy = ["read_history", "write_history", "read_checkpoints", "write_checkpoints"]
        elif isinstance(cache_policy, str):
            cache_policy = [cache_policy, ]
        if "all_history" in cache_policy:
            cache_policy.remove("all_history")
            cache_policy.extend(["read_history", "write_history"])
        if "all_checkpoints" in cache_policy:
            cache_policy.remove("all_checkpoints")
            cache_policy.extend(["read_checkpoints", "write_checkpoints"])

        self.seed = seed
        self.main_score = main_score
        self.n_save_checkpoints = n_save_checkpoints
        if with_writer and cache is not None:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(logdir=self.cache.entry('logs'))
        else:
            self.writer = None
        self.cache_policy = cache_policy
        self.cache = cache
        self.history = (cache.load("history.yaml", loader=yaml_load) if "read_history" in cache_policy else []) or []
        self.score_logger = TrainingLogger(key=main_score,
                                           patience_warmup=patience_warmup, patience=patience,
                                           formatter=metrics_info)
        self.monitor = TrainingState(goal=metrics_info[main_score]['goal'] if main_score is not None else main_score,
                                     patience_warmup=patience_warmup, patience=patience,
                                     patience_rate=patience_rate, max_epoch=max_epoch,
                                     exit_on_score=exit_on_score)
        assert self.state.setdefault("epoch", 0) == 0, "Init epoch must be 0"
        self.dumps = {}

    def __iter__(self):
        while self.monitor.keep_going:
            # Example 1: self.monitor.epoch = 12, self.state["epoch"] = 7, len(self.history) == 12
            # -> no more scores in self.history: we must train the model, so we go the "else"
            scores = {}
            model_has_been_trained = False
            if len(self.history) > self.monitor.epoch:
                scores = self.history[self.monitor.epoch]
            else:
                # Try to find the closest checkpointed model, starting from the required epoch
                # Iterate over (13, 12, 11, 10, 9, 8) to try and find a checkpointed model
                # region load closest checkpoint to required epoch
                for epoch in range(self.monitor.epoch + 1, self.state["epoch"], -1):
                    dumped = self.cache.load(f"checkpoint-{str(epoch)}.pt", map_location=tg.device) if "read_checkpoints" in self.cache_policy else None
                    if dumped is not None:
                        for name in dumped.keys():
                            persistable = self.state.get(name, None)
                            if name in self.state and hasattr(persistable, 'load_state_dict'):
                                persistable.load_state_dict(dumped[name])
                            else:
                                self.state[name] = dumped[name]
                            del persistable
                        # ex: now self.state["epoch"] = 11
                        self.state["epoch"] = epoch
                        del dumped
                        break

                # endregion
                # Train over the remaining epochs if needed
                # Iterate over self.state["epoch"] (11, 12)
                # region train until required epoch
                while self.state["epoch"] < self.monitor.epoch + 1:
                    seed_all(self.seed + self.state["epoch"])
                    time_start = time.time()
                    epoch_before = self.state["epoch"]
                    epoch_scores = {}
                    yield epoch_before, self.state, self.history, lambda update: epoch_scores.update(update)

                    time_end = time.time()
                    scores = {
                        **{key: value.item() if hasattr(value, 'item') else value for key, value in epoch_scores.items()},
                        "duration": time_end - time_start}
                    if self.writer is not None:
                        for score_name, score in scores.items():
                            self.writer.add_scalar(score_name, score, self.state["epoch"])

                    # Update epoch and declare that we have some changes to checkpoint
                    self.state["epoch"] = epoch_before + 1
                    model_has_been_trained = True

                    # Update the scheduler with the computed metrics (might lower LR or weight decay)
                    # scheduler.step(metrics=scores.get(self.main_score, 0), epoch=self.state["epoch"])
                # endregion
                # region update training self.state and dump model and self.history
            self.history = self.history[:self.monitor.epoch] + [scores] + self.history[self.monitor.epoch + 1:]
            self.monitor.record(scores[self.main_score] if self.main_score is not None else None)
            if "write_history" in self.cache_policy:
                self.cache.dump(self.history, "history.yaml", dumper=yaml_dump)
            if model_has_been_trained:
                dump_dict = {}
                for name, persistable in self.state.items():
                    if hasattr(persistable, 'state_dict'):
                        dump_dict[name] = persistable.state_dict()
                    else:
                        dump_dict[name] = persistable
                if "write_checkpoints" in self.cache_policy:
                    self.dumps[self.state["epoch"]] = self.cache.dump(dump_dict, dest="checkpoint-{}.pt".format(self.state["epoch"]))
            # Delete other self.dumps except the best one
            for dump_epoch, dest in list(self.dumps.items()):
                if dump_epoch <= self.state["epoch"] - self.n_save_checkpoints and dump_epoch != self.monitor.best_epoch:
                    sh.rm(dest)
                    self.dumps.pop(dump_epoch)
            self.score_logger.display({"epoch": self.monitor.epoch, **scores})
        if self.state["epoch"] != self.monitor.best_epoch and "read_checkpoints" in self.cache_policy:
            dumped = self.cache.load(f"checkpoint-{str(self.monitor.best_epoch)}.pt", map_location=tg.device) if self.cache is not None else None
            if dumped is not None:
                logging.info(f"Model restored to its best state: {self.monitor.best_epoch}")
                for name in dumped.keys():
                    persistable = self.state.get(name, None)
                    if name in self.state and hasattr(persistable, 'load_state_dict'):
                        persistable.load_state_dict(dumped[name])
                    else:
                        self.state[name] = dumped[name]
            else:
                logging.info(f"Could not restore model to its best state: {self.monitor.best_epoch}")


iter_optimization = OptimizationIterator


def run_optimization(
      metrics_info,
      max_epoch,
      epoch_fn,
      main_score=None,
      patience_warmup=None,
      patience_rate=None,
      patience=None,
      state=None,
      n_save_checkpoints=1,
      exit_on_score=None,
      cache=None,
      cache_policy="all",
      with_writer=None,
      seed=42,
      mutable_state_policy="warn",
      as_thread=False,
      _thread_should_stop=None,
):
    if as_thread:
        x = StoppableThread(
            target=run_optimization, args=(
                metrics_info,
                max_epoch,
                epoch_fn,
                main_score,
                patience_warmup,
                patience_rate,
                patience,
                state,
                n_save_checkpoints,
                exit_on_score,
                cache,
                cache_policy,
                with_writer,
                seed,
                mutable_state_policy,
                False,  # as_thread
                lambda: x.stopped()  # _thread_should_stop
            ))
        x.start()
        return x
    iterator = iter_optimization(
        metrics_info,
        max_epoch,
        main_score,
        patience_warmup,
        patience_rate,
        patience,
        state,
        n_save_checkpoints,
        exit_on_score,
        cache,
        cache_policy,
        with_writer,
        seed,
        mutable_state_policy,
    )
    for epoch_before, state, history, record in iterator:
        epoch_scores = epoch_fn()
        record(epoch_scores)
    return [{**iterator.history[iterator.monitor.best_epoch - 1], "best_epoch": iterator.monitor.best_epoch}, iterator.history]
