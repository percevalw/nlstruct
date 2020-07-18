import logging
from functools import partial
from math import inf, pi, cos

from torch.optim.optimizer import Optimizer

from nlstruct.utils.deep_attributes import get_deep_attr, set_deep_attr

logger = logging.getLogger("nlstruct")


class Schedule(object):
    def __init__(self, name, optimizer, refresh_base_vals=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.refresh_base_vals = refresh_base_vals
        self.optimizer = optimizer
        self.name = name
        self.last_epoch = -1

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        if "optimizer" in state_dict:
            del state_dict["optimizer"]
        self.__dict__.update(state_dict)

    def get_val(self):
        raise NotImplementedError

    def step(self, metrics=None, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch == 0 or self.refresh_base_vals:
            self.base_vals = list(map(lambda group: get_deep_attr(group, self.name), self.optimizer.param_groups))
        self.last_epoch = epoch
        for param_group, val in zip(self.optimizer.param_groups, self.get_val()):
            if val is not None:
                set_deep_attr(param_group, self.name, val)

    def done(self):
        raise NotImplementedError()


class LinearSchedule(Schedule):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_val (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
    """

    def __init__(self, name, optimizer, end_val, num_iter):
        self.end_val = end_val
        self.num_iter = num_iter
        super(LinearSchedule, self).__init__(name, optimizer)

    def get_val(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        if not isinstance(self.end_val, (tuple, list)):
            end_vals = [self.end_val for _ in self.base_vals]
        else:
            end_vals = self.end_val
        return [base_val + r * (end_val - base_val) for end_val, base_val in zip(end_vals, self.base_vals)]

    def done(self):
        return self.num_iter <= self.last_epoch + 1


class ExponentialSchedule(Schedule):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_val (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
    """

    def __init__(self, name, optimizer, end_val, num_iter):
        self.end_val = end_val
        self.num_iter = num_iter
        super(ExponentialSchedule, self).__init__(name, optimizer)

    def get_val(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        if not isinstance(self.end_val, (tuple, list)):
            end_vals = [self.end_val for _ in self.base_vals]
        else:
            end_vals = self.end_val
        return [base_val * (end_val / base_val) ** r for end_val, base_val in zip(end_vals, self.base_vals)]

    def done(self):
        return self.num_iter <= self.last_epoch + 1


class CosineSchedule(Schedule):
    """Cosine annealing schedule as introduced in
    SGDR: Stochastic Gradient Descent With Warm Restarts, Ilya Loshchilov & Frank Hutter

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_val (float, optional): the initial learning rate which is the lower
            boundary of the test
        num_iter (int, optional): the number of iterations over which the test
            occurs
    """

    def __init__(self, name, optimizer, end_val, num_iter):
        self.end_val = end_val
        self.num_iter = num_iter
        super(CosineSchedule, self).__init__(name, optimizer)

    def get_val(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        if not isinstance(self.end_val, (tuple, list)):
            end_vals = [self.end_val for _ in self.base_vals]
        else:
            end_vals = self.end_val
        return [end_val + (base_val - end_val) / 2 * (1 + cos(r * pi)) for end_val, base_val in zip(end_vals, self.base_vals)]

    def done(self):
        return self.num_iter <= self.last_epoch + 1


class ConcatSchedule(Schedule):
    def __init__(self, schedules):
        self.schedules = schedules
        self.current_schedule_i = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            "schedules": [schedule.state_dict() for schedule in self.schedules],
            "current_schedule_i": self.current_schedule_i,
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        for schedule, schedule_state_dict in zip(self.schedules, state_dict["schedules"]):
            schedule.load_state_dict(schedule_state_dict)
        self.current_schedule_i = state_dict["current_schedule_i"]

    def step(self, metrics=None, epoch=None):
        if self.schedules[self.current_schedule_i].done():
            if self.current_schedule_i + 1 >= len(self.schedules):
                return
            self.current_schedule_i += 1
        self.schedules[self.current_schedule_i].step(metrics, epoch)

    def get_val(self):
        return self.schedules[self.current_schedule_i].get_val()

    def done(self):
        return self.current_schedule_i >= len(self.schedules)


class ConstantSchedule(Schedule):
    def __init__(self, name, optim, end_val=None, num_iter=None):
        super().__init__(name, optim)
        self.end_val = end_val
        self.num_iter = num_iter

    def done(self):
        return self.last_epoch + 1 >= self.num_iter

    def get_val(self):
        if self.end_val is None:
            return self.base_vals
        if not isinstance(self.end_val, (tuple, list)):
            end_vals = [self.end_val for _ in self.base_vals]
        else:
            end_vals = self.end_val
        return end_vals


class ScaleOnPlateauSchedule(Schedule):
    """Reduce value when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, val will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_val = val * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after val has been reduced. Default: 0.
        bound_val (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to val. If the difference
            between new and old val is smaller than eps, the update is
            ignored. Default: 1e-8.
    """

    def __init__(self, name, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, bound_val=0, eps=1e-8):

        super(ScaleOnPlateauSchedule, self).__init__(name, optimizer, refresh_base_vals=True)

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(bound_val, list) or isinstance(bound_val, tuple):
            if len(bound_val) != len(optimizer.param_groups):
                raise ValueError("expected {} bound_vals, got {}".format(
                    len(optimizer.param_groups), len(bound_val)))
            self.bound_vals = list(bound_val)
        else:
            self.bound_vals = [bound_val] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        self.metrics = metrics
        super(ScaleOnPlateauSchedule, self).step(epoch)

    def done(self):
        return False

    def get_val(self):
        if self.metrics is None:
            return self.base_vals

        if self.is_better(self.metrics, self.best):
            self.best = self.metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs <= self.patience:
            return self.base_vals

        self.cooldown_counter = self.cooldown
        self.num_bad_epochs = 0

        vals = []
        for (i, param_group), old_val in zip(enumerate(self.optimizer.param_groups), self.base_vals):
            new_val = max(old_val * self.factor, self.bound_vals[i])
            if old_val - new_val > self.eps:
                vals.append(new_val)
                if self.verbose:
                    logger.info('Epoch {:5d}: reducing learning rate'
                                ' of group {} to {:.4e}.'.format(self.last_epoch, i, new_val))
            else:
                vals.append(old_val)
        return vals

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        if "optimizer" in state_dict:
            del state_dict["optimizer"]
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
