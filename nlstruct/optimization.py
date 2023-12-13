import torch


def split_name(names):
    _names = []
    for part in names.split("."):
        try:
            _names.append(int(part))
        except ValueError:
            _names.append(part)
    return _names


class ScheduledOptimizer(torch.optim.Optimizer):
    def __init__(self, optim):
        self.optim = optim
        for group in self.optim.param_groups:
            if "schedules" in group:
                if not isinstance(group["schedules"], list):
                    group["schedules"] = [group["schedules"]]
                group["schedules"] = list(group["schedules"])
                for schedule in group["schedules"]:
                    schedule.step(group)

    def zero_grad(self):
        return self.optim.zero_grad()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @property
    def state(self):
        return self.optim.state

    @state.setter
    def state(self, value):
        self.optim.state = value

    @property
    def defaults(self):
        return self.optim.defaults

    def state_dict(self):
        state = {
            "optim": self.optim.state_dict(),
            "lr": [
                group.get("lr") for group in self.optim.param_groups
            ],
            "schedules": [
                [schedule.state_dict() for schedule in group.get("schedules")]
                for group in self.optim.param_groups
            ]
        }
        for group in state["optim"]["param_groups"]:
            del group["schedules"]
        return state

    def load_state_dict(self, state):
        optim_schedules = [group["schedules"] for group in self.optim.param_groups]
        self.optim.load_state_dict(state["optim"])
        for group, group_schedule, group_schedules_state, lr in zip(self.optim.param_groups, optim_schedules, state["schedules"], state["lr"]):
            group["schedules"] = group_schedule
            for schedule, schedule_state in zip(group["schedules"], group_schedules_state):
                schedule.load_state_dict(schedule_state)
            group["lr"] = lr

    def step(self, closure=None):
        self.optim.step(closure=closure)
        for group in self.optim.param_groups:
            if "schedules" in group:
                for schedule in group["schedules"]:
                    schedule.step(group)


class LinearSchedule:
    def __init__(self, total_steps, max_value=None, start_value=0., path="lr", warmup=True, warmup_rate=0.1):
        self.path = path
        self.start_value = start_value
        self.max_value = max_value
        self.warmup = warmup
        self.warmup_rate = warmup_rate
        self.total_steps = total_steps
        self.idx = 0

    def state_dict(self):
        return {
            "idx": self.idx,
        }

    def load_state_dict(self, state):
        self.idx = state["idx"]

    def step(self, group, closure=None):
        if self.max_value is None:
            self.max_value = get_deep_attr(group, self.path)
        warmup_steps = self.total_steps * self.warmup_rate
        if self.idx < warmup_steps:
            progress = self.idx / warmup_steps
            value = self.start_value + (self.max_value - self.start_value) * progress
        else:
            progress = (self.idx - warmup_steps) / (self.total_steps - warmup_steps)
            value = self.max_value + (0 - self.max_value) * progress
        self.idx += 1
        set_deep_attr(group, self.path, value)


def get_attr_item(base, attr):
    if isinstance(base, (dict, list, tuple)):
        return base[attr]
    else:
        return getattr(base, attr)


def get_deep_attr(base, names):
    if isinstance(names, str):
        names = split_name(names)
    if len(names) == 0:
        return base
    [current, *remaining] = names
    return get_deep_attr(get_attr_item(base, current), remaining)


def set_attr_item(base, attr, val):
    if isinstance(base, (dict, list, tuple)):
        base[attr] = val
    else:
        setattr(base, attr, val)
    return base


def set_deep_attr(base, names, val):
    if isinstance(names, str):
        names = split_name(names)
    if len(names) == 0:
        return val
    if len(names) == 1:
        if isinstance(base, (dict, list)):
            base[names[0]] = val
        else:
            setattr(base, names[0], val)
    [current, *remaining] = names
    attr = base[current] if isinstance(base, (dict, list)) else getattr(base, current)
    if isinstance(base, tuple):
        set_deep_attr(attr, remaining, val)
    else:
        try:
            set_deep_attr(attr, remaining, val)
        except TypeError:
            new_attr = list(attr)
            set_deep_attr(new_attr, remaining, val)
            return set_attr_item(base, current, tuple(new_attr))
    return base
