import logging
from collections import defaultdict

from termcolor import colored

logger = logging.getLogger("nlstruct")


class TrainingLogger(object):
    default_format_map = {
        str: (5, lambda x: (x[:10] if len(x) > 10 else x)),
        int: (5, lambda x: x),
        float: (10, "{:.2E}".format),
        bool: (3, lambda x: ("YES" if x else " NO")),
        type(None): (5, lambda x: str(x)),
    }

    def __init__(self, key, formatter, patience_warmup=None, patience=None):
        self.best = {}
        self.fields = None
        self.widths = []
        self.key = key
        self.patience_warmup = patience_warmup
        self.patience = patience
        if isinstance(formatter, dict):
            self.formatter = defaultdict(lambda: False)
            self.formatter.update(formatter)
        if "epoch" not in self.formatter:
            self.formatter["epoch"] = {}

    def display(self, info):
        if self.fields is None:
            s = ""
            self.fields = []
            i = 0
            for field in [*info.keys(), "patience_warmup", "patience"]:
                if field not in info:
                    continue
                try:
                    format_info = self.formatter[field]
                except KeyError:
                    format_info = {}
                if format_info is False:
                    continue
                self.fields.append(field)
                if i > 0:
                    s += " | "
                if field in ("patience_warmup", "patience"):
                    min_width, field_formatter = format_info.get("format", self.default_format_map[str])
                    name = "warmup" if field == "patience_warmup" else "patience"
                    width = max(len(name), len(str(field_formatter(f"{info[field]}/.."))), min_width)
                    self.widths.append(width)
                else:
                    name = format_info.get("name", field if isinstance(field, str) else "_".join(map(str, field)))
                    min_width, field_formatter = format_info["format"] if "format" in format_info else self.default_format_map[type(info[field])]
                    name = "_".join(name) if isinstance(name, (list, tuple)) else name
                    formatted = str(field_formatter(info[field])) if info[field] is not None else 'None'
                    width = max(len(name), len(formatted), min_width)
                    self.widths.append(width)
                name_length = len(name)
                if field == self.key:
                    name = colored(name, "red")
                s += " " * (width - name_length) + name
                i += 1
            logger.info(s)

        s = ""
        for i, (field, width) in enumerate(zip(self.fields, self.widths)):
            if i > 0:
                s += " | "
            format_info = self.formatter.get(field, {})
            min_width, field_formatter = format_info["format"] if "format" in format_info else self.default_format_map[type(info[field])]
            goal = format_info.get("goal", None)

            formatted = str(field_formatter(info[field])) if info[field] is not None else 'None'
            if field == "patience_warmup" and self.patience_warmup is not None:
                assert 'epoch' in info, "Cannot display 'patience_warmup' column without an 'epoch' column"
                formatted = f"{str(min(info['epoch'], self.patience_warmup))}/{self.patience_warmup}"
                obj_width = len(formatted)
                if info['epoch'] < self.patience_warmup:
                    formatted = colored(formatted, "green")
            elif field == "patience" and self.patience is not None:
                formatted = f"{str(info[field])}/{self.patience}"
                obj_width = len(formatted)
                if info[field] > 0:
                    formatted = colored(formatted, "red")
            else:
                obj_width = len(formatted)
                if goal is not None and info[field] is not None and ('epoch' not in info or (self.patience_warmup is None or info["epoch"] >= self.patience_warmup)):
                    if field not in self.best or abs(self.best[field] - goal) > abs(info[field] - goal):
                        formatted = colored(formatted, 'green')
                        self.best[field] = info[field]
                    else:
                        formatted = colored(formatted, 'red')
            width = max(obj_width, width)

            s += " " * (width - obj_width) + formatted
        logger.info(s)
