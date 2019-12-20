def split_name(names):
    _names = []
    for part in names.split("."):
        try:
            _names.append(int(part))
        except ValueError:
            _names.append(part)
    return _names


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

