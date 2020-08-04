# This file was adapted from "https://github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp/random.py"

import functools
import random
from collections import namedtuple
from contextlib import contextmanager

import numpy as np
import torch

RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


def get_random_generator_state(cuda=torch.cuda.is_available()):
    """ Get the `torch`, `numpy` and `random` random generator state.

    Parameters
    ----------
    cuda: bool  If `True` saves the `cuda` seed also. Note that getting and setting
            the random generator state for CUDA can be quite slow if you have a lot of GPUs.

    Returns
    -------
    RandomGeneratorState
    """
    return RandomGeneratorState(random.getstate(), torch.random.get_rng_state(),
                                np.random.get_state(),
                                torch.cuda.get_rng_state_all() if cuda else None)


def set_random_generator_state(state):
    """
    Set the `torch`, `numpy` and `random` random generator state.

    Parameters
    ----------
    state: RandomGeneratorState
    """
    random.setstate(state.random)
    torch.random.set_rng_state(state.torch)
    np.random.set_state(state.numpy)
    if state.torch_cuda is not None and torch.cuda.is_available() and len(
          state.torch_cuda) == torch.cuda.device_count():  # pragma: no cover
        torch.cuda.set_rng_state_all(state.torch_cuda)


@contextmanager
def fork_rng(seed=None, cuda=torch.cuda.is_available()):
    """
    Forks the `torch`, `numpy` and `random` random generators, so that when you return, the
    random generators are reset to the state that they were previously in.

    Parameters
    ----------
    seed: int or None
        If defined this sets the seed values for the random
        generator fork. This is a convenience parameter.
    cuda: bool
        If `True` saves the `cuda` seed also. Getting and setting the random
        generator state can be quite slow if you have a lot of GPUs.
    """
    state = get_random_generator_state(cuda)
    if seed is not None:
        set_seed(seed, cuda)
    try:
        yield
    finally:
        set_random_generator_state(state)


def fork_rng_wrap(function=None, **kwargs):
    """ Decorator alias for `fork_rng`.
    """
    if not function:
        return functools.partial(fork_rng_wrap, **kwargs)

    @functools.wraps(function)
    def wrapper():
        with fork_rng(**kwargs):
            return function()

    return wrapper


def set_seed(seed, cuda=torch.cuda.is_available()):
    """ Set seed values for random generators.
    Parameters
    ----------
    seed: int
        Value used as a seed.
    cuda: bool
        If `True` sets the `cuda` seed also.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:  # pragma: no cover
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_all = set_seed


# Pytorch specific overrides to make random function more reproducible
def torch_normal_(tensor, mean=0, std=1):
    """

    Parameters
    ----------
    tensor: torch.Tensor
    mean
    std

    Returns
    -------

    """
    r"""Fills the input Tensor with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std})`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    with torch.no_grad():
        state = np.random.get_state()
        np.random.seed(torch.randint(0, np.iinfo(np.int32).max, (1,)))
        try:
            tensor.copy_(torch.as_tensor(np.random.randn(*tensor.shape) * std + mean))
        finally:
            np.random.set_state(state)
    return tensor


def torch_randn(*args, **kwargs):
    tensor = torch.empty(*args, **kwargs)
    torch_normal_(tensor)
    return tensor


torch.Tensor.normal_ = torch_normal_
torch.Tensor.random_ = torch_normal_
torch.randn = torch_randn
