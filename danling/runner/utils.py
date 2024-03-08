from __future__ import annotations

import os
import sys
from contextlib import suppress
from datetime import datetime
import socket
from collections.abc import Callable
from enum import auto
from functools import wraps
from typing import Any
from warnings import warn

from danling.utils import base62
from torch import nn
from yaml import add_representer
from yaml.representer import SafeRepresenter

try:
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:
    from strenum import LowercaseStrEnum as StrEnum  # type: ignore[no-redef]


class RunnerMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__()
        return instance


class RunnerMode(StrEnum):  # pylint: disable=too-few-public-methods
    r"""
    `RunnerMode` is an enumeration of running modes.

    Attributes:
        train: Training mode.
        eval: Evaluation mode.
        inf: Inference mode.
    """

    train = auto()
    eval = auto()
    inf = auto()


def get_time_str() -> str:
    time = datetime.now()
    time_tuple = time.isocalendar()[1:] + (
        time.hour,
        time.minute,
        time.second,
        time.microsecond,
    )
    return "".join(base62.encode(i) for i in time_tuple)


def get_git_hash() -> str | None:
    try:
        from git.exc import InvalidGitRepositoryError
        from git.repo import Repo

        try:
            return Repo(search_parent_directories=True).head.object.hexsha
        except ImportError:
            pass  # handle at last
        except (InvalidGitRepositoryError, ValueError):
            warn(
                "Unable to get git hash from CWD, fallback to git has of top-level code environment.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            path = os.path.dirname(os.path.abspath(sys.argv[0]))
            with suppress(InvalidGitRepositoryError, ValueError):
                return Repo(path=path, search_parent_directories=True).head.object.hexsha
    except ImportError:
        warn(
            "GitPython is not installed, unable to fing git hash",
            category=RuntimeWarning,
            stacklevel=2,
        )
    return None


class Precision(StrEnum):
    r"""
    `Precision` is an enumeration of data precision in running.

    Attributes:
        notset: Not set.
        fp64: Double precision floating point.
        fp32: Single precision floating point.
        bf16: Brain floating point.
        fp16: Half precision floating point.
        fp8: Quarter precision floating point.
        int8: 8 bit integer.
    """

    notset = auto()
    fp64 = auto()
    fp32 = auto()
    bf16 = auto()
    fp16 = auto()
    fp8 = auto()
    int8 = auto()


class UniqueList(list):
    elements: set[Any] = set()

    def append(self, item) -> None:
        if item not in self.elements:
            self.elements.add(item)
            super().append(item)

    def add(self, item) -> None:
        return self.append(item)

    def __contains__(self, item: object) -> bool:
        return item in self.elements


def on_main_process(func: Callable):
    """
    Decorator to run func only on main process.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any | None:
        if self.is_main_process or not self.distributed:
            return func(self, *args, **kwargs)
        return None

    return wrapper


def on_local_main_process(func: Callable):
    """
    Decorator to run func only on local main process.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any | None:
        if self.is_local_main_process or not self.distributed:
            return func(self, *args, **kwargs)
        return None

    return wrapper


def get_port() -> str:
    if "MASTER_PORT" in os.environ:
        return os.environ["MASTER_PORT"]
    sock = socket.socket()
    sock.bind(("", 0))
    os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
    warn(f"MASTER_PORT is not set. Setting MASTER_PORT to {os.environ['MASTER_PORT']}")
    return os.environ["MASTER_PORT"]


def is_criterion(module: nn.Module):
    has_parameters = any(p.requires_grad for p in module.parameters())
    if has_parameters:
        return False

    forward_params = list(module.forward.__code__.co_varnames)
    if "input" in forward_params and "target" in forward_params:
        return True

    return False


add_representer(Precision, SafeRepresenter.represent_str)
SafeRepresenter.add_representer(Precision, SafeRepresenter.represent_str)
