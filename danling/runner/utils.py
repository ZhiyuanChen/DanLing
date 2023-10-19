from __future__ import annotations

from collections.abc import Callable
from enum import auto
from functools import wraps
from typing import Any

from torch import nn

try:
    from enum import StrEnum  # type: ignore # pylint: disable = C0412
except ImportError:
    from strenum import LowercaseStrEnum as StrEnum  # type: ignore


class RunnerMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__()
        return instance


class RunnerMode(StrEnum):
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


def is_criterion(module: nn.Module):
    has_parameters = any(p.requires_grad for p in module.parameters())
    if has_parameters:
        return False

    forward_params = list(module.forward.__code__.co_varnames)
    if "input" in forward_params and "target" in forward_params:
        return True

    return False
