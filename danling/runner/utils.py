from __future__ import annotations

from enum import auto
from functools import wraps
from typing import Any

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


def on_main_process(func):
    """
    Decorator to run func only on main process.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any | None:
        if self.is_main_process or not self.distributed:
            return func(self, *args, **kwargs)
        return None

    return wrapper


def on_local_main_process(func):
    """
    Decorator to run func only on local main process.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any | None:
        if self.is_local_main_process or not self.distributed:
            return func(self, *args, **kwargs)
        return None

    return wrapper
