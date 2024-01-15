from __future__ import annotations

import os
import sys
from contextlib import suppress
from datetime import datetime
from enum import auto
from functools import wraps
from typing import Any
from warnings import warn

from danling.utils import base62

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
