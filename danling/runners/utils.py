# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from enum import auto
from functools import wraps
from typing import Any

import torch

from danling.utils import RoundDict, base62

try:
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:
    from strenum import LowercaseStrEnum as StrEnum  # type: ignore[no-redef]


class RunnerMode(StrEnum):  # pylint: disable=too-few-public-methods
    r"""
    `RunnerMode` is an enumeration of running modes.

    Attributes:
        train: Training mode.
        evaluate: Evaluation mode.
        infer: Inference mode.
    """

    train = auto()
    evaluate = auto()
    infer = auto()


class MetaRunner(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__()
        return instance


def get_precision(precision: str) -> torch.dtype:
    precision = str(precision).strip().lower().replace("-", "_")
    if precision in ("fp16", "float16", "half"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {precision}")


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
        from git.repo import Repo
    except Exception:
        return None

    try:
        repo = Repo(path=os.getcwd(), search_parent_directories=True)
        short_sha = repo.head.commit.hexsha[:8]
    except Exception:
        return None

    if not short_sha:
        return None

    if not bool(repo.is_dirty(untracked_files=False)):
        return short_sha

    diff_content = ""
    try:
        diff_content = str(repo.git.diff("HEAD", "--binary"))
    except Exception:
        diff_content = ""
    if not diff_content:
        try:
            diff_content = str(repo.git.status("--porcelain=v1"))
        except Exception:
            diff_content = "dirty"

    diff_sha = hashlib.sha1(diff_content.encode("utf-8")).hexdigest()[:10]
    return f"{short_sha}-d{diff_sha}"


def get_git_diff() -> str | None:
    try:
        from git.repo import Repo
    except Exception:
        return None

    try:
        repo = Repo(path=os.getcwd(), search_parent_directories=True)
        return str(repo.git.diff("HEAD", "--binary"))
    except Exception:
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


def format_result(result, format_spec: str = ".4f", depth: int = 0) -> str:
    del depth  # kept for API compatibility
    if not isinstance(result, RoundDict):
        result = RoundDict(result).round(4)
    return format(result, format_spec)
