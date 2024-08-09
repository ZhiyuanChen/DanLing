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

import builtins
import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from chanfig import NestedDict

from danling.utils import cached_ensure_dir, cached_ensure_parent_dir, catch

from .config import RunnerConfig
from .utils import get_git_diff

if TYPE_CHECKING:
    from .base_runner import BaseRunner


def _stream_supports_color(stream) -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty is not None and isatty())


class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    DIM = "\033[2m"
    LEVEL_COLORS = {
        logging.DEBUG: "#4298B5",
        logging.INFO: "#6FA287",
        logging.WARNING: "#E98300",
        logging.ERROR: "#E04F39",
        logging.CRITICAL: "#8C1515",
    }
    MESSAGE_COLORS = {
        logging.DEBUG: "#67AFD2",
        logging.INFO: "#8AB8A7",
        logging.WARNING: "#F9A44A",
        logging.ERROR: "#F4795B",
        logging.CRITICAL: "#B83A4B",
    }
    TIMESTAMP_COLOR = "#544948"

    @staticmethod
    def _hex_color(code: str) -> str:
        code = code.lstrip("#")
        red = int(code[0:2], 16)
        green = int(code[2:4], 16)
        blue = int(code[4:6], 16)
        return f"\033[38;2;{red};{green};{blue}m"

    @classmethod
    def _colorize(cls, text: str, code: str | None, *, dim: bool = False) -> str:
        prefix = cls._hex_color(code) if code is not None else ""
        if dim:
            prefix += cls.DIM
        return f"{prefix}{text}{cls.RESET}" if prefix else text

    def format(self, record: logging.LogRecord) -> str:
        message = self._colorize(record.getMessage(), self.MESSAGE_COLORS.get(record.levelno))
        timestamp = self._colorize(self.formatTime(record, self.datefmt), self.TIMESTAMP_COLOR, dim=True)
        level_code = self.LEVEL_COLORS.get(record.levelno)
        level = self._colorize(f"[{record.levelname}]", level_code)
        formatted = f"{timestamp} {level} {message}"

        if record.exc_info:
            formatted = f"{formatted}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            formatted = f"{formatted}\n{self.formatStack(record.stack_info)}"
        return formatted


def _delegate_print(
    printer: Any,
    *args: Any,
    sep: str = " ",
    end: str = "\n",
    file: Any = None,
    flush: bool = False,
    force: bool = False,
    **kwargs: Any,
) -> None:
    if printer is builtins.__dict__.get("print"):
        printer(*args, sep=sep, end=end, file=file, flush=flush, **kwargs)
        return
    try:
        printer(*args, sep=sep, end=end, file=file, flush=flush, force=force, **kwargs)
    except TypeError:
        printer(*args, sep=sep, end=end, file=file, flush=flush, **kwargs)


class _PrintRouter:
    def __init__(self, runner: BaseRunner, previous: Any, process: int) -> None:
        self.runner = runner
        self.previous = previous
        self.process = int(process)
        self.active = True

    def __call__(
        self,
        *args: Any,
        sep: str = " ",
        end: str = "\n",
        file: Any = None,
        flush: bool = False,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        if not self.active:
            _delegate_print(self.previous, *args, sep=sep, end=end, file=file, flush=flush, force=force, **kwargs)
            return

        if not force and self.runner.rank != self.process:
            return

        logger = self.runner.logger
        if logger is not None:
            msg = sep.join(str(a) for a in args)
            if end and end != "\n":
                msg += end.rstrip("\n")
            logger.info(msg)
            return

        _delegate_print(self.previous, *args, sep=sep, end=end, file=file, flush=flush, force=force, **kwargs)


class _PrintPatchGuard:
    def __init__(self, runner: BaseRunner) -> None:
        self.runner = runner
        self.router: _PrintRouter | None = None
        self.process = 0
        self.installed = False

    def install(self, process: int) -> _PrintRouter:
        process = int(process)
        if self.installed and self.router is not None and self.process == process:
            return self.router

        self.uninstall()
        current = builtins.print
        if (
            isinstance(current, _PrintRouter)
            and current.active
            and current.runner is self.runner
            and current.process == process
        ):
            self.router = current
            self.process = process
            self.installed = True
            return current

        router = _PrintRouter(self.runner, current, process)
        self.router = router
        self.process = process
        self.installed = True
        cast(Any, builtins).print = router
        return router

    def uninstall(self) -> None:
        router = self.router
        self.router = None
        self.installed = False
        if router is None:
            return
        router.active = False
        if builtins.print is router:
            cast(Any, builtins).print = _active_print_target(router.previous)


def _active_print_target(printer: Any) -> Any:
    current = printer
    while isinstance(current, _PrintRouter) and not current.active:
        current = current.previous
    return current


class RunnerWorkspace:
    """Workspace layout, metadata, and logging helpers for a runner."""

    def __init__(self, runner: BaseRunner) -> None:
        self.runner = runner
        self._print_guard = _PrintPatchGuard(runner)

    @cached_ensure_dir
    def workspace_root(self) -> str:
        return self.runner.config.get("workspace_root", "experiments")

    @cached_property
    def lineage(self) -> str:
        return self.runner.config.get("lineage", "lin")

    @property
    def experiment(self) -> str:
        return self.runner.config.get("experiment", "exp")

    @property
    def id(self) -> str:
        return self.runner.id

    @cached_property
    def code_id(self) -> str | None:
        return self.runner.code_id

    @cached_ensure_dir
    def dir(self) -> str:
        if "dir" in self.runner.config:
            return self.runner.config.dir
        lineage = self.lineage
        if self.code_id is not None:
            lineage += f"-{self.code_id}"
        return os.path.join(self.workspace_root, lineage, self.id)

    @cached_ensure_parent_dir
    def log_file(self) -> str:
        if "log_file" in self.runner.config:
            return self.runner.config.log_file
        return os.path.join(self.dir, "logs", f"{self.runner.timestamp}.log")

    @cached_ensure_dir
    def checkpoint_dir(self) -> str:
        if "checkpoint_dir" in self.runner.config:
            return self.runner.config.checkpoint_dir
        return os.path.join(self.dir, self.runner.config.get("checkpoint.dir_name", "checkpoints"))

    def init_logging(self) -> None:
        if not (self.runner.is_main_process or not self.runner.distributed):
            return
        logger = logging.getLogger(f"danling.runner.{self.id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            plain_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                ColorFormatter("%(message)s") if _stream_supports_color(stream_handler.stream) else plain_formatter
            )
            logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(plain_formatter)
            logger.addHandler(file_handler)

        self.runner.logger = logger

    def init_print(self, process: int = 0) -> None:
        self.runner._print_process = int(process)
        self._print_guard.install(process)

    @catch
    def save_metadata(self) -> None:
        if not (self.runner.is_main_process or not self.runner.distributed):
            return
        metadata_dir = os.path.join(self.dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        full_config = self.runner.config
        canonical_config = self.runner.config.canonical()
        default_config = RunnerConfig().canonical()

        self.runner.save(full_config, os.path.join(metadata_dir, "config.full.yaml"))
        self.runner.save(canonical_config, os.path.join(metadata_dir, "config.canonical.yaml"))

        diff = NestedDict(default_config).diff(canonical_config)
        self.runner.save(diff, os.path.join(metadata_dir, "config.diff.yaml"))

        git_metadata = NestedDict({"commit": self.code_id, "id": self.id, "timestamp": self.runner.timestamp})
        self.runner.save(git_metadata, os.path.join(metadata_dir, "git.yaml"))

        git_diff = get_git_diff()
        if git_diff is not None:
            with open(os.path.join(metadata_dir, "git.diff"), "w", encoding="utf-8") as fp:
                fp.write(git_diff)

    def close(self) -> None:
        logger = self.runner.logger
        if logger is not None:
            handlers = list(logger.handlers)
            for handler in handlers:
                handler.flush()
                handler.close()
                logger.removeHandler(handler)
            self.runner.logger = None

        self._print_guard.uninstall()
