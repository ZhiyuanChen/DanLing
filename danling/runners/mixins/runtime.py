# DanLing
# Copyright (C) 2022-Present  DanLing
#
# This file is part of DanLing.
#
# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0
#
# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

import builtins
import logging
import os
from functools import cached_property

from chanfig import NestedDict

from danling.utils import cached_ensure_dir, cached_ensure_parent_dir, catch

from ..config import RunnerConfig
from ..utils import get_git_diff, get_git_hash, on_main_process


class RuntimeMixin:
    """Cold-path runtime setup helpers extracted from BaseRunner."""

    config: RunnerConfig
    logger: logging.Logger | None = None
    timestamp: str
    _print_process: int

    @property
    def world_size(self) -> int:
        """Distributed world size from environment."""
        return int(os.getenv("WORLD_SIZE", "1"))

    @property
    def rank(self) -> int:
        """Global rank from environment."""
        return int(os.getenv("RANK", "0"))

    @property
    def local_rank(self) -> int:
        """Local rank from environment."""
        return int(os.getenv("LOCAL_RANK", "0"))

    @property
    def distributed(self) -> bool:
        """Whether distributed mode is active."""
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """Whether current rank is global main process."""
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        """Whether current rank is local main process."""
        return self.local_rank == 0

    @on_main_process
    def init_logging(self) -> None:
        logger = logging.getLogger(f"danling.runner.{self.id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def init_print(self, process: int = 0) -> None:
        builtin_print = builtins.print
        self._print_process = int(process)

        def _print(*args, sep=" ", end="\n", file=None, flush=False, force=False, **kwargs):
            if not force and self.rank != self._print_process:
                return

            if self.logger is not None:
                msg = sep.join(str(a) for a in args)
                if end and end != "\n":
                    msg += end.rstrip("\n")
                self.logger.info(msg)
            else:
                builtin_print(*args, sep=sep, end=end, file=file, flush=flush, **kwargs)

        builtins.__dict__["_print"] = builtins.print
        builtins.print = _print

    @catch
    @on_main_process
    def save_metadata(self) -> None:
        metadata_dir = os.path.join(self.dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        full_config = self.config
        canonical_config = self.config.canonical()
        default_config = RunnerConfig().canonical()

        full_config.yaml(os.path.join(metadata_dir, "config.full.yaml"))
        canonical_config.yaml(os.path.join(metadata_dir, "config.canonical.yaml"))

        diff = NestedDict(default_config).diff(canonical_config)
        diff.yaml(os.path.join(metadata_dir, "config.diff.yaml"))

        git_metadata = NestedDict({"commit": self.code_id, "id": self.id, "timestamp": self.timestamp})
        git_metadata.yaml(os.path.join(metadata_dir, "git.yaml"))

        git_diff = get_git_diff()
        if git_diff is not None:
            with open(os.path.join(metadata_dir, "git.diff"), "w", encoding="utf-8") as fp:
                fp.write(git_diff)

    @cached_ensure_dir
    def workspace_root(self) -> str:
        """Workspace root directory for experiment outputs."""
        return self.config.get("workspace_root", "experiments")

    @cached_property
    def lineage(self) -> str:
        """Configured lineage identifier."""
        return self.config.get("lineage", "lin")

    @property
    def experiment(self) -> str:
        """Configured experiment name."""
        return self.config.get("experiment", "exp")

    @cached_property
    def config_id(self) -> str:
        """Deterministic short id derived from config hash."""
        return format(hash(self.config) & ((1 << 48) - 1), "012x")

    @property
    def id(self) -> str:
        """Run attempt id (timestamp)."""
        return self.timestamp

    @cached_property
    def code_id(self) -> str | None:
        """Current git code identifier (or `None` when unavailable)."""
        return get_git_hash()

    @cached_ensure_dir
    def dir(self) -> str:
        """Experiment directory path."""
        if "dir" in self.config:
            return self.config.dir
        lineage = self.lineage
        if self.code_id is not None:
            lineage += f"-{self.code_id}"
        experiment = f"{self.experiment}-{self.config_id}"
        return os.path.join(self.workspace_root, lineage, experiment)

    @cached_ensure_parent_dir
    def log_file(self) -> str:
        """Log file path for this run id."""
        if "log_file" in self.config:
            return self.config.log_file
        return os.path.join(self.dir, "logs", f"{self.id}.log")

    @cached_ensure_dir
    def checkpoint_dir(self) -> str:
        """Checkpoint directory path."""
        if "checkpoint_dir" in self.config:
            return self.config.checkpoint_dir
        return os.path.join(self.dir, self.config.get("checkpoint.dir_name", "checkpoints"))
