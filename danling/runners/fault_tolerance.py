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

import importlib.util
from collections.abc import Mapping
from contextlib import suppress
from datetime import timedelta
from typing import Any

_torchft_spec = importlib.util.find_spec("torchft")
if _torchft_spec is not None:
    import torchft as torchft

    has_torchft = True
else:
    torchft = None
    has_torchft = False


class FaultTolerance:
    """Small fault-tolerance runtime handle owned by a runner."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self._manager: Any | None = None
        self._process_group: Any | None = None
        self._replicate_process_group: Any | None = None
        self.replica_id: int = 0
        self.group_size: int = 1

        config = runner.config.get("ft")
        if not config or not bool(config.get("enabled", False)):
            return

        if not has_torchft or torchft is None:
            raise ImportError("torchft is not installed. Install `torchft-nightly` to enable config.ft.enabled.")

        process_group = str(config.get("process_group", "gloo")).strip().lower()
        process_group_timeout_ms = int(config.get("process_group_timeout_ms", 10000))
        self.replica_id = int(config.get("replica_id", 0))
        self.group_size = int(config.get("group_size", 1))
        min_replica_size = int(config.get("min_replica_size", 1))

        if process_group_timeout_ms <= 0:
            raise ValueError(
                f"config.ft.process_group_timeout_ms must be a positive integer, got {process_group_timeout_ms}"
            )
        if self.group_size <= 0:
            raise ValueError(f"config.ft.group_size must be a positive integer, got {self.group_size}")
        if self.replica_id < 0 or self.replica_id >= self.group_size:
            raise ValueError(f"config.ft.replica_id must be in [0, {self.group_size - 1}], got {self.replica_id}")
        if min_replica_size <= 0:
            raise ValueError(f"config.ft.min_replica_size must be a positive integer, got {min_replica_size}")

        timeout = timedelta(milliseconds=process_group_timeout_ms)
        if process_group == "gloo":
            pg = torchft.ProcessGroupGloo(timeout=timeout)
        elif process_group == "nccl":
            pg = torchft.ProcessGroupNCCL(timeout=timeout)
        else:
            raise ValueError(f"unsupported config.ft.process_group: {process_group!r}; expected 'gloo' or 'nccl'")

        self._process_group = pg
        try:
            self._manager = torchft.Manager(
                pg=pg,
                min_replica_size=min_replica_size,
                load_state_dict=self._load_state_dict,
                state_dict=self._state_dict,
                use_async_quorum=True,
                replica_id=f"danling_ft_{self.replica_id}",
            )
            self._replicate_process_group = torchft.process_group.ManagedProcessGroup(self._manager)
            self._replicate_process_group.register("dp_replicate")
        except Exception:
            self.close()
            raise

    @property
    def enabled(self) -> bool:
        return self._manager is not None

    @property
    def replicate_process_group(self) -> Any | None:
        if self.enabled:
            return self._replicate_process_group
        return None

    def data_parallel_info(self, degree: int, rank: int) -> tuple[int, int]:
        if not self.enabled:
            return degree, rank
        return degree * self.group_size, degree * self.replica_id + rank

    def participating_rank(self) -> int:
        if self._manager is None:
            return 0
        return int(self._manager.participating_rank())

    @staticmethod
    def _close_resource(resource: Any | None) -> None:
        if resource is None:
            return
        for method_name in ("close", "shutdown", "stop"):
            method = getattr(resource, method_name, None)
            if callable(method):
                with suppress(Exception):
                    method()
                return

    def close(self) -> None:
        self._close_resource(self._replicate_process_group)
        self._replicate_process_group = None
        self._close_resource(self._manager)
        self._manager = None
        self._close_resource(self._process_group)
        self._process_group = None

    def _state_dict(self) -> Mapping[str, Any]:
        return self.runner.state_dict()

    def _load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.runner.load_checkpoint(state_dict)
