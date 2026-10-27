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
        self._quorum_active = False
        self._nonfinite_loss = False
        self._consecutive_nonfinite_steps = 0
        self._max_nonfinite_steps = int(runner.config.get("ft.max_consecutive_nonfinite_steps", 10))

        config = runner.config.get("ft")
        if not config or not bool(config.get("enabled", False)):
            return

        if config.get("use_async_quorum", False):
            raise NotImplementedError(
                "DanLing requires synchronous TorchFT quorum to restore state before selecting data"
            )

        if not has_torchft or torchft is None:
            raise ImportError("torchft is not installed. Install `torchft-nightly` to enable config.ft.enabled.")

        process_group = str(config.get("process_group", "gloo")).strip().lower()
        process_group_timeout_seconds = float(config.get("process_group_timeout_seconds", 10.0))
        self.replica_id = int(config.get("replica_id", 0))
        self.group_size = int(config.get("group_size", 1))
        min_replica_size = int(config.get("min_replica_size", 1))
        manager_timeout_seconds = float(config.get("manager_timeout_seconds", 60.0))
        if manager_timeout_seconds <= 0 or self._max_nonfinite_steps <= 0:
            raise ValueError("FT manager timeout and nonfinite retry limit must be positive")

        if process_group_timeout_seconds <= 0:
            raise ValueError(
                "config.ft.process_group_timeout_seconds must be positive, " f"got {process_group_timeout_seconds}"
            )
        if self.group_size <= 0:
            raise ValueError(f"config.ft.group_size must be a positive integer, got {self.group_size}")
        if self.replica_id < 0 or self.replica_id >= self.group_size:
            raise ValueError(f"config.ft.replica_id must be in [0, {self.group_size - 1}], got {self.replica_id}")
        if min_replica_size <= 0:
            raise ValueError(f"config.ft.min_replica_size must be a positive integer, got {min_replica_size}")

        timeout = timedelta(seconds=process_group_timeout_seconds)
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
                use_async_quorum=False,
                timeout=timedelta(seconds=manager_timeout_seconds),
                replica_id=f"danling_fault_tolerance_{self.replica_id}",
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

    def begin_step(self) -> None:
        """Start one quorum before consuming an optimizer window's data.

        Synchronous quorum is the default so healing restores model and loader
        state before the next forward. The guard spans gradient accumulation.
        """
        if self._manager is not None and not self._quorum_active:
            self._manager.start_quorum()
            self._quorum_active = True
            self._nonfinite_loss = False

    def record_nonfinite_loss(self) -> None:
        """Reject the whole window even when a bad micro-batch is zeroed."""
        self._nonfinite_loss = True

    def should_commit(self, *, nonfinite: bool = False) -> bool:
        """Finish the quorum once, before optimizer or scheduler mutation."""
        if self._manager is None:
            return not nonfinite
        self.begin_step()
        nonfinite = nonfinite or self._nonfinite_loss
        if nonfinite:
            self._consecutive_nonfinite_steps += 1
            self._manager.report_error(RuntimeError("Loss or gradients are not finite"))
        else:
            self._consecutive_nonfinite_steps = 0
        try:
            commit = bool(self._manager.should_commit())
        finally:
            self._quorum_active = False
        if self._consecutive_nonfinite_steps > self._max_nonfinite_steps:
            raise RuntimeError("Nonfinite gradients persisted beyond the configured FT retry limit")
        return commit and not nonfinite

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
        self._quorum_active = False
        self._close_resource(self._replicate_process_group)
        self._replicate_process_group = None
        self._close_resource(self._manager)
        self._manager = None
        self._close_resource(self._process_group)
        self._process_group = None

    def state_dict(self) -> Mapping[str, Any]:
        """Persist the manager's committed-step and participation metadata."""
        return {} if self._manager is None else self._manager.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._manager is not None:
            self._manager.load_state_dict(dict(state))

    def _state_dict(self) -> Mapping[str, Any]:
        state = dict(self.runner.state_dict())
        # The manager transports its own metadata during healing. Its callback
        # only owns runner components, not the active quorum's manager state.
        state.pop("fault_tolerance", None)
        return state

    def _load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.runner.load_checkpoint(state_dict)
