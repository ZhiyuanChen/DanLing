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

import gc as python_gc
import os
import signal
import socket
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from warnings import warn

from chanfig import FlatDict

if TYPE_CHECKING:
    from .base_runner import BaseRunner


class RunnerSupervisor:
    """Supervisor for runner GC, heartbeat, and signal handling."""

    def __init__(self, runner: BaseRunner) -> None:
        self.runner = runner
        self._signal_handlers: dict[int, Any] = {}
        self._termination_signal: int | None = None
        self._termination_request_count = 0

        self._gc_managed = False
        self._gc_was_enabled: bool | None = None
        self._gc_last_collection: dict[str, int] = {}
        self._gc_interval_value: int | None = None
        self._gc_generation_value: int = 1
        self._gc_disable_automatic = True

        self._heartbeat_enabled = False
        self._heartbeat_interval_seconds: float | None = None
        self._heartbeat_dir_name = "heartbeats"
        self._heartbeat_last_progress_time = time.time()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_lock = threading.Lock()
        self._heartbeat_warned = False

    def init_heartbeat(self) -> None:
        config = self.runner.config.get("heartbeat")
        if not isinstance(config, Mapping):
            return
        if not bool(config.get("enabled", False)):
            return

        interval_seconds = float(config.get("interval_seconds", 60.0))
        if interval_seconds <= 0:
            raise ValueError(f"heartbeat.interval_seconds must be a positive number, got {interval_seconds}")

        dir_name = os.fsdecode(config.get("dir_name", "heartbeats"))
        self._heartbeat_enabled = True
        self._heartbeat_interval_seconds = interval_seconds
        self._heartbeat_dir_name = dir_name
        self._heartbeat_last_progress_time = time.time()
        self._heartbeat_stop_event.clear()
        self.write_heartbeat(status="starting", event="startup")
        self.start_heartbeat_thread()

    def heartbeat_payload(self, *, status: str, event: str | None = None) -> FlatDict:
        now = time.time()
        with self._heartbeat_lock:
            last_progress_at = self._heartbeat_last_progress_time

        try:
            progress = float(self.runner.progress)
        except ValueError:
            progress = None

        payload = {
            "status": status,
            "event": event,
            "updated_at_unix": now,
            "last_progress_at_unix": last_progress_at,
            "seconds_since_progress": max(now - last_progress_at, 0.0),
            "id": self.runner.id,
            "name": self.runner.name,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "rank": self.runner.rank,
            "local_rank": self.runner.local_rank,
            "world_size": self.runner.world_size,
            "mode": self.runner.mode.value,
            "split": self.runner.split,
            "global_step": self.runner.train_state.global_step,
            "micro_step": self.runner.train_state.micro_step,
            "epoch": self.runner.train_state.epoch,
            "restart_count": self.runner.elastic_state.restart_count,
            "progress": progress,
        }
        total_steps = self.runner.steps
        if total_steps is not None:
            payload["steps"] = total_steps
        total_epochs = self.runner.epochs
        if total_epochs is not None:
            payload["epochs"] = total_epochs
        return FlatDict(payload)

    def write_heartbeat(self, *, status: str = "running", event: str | None = None) -> None:
        if not self._heartbeat_enabled:
            return

        path = os.path.join(self.runner.workspace.dir, self._heartbeat_dir_name, f"rank-{self.runner.rank:05d}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
        try:
            payload = self.heartbeat_payload(status=status, event=event)
            payload.json(tmp_path, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except OSError as exc:
            if os.path.exists(tmp_path):
                with suppress(OSError):
                    os.remove(tmp_path)
            if not self._heartbeat_warned:
                warn(f"heartbeat write failed: {exc}", RuntimeWarning, stacklevel=2)
                self._heartbeat_warned = True

    def _heartbeat_loop(self) -> None:
        interval_seconds = self._heartbeat_interval_seconds
        if interval_seconds is None:
            return
        while not self._heartbeat_stop_event.wait(interval_seconds):
            self.write_heartbeat()

    def start_heartbeat_thread(self) -> None:
        thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"danling-heartbeat-rank{self.runner.rank}",
            daemon=True,
        )
        thread.start()
        self._heartbeat_thread = thread

    def mark_heartbeat_progress(self) -> None:
        if not self._heartbeat_enabled:
            return
        with self._heartbeat_lock:
            self._heartbeat_last_progress_time = time.time()

    def stop_heartbeat(self, *, status: str = "closed", event: str | None = None) -> None:
        if not self._heartbeat_enabled:
            return
        self._heartbeat_stop_event.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._heartbeat_thread = None
        self.write_heartbeat(status=status, event=event)
        self._heartbeat_enabled = False
        self._heartbeat_interval_seconds = None
        self._heartbeat_dir_name = "heartbeats"
        self._heartbeat_stop_event.clear()
        self._heartbeat_warned = False

    def init_garbage_collection(self) -> None:
        value = self.runner.config.get("gc.interval")
        if value is None:
            return
        interval = int(value)
        if interval <= 0:
            raise ValueError(f"gc.interval must be a positive integer, got {interval}")
        generation = int(self.runner.config.get("gc.generation", 1))
        if generation not in (0, 1, 2):
            raise ValueError(f"gc.generation must be one of 0, 1, or 2, got {generation}")
        self._gc_interval_value = interval
        self._gc_generation_value = generation
        self._gc_disable_automatic = bool(self.runner.config.get("gc.disable_automatic", True))
        self._gc_managed = True
        self._gc_last_collection.clear()
        self._gc_was_enabled = python_gc.isenabled()
        if self._gc_disable_automatic and self._gc_was_enabled:
            python_gc.disable()
        self.run_gc_collection(scope="startup", progress=0)

    def run_gc_collection(self, *, scope: str, progress: int) -> None:
        del scope, progress
        python_gc.collect(self._gc_generation_value)

    def maybe_collect_garbage(self, progress: int, *, scope: str) -> bool:
        interval = self._gc_interval_value
        if interval is None or progress <= 0 or progress % interval != 0:
            return False
        if self._gc_last_collection.get(scope) == progress:
            return False
        self.run_gc_collection(scope=scope, progress=progress)
        self._gc_last_collection[scope] = progress
        return True

    def restore_gc(self) -> None:
        if not self._gc_managed:
            return
        if self._gc_disable_automatic and self._gc_was_enabled:
            python_gc.enable()
        self._gc_managed = False
        self._gc_was_enabled = None
        self._gc_last_collection.clear()
        self._gc_interval_value = None
        self._gc_generation_value = 1
        self._gc_disable_automatic = True

    def init_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        if not hasattr(signal, "SIGTERM"):
            return
        self.install_signal_handler(signal.SIGTERM)

    def install_signal_handler(self, signum: int) -> None:
        if signum in self._signal_handlers:
            return
        previous_handler = signal.getsignal(signum)
        signal.signal(signum, self.request_shutdown)
        self._signal_handlers[signum] = previous_handler

    def restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        while self._signal_handlers:
            signum, handler = self._signal_handlers.popitem()
            signal.signal(signum, handler)

    def request_shutdown(self, signum: int, frame: Any) -> None:
        del frame
        self._termination_signal = int(signum)
        self._termination_request_count += 1
        if self._termination_request_count > 1:
            raise SystemExit(128 + int(signum))

    def maybe_handle_termination_signal(self) -> None:
        signum = self._termination_signal
        if signum is None:
            return

        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"signal-{signum}"

        print(f"runner: received {signal_name}; saving checkpoint and exiting")
        self.write_heartbeat(status="terminating", event=signal_name)
        self.runner.prepare_for_shutdown_checkpoint()
        self.runner.save_checkpoint(save_best=False, force=True)

        drained = self.runner.close(timeout=self.runner.config.get("checkpoint.wait_timeout"))
        if not drained:
            warn("runner shutdown: timed out while draining async checkpoints", RuntimeWarning, stacklevel=2)

        raise SystemExit(128 + signum)

    def close(self) -> None:
        self.stop_heartbeat(status="closed", event="shutdown")
        self.restore_gc()
