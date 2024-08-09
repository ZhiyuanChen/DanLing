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

import os
import shutil
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from warnings import warn

import torch

from .base import CheckpointManager


@dataclass
class CheckpointTask:
    payload: Mapping[str, Any]
    name: str
    history_name: str | None
    should_update_best: bool


class FileCheckpointManager(CheckpointManager):
    """Filesystem checkpoint manager with reliable history queue + latest-wins coalescing."""

    _executor: ThreadPoolExecutor | None
    _inflight: Future | None
    _history_names_inflight_or_pending: set[str]
    _warned_unsupported_staging_mode: bool
    _retention_history: deque[str]
    _closing: bool
    _warned_submit_after_shutdown: bool

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        self._executor = None
        self._history_names_inflight_or_pending = set()
        self._warned_unsupported_staging_mode = False
        self._retention_history = deque()
        self._warned_submit_after_shutdown = False

    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
        force: bool = False,
    ) -> None:
        epochs = self.runner.train_state.epoch if epochs is None else epochs
        if not self.should_persist_checkpoint(epochs=epochs, last_step=last_step, force=force):
            return

        payload = self.build_checkpoint_payload(last_step=last_step)
        async_mode = self.checkpoint_async_mode()
        if async_mode != "disabled":
            payload = self._snapshot_payload(payload)
        task = CheckpointTask(
            payload=payload,
            name=name,
            history_name=self.resolve_history_name(epochs, suffix=".pth"),
            should_update_best=bool(save_best and self.runner.is_best),
        )
        if async_mode == "async_with_pinned_mem" and not self._warned_unsupported_staging_mode:
            warn(
                "checkpoint.async_mode='async_with_pinned_mem' is not supported by file backend; "
                "falling back to regular async save",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_unsupported_staging_mode = True
        if async_mode == "disabled":
            self._persist_checkpoint_payload(task)
            return
        with self._lock:
            if task.history_name is not None:
                if task.history_name in self._history_names_inflight_or_pending:
                    task.history_name = None
                else:
                    self._history_names_inflight_or_pending.add(task.history_name)
        self._enqueue_async_task(task)

    @classmethod
    def _snapshot_payload(cls, payload: Any) -> Any:
        if torch.is_tensor(payload):
            tensor = payload.detach()
            if tensor.device.type != "cpu":
                return tensor.to(device="cpu", copy=True)
            return tensor.clone()
        if isinstance(payload, Mapping):
            return {key: cls._snapshot_payload(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [cls._snapshot_payload(value) for value in payload]
        if isinstance(payload, tuple):
            return tuple(cls._snapshot_payload(value) for value in payload)
        return payload

    def load_model_state(
        self,
        *,
        model: Any,
        model_state_dict: Mapping[str, Any],
        options_cls: Any = None,
        strict: bool = True,
    ) -> None:
        del options_cls
        model.load_state_dict(model_state_dict, strict=strict)

    def load_optimizer_state(
        self,
        *,
        model: Any,
        optimizer: Any,
        optimizer_state_dict: Mapping[str, Any],
        options_cls: Any = None,
        strict: bool = True,
    ) -> None:
        del model, options_cls, strict
        optimizer.load_state_dict(optimizer_state_dict)

    def _persist_checkpoint_payload(self, task: CheckpointTask) -> None:
        checkpoint_dir = self.runner.workspace.checkpoint_dir
        latest_path = os.path.join(checkpoint_dir, f"{task.name}.pth")
        latest_tmp_path = os.path.join(checkpoint_dir, f"{task.name}.tmp-{self.runner.id}.pth")
        self.runner.save(task.payload, latest_tmp_path)
        if not os.path.exists(latest_tmp_path):
            raise RuntimeError(f"checkpoint temp file was not materialized for {latest_tmp_path!r}")
        os.replace(latest_tmp_path, latest_path)

        if task.history_name is not None:
            self._update_checkpoint_alias(latest_path, os.path.join(checkpoint_dir, task.history_name))
            to_delete = self._record_retention_entry(self._retention_history, task.history_name)
            for stale_history in to_delete:
                stale_path = os.path.join(checkpoint_dir, stale_history)
                self._enqueue_purge_path(stale_path)

        if task.should_update_best:
            self._update_checkpoint_alias(latest_path, os.path.join(checkpoint_dir, "best.pth"))

    def _is_async_task_reliable(self, task: CheckpointTask) -> bool:
        return task.history_name is not None or task.should_update_best

    def _start_async_task_locked(self, task: CheckpointTask) -> Future | None:
        if self._closing:
            return None
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"danling-ckpt-{self.runner.id[:8]}",
            )
        try:
            future = self._executor.submit(self._persist_checkpoint_payload, task)
        except RuntimeError as exc:
            message = str(exc)
            if "cannot schedule new futures after" not in message:
                raise

            self._closing = True
            self._inflight = None
            self._pending_latest = None
            self._pending_reliable.clear()
            self._history_names_inflight_or_pending.clear()
            if not self._warned_submit_after_shutdown:
                warn("checkpoint save skipped during shutdown", RuntimeWarning, stacklevel=2)
                self._warned_submit_after_shutdown = True
            return None
        return future

    def _release_async_task_locked(self, task: CheckpointTask) -> None:
        if task.history_name is not None:
            self._history_names_inflight_or_pending.discard(task.history_name)

    def _on_async_task_failed(self, task: CheckpointTask, exc: Exception) -> None:
        del task
        self._record_async_error(exc)
        warn(f"checkpoint save failed: {exc}", RuntimeWarning, stacklevel=2)

    def _on_async_task_start_failed(self, task: CheckpointTask, exc: Exception) -> None:
        if task.history_name is not None:
            with self._lock:
                self._history_names_inflight_or_pending.discard(task.history_name)
        super()._on_async_task_start_failed(task, exc)

    def wait(self, timeout: float | None = None) -> bool:
        return self._wait_for_async_tasks(timeout=timeout)

    def close(self, timeout: float | None = None) -> bool:
        drained = False
        close_error: Exception | None = None
        try:
            drained = self.wait(timeout=timeout)
        except Exception as exc:
            close_error = exc

        with self._lock:
            self._closing = True
            executor = self._executor
            self._executor = None
            self._inflight = None
            self._pending_latest = None
            self._pending_reliable.clear()
            self._history_names_inflight_or_pending.clear()
            self._retention_history.clear()

        if executor is not None:
            if drained:
                executor.shutdown(wait=True)
            else:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)

        self._close_purge_worker()
        if close_error is not None:
            raise close_error
        return drained

    def _update_checkpoint_alias(self, source_path: str, alias_path: str) -> None:
        """Update one checkpoint alias using hardlink first, copy as fallback."""

        if not os.path.exists(source_path):
            warn(f"checkpoint source path does not exist: {source_path!r}", RuntimeWarning, stacklevel=2)
            return

        alias_tmp_path = f"{alias_path}.tmp-{self.runner.id}"
        try:
            if os.path.lexists(alias_tmp_path):
                os.remove(alias_tmp_path)
            os.link(source_path, alias_tmp_path)
            os.replace(alias_tmp_path, alias_path)
            return
        except OSError:
            if os.path.lexists(alias_tmp_path):
                os.remove(alias_tmp_path)

        try:
            shutil.copy2(source_path, alias_path)
        except OSError as exc:
            warn(
                f"failed to update checkpoint alias {alias_path!r} from {source_path!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
