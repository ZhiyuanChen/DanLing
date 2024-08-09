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
import queue
import shutil
import threading
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from dataclasses import dataclass
from threading import Lock
from time import monotonic, sleep
from typing import Any
from warnings import warn

from .base import CheckpointManager


@dataclass
class CheckpointTask:
    payload: Mapping[str, Any]
    name: str
    archive_name: str | None
    should_update_best: bool


class _PurgeTerminate:
    pass


class FileCheckpointManager(CheckpointManager):
    """Filesystem checkpoint manager with reliable archive queue + latest-wins coalescing."""

    _executor: ThreadPoolExecutor | None
    _inflight: Future | None
    _pending_latest: CheckpointTask | None
    _pending_reliable: deque[CheckpointTask]
    _archive_names_inflight_or_pending: set[str]
    _lock: Lock
    _warned_unsupported_staging_mode: bool
    _keep_latest_k: int
    _archive_history: deque[str]
    _purge_queue: queue.Queue[str | _PurgeTerminate] | None
    _purge_thread: threading.Thread | None
    _closing: bool
    _warned_submit_after_shutdown: bool

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        self._executor = None
        self._inflight = None
        self._pending_latest = None
        self._pending_reliable = deque()
        self._archive_names_inflight_or_pending = set()
        self._lock = Lock()
        self._warned_unsupported_staging_mode = False
        self._keep_latest_k = int(self.runner.config.get("checkpoint.keep_latest_k", 0) or 0)
        if self._keep_latest_k < 0:
            raise ValueError(f"keep_latest_k must be non-negative, got {self._keep_latest_k}")
        self._archive_history = deque()
        self._purge_queue = None
        self._purge_thread = None
        self._closing = False
        self._warned_submit_after_shutdown = False
        if self._keep_latest_k > 0:
            self._purge_queue = queue.Queue()
            self._purge_thread = threading.Thread(
                target=self._purge_worker,
                daemon=True,
                name=f"danling-ckpt-file-purge-{self.runner.id[:8]}",
            )
            self._purge_thread.start()

    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
    ) -> None:
        epochs = self.runner.train_state.epoch if epochs is None else epochs
        if not self.should_persist_checkpoint(epochs=epochs, last_step=last_step):
            return

        payload = self.build_checkpoint_payload(last_step=last_step)
        task = CheckpointTask(
            payload=payload,
            name=name,
            archive_name=self.resolve_archive_name(epochs, suffix=".pth"),
            should_update_best=bool(save_best and self.runner.is_best),
        )
        async_mode = self.checkpoint_async_mode()
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
        self._enqueue_checkpoint_payload(task)

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
        latest_path = os.path.join(self.runner.checkpoint_dir, f"{task.name}.pth")
        latest_tmp_path = os.path.join(self.runner.checkpoint_dir, f"{task.name}.tmp-{self.runner.id}.pth")
        self.runner.save(task.payload, latest_tmp_path)
        if not os.path.exists(latest_tmp_path):
            return
        os.replace(latest_tmp_path, latest_path)

        if task.archive_name is not None:
            self._update_checkpoint_alias(latest_path, os.path.join(self.runner.checkpoint_dir, task.archive_name))
            self._record_archive(task.archive_name)

        if task.should_update_best:
            self._update_checkpoint_alias(latest_path, os.path.join(self.runner.checkpoint_dir, "best.pth"))

    def _submit_task_locked(self, task: CheckpointTask) -> None:
        if self._closing:
            return
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
            self._archive_names_inflight_or_pending.clear()
            if not self._warned_submit_after_shutdown:
                warn("checkpoint save skipped during shutdown", RuntimeWarning, stacklevel=2)
                self._warned_submit_after_shutdown = True
            return
        self._inflight = future

        def _on_done(completed: Future[Any], checkpoint_task: CheckpointTask = task) -> None:
            self._on_checkpoint_payload_done(completed, checkpoint_task)

        future.add_done_callback(_on_done)

    def _pop_next_task_locked(self) -> CheckpointTask | None:
        if self._pending_reliable:
            return self._pending_reliable.popleft()
        if self._pending_latest is None:
            return None
        task = self._pending_latest
        self._pending_latest = None
        return task

    def _enqueue_checkpoint_payload(self, task: CheckpointTask) -> None:
        with self._lock:
            if self._closing:
                return
            inflight = self._inflight
            if task.archive_name is not None:
                if task.archive_name in self._archive_names_inflight_or_pending:
                    task.archive_name = None
                else:
                    self._archive_names_inflight_or_pending.add(task.archive_name)

            if inflight is None or inflight.done():
                self._submit_task_locked(task)
                return

            if task.archive_name is not None or task.should_update_best:
                self._pending_reliable.append(task)
                return

            # Keep only the latest non-critical queued save request to bound backlog.
            self._pending_latest = task

    def _on_checkpoint_payload_done(self, future: Future, task: CheckpointTask) -> None:
        exception = future.exception()

        if exception is not None:
            warn(
                f"checkpoint save failed: {exception}",
                RuntimeWarning,
                stacklevel=2,
            )

        with self._lock:
            if self._inflight is future:
                self._inflight = None

            if task.archive_name is not None:
                self._archive_names_inflight_or_pending.discard(task.archive_name)

            next_task = self._pop_next_task_locked()
            if next_task is None:
                return
            self._submit_task_locked(next_task)

    def wait(self, timeout: float | None = None) -> bool:
        if not self.checkpoint_async_enabled():
            return True

        deadline = None if timeout is None else monotonic() + max(float(timeout), 0.0)
        while True:
            with self._lock:
                inflight = self._inflight
                pending_latest = self._pending_latest
                pending_reliable = bool(self._pending_reliable)

            if inflight is None and pending_latest is None and not pending_reliable:
                return True

            remaining = None if deadline is None else deadline - monotonic()
            if remaining is not None and remaining <= 0:
                return False

            if inflight is not None:
                try:
                    inflight.result(timeout=remaining)
                except FutureTimeoutError:
                    return False
                except Exception:
                    # Exceptions are surfaced as warnings in `_on_checkpoint_payload_done`.
                    continue
                continue

            sleep(min(max(remaining, 0.0), 0.01) if remaining is not None else 0.01)

    def close(self, timeout: float | None = None) -> bool:
        drained = self.wait(timeout=timeout)

        with self._lock:
            self._closing = True
            executor = self._executor
            self._executor = None
            self._inflight = None
            self._pending_latest = None
            self._pending_reliable.clear()
            self._archive_names_inflight_or_pending.clear()
            self._archive_history.clear()

        if executor is not None:
            if drained:
                executor.shutdown(wait=True)
            else:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)

        purge_thread = self._purge_thread
        purge_queue = self._purge_queue
        self._purge_thread = None
        self._purge_queue = None
        if purge_thread is not None and purge_queue is not None:
            purge_queue.put(_PurgeTerminate())
            purge_thread.join()

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

    def _record_archive(self, archive_name: str) -> None:
        if self._keep_latest_k <= 0:
            return

        to_delete: list[str] = []
        with self._lock:
            with suppress(ValueError):
                self._archive_history.remove(archive_name)
            self._archive_history.append(archive_name)
            while len(self._archive_history) > self._keep_latest_k:
                to_delete.append(self._archive_history.popleft())

        for stale_archive in to_delete:
            stale_path = os.path.join(self.runner.checkpoint_dir, stale_archive)
            self._enqueue_purge_path(stale_path)

    def _enqueue_purge_path(self, path: str) -> None:
        purge_queue = self._purge_queue
        if purge_queue is not None:
            purge_queue.put(path)
            return

        if os.path.isfile(path):
            with suppress(OSError):
                os.remove(path)
            return
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    def _purge_worker(self) -> None:
        purge_queue = self._purge_queue
        if purge_queue is None:
            return
        while True:
            path = purge_queue.get()
            if isinstance(path, _PurgeTerminate):
                return
            if os.path.isfile(path):
                with suppress(OSError):
                    os.remove(path)
                continue
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
