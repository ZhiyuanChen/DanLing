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
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from threading import Lock
from time import monotonic, sleep
from typing import Any
from warnings import warn

from danling.utils import load

from ..config import RunnerConfig


class _PurgeTerminate:
    pass


class CheckpointManager(ABC):
    """Backend-agnostic checkpoint management contract."""

    _VALID_ASYNC_MODES = frozenset({"disabled", "async", "async_with_pinned_mem"})

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self._inflight: Future | None = None
        self._pending_latest: Any | None = None
        self._pending_reliable: deque[Any] = deque()
        self._lock = Lock()
        self._closing = False
        self._async_error: Exception | None = None
        self._keep_latest_k = int(self.runner.config.get("checkpoint.keep_latest_k", 0) or 0)
        if self._keep_latest_k < 0:
            raise ValueError(f"keep_latest_k must be non-negative, got {self._keep_latest_k}")
        self._purge_queue: queue.Queue[str | _PurgeTerminate] | None = None
        self._purge_thread: threading.Thread | None = None
        if self._keep_latest_k > 0:
            self._purge_queue = queue.Queue()
            self._purge_thread = threading.Thread(
                target=self._purge_worker,
                daemon=True,
                name=f"danling-ckpt-purge-{self.__class__.__name__.lower()}-{self.runner.id[:8]}",
            )
            self._purge_thread.start()

    @abstractmethod
    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
        force: bool = False,
    ) -> None:
        """Persist one checkpoint update."""

    def checkpoint_async_mode(self) -> str:
        mode = self.runner.config.get("checkpoint.async_mode")
        if mode is None:
            mode = self.runner.config.get("checkpoint.async_enabled", True)

        if isinstance(mode, bool):
            return "async" if mode else "disabled"

        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized in {"true", "on", "enable", "enabled"}:
            return "async"
        if normalized in {"false", "off", "disable"}:
            return "disabled"
        if normalized not in self._VALID_ASYNC_MODES:
            valid = ", ".join(sorted(self._VALID_ASYNC_MODES))
            raise ValueError(f"Unknown checkpoint.async_mode: {mode!r}. Valid options are: {valid}")
        return normalized

    def _is_async_task_reliable(self, task: Any) -> bool:
        del task
        return False

    def _start_async_task_locked(self, task: Any) -> Future | None:
        del task
        raise NotImplementedError

    def _on_async_task_succeeded(self, task: Any) -> None:
        del task

    def _on_async_task_failed(self, task: Any, exc: Exception) -> None:
        del task
        self._record_async_error(exc)
        warn(f"checkpoint save failed: {exc}", RuntimeWarning, stacklevel=2)

    def _on_async_task_start_failed(self, task: Any, exc: Exception) -> None:
        self._on_async_task_failed(task, exc)

    def _on_async_task_dropped(self, task: Any) -> None:
        del task

    def _release_async_task_locked(self, task: Any) -> None:
        del task

    def _after_async_task_done_locked(self, future: Future, task: Any) -> None:
        del future, task

    def _enqueue_async_task(self, task: Any) -> None:
        task_to_start: Any | None = None
        dropped_task: Any | None = None
        with self._lock:
            if self._closing:
                return
            inflight = self._inflight
            if inflight is None or inflight.done():
                task_to_start = task
            elif self._is_async_task_reliable(task):
                self._pending_reliable.append(task)
            else:
                dropped_task = self._pending_latest
                self._pending_latest = task
        if dropped_task is not None:
            self._on_async_task_dropped(dropped_task)
        if task_to_start is not None:
            self._start_or_queue_async_task(task_to_start)

    def _start_or_queue_async_task(self, task: Any) -> None:
        dropped_task: Any | None = None
        future: Future | None = None
        start_error: Exception | None = None
        with self._lock:
            if self._closing:
                return
            inflight = self._inflight
            if inflight is not None and not inflight.done():
                if self._is_async_task_reliable(task):
                    self._pending_reliable.appendleft(task)
                else:
                    dropped_task = self._pending_latest
                    self._pending_latest = task
            else:
                try:
                    future = self._start_async_task_locked(task)
                except Exception as exc:  # pragma: no cover - delegated to backend-specific tests.
                    start_error = exc
                else:
                    if future is not None:
                        self._inflight = future
        if dropped_task is not None:
            self._on_async_task_dropped(dropped_task)
        if start_error is not None:
            self._on_async_task_start_failed(task, start_error)
            raise start_error
        if future is None:
            return

        def _on_done(completed: Future[Any], checkpoint_task: Any = task) -> None:
            self._on_async_task_done(completed, checkpoint_task)

        future.add_done_callback(_on_done)

    def _on_async_task_done(self, future: Future, task: Any) -> None:
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - backend-specific behavior is tested in subclasses.
            self._on_async_task_failed(task, exc)
        else:
            self._on_async_task_succeeded(task)

        next_task: Any | None = None
        with self._lock:
            if self._inflight is future:
                self._inflight = None
            self._release_async_task_locked(task)
            self._after_async_task_done_locked(future, task)
            if self._pending_reliable:
                next_task = self._pending_reliable.popleft()
            elif self._pending_latest is not None:
                next_task = self._pending_latest
                self._pending_latest = None
        if next_task is not None:
            self._start_or_queue_async_task(next_task)

    def _wait_for_async_tasks(self, timeout: float | None = None) -> bool:
        if self.checkpoint_async_mode() == "disabled":
            if self._async_error is not None:
                failure = self._async_error
                self._async_error = None
                raise failure
            return True

        deadline = None if timeout is None else monotonic() + max(float(timeout), 0.0)
        while True:
            with self._lock:
                inflight = self._inflight
                pending_latest = self._pending_latest
                pending_reliable = bool(self._pending_reliable)

            if inflight is None and pending_latest is None and not pending_reliable:
                if self._async_error is not None:
                    failure = self._async_error
                    self._async_error = None
                    raise failure
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
                    continue
                continue

            sleep(min(max(remaining, 0.0), 0.01) if remaining is not None else 0.01)

    def _record_async_error(self, exc: Exception) -> None:
        if self._async_error is None:
            self._async_error = exc

    def _record_retention_entry(
        self,
        history: deque[str],
        entry: str,
        *,
        protected_entries: Sequence[str | None] = (),
    ) -> list[str]:
        if self._keep_latest_k <= 0:
            return []

        protected = {value for value in protected_entries if value}
        to_delete: list[str] = []
        with self._lock:
            with suppress(ValueError):
                history.remove(entry)
            history.append(entry)
            attempts = len(history)
            while len(history) > self._keep_latest_k and attempts > 0:
                candidate = history.popleft()
                if candidate in protected:
                    history.append(candidate)
                else:
                    to_delete.append(candidate)
                attempts -= 1
        return to_delete

    def _enqueue_purge_path(self, path: str) -> None:
        purge_queue = self._purge_queue
        if purge_queue is not None:
            purge_queue.put(path)
            return
        self._purge_path(path)

    def _purge_path(self, path: str) -> None:
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
            self._purge_path(path)

    def _close_purge_worker(self) -> None:
        purge_thread = self._purge_thread
        purge_queue = self._purge_queue
        self._purge_thread = None
        self._purge_queue = None
        if purge_thread is not None and purge_queue is not None:
            purge_queue.put(_PurgeTerminate())
            purge_thread.join()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until pending checkpoint work is drained."""

        return True

    def maybe_wait_for_staging(self, timeout: float | None = None) -> bool:
        """Wait for async staging completion when supported by backend."""

        del timeout
        return True

    def close(self, timeout: float | None = None) -> bool:
        """Finalize checkpoint I/O before runner shutdown."""

        return self.wait(timeout=timeout)

    def load_model_state(
        self,
        *,
        model: Any,
        model_state_dict: Mapping[str, Any],
        options_cls: Any = None,
        strict: bool = True,
    ) -> None:
        """Load model state for the active checkpoint backend."""
        raise NotImplementedError

    def load_optimizer_state(
        self,
        *,
        model: Any,
        optimizer: Any,
        optimizer_state_dict: Mapping[str, Any],
        options_cls: Any = None,
        strict: bool = True,
    ) -> None:
        """Load optimizer state for the active checkpoint backend."""
        raise NotImplementedError

    def load_checkpoint(self, checkpoint: bytes | str | os.PathLike) -> dict[str, Any]:
        """Load a checkpoint payload from backend storage."""

        checkpoint_path = self.resolve_checkpoint_path(checkpoint)
        checkpoint_state = load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint_state, Mapping):
            raise ValueError(
                "invalid checkpoint payload: expected a mapping, "
                f"got {type(checkpoint_state).__name__}: {checkpoint_state!r}"
            )
        return dict(checkpoint_state)

    @classmethod
    def resolve_checkpoint_path(cls, checkpoint: bytes | str | os.PathLike) -> str:
        """Resolve checkpoint input path for this backend."""

        del cls
        return os.fsdecode(checkpoint)

    @classmethod
    def is_checkpoint_path(cls, checkpoint: bytes | str | os.PathLike) -> bool:
        """Return whether path points to a backend checkpoint payload."""

        return os.path.isfile(cls.resolve_checkpoint_path(checkpoint))

    @classmethod
    def read_config(cls, checkpoint: bytes | str | os.PathLike) -> RunnerConfig:
        """Read runner config from checkpoint payload path."""

        ckpt = load(cls.resolve_checkpoint_path(checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, Mapping):
            raise ValueError(
                "cannot read runner config: checkpoint payload must be a mapping, "
                f"got {type(ckpt).__name__}: {ckpt!r}"
            )
        if "runner" not in ckpt:
            raise ValueError(
                "cannot read runner config: checkpoint is missing key 'runner'; "
                "use from_pretrained(...) for model-only checkpoints"
            )
        return RunnerConfig(ckpt["runner"])

    def export_model_optimizer_state(
        self,
        *,
        model: Any,
        optimizer: Any,
        options_cls: Any = None,
        strict: bool = True,
    ) -> tuple[Any, Any]:
        """Export model/optimizer state for checkpoint payload composition."""

        del options_cls, strict
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict() if optimizer is not None else {}
        return model_state, optimizer_state

    def should_persist_checkpoint(self, *, epochs: int, last_step: bool = False, force: bool = False) -> bool:
        if self.runner.config.get("checkpoint.load_only", False):
            return False

        if force:
            return True

        if last_step:
            return True

        interval = self.runner.checkpoint_interval
        if self.runner.is_step_mode:
            progress_index = self.runner.train_state.global_step
        else:
            progress_index = epochs + 1

        if interval <= 0:
            return False

        return progress_index > 0 and progress_index % interval == 0

    @staticmethod
    def _to_model_only_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        model_only: dict[str, Any] = {}
        for key in ("model", "model_parts", "module", "parallel", "fsdp"):
            if key in payload:
                model_only[key] = payload[key]
        return model_only or dict(payload)

    @staticmethod
    def _resolve_export_dtype(dtype_name: str):
        import torch  # pylint: disable=C0415

        normalized = str(dtype_name).strip().lower().replace("-", "_")
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float64": torch.float64,
            "fp64": torch.float64,
            "double": torch.float64,
        }
        if normalized not in mapping:
            valid = ", ".join(sorted(mapping))
            raise ValueError(f"Unknown checkpoint.export_dtype: {dtype_name!r}. Valid options are: {valid}")
        return mapping[normalized]

    def _cast_payload_tensors_dtype(self, payload: Any, dtype: Any) -> Any:
        import torch  # pylint: disable=C0415

        if torch.is_tensor(payload):
            try:
                return payload.to(dtype=dtype)
            except Exception:  # pragma: no cover - dtype cast support depends on tensor/storage implementations.
                return payload

        if isinstance(payload, Mapping):
            return {key: self._cast_payload_tensors_dtype(value, dtype) for key, value in payload.items()}

        if isinstance(payload, list):
            return [self._cast_payload_tensors_dtype(value, dtype) for value in payload]
        if isinstance(payload, tuple):
            return tuple(self._cast_payload_tensors_dtype(value, dtype) for value in payload)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return [self._cast_payload_tensors_dtype(value, dtype) for value in payload]
        return payload

    def build_checkpoint_payload(self, *, last_step: bool = False) -> Mapping[str, Any]:
        payload = self.runner.state_dict()
        if not isinstance(payload, Mapping):
            raise ValueError(f"runner.state_dict() must return a mapping, got {type(payload).__name__}")

        if not last_step or not self.runner.config.get("checkpoint.last_save_model_only", False):
            return payload

        model_only_payload = self._to_model_only_payload(payload)

        export_dtype_name = self.runner.config.get("checkpoint.export_dtype")
        if export_dtype_name is None:
            export_dtype_name = self.runner.config.get("export_dtype")
        if export_dtype_name is None:
            return model_only_payload

        dtype = self._resolve_export_dtype(str(export_dtype_name))
        return self._cast_payload_tensors_dtype(model_only_payload, dtype)

    def resolve_history_name(self, epochs: int, *, suffix: str = "") -> str | None:
        """Return periodic history checkpoint name for the active train mode."""

        is_step_mode = self.runner.is_step_mode
        checkpoint_interval = self.runner.checkpoint_interval
        if is_step_mode:
            history_index = self.runner.train_state.global_step
            should_save_history = (
                checkpoint_interval > 0 and history_index > 0 and history_index % checkpoint_interval == 0
            )
        else:
            history_index = epochs
            should_save_history = checkpoint_interval > 0 and (history_index + 1) % checkpoint_interval == 0
        if not should_save_history:
            return None

        if is_step_mode:
            return f"ckpt-s{history_index:012d}{suffix}"
        return f"ckpt-e{history_index + 1:06d}{suffix}"
