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

from lazy_imports import try_import
from torch import distributed as dist

from ..config import RunnerConfig
from .base import CheckpointManager

with try_import() as dcp_runtime:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
    )
    from torch.distributed.checkpoint.state_dict import get_state_dict as dcp_get_state_dict
    from torch.distributed.checkpoint.state_dict import set_model_state_dict as dcp_set_model_state_dict
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict as dcp_set_optimizer_state_dict
    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType


@dataclass
class _PendingPointers:
    target_name: str
    aliases: tuple[str, ...]
    update_best: bool
    cleanup_previous_latest: bool = False
    archive_alias: str | None = None


@dataclass
class _CheckpointTask:
    state: Mapping[str, Any]
    checkpoint_id: str
    pointers: _PendingPointers
    no_dist: bool
    async_mode: str
    reliable: bool


@dataclass
class _FTCheckpointTask:
    checkpoint_id: str
    checkpoint_name: str


@dataclass
class _AsyncPointerState:
    pointers: _PendingPointers
    ft_required: bool
    main_saved: bool = False
    ft_saved: bool = False


class _PurgeTerminate:
    pass


class _DataloaderState:
    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        self._state = dict(state or {})

    def state_dict(self) -> dict[str, Any]:
        return dict(self._state)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._state = dict(state_dict)


class TorchDistributedCheckpointManager(CheckpointManager):
    """Torch DCP checkpoint manager used by TorchRunner."""

    _inflight: Future | None
    _staging_future: Future | None
    _pending_latest: _CheckpointTask | None
    _pending_reliable: deque[_CheckpointTask]
    _save_sequence: int
    _stager: Any | None
    _lock: Lock
    _keep_latest_k: int
    _archive_history: deque[str]
    _purge_queue: queue.Queue[str | _PurgeTerminate] | None
    _purge_thread: threading.Thread | None
    _async_process_group: Any | None
    _owns_async_process_group: bool
    _warned_async_process_group: bool
    _ft_enabled: bool
    _ft_prefix: str
    _ft_replica_id: str
    _ft_inflight: Future | None
    _ft_pending: deque[_FTCheckpointTask]
    _ft_upload_futures: set[Future]
    _ft_archive_history: deque[str]
    _ft_executor: ThreadPoolExecutor | None
    _async_pointer_states: dict[str, _AsyncPointerState]

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        dcp_runtime.check()
        self._inflight = None
        self._staging_future = None
        self._pending_latest = None
        self._pending_reliable = deque()
        self._save_sequence = 0
        self._stager = None
        self._lock = Lock()
        self._async_process_group = None
        self._owns_async_process_group = False
        self._warned_async_process_group = False
        self._keep_latest_k = self.runner.config.get("checkpoint.keep_latest_k", 0) or 0
        if self._keep_latest_k < 0:
            raise ValueError(f"keep_latest_k must be non-negative, got {self._keep_latest_k}")
        self._archive_history = deque()
        self._purge_queue = None
        self._purge_thread = None
        self._ft_enabled = self.runner.config.get("checkpoint.enable_ft_dataloader_checkpoints", False)
        self._ft_prefix = self.runner.config.get("checkpoint.ft_dataloader_checkpoint_prefix", "ft-replica")
        if self._ft_enabled:
            configured_replica_id = self.runner.config.get("checkpoint.ft_replica_id")
            if configured_replica_id is None:
                configured_replica_id = os.getenv("FT_REPLICA_ID")
            if configured_replica_id is None:
                configured_replica_id = self.runner.rank
            self._ft_replica_id = str(configured_replica_id)
        else:
            self._ft_replica_id = "0"
        self._ft_inflight = None
        self._ft_pending = deque()
        self._ft_upload_futures = set()
        self._ft_archive_history = deque()
        self._ft_executor = None
        self._async_pointer_states = {}
        if self._keep_latest_k > 0:
            self._purge_queue = queue.Queue()
            self._purge_thread = threading.Thread(
                target=self._purge_worker,
                daemon=True,
                name=f"danling-ckpt-dcp-purge-{self.runner.id[:8]}",
            )
            self._purge_thread.start()

    @staticmethod
    def resolve_checkpoint_path(checkpoint: bytes | str | os.PathLike) -> str:
        """Resolve DCP checkpoint directory from directory/pointer input."""

        checkpoint_id = os.fsdecode(checkpoint)
        pointer_path = checkpoint_id if checkpoint_id.endswith(".pointer") else f"{checkpoint_id}.pointer"
        if os.path.isfile(pointer_path):
            with open(pointer_path, encoding="utf-8") as pointer_fp:
                pointed = pointer_fp.read().strip()
            if pointed:
                if os.path.isabs(pointed):
                    return pointed
                return os.path.join(os.path.dirname(pointer_path), pointed)
        return checkpoint_id

    @classmethod
    def is_checkpoint_path(cls, checkpoint: bytes | str | os.PathLike) -> bool:
        """Return whether input points to a DCP checkpoint directory."""

        if not isinstance(checkpoint, (bytes, str, os.PathLike)):
            return False
        checkpoint_id = cls.resolve_checkpoint_path(checkpoint)
        metadata_file = os.path.join(checkpoint_id, ".metadata")
        return os.path.isdir(checkpoint_id) and os.path.exists(metadata_file)

    @classmethod
    def read_config(cls, checkpoint: bytes | str | os.PathLike) -> RunnerConfig:
        """Read runner config from DCP checkpoint payload."""

        dcp_runtime.check()
        checkpoint_id = cls.resolve_checkpoint_path(checkpoint)
        ckpt: dict[str, Any] = {"runner": {}}
        dcp.load(ckpt, checkpoint_id=checkpoint_id, no_dist=True)
        if "runner" not in ckpt:
            raise ValueError(
                "cannot read runner config: checkpoint is missing key 'runner'; "
                "use from_pretrained(...) for model-only checkpoints"
            )
        return RunnerConfig(ckpt["runner"])

    @staticmethod
    def state_dict_options(options_cls: Any, **kwargs) -> Any:
        """Build torch.distributed.checkpoint state-dict options."""

        if options_cls is None:
            options_cls = StateDictOptions
        options_kwargs = dict(kwargs)
        options_kwargs.setdefault("flatten_optimizer_state_dict", True)
        try:
            return options_cls(**options_kwargs)  # type: ignore[misc]
        except TypeError:
            options_kwargs.pop("flatten_optimizer_state_dict", None)
            return options_cls(**options_kwargs)  # type: ignore[misc]

    @classmethod
    def export_model_optimizer_state(
        cls,
        *,
        model: Any,
        optimizer: Any,
        options_cls: Any = None,
        strict: bool = True,
    ) -> tuple[Any, Any]:
        """Export model/optimizer state via DCP state-dict API."""

        options = cls.state_dict_options(options_cls, full_state_dict=False, cpu_offload=False, strict=strict)
        optimizers: Any = optimizer if optimizer is not None else []
        return dcp_get_state_dict(model, optimizers, options=options)

    @classmethod
    def load_model_state(
        cls,
        *,
        model: Any,
        model_state_dict: Mapping[str, Any],
        options_cls: Any = None,
        strict: bool = True,
    ) -> None:
        """Load model state via DCP state-dict API."""

        options = cls.state_dict_options(options_cls, strict=strict)
        dcp_set_model_state_dict(model, model_state_dict, options=options)

    @classmethod
    def load_optimizer_state(
        cls,
        *,
        model: Any,
        optimizer: Any,
        optimizer_state_dict: Mapping[str, Any],
        options_cls: Any = None,
        strict: bool = True,
    ) -> None:
        """Load optimizer state via DCP state-dict API."""

        options = cls.state_dict_options(options_cls, strict=strict)
        dcp_set_optimizer_state_dict(
            model,
            optimizer,
            optim_state_dict=optimizer_state_dict,
            options=options,
        )

    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
    ) -> None:
        epochs = self.runner.train_state.epoch if epochs is None else epochs
        if self.runner.config.get("checkpoint.load_only", False):
            return
        if not self.should_persist_checkpoint(epochs=epochs, last_step=last_step):
            return

        normalized_name = self._normalize_name(name)
        archive_alias = self.resolve_archive_name(epochs)
        if archive_alias is not None:
            archive_alias = self._normalize_name(archive_alias)
        async_mode = self.checkpoint_async_mode()

        target_base = archive_alias or normalized_name
        target_name = self._build_target_name(target_base)
        checkpoint_id = os.path.join(self.runner.checkpoint_dir, target_name)

        should_update_best = save_best and self.runner.is_best
        state = self.build_checkpoint_payload(last_step=last_step)

        has_dist = dist.is_available() and dist.is_initialized()
        no_dist = not has_dist

        aliases = ["latest"]
        if archive_alias is not None:
            aliases.append(archive_alias)
        if normalized_name == "best":
            aliases.append("best")
        elif normalized_name != "latest":
            aliases.append(normalized_name)
        aliases = list(dict.fromkeys(aliases))

        pointers = _PendingPointers(
            target_name=target_name,
            aliases=tuple(aliases),
            update_best=should_update_best,
            cleanup_previous_latest=archive_alias is None and normalized_name == "latest",
            archive_alias=archive_alias,
        )
        task = _CheckpointTask(
            state=state,
            checkpoint_id=checkpoint_id,
            pointers=pointers,
            no_dist=no_dist,
            async_mode=async_mode,
            reliable=archive_alias is not None or should_update_best or normalized_name != "latest",
        )

        if async_mode != "disabled":
            ft_required = self._ft_enabled and self._has_stateful_dataloaders()
            with self._lock:
                self._async_pointer_states[target_name] = _AsyncPointerState(
                    pointers=pointers,
                    ft_required=ft_required,
                    ft_saved=not ft_required,
                )
            if ft_required:
                try:
                    self._save_ft_dataloader_checkpoint(checkpoint_name=target_name)
                except Exception:
                    with self._lock:
                        self._async_pointer_states.pop(target_name, None)
                    raise
            self._enqueue_checkpoint_task(task)
            return

        self._save_task(task, apply_pointers=False)
        ft_future = self._save_ft_dataloader_checkpoint(checkpoint_name=target_name)
        ft_saved = self._wait_ft_save_future(ft_future)
        if ft_saved and self._is_io_rank():
            self._apply_pointer_updates(task.pointers)

    def load_checkpoint(self, checkpoint: bytes | str | os.PathLike) -> dict[str, Any]:
        checkpoint_id = self._resolve_checkpoint_id(checkpoint)
        state = self.runner.state_dict()
        no_dist = not (dist.is_available() and dist.is_initialized())
        dcp.load(state, checkpoint_id=checkpoint_id, no_dist=no_dist)
        self._load_ft_dataloader_checkpoint(checkpoint_id=checkpoint_id)
        return dict(state)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else monotonic() + max(float(timeout), 0.0)
        while True:
            with self._lock:
                inflight = self._inflight
                staging_future = self._staging_future
                pending_latest = self._pending_latest
                pending_reliable = self._pending_reliable
                ft_inflight = self._ft_inflight
                ft_pending = self._ft_pending
                ft_upload_pending = self._ft_upload_futures
            if (
                inflight is None
                and staging_future is None
                and pending_latest is None
                and not pending_reliable
                and ft_inflight is None
                and not ft_pending
                and not ft_upload_pending
            ):
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
                    # Exceptions are surfaced by `_on_async_save_done`.
                    continue
                sleep(0)
                continue

            if ft_inflight is not None:
                try:
                    ft_inflight.result(timeout=remaining)
                except FutureTimeoutError:
                    return False
                except Exception:
                    continue
                sleep(0)
                continue

            if staging_future is not None:
                try:
                    staging_future.result(timeout=remaining)
                except FutureTimeoutError:
                    return False
                except Exception as exc:
                    warn(f"dcp checkpoint staging failed: {exc}", RuntimeWarning, stacklevel=2)
                    return False
                with self._lock:
                    if self._staging_future is staging_future:
                        self._staging_future = None
                sleep(0)
                continue

            sleep(min(max(remaining, 0.0), 0.01) if remaining is not None else 0.01)

    def close(self, timeout: float | None = None) -> bool:
        drained = self.wait(timeout=timeout)
        staged = self.maybe_wait_for_staging(timeout=timeout)

        stager = self._stager
        self._stager = None
        self._async_process_group = None
        self._owns_async_process_group = False
        with self._lock:
            self._archive_history.clear()
            self._ft_archive_history.clear()
            self._ft_inflight = None
            self._ft_pending.clear()
            self._ft_upload_futures.clear()
            self._async_pointer_states.clear()
        ft_executor = self._ft_executor
        self._ft_executor = None
        if stager is not None:
            stager.close()
        if ft_executor is not None:
            ft_executor.shutdown(wait=drained)

        purge_thread = self._purge_thread
        purge_queue = self._purge_queue
        self._purge_thread = None
        self._purge_queue = None
        if purge_thread is not None and purge_queue is not None:
            purge_queue.put(_PurgeTerminate())
            purge_thread.join()

        return drained and staged

    def maybe_wait_for_staging(self, timeout: float | None = None) -> bool:
        with self._lock:
            staging_future = self._staging_future
        if staging_future is None:
            return True

        try:
            staging_future.result(timeout=timeout)
        except FutureTimeoutError:
            return False
        except Exception as exc:
            warn(f"dcp checkpoint staging failed: {exc}", RuntimeWarning, stacklevel=2)
            return False

        with self._lock:
            if self._staging_future is staging_future:
                self._staging_future = None
        return True

    @staticmethod
    def _normalize_name(name: str) -> str:
        if name.endswith(".pth"):
            return name[:-4]
        return name

    def _pointer_path(self, pointer: str) -> str:
        return os.path.join(self.runner.checkpoint_dir, f"{pointer}.pointer")

    def _write_pointer(self, pointer: str, target_name: str) -> None:
        pointer_path = self._pointer_path(pointer)
        tmp_path = f"{pointer_path}.tmp-{self.runner.id}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(target_name)
            handle.write("\n")
        os.replace(tmp_path, pointer_path)

    def _read_pointer(self, pointer: str) -> str | None:
        pointer_path = self._pointer_path(pointer)
        if not os.path.isfile(pointer_path):
            return None
        with open(pointer_path, encoding="utf-8") as handle:
            value = handle.read().strip()
        return value or None

    def _remove_stale_archive_pointers(self, target_name: str) -> None:
        checkpoint_dir = self.runner.checkpoint_dir
        try:
            entries = os.listdir(checkpoint_dir)
        except OSError:
            return

        for entry in entries:
            if not entry.endswith(".pointer"):
                continue
            alias = entry[: -len(".pointer")]
            if alias in {"latest", "best"}:
                continue
            pointer_path = os.path.join(checkpoint_dir, entry)
            try:
                with open(pointer_path, encoding="utf-8") as handle:
                    pointed = handle.read().strip()
            except OSError:
                continue
            if pointed != target_name:
                continue
            with suppress(OSError):
                os.remove(pointer_path)

    @staticmethod
    def _metadata_path(checkpoint_id: str) -> str:
        return os.path.join(checkpoint_id, ".metadata")

    def _resolve_checkpoint_id(self, checkpoint: bytes | str | os.PathLike) -> str:
        checkpoint_str = os.fsdecode(checkpoint)

        def resolve_local(path_or_name: str) -> str:
            resolved_path = self.resolve_checkpoint_path(path_or_name)
            if os.path.isdir(resolved_path) and os.path.exists(self._metadata_path(resolved_path)):
                return resolved_path

            if os.sep not in path_or_name:
                normalized = self._normalize_name(path_or_name)
                candidate = os.path.join(self.runner.checkpoint_dir, normalized)
                if os.path.isdir(candidate) and os.path.exists(self._metadata_path(candidate)):
                    return candidate
                pointed = self._read_pointer(normalized)
                if pointed is not None:
                    pointer_candidate = os.path.join(self.runner.checkpoint_dir, pointed)
                    if os.path.isdir(pointer_candidate) and os.path.exists(self._metadata_path(pointer_candidate)):
                        return pointer_candidate

            raise FileNotFoundError(f"dcp checkpoint not found or invalid: {path_or_name!r}")

        has_dist = dist.is_available() and dist.is_initialized()
        should_broadcast = has_dist and self.runner.distributed
        if not should_broadcast:
            return resolve_local(checkpoint_str)

        payload: list[Any] = [None, None]
        if self._is_io_rank():
            try:
                payload = [True, resolve_local(checkpoint_str)]
            except Exception as exc:
                payload = [False, f"{type(exc).__name__}: {exc}"]
        dist.broadcast_object_list(payload, src=0)
        if not payload[0]:
            raise FileNotFoundError(str(payload[1]))
        return str(payload[1])

    def _on_async_save_done(self, future: Future, task: _CheckpointTask) -> None:
        next_task: _CheckpointTask | None = None
        try:
            future.result()
        except Exception as exc:
            warn(f"dcp checkpoint save failed: {exc}", RuntimeWarning, stacklevel=2)
            self._mark_async_pointer_state(task.pointers.target_name, failed=True)
            with self._lock:
                if self._inflight is future:
                    self._inflight = None
                if self._staging_future is not None and self._staging_future.done():
                    self._staging_future = None
                next_task = self._pop_next_task_locked()
            if next_task is not None:
                self._start_task(next_task)
            return

        self._mark_async_pointer_state(task.pointers.target_name, main_saved=True)

        with self._lock:
            if self._inflight is future:
                self._inflight = None
            if self._staging_future is not None and self._staging_future.done():
                self._staging_future = None
            next_task = self._pop_next_task_locked()
        if next_task is not None:
            self._start_task(next_task)

    @staticmethod
    def _as_future(response: Any) -> Future | None:
        if isinstance(response, Future):
            return response
        upload_completion = getattr(response, "upload_completion", None)
        if isinstance(upload_completion, Future):
            return upload_completion
        return None

    @staticmethod
    def _as_staging_future(response: Any) -> Future | None:
        staging_completion = getattr(response, "staging_completion", None)
        if isinstance(staging_completion, Future):
            return staging_completion
        return None

    def _is_io_rank(self) -> bool:
        return (not self.runner.distributed) or self.runner.is_main_process

    def _ensure_async_process_group(self) -> Any | None:
        if self._async_process_group is not None:
            return self._async_process_group

        if not self.runner.config.get("checkpoint.dedicated_async_process_group", True):
            return None
        if not (dist.is_available() and dist.is_initialized()):
            return None

        backend = self.runner.config.get("checkpoint.async_process_group_backend", "gloo")
        try:
            process_group = dist.new_group(backend=backend)
        except Exception as exc:
            if not self._warned_async_process_group:
                warn(
                    f"failed to initialize async checkpoint process group with backend={backend!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_async_process_group = True
            return None

        self._async_process_group = process_group
        self._owns_async_process_group = True
        return process_group

    def _ft_checkpoint_dir(self) -> str:
        return os.path.join(self.runner.checkpoint_dir, f"{self._ft_prefix}-{self._ft_replica_id}")

    def _has_stateful_dataloaders(self) -> bool:
        dataloaders = getattr(self.runner, "dataloaders", None)
        if not dataloaders:
            return False
        for loader in dataloaders.values():
            state_dict_fn = getattr(loader, "state_dict", None)
            if callable(state_dict_fn):
                return True
        return False

    def _collect_dataloader_state(self) -> Mapping[str, Any]:
        return self.runner.dataloaders.state_dict()

    def _save_ft_dataloader_checkpoint(
        self,
        *,
        checkpoint_name: str,
    ) -> Future | None:
        if not self._ft_enabled:
            return None

        if not self._has_stateful_dataloaders():
            return None

        checkpoint_dir = self._ft_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_id = os.path.join(checkpoint_dir, checkpoint_name)
        task = _FTCheckpointTask(
            checkpoint_id=checkpoint_id,
            checkpoint_name=checkpoint_name,
        )

        ft_future: Future | None = None
        replaced_pending_checkpoint_names: list[str] = []
        queued = False
        with self._lock:
            ft_inflight = self._ft_inflight
            if ft_inflight is not None and not ft_inflight.done():
                if self._ft_pending:
                    replaced_pending_checkpoint_names = [pending.checkpoint_name for pending in self._ft_pending]
                    self._ft_pending.clear()
                # Keep only the latest pending FT task to bound backlog under I/O lag.
                self._ft_pending.append(task)
                queued = True
            else:
                ft_future = self._start_ft_checkpoint_task(task)
                self._ft_inflight = ft_future
        for dropped_checkpoint_name in replaced_pending_checkpoint_names:
            self._mark_async_pointer_state(dropped_checkpoint_name, failed=True)
        if queued:
            return None
        if ft_future is None:
            return None

        def _on_ft_done(completed: Future[Any], saved_name: str = task.checkpoint_name) -> None:
            self._on_ft_save_done(completed, saved_name)

        ft_future.add_done_callback(_on_ft_done)
        return ft_future

    @staticmethod
    def _wait_ft_save_future(future: Future | None) -> bool:
        if future is None:
            return True
        try:
            return future.result()
        except Exception as exc:
            warn(f"ft dataloader checkpoint save failed: {exc}", RuntimeWarning, stacklevel=2)
            return False

    def _on_ft_save_done(self, future: Future, checkpoint_name: str) -> None:
        pending_task: _FTCheckpointTask | None = None
        pending_future: Future | None = None
        pending_start_error: Exception | None = None
        saved = False
        try:
            saved = future.result()
        except Exception as exc:
            warn(f"ft dataloader checkpoint save failed: {exc}", RuntimeWarning, stacklevel=2)
            self._mark_async_pointer_state(checkpoint_name, failed=True)
        else:
            if not saved:
                warn(
                    f"ft dataloader checkpoint skipped for {checkpoint_name!r}: empty dataloader state",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._mark_async_pointer_state(checkpoint_name, failed=True)
        finally:
            with self._lock:
                if self._ft_inflight is future:
                    self._ft_inflight = None
                pending_task = self._ft_pending.popleft() if self._ft_pending else None
                if pending_task is not None:
                    try:
                        pending_future = self._start_ft_checkpoint_task(pending_task)
                    except Exception as exc:  # pragma: no cover - async startup failure path.
                        pending_start_error = exc
                    else:
                        self._ft_inflight = pending_future

        if pending_start_error is not None:
            warn(f"ft dataloader checkpoint save failed: {pending_start_error}", RuntimeWarning, stacklevel=2)
            return
        if pending_task is not None and pending_future is not None:
            saved_name = pending_task.checkpoint_name

            def _on_pending_ft_done(completed: Future[Any], target_name: str = saved_name) -> None:
                self._on_ft_save_done(completed, target_name)

            pending_future.add_done_callback(_on_pending_ft_done)

    def _on_ft_upload_done(self, future: Future, checkpoint_name: str) -> None:
        try:
            future.result()
        except Exception as exc:
            warn(f"ft dataloader checkpoint save failed: {exc}", RuntimeWarning, stacklevel=2)
            self._mark_async_pointer_state(checkpoint_name, failed=True)
        else:
            self._record_ft_archive_target(checkpoint_name)
            self._mark_async_pointer_state(checkpoint_name, ft_saved=True)
        finally:
            with self._lock:
                self._ft_upload_futures.discard(future)

    def _mark_async_pointer_state(
        self,
        checkpoint_name: str,
        *,
        main_saved: bool = False,
        ft_saved: bool = False,
        failed: bool = False,
    ) -> None:
        pointers: _PendingPointers | None = None
        with self._lock:
            state = self._async_pointer_states.get(checkpoint_name)
            if state is None:
                return
            if failed:
                self._async_pointer_states.pop(checkpoint_name, None)
                return
            if main_saved:
                state.main_saved = True
            if ft_saved:
                state.ft_saved = True
            if not state.main_saved or (state.ft_required and not state.ft_saved):
                return
            pointers = state.pointers
            self._async_pointer_states.pop(checkpoint_name, None)

        if pointers is None or not self._is_io_rank():
            return
        try:
            self._apply_pointer_updates(pointers)
        except Exception as exc:  # pragma: no cover - defensive for pointer I/O failures.
            warn(f"failed to update dcp checkpoint pointers: {exc}", RuntimeWarning, stacklevel=2)

    def _start_ft_checkpoint_task(self, task: _FTCheckpointTask) -> Future:
        executor = self._ft_executor
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"danling-ckpt-dcp-ft-{self.runner.id[:8]}",
            )
            self._ft_executor = executor

        control_future: Future = Future()

        def run() -> None:
            try:
                dataloader_state = self._collect_dataloader_state()
                if not dataloader_state:
                    if not control_future.done():
                        control_future.set_result(False)
                    return

                state = {"dataloader": _DataloaderState(dataloader_state)}

                async_mode = self.checkpoint_async_mode()
                if async_mode == "disabled":
                    dcp.save(state, checkpoint_id=task.checkpoint_id, no_dist=True)
                    self._record_ft_archive_target(task.checkpoint_name)
                    if not control_future.done():
                        control_future.set_result(True)
                    return

                response = dcp.async_save(state, checkpoint_id=task.checkpoint_id, no_dist=True)
                upload_future = self._as_future(response)
                if upload_future is None:
                    raise RuntimeError(
                        "dcp.async_save did not return a Future-compatible handle for FT dataloader checkpoint"
                    )
                with self._lock:
                    self._ft_upload_futures.add(upload_future)

                def _on_upload_done(completed: Future[Any], saved_name: str = task.checkpoint_name) -> None:
                    self._on_ft_upload_done(completed, saved_name)

                upload_future.add_done_callback(_on_upload_done)
                if not control_future.done():
                    control_future.set_result(True)
            except Exception as exc:
                if not control_future.done():
                    control_future.set_exception(exc)

        executor.submit(run)
        return control_future

    def _record_ft_archive_target(self, checkpoint_name: str) -> None:
        if self._keep_latest_k <= 0:
            return
        to_delete: list[str] = []
        with self._lock:
            with suppress(ValueError):
                self._ft_archive_history.remove(checkpoint_name)
            self._ft_archive_history.append(checkpoint_name)
            while len(self._ft_archive_history) > self._keep_latest_k:
                to_delete.append(self._ft_archive_history.popleft())
        for stale_name in to_delete:
            self._enqueue_purge_target(os.path.join(f"{self._ft_prefix}-{self._ft_replica_id}", stale_name))

    def _load_ft_dataloader_checkpoint(self, *, checkpoint_id: str | None = None) -> None:
        if not self._ft_enabled:
            return

        dataloader_state = self._collect_dataloader_state()
        if not dataloader_state:
            return

        if checkpoint_id is None:
            return
        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_id))
        ft_checkpoint_id = os.path.join(self._ft_checkpoint_dir(), checkpoint_name)
        metadata_file = os.path.join(ft_checkpoint_id, ".metadata")
        if not os.path.isfile(metadata_file):
            return

        wrapper = _DataloaderState(dataloader_state)
        dcp.load({"dataloader": wrapper}, checkpoint_id=ft_checkpoint_id, no_dist=True)
        self.runner.dataloaders.load_state_dict(wrapper.state_dict())

    def _build_target_name(self, base_name: str) -> str:
        with self._lock:
            self._save_sequence += 1
            save_sequence = self._save_sequence
        global_step = self.runner.train_state.global_step
        return f"{base_name}-g{global_step:012d}-q{save_sequence:06d}"

    def _save_task(self, task: _CheckpointTask, *, apply_pointers: bool = True) -> None:
        dcp.save(task.state, checkpoint_id=task.checkpoint_id, no_dist=task.no_dist)
        if apply_pointers and self._is_io_rank():
            self._apply_pointer_updates(task.pointers)

    def _enqueue_checkpoint_task(self, task: _CheckpointTask) -> None:
        task_to_start: _CheckpointTask | None = None
        dropped_checkpoint_name: str | None = None
        with self._lock:
            inflight = self._inflight
            if inflight is None or inflight.done():
                task_to_start = task
            elif task.reliable:
                self._pending_reliable.append(task)
            else:
                # Keep only the latest non-critical save request to bound backlog.
                previous_pending = self._pending_latest
                self._pending_latest = task
                if previous_pending is not None:
                    dropped_checkpoint_name = previous_pending.pointers.target_name

        if dropped_checkpoint_name is not None:
            self._mark_async_pointer_state(dropped_checkpoint_name, failed=True)
        if task_to_start is not None:
            self._start_task(task_to_start)

    def _start_task(self, task: _CheckpointTask) -> None:
        dropped_checkpoint_name: str | None = None
        future: Future | None = None
        with self._lock:
            inflight = self._inflight
            if inflight is not None and not inflight.done():
                if task.reliable:
                    self._pending_reliable.appendleft(task)
                else:
                    previous_pending = self._pending_latest
                    self._pending_latest = task
                    if previous_pending is not None:
                        dropped_checkpoint_name = previous_pending.pointers.target_name
            else:
                # Launch and publish inflight atomically under the same lock to avoid
                # races with concurrent enqueue/callback threads.
                save_kwargs: dict[str, Any] = {
                    "checkpoint_id": task.checkpoint_id,
                    "no_dist": task.no_dist,
                }
                if not task.no_dist:
                    process_group = self._ensure_async_process_group()
                    if process_group is not None:
                        save_kwargs["process_group"] = process_group
                if task.async_mode == "async_with_pinned_mem":
                    if self._stager is None:
                        self._stager = DefaultStager(StagingOptions(True, True, True, True))
                    save_kwargs["async_checkpointer_type"] = AsyncCheckpointerType.PROCESS
                    save_kwargs["async_stager"] = self._stager
                response = dcp.async_save(task.state, **save_kwargs)
                future = self._as_future(response)
                if future is None:
                    raise RuntimeError("dcp.async_save did not return a Future-compatible handle")
                staging_future = (
                    self._as_staging_future(response) if task.async_mode == "async_with_pinned_mem" else None
                )
                self._inflight = future
                self._staging_future = staging_future

        if dropped_checkpoint_name is not None:
            self._mark_async_pointer_state(dropped_checkpoint_name, failed=True)
        if future is None:
            return

        def _on_async_done(completed: Future[Any], checkpoint_task: _CheckpointTask = task) -> None:
            self._on_async_save_done(completed, checkpoint_task)

        future.add_done_callback(_on_async_done)

    def _pop_next_task_locked(self) -> _CheckpointTask | None:
        if self._pending_reliable:
            return self._pending_reliable.popleft()
        if self._pending_latest is None:
            return None
        task = self._pending_latest
        self._pending_latest = None
        return task

    def _apply_pointer_updates(self, pointers: _PendingPointers) -> None:
        previous_latest = None
        if pointers.cleanup_previous_latest and "latest" in pointers.aliases:
            previous_latest = self._read_pointer("latest")

        for alias in pointers.aliases:
            self._write_pointer(alias, pointers.target_name)

        if pointers.update_best:
            self._write_pointer("best", pointers.target_name)

        if pointers.archive_alias is not None:
            self._record_archive_target(pointers.target_name)

        if previous_latest is None or previous_latest == pointers.target_name:
            return

        best_target = self._read_pointer("best")
        if previous_latest == best_target:
            return

        self._remove_stale_archive_pointers(previous_latest)
        old_checkpoint_path = os.path.join(self.runner.checkpoint_dir, previous_latest)
        if os.path.isdir(old_checkpoint_path):
            shutil.rmtree(old_checkpoint_path, ignore_errors=True)

    def _record_archive_target(self, target_name: str) -> None:
        if self._keep_latest_k <= 0:
            return

        to_delete: list[str] = []
        with self._lock:
            with suppress(ValueError):
                self._archive_history.remove(target_name)
            self._archive_history.append(target_name)

            protected_targets = {
                self._read_pointer("latest"),
                self._read_pointer("best"),
            }
            attempts = len(self._archive_history)
            while len(self._archive_history) > self._keep_latest_k and attempts > 0:
                candidate = self._archive_history.popleft()
                if candidate in protected_targets:
                    self._archive_history.append(candidate)
                else:
                    to_delete.append(candidate)
                attempts -= 1

        for stale_target in to_delete:
            self._remove_stale_archive_pointers(stale_target)
            self._enqueue_purge_target(stale_target)

    def _enqueue_purge_target(self, target_name: str) -> None:
        path = os.path.join(self.runner.checkpoint_dir, target_name)
        purge_queue = self._purge_queue
        if purge_queue is not None:
            purge_queue.put(path)
            return
        shutil.rmtree(path, ignore_errors=True)

    def _purge_worker(self) -> None:
        purge_queue = self._purge_queue
        if purge_queue is None:
            return
        while True:
            target = purge_queue.get()
            if isinstance(target, _PurgeTerminate):
                return
            shutil.rmtree(target, ignore_errors=True)
