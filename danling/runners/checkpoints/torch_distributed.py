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
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Any
from warnings import warn

from lazy_imports import try_import
from torch import distributed as dist

from ..config import RunnerConfig
from .base import CheckpointManager

with try_import() as dcp_runtime:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.checkpoint.state_dict import get_state_dict as dcp_get_state_dict
    from torch.distributed.checkpoint.state_dict import set_model_state_dict as dcp_set_model_state_dict
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict as dcp_set_optimizer_state_dict

AsyncCheckpointerType: Any
DefaultStager: Any
StagingOptions: Any

try:
    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType as _AsyncCheckpointerType
except ImportError:
    AsyncCheckpointerType = None
else:
    AsyncCheckpointerType = _AsyncCheckpointerType

try:
    from torch.distributed.checkpoint.staging import DefaultStager as _DefaultStager
    from torch.distributed.checkpoint.staging import StagingOptions as _StagingOptions
except ImportError:
    DefaultStager = None
    StagingOptions = None
else:
    DefaultStager = _DefaultStager
    StagingOptions = _StagingOptions

DCP_PINNED_MEMORY_STAGING_AVAILABLE = (
    AsyncCheckpointerType is not None and DefaultStager is not None and StagingOptions is not None
)


@dataclass(frozen=True)
class _PendingPointers:
    checkpoint_id: str
    target_name: str
    aliases: tuple[str, ...]
    update_best: bool
    track_for_retention: bool = False


@dataclass(frozen=True)
class TorchDistributedCheckpointTask:
    state: Mapping[str, Any]
    checkpoint_id: str
    pointers: _PendingPointers
    no_dist: bool
    async_mode: str
    reliable: bool
    write_runner_config: bool = True


class _PointerUpdateError(RuntimeError):
    def __init__(
        self,
        alias: str,
        target_name: str,
        published_aliases: tuple[str, ...],
        cause: Exception,
    ) -> None:
        super().__init__(alias, target_name, published_aliases, cause)

    @property
    def alias(self) -> str:
        return self.args[0]

    @property
    def target_name(self) -> str:
        return self.args[1]

    @property
    def published_aliases(self) -> tuple[str, ...]:
        return self.args[2]

    @property
    def cause(self) -> Exception:
        return self.args[3]

    def __str__(self) -> str:
        return f"failed to update pointer {self.alias!r} for {self.target_name!r}: {self.cause}"


class CheckpointPointerStore:
    def __init__(self, checkpoint_dir: str, runner_id: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.runner_id = runner_id

    def path(self, pointer: str) -> str:
        return os.path.join(self.checkpoint_dir, f"{pointer}.pointer")

    def write(self, pointer: str, target_name: str) -> None:
        pointer_path = self.path(pointer)
        tmp_path = f"{pointer_path}.tmp-{self.runner_id}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(target_name)
            handle.write("\n")
        os.replace(tmp_path, pointer_path)

    def read(self, pointer: str) -> str | None:
        pointer_path = self.path(pointer)
        if not os.path.isfile(pointer_path):
            return None
        with open(pointer_path, encoding="utf-8") as handle:
            value = handle.read().strip()
        return value or None

    def is_target_published(self, target_name: str) -> bool:
        try:
            entries = os.listdir(self.checkpoint_dir)
        except OSError:
            return False

        for entry in entries:
            if not entry.endswith(".pointer"):
                continue
            pointer_path = os.path.join(self.checkpoint_dir, entry)
            try:
                with open(pointer_path, encoding="utf-8") as handle:
                    if handle.read().strip() == target_name:
                        return True
            except OSError:
                continue
        return False

    def remove_stale_history_pointers(self, target_name: str) -> None:
        try:
            entries = os.listdir(self.checkpoint_dir)
        except OSError:
            return

        for entry in entries:
            if not entry.endswith(".pointer"):
                continue
            alias = entry[: -len(".pointer")]
            if alias in {"latest", "best"}:
                continue
            pointer_path = os.path.join(self.checkpoint_dir, entry)
            try:
                with open(pointer_path, encoding="utf-8") as handle:
                    pointed = handle.read().strip()
            except OSError:
                continue
            if pointed != target_name:
                continue
            with suppress(OSError):
                os.remove(pointer_path)


class TorchDistributedCheckpointManager(CheckpointManager):
    """Torch DCP checkpoint manager used by TorchRunner.

    ``TorchFTCheckpointManager`` is a package-internal subclass that relies on
    the protected checkpoint task, pointer, purge, and async lifecycle helpers
    here. Those helpers are an internal subclass contract, not public user API;
    refactor them together with the FT subclass and checkpoint tests.
    """

    @property
    def is_collective(self) -> bool:
        return True

    _inflight: Future | None
    _staging_future: Future | None
    _save_sequence: int
    _stager: Any | None
    _retention_history: deque[str]
    _async_process_group: Any | None
    _owns_async_process_group: bool
    _warned_async_process_group: bool
    _pinned_memory_async_staging: bool
    _pointer_store: CheckpointPointerStore

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        dcp_runtime.check()
        self._staging_future = None
        self._save_sequence = 0
        self._stager = None
        self._async_process_group = None
        self._owns_async_process_group = False
        self._warned_async_process_group = False
        self._pinned_memory_async_staging = DCP_PINNED_MEMORY_STAGING_AVAILABLE
        if self.checkpoint_async_mode() == "async_with_pinned_mem" and not self._pinned_memory_async_staging:
            warn(
                "ckpt.async_mode='async_with_pinned_mem' requested, but this PyTorch version does not expose "
                "pinned-memory async staging; falling back to DCP async save",
                RuntimeWarning,
                stacklevel=2,
            )
        self._retention_history = deque()
        self._pointer_store = CheckpointPointerStore(self.runner.workspace.checkpoint_dir, self.runner.id)

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
        runner_yaml = os.path.join(checkpoint_id, "runner.yaml")
        if os.path.isfile(runner_yaml):
            return RunnerConfig.from_yaml(runner_yaml)

        raise ValueError(f"cannot read runner config from DCP checkpoint: missing {runner_yaml!r}")

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
        dcp_set_model_state_dict(model, dict(model_state_dict), options=options)

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
            optim_state_dict=dict(optimizer_state_dict),
            options=options,
        )

    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
        force: bool = False,
    ) -> None:
        epochs = self.runner.train_state.epoch if epochs is None else epochs
        if not self.runner.config.get("ckpt.enabled", True):
            return
        should_persist = self.should_persist_checkpoint(epochs=epochs, last_step=last_step, force=force)
        should_update_best = bool(save_best and self.runner.is_best)
        if not should_persist and not should_update_best:
            return

        async_mode = self.checkpoint_async_mode()
        state = self.build_checkpoint_payload(last_step=last_step)
        pointers, reliable = self._build_checkpoint_pointers(name=name, epochs=epochs, save_best=save_best)
        task = TorchDistributedCheckpointTask(
            state=state,
            checkpoint_id=pointers.checkpoint_id,
            pointers=pointers,
            no_dist=not (dist.is_available() and dist.is_initialized()),
            async_mode=async_mode,
            reliable=reliable,
        )

        if async_mode != "disabled":
            self._enqueue_async_task(task)
            return

        try:
            self._save_task(task)
        except Exception as exc:
            if isinstance(exc, _PointerUpdateError):
                self._on_pointer_update_failed(task, exc)
            else:
                self._on_checkpoint_task_failed(task, exc)
            self.raise_checkpoint_error_if_requested()

    def save_model_checkpoint(self, name: str = "model") -> None:
        if not self.runner.config.get("ckpt.enabled", True):
            return

        async_mode = self.checkpoint_async_mode()
        normalized_name = self._normalize_name(name)
        target_name = self._build_target_name(normalized_name)
        pointers = _PendingPointers(
            checkpoint_id=os.path.join(self.runner.workspace.checkpoint_dir, target_name),
            target_name=target_name,
            aliases=(normalized_name,),
            update_best=False,
            track_for_retention=False,
        )
        task = TorchDistributedCheckpointTask(
            state=self.build_model_checkpoint_payload(),
            checkpoint_id=pointers.checkpoint_id,
            pointers=pointers,
            no_dist=not (dist.is_available() and dist.is_initialized()),
            async_mode=async_mode,
            reliable=True,
            write_runner_config=False,
        )

        if async_mode != "disabled":
            self._enqueue_async_task(task)
            return

        try:
            self._save_task(task)
        except Exception as exc:
            if isinstance(exc, _PointerUpdateError):
                self._on_pointer_update_failed(task, exc)
            else:
                self._on_checkpoint_task_failed(task, exc)
            self.raise_checkpoint_error_if_requested()

    def load_checkpoint(self, checkpoint: bytes | str | os.PathLike) -> dict[str, Any]:
        checkpoint_id = self._resolve_checkpoint_id(checkpoint)
        state = self.runner.state_dict()
        no_dist = not (dist.is_available() and dist.is_initialized())
        dcp.load(state, checkpoint_id=checkpoint_id, no_dist=no_dist)
        return dict(state)

    def load_model_checkpoint(self, checkpoint: bytes | str | os.PathLike) -> dict[str, Any]:
        checkpoint_id = self._resolve_checkpoint_id(checkpoint)
        state = dict(self.build_model_checkpoint_payload())
        no_dist = not (dist.is_available() and dist.is_initialized())
        dcp.load(state, checkpoint_id=checkpoint_id, no_dist=no_dist)
        return dict(state)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else monotonic() + max(float(timeout), 0.0)
        while True:
            with self._lock:
                inflight = self._inflight
                staging_future = self._staging_future
                pending_latest = self._pending_latest
                pending_reliable = bool(self._pending_reliable)
            if inflight is None and staging_future is None and pending_latest is None and not pending_reliable:
                self.raise_checkpoint_error_if_requested()
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
                    # Exceptions are surfaced by `_on_async_task_done`.
                    continue
                sleep(0)
                continue

            if staging_future is not None:
                try:
                    staging_future.result(timeout=remaining)
                except FutureTimeoutError:
                    return False
                except Exception as exc:
                    self.record_checkpoint_failure(exc)
                    self.raise_checkpoint_error_if_requested()
                with self._lock:
                    if self._staging_future is staging_future:
                        self._staging_future = None
                sleep(0)
                continue

            sleep(min(max(remaining, 0.0), 0.01) if remaining is not None else 0.01)

    def close(self, timeout: float | None = None) -> bool:
        drained = False
        staged = False
        close_error: Exception | None = None
        deadline = None if timeout is None else monotonic() + max(float(timeout), 0.0)

        def remaining_timeout() -> float | None:
            if deadline is None:
                return None
            return max(deadline - monotonic(), 0.0)

        try:
            drained = self.wait(timeout=remaining_timeout())
            if drained:
                staged = self.maybe_wait_for_staging(timeout=remaining_timeout())
        except Exception as exc:
            close_error = exc
        if close_error is None and not (drained and staged):
            return False

        stager = self._stager
        self._stager = None
        async_process_group = self._async_process_group
        owns_async_process_group = self._owns_async_process_group
        self._async_process_group = None
        self._owns_async_process_group = False
        self._clear_drained_state()
        if stager is not None:
            stager.close()
        if owns_async_process_group and async_process_group is not None:
            try:
                dist.destroy_process_group(group=async_process_group)
            except Exception as exc:
                warn(f"failed to destroy async checkpoint process group: {exc}", RuntimeWarning, stacklevel=2)

        self._close_purge_worker()
        if close_error is not None:
            raise close_error
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
            self.record_checkpoint_failure(exc)
            self.raise_checkpoint_error_if_requested()

        with self._lock:
            if self._staging_future is staging_future:
                self._staging_future = None
        self.raise_checkpoint_error_if_requested()
        return True

    @staticmethod
    def _normalize_name(name: str) -> str:
        if name.endswith(".pth"):
            return name[:-4]
        return name

    def _clear_drained_state(self) -> None:
        with self._lock:
            self._closing = True
            self._inflight = None
            self._pending_latest = None
            self._pending_reliable.clear()
            self._retention_history.clear()
            self._staging_future = None

    def _pointer_path(self, pointer: str) -> str:
        return self._pointer_store.path(pointer)

    def _write_pointer(self, pointer: str, target_name: str) -> None:
        self._pointer_store.write(pointer, target_name)

    def read_pointer(self, pointer: str) -> str | None:
        return self._pointer_store.read(pointer)

    def is_target_published(self, target_name: str) -> bool:
        return self._pointer_store.is_target_published(target_name)

    def purge_unpublished_checkpoint(self, task: TorchDistributedCheckpointTask) -> None:
        self.purge_unpublished_target(task.checkpoint_id, task.pointers.target_name)

    def purge_unpublished_target(self, checkpoint_id: str, target_name: str) -> None:
        if not self._is_io_rank():
            return
        if self.is_target_published(target_name):
            return
        self.purge_path(checkpoint_id)

    def _remove_stale_history_pointers(self, target_name: str) -> None:
        self._pointer_store.remove_stale_history_pointers(target_name)

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
                candidate = os.path.join(self.runner.workspace.checkpoint_dir, normalized)
                if os.path.isdir(candidate) and os.path.exists(self._metadata_path(candidate)):
                    return candidate
                pointed = self.read_pointer(normalized)
                if pointed is not None:
                    pointer_candidate = os.path.join(self.runner.workspace.checkpoint_dir, pointed)
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

    def _is_async_task_reliable(self, task: TorchDistributedCheckpointTask) -> bool:
        return task.reliable

    def _on_async_task_succeeded(self, task: TorchDistributedCheckpointTask) -> None:
        if task.write_runner_config:
            try:
                self._write_runner_config(task.checkpoint_id)
            except Exception as exc:
                self._on_checkpoint_task_failed(task, exc)
                return
        if not self._is_io_rank():
            return
        try:
            self.apply_pointer_updates(task.pointers)
        except Exception as exc:
            self._on_pointer_update_failed(task, exc)

    def _on_checkpoint_task_failed(self, task: TorchDistributedCheckpointTask, exc: Exception) -> None:
        self.record_checkpoint_failure(exc, target=task.pointers.target_name)
        self.purge_unpublished_checkpoint(task)

    def _on_pointer_update_failed(self, task: TorchDistributedCheckpointTask, exc: Exception) -> None:
        failed_alias = None
        published_aliases: tuple[str, ...] = ()
        failure = exc
        if isinstance(exc, _PointerUpdateError):
            failed_alias = exc.alias
            published_aliases = exc.published_aliases
            failure = exc.cause
        if published_aliases:
            self._record_published_pointer_target(task.pointers, published_aliases, emit=False)
        self.record_checkpoint_failure(failure, target=task.pointers.target_name, alias=failed_alias)
        self.purge_unpublished_checkpoint(task)

    def _on_async_task_failed(self, task: TorchDistributedCheckpointTask, exc: Exception) -> None:
        self._on_checkpoint_task_failed(task, exc)

    def _on_async_task_dropped(self, task: TorchDistributedCheckpointTask) -> None:
        self.purge_unpublished_checkpoint(task)

    def _after_async_task_done_locked(self, future: Future, task: TorchDistributedCheckpointTask) -> None:
        del future, task
        if self._staging_future is not None and self._staging_future.done():
            self._staging_future = None

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

        if not self.runner.config.get("ckpt.dedicated_async_process_group", True):
            return None
        if not (dist.is_available() and dist.is_initialized()):
            return None

        backend = self.runner.config.get("ckpt.async_process_group_backend", "gloo")
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

    def _build_checkpoint_pointers(
        self,
        *,
        name: str,
        epochs: int,
        save_best: bool,
    ) -> tuple[_PendingPointers, bool]:
        normalized_name = self._normalize_name(name)
        history_alias = self.resolve_history_name(epochs)
        if history_alias is not None:
            history_alias = self._normalize_name(history_alias)

        target_base = history_alias or normalized_name
        target_name = self._build_target_name(target_base)
        checkpoint_id = os.path.join(self.runner.workspace.checkpoint_dir, target_name)
        should_update_best = save_best and self.runner.is_best
        reliable = history_alias is not None or should_update_best or normalized_name != "latest"

        aliases = ["latest"]
        if history_alias is not None:
            aliases.append(history_alias)
        if normalized_name == "best":
            aliases.append("best")
        elif normalized_name != "latest":
            aliases.append(normalized_name)

        pointers = _PendingPointers(
            checkpoint_id=checkpoint_id,
            target_name=target_name,
            aliases=tuple(dict.fromkeys(aliases)),
            update_best=should_update_best,
            track_for_retention=history_alias is not None or normalized_name == "latest",
        )
        return pointers, reliable

    def _build_target_name(self, base_name: str) -> str:
        with self._lock:
            self._save_sequence += 1
            save_sequence = self._save_sequence
        global_step = self.runner.train_state.global_step
        return f"{base_name}-g{global_step:012d}-q{save_sequence:06d}"

    def _save_task(self, task: TorchDistributedCheckpointTask, *, apply_pointers: bool = True) -> None:
        dcp.save(task.state, checkpoint_id=task.checkpoint_id, no_dist=task.no_dist)
        if task.write_runner_config:
            try:
                self._write_runner_config(task.checkpoint_id)
            except Exception:
                self.purge_unpublished_checkpoint(task)
                raise
        if apply_pointers and self._is_io_rank():
            self.apply_pointer_updates(task.pointers)

    def _write_runner_config(self, checkpoint_id: str) -> None:
        if not self._is_io_rank():
            return
        if not os.path.isdir(checkpoint_id):
            return
        config = self.runner.config
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        config.yaml(os.path.join(checkpoint_id, "runner.yaml"))

    def _start_async_task_locked(self, task: TorchDistributedCheckpointTask) -> Future | None:
        save_kwargs: dict[str, Any] = {
            "checkpoint_id": task.checkpoint_id,
        }
        if not task.no_dist:
            process_group = self._ensure_async_process_group()
            if process_group is None and self.runner.config.get("ckpt.dedicated_async_process_group", True):
                raise RuntimeError(
                    "async DCP checkpoints require a dedicated async checkpoint process group; "
                    "set ckpt.dedicated_async_process_group=False to explicitly use the default process group "
                    "or use ckpt.async_mode='disabled'"
                )
            if process_group is not None:
                save_kwargs["process_group"] = process_group
        if task.async_mode == "async_with_pinned_mem" and self._pinned_memory_async_staging:
            save_kwargs["async_checkpointer_type"] = AsyncCheckpointerType.PROCESS
            if self._stager is None:
                self._stager = DefaultStager(StagingOptions(True, True, True, True))
            save_kwargs["async_stager"] = self._stager
        response = dcp.async_save(dict(task.state), **save_kwargs)
        future = self._as_future(response)
        if future is None:
            raise RuntimeError("dcp.async_save did not return a Future-compatible handle")
        self._staging_future = self._as_staging_future(response) if task.async_mode == "async_with_pinned_mem" else None
        return future

    def apply_pointer_updates(self, pointers: _PendingPointers) -> None:
        published_aliases: list[str] = []
        for alias in pointers.aliases:
            try:
                self._write_pointer(alias, pointers.target_name)
            except Exception as exc:
                raise _PointerUpdateError(
                    alias,
                    pointers.target_name,
                    tuple(published_aliases),
                    exc,
                ) from exc
            published_aliases.append(alias)

        if pointers.update_best and "best" not in published_aliases:
            try:
                self._write_pointer("best", pointers.target_name)
            except Exception as exc:
                raise _PointerUpdateError(
                    "best",
                    pointers.target_name,
                    tuple(published_aliases),
                    exc,
                ) from exc
            published_aliases.append("best")

        self._record_published_pointer_target(pointers, tuple(published_aliases))

    def _record_published_pointer_target(
        self,
        pointers: _PendingPointers,
        aliases: tuple[str, ...],
        *,
        emit: bool = True,
    ) -> None:
        if pointers.track_for_retention:
            self._record_retained_target(pointers.target_name)
        self.record_checkpoint_success(target=pointers.target_name, aliases=aliases, emit=emit)

    def _record_retained_target(self, target_name: str) -> None:
        to_delete = self._record_retention_entry(
            self._retention_history,
            target_name,
            protected_entries=(self.read_pointer("latest"), self.read_pointer("best")),
        )
        for stale_target in to_delete:
            self._remove_stale_history_pointers(stale_target)
            self._enqueue_purge_target(stale_target)

    def _enqueue_purge_target(self, target_name: str) -> None:
        path = os.path.join(self.runner.workspace.checkpoint_dir, target_name)
        self.enqueue_purge_path(path)
