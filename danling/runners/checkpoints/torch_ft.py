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
from dataclasses import dataclass
from typing import Any
from warnings import warn

from torch import distributed as dist

from .torch_distributed import TorchDistributedCheckpointManager, TorchDistributedCheckpointTask, dcp


@dataclass(frozen=True)
class FTDataLoaderCheckpointTask:
    state: Mapping[str, Any]
    checkpoint_name: str
    checkpoint_id: str
    no_dist: bool
    track_for_retention: bool
    process_group: Any | None = None


class FTDataLoaderCheckpointer:
    """DCP persistence helper for per-replica stateful dataloaders."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    def checkpoint_id(self, checkpoint_name: str) -> str:
        return os.path.join(self.root_dir(), checkpoint_name)

    def root_dir(self) -> str:
        prefix = str(self.runner.config.get("checkpoint.ft_dataloader_checkpoint_prefix", "ft-replica"))
        return os.path.join(self.runner.workspace.checkpoint_dir, f"{prefix}-{self.replica_id()}")

    def replica_id(self) -> str:
        configured = self.runner.config.get("checkpoint.ft_replica_id")
        if configured is not None:
            return str(configured)

        env_replica_id = os.getenv("FT_REPLICA_ID")
        if env_replica_id:
            return env_replica_id

        ft_manager = getattr(self.runner, "ft", None)
        if ft_manager is not None and hasattr(ft_manager, "replica_id"):
            return str(ft_manager.replica_id)
        return str(getattr(self.runner, "rank", 0))

    def build_task(
        self,
        *,
        checkpoint_name: str,
        track_for_retention: bool,
    ) -> FTDataLoaderCheckpointTask | None:
        state = self.state_dict()
        if not state:
            return None
        return FTDataLoaderCheckpointTask(
            state=state,
            checkpoint_name=checkpoint_name,
            checkpoint_id=self.checkpoint_id(checkpoint_name),
            no_dist=not (dist.is_available() and dist.is_initialized()),
            track_for_retention=track_for_retention,
        )

    def state_dict(self) -> Mapping[str, Any]:
        dataloaders = getattr(self.runner, "dataloaders", None)
        state_dict_fn = getattr(dataloaders, "state_dict", None)
        if not callable(state_dict_fn):
            return {}
        state = state_dict_fn()
        if state is None:
            return {}
        if not isinstance(state, Mapping):
            raise ValueError(f"dataloaders.state_dict() must return a mapping, got {type(state).__name__}")
        return state

    def save(self, task: FTDataLoaderCheckpointTask) -> None:
        os.makedirs(self.root_dir(), exist_ok=True)
        save_kwargs: dict[str, Any] = {
            "checkpoint_id": task.checkpoint_id,
            "no_dist": task.no_dist,
        }
        if not task.no_dist and task.process_group is not None:
            save_kwargs["process_group"] = task.process_group
        dcp.save(task.state, **save_kwargs)

    def load(self, *, checkpoint_id: str | None = None) -> Mapping[str, Any] | None:
        if checkpoint_id is None:
            return None

        checkpoint_name = os.path.basename(os.path.normpath(os.fsdecode(checkpoint_id)))
        ft_checkpoint_id = self.checkpoint_id(checkpoint_name)
        if not os.path.exists(os.path.join(ft_checkpoint_id, ".metadata")):
            return None

        state = dict(self.state_dict())
        if not state:
            return None
        no_dist = not (dist.is_available() and dist.is_initialized())
        dcp.load(state, checkpoint_id=ft_checkpoint_id, no_dist=no_dist)

        dataloaders = getattr(self.runner, "dataloaders", None)
        load_state_dict_fn = getattr(dataloaders, "load_state_dict", None)
        if callable(load_state_dict_fn):
            load_state_dict_fn(state)
        return state


class TorchFTCheckpointManager(TorchDistributedCheckpointManager):
    """Torch DCP checkpoint manager with per-replica dataloader checkpoints for TorchFT."""

    _async_dataloader_tasks: dict[str, FTDataLoaderCheckpointTask]
    _ft_dataloader_retention_history: deque[str]

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        self._ft_dataloader_checkpointer = FTDataLoaderCheckpointer(self.runner)
        self._async_dataloader_tasks = {}
        self._ft_dataloader_retention_history = deque()

    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
        force: bool = False,
    ) -> None:
        epochs = self.runner.train_state.epoch if epochs is None else epochs
        if self.runner.config.get("checkpoint.load_only", False):
            return
        should_persist = self.should_persist_checkpoint(epochs=epochs, last_step=last_step, force=force)
        should_update_best = bool(save_best and self.runner.is_best)
        if not should_persist and not should_update_best:
            return

        async_mode = self.checkpoint_async_mode()
        pointers, reliable = self._build_checkpoint_pointers(name=name, epochs=epochs, save_best=save_best)
        dataloader_task, should_continue = self._prepare_ft_dataloader_task(
            checkpoint_name=pointers.target_name,
            track_for_retention=pointers.track_for_retention,
            async_mode=async_mode,
        )
        if not should_continue:
            return

        if not self._should_save_full_checkpoint():
            if dataloader_task is not None:
                if self._save_ft_checkpoint_task(dataloader_task):
                    self._record_ft_checkpoint_success(dataloader_task)
                self.raise_checkpoint_error_if_requested()
            return

        task = TorchDistributedCheckpointTask(
            state=self.build_checkpoint_payload(last_step=last_step),
            checkpoint_id=pointers.checkpoint_id,
            pointers=pointers,
            no_dist=not (dist.is_available() and dist.is_initialized()),
            async_mode=async_mode,
            reliable=reliable,
        )

        if async_mode != "disabled":
            if dataloader_task is not None:
                with self._lock:
                    self._async_dataloader_tasks[pointers.target_name] = dataloader_task
            self._enqueue_async_task(task)
            return

        try:
            self._save_task(task, apply_pointers=False)
        except Exception as exc:
            self._on_checkpoint_task_failed(task, exc)
            self.raise_checkpoint_error_if_requested()
            return

        if dataloader_task is not None and not self._save_ft_checkpoint_task(dataloader_task):
            self.purge_unpublished_checkpoint(task)
            self.raise_checkpoint_error_if_requested()
            return

        if self._is_io_rank():
            try:
                self.apply_pointer_updates(task.pointers)
            except Exception as exc:
                self._on_ft_pointer_update_failed(task, dataloader_task, exc)
                self.raise_checkpoint_error_if_requested()
                return

        if dataloader_task is not None:
            self._record_ft_checkpoint_success(dataloader_task)

    def load_checkpoint(self, checkpoint: bytes | str | os.PathLike) -> dict[str, Any]:
        checkpoint_id = self._resolve_checkpoint_id(checkpoint)
        state = super().load_checkpoint(checkpoint_id)
        dataloader_state = self._ft_dataloader_checkpointer.load(checkpoint_id=checkpoint_id)
        if dataloader_state is not None:
            state["dataloaders"] = dataloader_state
        return state

    def _should_save_full_checkpoint(self) -> bool:
        ft_manager = getattr(self.runner, "ft", None)
        if ft_manager is None or not getattr(ft_manager, "enabled", False):
            return True
        return ft_manager.participating_rank() == 0

    def _prepare_ft_dataloader_task(
        self,
        *,
        checkpoint_name: str,
        track_for_retention: bool,
        async_mode: str,
    ) -> tuple[FTDataLoaderCheckpointTask | None, bool]:
        task = self._ft_dataloader_checkpointer.build_task(
            checkpoint_name=checkpoint_name,
            track_for_retention=track_for_retention,
        )
        if task is None:
            return None, True
        if async_mode == "disabled" or task.no_dist:
            return task, True

        process_group = self._ensure_async_process_group()
        if process_group is not None:
            return (
                FTDataLoaderCheckpointTask(
                    state=task.state,
                    checkpoint_name=task.checkpoint_name,
                    checkpoint_id=task.checkpoint_id,
                    no_dist=task.no_dist,
                    track_for_retention=task.track_for_retention,
                    process_group=process_group,
                ),
                True,
            )

        exc = RuntimeError(
            "async FT dataloader checkpoints require a dedicated async checkpoint process group; "
            "set checkpoint.dedicated_async_process_group=True or use checkpoint.async_mode='disabled'"
        )
        self.record_checkpoint_failure(exc, target=task.checkpoint_name)
        warn(str(exc), RuntimeWarning, stacklevel=2)
        self.raise_checkpoint_error_if_requested()
        return None, False

    def _save_ft_checkpoint_task(self, task: FTDataLoaderCheckpointTask) -> bool:
        try:
            self._ft_dataloader_checkpointer.save(task)
        except Exception as exc:
            self.record_checkpoint_failure(exc, target=task.checkpoint_name)
            warn(f"ft dataloader checkpoint save failed: {exc}", RuntimeWarning, stacklevel=2)
            self._purge_ft_checkpoint_task(task)
            return False
        return True

    def _on_async_task_succeeded(self, task: TorchDistributedCheckpointTask) -> None:
        dataloader_task = self._pop_async_dataloader_task(task)
        try:
            self._write_runner_config(task.checkpoint_id)
        except Exception as exc:
            self._on_checkpoint_task_failed(task, exc)
            if dataloader_task is not None:
                self._purge_ft_checkpoint_task(dataloader_task)
            return

        if dataloader_task is not None and not self._save_ft_checkpoint_task(dataloader_task):
            self.purge_unpublished_checkpoint(task)
            return

        if self._is_io_rank():
            try:
                self.apply_pointer_updates(task.pointers)
            except Exception as exc:
                self._on_ft_pointer_update_failed(task, dataloader_task, exc)
                return

        if dataloader_task is not None:
            self._record_ft_checkpoint_success(dataloader_task)

    def _on_checkpoint_task_failed(self, task: TorchDistributedCheckpointTask, exc: Exception) -> None:
        dataloader_task = self._pop_async_dataloader_task(task)
        super()._on_checkpoint_task_failed(task, exc)
        if dataloader_task is not None:
            self._purge_ft_checkpoint_task(dataloader_task)

    def _on_async_task_dropped(self, task: TorchDistributedCheckpointTask) -> None:
        dataloader_task = self._pop_async_dataloader_task(task)
        self.purge_unpublished_checkpoint(task)
        if dataloader_task is not None:
            self._purge_ft_checkpoint_task(dataloader_task)

    def _clear_drained_state(self) -> None:
        super()._clear_drained_state()
        with self._lock:
            self._async_dataloader_tasks.clear()
            self._ft_dataloader_retention_history.clear()

    def _pop_async_dataloader_task(self, task: TorchDistributedCheckpointTask) -> FTDataLoaderCheckpointTask | None:
        with self._lock:
            return self._async_dataloader_tasks.pop(task.pointers.target_name, None)

    def _record_ft_checkpoint_success(self, task: FTDataLoaderCheckpointTask) -> None:
        if task.track_for_retention:
            to_delete = self._record_retention_entry(
                self._ft_dataloader_retention_history,
                task.checkpoint_name,
                protected_entries=(self.read_pointer("latest"), self.read_pointer("best"), task.checkpoint_name),
            )
            for stale_target in to_delete:
                self.enqueue_purge_path(self._ft_dataloader_checkpointer.checkpoint_id(stale_target))

    def _purge_ft_checkpoint_task(self, task: FTDataLoaderCheckpointTask) -> None:
        self.purge_path(task.checkpoint_id)

    def _on_ft_pointer_update_failed(
        self,
        task: TorchDistributedCheckpointTask,
        dataloader_task: FTDataLoaderCheckpointTask | None,
        exc: Exception,
    ) -> None:
        self._on_pointer_update_failed(task, exc)
        if dataloader_task is None:
            return
        if self.is_target_published(task.pointers.target_name):
            self._record_ft_checkpoint_success(dataloader_task)
            return
        self._purge_ft_checkpoint_task(dataloader_task)
