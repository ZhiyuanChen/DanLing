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

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from danling.runners.checkpoints.file import CheckpointTask, FileCheckpointManager


@dataclass
class _TrainState:
    epoch: int = 0
    global_step: int = 0


@dataclass
class _CheckpointWorkspace:
    id: str
    checkpoint_dir: str


class _CheckpointRunner:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.train_state = _TrainState()
        self.config = {"checkpoint.async_mode": "disabled", "checkpoint.interval": None}
        self.id = "test-runner"
        self.workspace = _CheckpointWorkspace(id=self.id, checkpoint_dir=str(checkpoint_dir))
        self.is_best = False
        self.is_step_mode = False

    def state_dict(self):
        return {
            "model": {"weight": torch.ones(2, 2, dtype=torch.float32)},
            "optimizer": {"param_groups": [1]},
            "state": {"train": {"global_step": self.train_state.global_step}},
        }

    @property
    def checkpoint_interval(self) -> int:
        interval = self.config.get("checkpoint.interval")
        return int(interval) if interval is not None else -1

    @staticmethod
    def save(obj, file):
        torch.save(obj, file)
        return file


class _FailingSaveRunner(_CheckpointRunner):
    @staticmethod
    def save(obj, file):
        del obj, file
        raise OSError("disk full")


class _TimeoutFileCheckpointManager(FileCheckpointManager):
    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        return False


class _CoalescingFileCheckpointManager(FileCheckpointManager):
    def __init__(self, runner) -> None:
        super().__init__(runner)
        self.started_steps: list[int] = []
        self.first_future: Future = Future()

    def _start_async_task_locked(self, task: CheckpointTask):
        self.started_steps.append(int(task.payload["state"]["train"]["global_step"]))
        if len(self.started_steps) == 1:
            return self.first_future
        future: Future = Future()
        future.set_result(None)
        return future


class _CapturingFileCheckpointManager(FileCheckpointManager):
    def __init__(self, runner) -> None:
        super().__init__(runner)
        self.task: CheckpointTask | None = None
        self.future: Future = Future()

    def _start_async_task_locked(self, task: CheckpointTask):
        self.task = task
        return self.future


def _retaining_file_manager(tmp_path: Path) -> tuple[_CheckpointRunner, FileCheckpointManager]:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"checkpoint.interval": 2, "checkpoint.keep_latest_k": 1})
    return runner, FileCheckpointManager(runner)


def _checkpoint_step(path: Path) -> int:
    return int(torch.load(path, map_location="cpu")["state"]["train"]["global_step"])


def _save_best_file_checkpoint(tmp_path: Path) -> FileCheckpointManager:
    runner, manager = _retaining_file_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    return manager


def test_file_last_step_model_only_export_with_dtype(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"checkpoint.last_save_model_only": True, "checkpoint.export_dtype": "fp16"})
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(last_step=True)
    assert manager.close(timeout=1.0) is True

    payload = torch.load(tmp_path / "latest.pth", map_location="cpu")
    assert list(payload) == ["model"]
    assert payload["model"]["weight"].dtype == torch.float16


def test_file_keep_latest_k_prunes_old_history_checkpoints(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"checkpoint.interval": 1, "checkpoint.keep_latest_k": 2})
    manager = FileCheckpointManager(runner)

    for epoch in range(3):
        runner.train_state.epoch = epoch
        manager.save_checkpoint(epochs=epoch)

    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "ckpt-e000001.pth").exists() is False
    assert (tmp_path / "ckpt-e000002.pth").exists() is True
    assert (tmp_path / "ckpt-e000003.pth").exists() is True


def test_file_checkpoint_waits_for_interval(tmp_path: Path) -> None:
    runner, manager = _retaining_file_manager(tmp_path)
    runner.train_state.global_step = 10

    manager.save_checkpoint(epochs=0)

    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "latest.pth").exists() is False


def test_file_checkpoint_writes_latest(tmp_path: Path) -> None:
    manager = _save_best_file_checkpoint(tmp_path)

    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "latest.pth").exists() is True


def test_file_checkpoint_writes_best(tmp_path: Path) -> None:
    manager = _save_best_file_checkpoint(tmp_path)

    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "best.pth").exists() is True


def test_file_checkpoint_writes_history(tmp_path: Path) -> None:
    manager = _save_best_file_checkpoint(tmp_path)

    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "ckpt-e000002.pth").exists() is True


def test_file_force_checkpoint_updates_latest_without_replacing_best(tmp_path: Path) -> None:
    runner, manager = _retaining_file_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)

    runner.is_best = False
    runner.train_state.global_step = 30
    manager.save_checkpoint(epochs=2, force=True)

    assert manager.close(timeout=1.0) is True
    assert _checkpoint_step(tmp_path / "latest.pth") == 30
    assert _checkpoint_step(tmp_path / "best.pth") == 20
    assert (tmp_path / "ckpt-e000002.pth").exists() is True


def test_file_checkpoint_retention_prunes_old_history(tmp_path: Path) -> None:
    runner, manager = _retaining_file_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    runner.is_best = False
    runner.train_state.global_step = 30
    manager.save_checkpoint(epochs=2, force=True)

    runner.train_state.global_step = 40
    manager.save_checkpoint(epochs=3)

    assert manager.close(timeout=1.0) is True
    assert _checkpoint_step(tmp_path / "latest.pth") == 40
    assert _checkpoint_step(tmp_path / "best.pth") == 20
    assert (tmp_path / "ckpt-e000002.pth").exists() is False
    assert (tmp_path / "ckpt-e000004.pth").exists() is True


def test_file_checkpoint_skips_periodic_writes_without_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    assert manager.close(timeout=1.0) is True

    assert (tmp_path / "latest.pth").exists() is False


def test_file_checkpoint_interval_controls_periodic_persist(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"checkpoint.interval": 1})
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    assert (tmp_path / "latest.pth").exists() is True
    assert (tmp_path / "ckpt-e000001.pth").exists() is True

    manager.save_checkpoint(epochs=1)
    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "ckpt-e000002.pth").exists() is True


def test_file_force_checkpoint_bypasses_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(epochs=0, force=True)
    assert manager.close(timeout=1.0) is True

    assert (tmp_path / "latest.pth").exists() is True


def test_file_checkpoint_sync_save_failure_raises(tmp_path: Path) -> None:
    runner = _FailingSaveRunner(tmp_path)
    manager = FileCheckpointManager(runner)

    with pytest.raises(OSError, match="disk full"):
        manager.save_checkpoint(force=True)

    assert (tmp_path / "latest.pth").exists() is False


def test_file_checkpoint_async_save_failure_surfaces_on_wait(tmp_path: Path) -> None:
    runner = _FailingSaveRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "async"})
    manager = FileCheckpointManager(runner)

    try:
        manager.save_checkpoint(force=True)
        with pytest.raises(OSError, match="disk full"):
            manager.wait(timeout=1.0)
    finally:
        manager.close(timeout=1.0)

    assert (tmp_path / "latest.pth").exists() is False


def test_file_close_cleans_up_when_wait_times_out(tmp_path: Path) -> None:
    class _RecordingExecutor:
        def __init__(self) -> None:
            self.calls = []

        def shutdown(self, wait, cancel_futures=False):
            self.calls.append((wait, cancel_futures))

    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"checkpoint.keep_latest_k": 1})
    manager = _TimeoutFileCheckpointManager(runner)

    executor = _RecordingExecutor()
    manager._executor = executor  # type: ignore[assignment]
    manager._inflight = object()  # type: ignore[assignment]
    manager._pending_latest = object()  # type: ignore[assignment]
    manager._pending_reliable.append(object())  # type: ignore[arg-type]
    manager._history_names_inflight_or_pending.add("ckpt-e000000.pth")
    manager._retention_history.append("ckpt-e000000.pth")

    purge_thread = manager._purge_thread
    assert purge_thread is not None
    assert purge_thread.is_alive()

    assert manager.close(timeout=0.0) is False

    assert executor.calls
    wait, _ = executor.calls[0]
    assert wait is False
    assert manager._executor is None
    assert manager._inflight is None
    assert manager._pending_latest is None
    assert len(manager._pending_reliable) == 0
    assert len(manager._history_names_inflight_or_pending) == 0
    assert len(manager._retention_history) == 0
    assert manager._purge_thread is None
    assert manager._purge_queue is None
    assert purge_thread.is_alive() is False


def test_file_callback_ignores_submit_after_shutdown(tmp_path: Path) -> None:
    class _ShutdownExecutor:
        @staticmethod
        def submit(*args, **kwargs):
            raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "async"})
    manager = FileCheckpointManager(runner)

    completed = Future()
    completed.set_result(None)
    manager._inflight = completed
    manager._executor = _ShutdownExecutor()  # type: ignore[assignment]
    manager._pending_reliable.append(
        CheckpointTask(
            payload=runner.state_dict(), name="latest", history_name="ckpt-e000000.pth", should_update_best=False
        )
    )

    manager._on_async_task_done(
        completed,
        CheckpointTask(payload=runner.state_dict(), name="latest", history_name=None, should_update_best=False),
    )

    assert manager._closing is True
    assert manager._inflight is None
    assert manager._pending_latest is None
    assert len(manager._pending_reliable) == 0
    assert len(manager._history_names_inflight_or_pending) == 0


def test_file_async_latest_wins_coalesces_pending_saves(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "async"})
    manager = _CoalescingFileCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(force=True)
    runner.train_state.global_step = 2
    manager.save_checkpoint(force=True)
    runner.train_state.global_step = 3
    manager.save_checkpoint(force=True)

    assert manager._pending_latest is not None
    assert manager._pending_latest.payload["state"]["train"]["global_step"] == 3
    manager.first_future.set_result(None)

    assert manager.started_steps == [1, 3]
    assert manager.wait(timeout=1.0) is True
    assert manager.close(timeout=1.0) is True


def test_file_async_checkpoint_snapshots_tensor_payload(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "async"})
    weight = torch.ones(2, 2)
    runner.state_dict = lambda: {"model": {"weight": weight}}  # type: ignore[method-assign]
    manager = _CapturingFileCheckpointManager(runner)

    try:
        manager.save_checkpoint(force=True)
        weight.fill_(9.0)

        assert manager.task is not None
        torch.testing.assert_close(manager.task.payload["model"]["weight"], torch.ones(2, 2))
    finally:
        manager.future.set_result(None)
        manager.close(timeout=1.0)
