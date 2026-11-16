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

import errno
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest
import torch

from danling.runners.checkpoints.file import FileCheckpointManager


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
        self.config = {"ckpt.async_mode": "disabled", "ckpt.interval": None}
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
        interval = self.config.get("ckpt.interval")
        return int(interval) if interval is not None else -1

    @staticmethod
    def save(obj, file):
        torch.save(obj, file)
        return file


class _FailingSaveRunner(_CheckpointRunner):
    @staticmethod
    def save(obj, file):
        del obj, file
        raise OSError(errno.ENOSPC, "disk full")


class BlockingRunner(_CheckpointRunner):
    def __init__(self, checkpoint_dir: Path) -> None:
        super().__init__(checkpoint_dir)
        self.config["ckpt.async_mode"] = "async"
        self.started = Event()
        self.release = Event()
        self.saved_payloads = []

    def save(self, obj, file):
        self.started.set()
        self.release.wait()
        result = super().save(obj, file)
        self.saved_payloads.append(obj)
        return result


def _retaining_file_manager(tmp_path: Path) -> tuple[_CheckpointRunner, FileCheckpointManager]:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"ckpt.interval": 2, "ckpt.keep_latest_k": 1})
    return runner, FileCheckpointManager(runner)


def _checkpoint_step(path: Path) -> int:
    return int(torch.load(path, map_location="cpu")["state"]["train"]["global_step"])


def _save_best_file_checkpoint(tmp_path: Path) -> FileCheckpointManager:
    runner, manager = _retaining_file_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    return manager


def _assert_checkpoint_failure(manager: FileCheckpointManager, exc_type: type[BaseException], target: str) -> None:
    health = manager.checkpoint_health
    assert isinstance(health.first_error, exc_type)
    assert isinstance(health.last_error, exc_type)
    assert health.error_count == 1
    assert health.last_failed_target == target


@pytest.fixture
def failing_file_checkpoint_manager(tmp_path: Path) -> FileCheckpointManager:
    return FileCheckpointManager(_FailingSaveRunner(tmp_path))


def test_model_export_is_separate(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"ckpt.export_dtype": "fp16"})
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(last_step=True)
    manager.save_model_checkpoint()
    assert manager.close(timeout=1.0) is True

    latest_payload = torch.load(tmp_path / "latest.pth", map_location="cpu", weights_only=False)
    assert "optimizer" in latest_payload
    assert latest_payload["model"]["weight"].dtype == torch.float32

    model_payload = torch.load(tmp_path / "model.pth", map_location="cpu", weights_only=False)
    assert list(model_payload) == ["model"]
    assert model_payload["model"]["weight"].dtype == torch.float16


def test_history_retention(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"ckpt.interval": 1, "ckpt.keep_latest_k": 2})
    manager = FileCheckpointManager(runner)

    for epoch in range(3):
        runner.train_state.epoch = epoch
        manager.save_checkpoint(epochs=epoch)

    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "ckpt-e000001.pth").exists() is False
    assert (tmp_path / "ckpt-e000002.pth").exists() is True
    assert (tmp_path / "ckpt-e000003.pth").exists() is True


def test_waits_for_interval(tmp_path: Path) -> None:
    runner, manager = _retaining_file_manager(tmp_path)
    runner.train_state.global_step = 10

    manager.save_checkpoint(epochs=0)

    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "latest.pth").exists() is False


def test_published_aliases(tmp_path: Path) -> None:
    manager = _save_best_file_checkpoint(tmp_path)
    assert manager.close(timeout=1.0)
    for name in ("latest.pth", "best.pth", "ckpt-e000002.pth"):
        assert _checkpoint_step(tmp_path / name) == 20


def test_partial_alias_failure(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"ckpt.interval": 1})
    runner.is_best = True
    manager = FileCheckpointManager(runner)
    best_dir = tmp_path / "best.pth"
    best_dir.mkdir()
    (best_dir / "latest.pth").mkdir()

    with pytest.warns(RuntimeWarning, match="checkpoint failed"):
        manager.save_checkpoint(epochs=0)

    health = manager.checkpoint_health
    assert health.error_count == 1
    assert health.last_failed_alias == "best.pth"
    assert health.last_successful_target == "ckpt-e000001.pth"
    assert health.last_successful_aliases == ("latest.pth", "ckpt-e000001.pth")
    assert (tmp_path / "latest.pth").is_file()
    assert (tmp_path / "ckpt-e000001.pth").is_file()
    assert manager.close(timeout=1.0) is True


def test_forced_latest_preserves_best(tmp_path: Path) -> None:
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


def test_retention_prunes_old_history(tmp_path: Path) -> None:
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


def test_skips_periodic_writes_without_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    assert manager.close(timeout=1.0) is True

    assert (tmp_path / "latest.pth").exists() is False


def test_interval_controls_periodic_persist(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.config.update({"ckpt.interval": 1})
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    assert (tmp_path / "latest.pth").exists() is True
    assert (tmp_path / "ckpt-e000001.pth").exists() is True

    manager.save_checkpoint(epochs=1)
    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "ckpt-e000002.pth").exists() is True


def test_force_checkpoint_bypasses_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(epochs=0, force=True)
    assert manager.close(timeout=1.0) is True

    assert (tmp_path / "latest.pth").exists() is True


@pytest.mark.parametrize("asynchronous", [False, True])
def test_save_failure(tmp_path: Path, failing_file_checkpoint_manager, asynchronous):
    manager = failing_file_checkpoint_manager
    manager.runner.config["ckpt.async_mode"] = "async" if asynchronous else "disabled"
    try:
        with pytest.warns(RuntimeWarning, match="storage failure: .*ENOSPC"):
            manager.save_checkpoint(force=True)
            assert manager.wait(timeout=1.0)
        _assert_checkpoint_failure(manager, OSError, "latest")
        assert manager.checkpoint_health.last_successful_target is None
        assert not (tmp_path / "latest.pth").exists()
    finally:
        manager.close(timeout=1.0)


def test_save_success_reports_event(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runner = _CheckpointRunner(tmp_path)
    runner.train_state.epoch = 2
    runner.train_state.global_step = 5
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(name="latest", force=True)

    output = capsys.readouterr().out
    assert "checkpoint saved:" in output
    assert "step=5" in output
    assert "epoch=2" in output
    assert "target=latest.pth" in output
    assert "aliases=latest.pth" in output


def test_close_timeout_retry(tmp_path: Path) -> None:
    runner = BlockingRunner(tmp_path)
    manager = FileCheckpointManager(runner)
    try:
        manager.save_checkpoint(force=True)
        assert runner.started.wait(timeout=5.0)
        assert manager.close(timeout=0.0) is False
        assert not (tmp_path / "latest.pth").exists()
        runner.release.set()
        assert manager.close(timeout=1.0) is True
        assert _checkpoint_step(tmp_path / "latest.pth") == 0
    finally:
        runner.release.set()
        manager.close(timeout=1.0)


def test_interpreter_shutdown(tmp_path: Path, run_isolated):
    run_isolated(f"""
        import atexit
        import json
        from pathlib import Path
        from danling.runners.checkpoints.file import FileCheckpointManager
        from tests.runners.checkpoints.test_file import _CheckpointRunner

        directory = Path({str(tmp_path)!r})
        runner = _CheckpointRunner(directory)
        runner.config["ckpt.async_mode"] = "async"
        manager = FileCheckpointManager(runner)
        manager.save_checkpoint(force=True)
        assert manager.wait(timeout=5.0)
        runner.train_state.global_step = 1

        def save_at_exit():
            try:
                manager.save_checkpoint(force=True)
                status = {{"closed": manager.close(timeout=1.0)}}
            except Exception as error:
                status = {{"error": repr(error)}}
            (directory / "shutdown.json").write_text(json.dumps(status))

        atexit.register(save_at_exit)
    """)
    assert json.loads((tmp_path / "shutdown.json").read_text()) == {"closed": True}
    assert _checkpoint_step(tmp_path / "latest.pth") == 0


def test_async_latest_wins(tmp_path: Path) -> None:
    runner = BlockingRunner(tmp_path)
    manager = FileCheckpointManager(runner)
    try:
        runner.train_state.global_step = 1
        manager.save_checkpoint(force=True)
        assert runner.started.wait(timeout=5.0)
        for step in (2, 3):
            runner.train_state.global_step = step
            manager.save_checkpoint(force=True)
        runner.release.set()
        assert manager.close(timeout=1.0) is True
        assert [payload["state"]["train"]["global_step"] for payload in runner.saved_payloads] == [1, 3]
        assert _checkpoint_step(tmp_path / "latest.pth") == 3
    finally:
        runner.release.set()
        manager.close(timeout=1.0)


def test_async_reliable_exports(tmp_path: Path) -> None:
    runner = BlockingRunner(tmp_path)
    runner.config["ckpt.export_dtype"] = "fp16"
    manager = FileCheckpointManager(runner)
    try:
        runner.train_state.global_step = 1
        manager.save_checkpoint(force=True)
        assert runner.started.wait(timeout=5.0)
        runner.train_state.global_step = 2
        manager.save_checkpoint(last_step=True)
        runner.train_state.global_step = 3
        manager.save_model_checkpoint()
        runner.release.set()
        assert manager.close(timeout=1.0) is True
        assert _checkpoint_step(tmp_path / "latest.pth") == 2
        model_payload = torch.load(tmp_path / "model.pth", map_location="cpu", weights_only=False)
        assert list(model_payload) == ["model"]
        assert model_payload["model"]["weight"].dtype == torch.float16
    finally:
        runner.release.set()
        manager.close(timeout=1.0)


def test_async_state_snapshot(tmp_path: Path) -> None:
    class SnapshotRunner(BlockingRunner):
        def __init__(self, directory):
            super().__init__(directory)
            self.weight = torch.ones(2, 2)

        def state_dict(self):
            return {"model": {"weight": self.weight}}

    runner = SnapshotRunner(tmp_path)
    manager = FileCheckpointManager(runner)
    try:
        manager.save_checkpoint(force=True)
        assert runner.started.wait(timeout=5.0)
        runner.weight.fill_(9.0)
        runner.release.set()
        assert manager.close(timeout=1.0) is True
        payload = torch.load(tmp_path / "latest.pth", map_location="cpu", weights_only=False)
        torch.testing.assert_close(payload["model"]["weight"], torch.ones(2, 2))
    finally:
        runner.release.set()
        manager.close(timeout=1.0)
