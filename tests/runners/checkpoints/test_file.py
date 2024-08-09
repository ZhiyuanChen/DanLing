from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import torch

from danling.runners.checkpoints.file import CheckpointTask, FileCheckpointManager


class _FakeRunner:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.train_state = SimpleNamespace(epoch=0, global_step=0)
        self.config = {"checkpoint.async_mode": "disabled", "checkpoint.interval": None}
        self.id = "test-runner"
        self.checkpoint_dir = str(checkpoint_dir)
        self.is_best = False
        self.is_step_mode = False

    def state_dict(self):
        return {
            "model": {"weight": torch.ones(2, 2, dtype=torch.float32)},
            "optimizer": {"param_groups": [1]},
            "state": {"train": {"global_step": self.train_state.global_step}},
        }

    @staticmethod
    def adapt_checkpoint_payload_for_save(payload):
        return payload

    @staticmethod
    def adapt_checkpoint_payload_for_load(payload):
        return payload

    @property
    def checkpoint_interval(self) -> int:
        interval = self.config.get("checkpoint.interval")
        return int(interval) if interval is not None else -1

    @staticmethod
    def save(obj, file):
        torch.save(obj, file)
        return file


def test_file_last_step_model_only_export_with_dtype(tmp_path: Path) -> None:
    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.last_save_model_only": True, "checkpoint.export_dtype": "fp16"})
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(last_step=True)
    assert manager.close(timeout=1.0) is True

    payload = torch.load(tmp_path / "latest.pth", map_location="cpu")
    assert list(payload) == ["model"]
    assert payload["model"]["weight"].dtype == torch.float16


def test_file_keep_latest_k_prunes_old_archives(tmp_path: Path) -> None:
    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.interval": 1, "checkpoint.keep_latest_k": 2})
    manager = FileCheckpointManager(runner)

    for epoch in range(3):
        runner.train_state.epoch = epoch
        manager.save_checkpoint(epochs=epoch)

    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "ckpt-e000000.pth").exists() is False
    assert (tmp_path / "ckpt-e000001.pth").exists() is True
    assert (tmp_path / "ckpt-e000002.pth").exists() is True


def test_file_checkpoint_skips_periodic_writes_without_interval(tmp_path: Path) -> None:
    runner = _FakeRunner(tmp_path)
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    assert manager.close(timeout=1.0) is True

    assert (tmp_path / "latest.pth").exists() is False


def test_file_checkpoint_interval_controls_periodic_persist(tmp_path: Path) -> None:
    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.interval": 1})
    manager = FileCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    assert (tmp_path / "latest.pth").exists() is True
    assert (tmp_path / "ckpt-e000000.pth").exists() is True

    manager.save_checkpoint(epochs=1)
    assert manager.close(timeout=1.0) is True
    assert (tmp_path / "ckpt-e000001.pth").exists() is True


def test_file_close_cleans_up_when_wait_times_out(tmp_path: Path, monkeypatch) -> None:
    class _FakeExecutor:
        def __init__(self) -> None:
            self.calls = []

        def shutdown(self, wait, cancel_futures=False):
            self.calls.append((wait, cancel_futures))

    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.keep_latest_k": 1})
    manager = FileCheckpointManager(runner)

    fake_executor = _FakeExecutor()
    manager._executor = fake_executor  # type: ignore[assignment]
    manager._inflight = object()  # type: ignore[assignment]
    manager._pending_latest = object()  # type: ignore[assignment]
    manager._pending_reliable.append(object())  # type: ignore[arg-type]
    manager._archive_names_inflight_or_pending.add("ckpt-e000000.pth")
    manager._archive_history.append("ckpt-e000000.pth")

    purge_thread = manager._purge_thread
    assert purge_thread is not None
    assert purge_thread.is_alive()

    monkeypatch.setattr(manager, "wait", lambda timeout=None: False)
    assert manager.close(timeout=0.0) is False

    assert fake_executor.calls
    wait, _ = fake_executor.calls[0]
    assert wait is False
    assert manager._executor is None
    assert manager._inflight is None
    assert manager._pending_latest is None
    assert len(manager._pending_reliable) == 0
    assert len(manager._archive_names_inflight_or_pending) == 0
    assert len(manager._archive_history) == 0
    assert manager._purge_thread is None
    assert manager._purge_queue is None
    assert purge_thread.is_alive() is False


def test_file_callback_ignores_submit_after_shutdown(tmp_path: Path) -> None:
    class _ShutdownExecutor:
        @staticmethod
        def submit(*args, **kwargs):
            raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "async"})
    manager = FileCheckpointManager(runner)

    completed = Future()
    completed.set_result(None)
    manager._inflight = completed
    manager._executor = _ShutdownExecutor()  # type: ignore[assignment]
    manager._pending_reliable.append(
        CheckpointTask(
            payload=runner.state_dict(), name="latest", archive_name="ckpt-e000000.pth", should_update_best=False
        )
    )

    manager._on_checkpoint_payload_done(
        completed,
        CheckpointTask(payload=runner.state_dict(), name="latest", archive_name=None, should_update_best=False),
    )

    assert manager._closing is True
    assert manager._inflight is None
    assert manager._pending_latest is None
    assert len(manager._pending_reliable) == 0
    assert len(manager._archive_names_inflight_or_pending) == 0
