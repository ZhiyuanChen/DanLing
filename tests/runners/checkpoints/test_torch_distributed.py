from __future__ import annotations

import copy
import os
import threading
import warnings
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import danling.runners.checkpoints.torch_distributed as dcp_module
from danling.data import DataLoaderDict
from danling.runners.checkpoints.torch_distributed import TorchDistributedCheckpointManager


class _FakeDCP:
    def __init__(self) -> None:
        self.async_calls: list[dict[str, object]] = []
        self.save_calls: list[dict[str, object]] = []
        self._saved: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _materialize_checkpoint_dir(checkpoint_id: str) -> None:
        os.makedirs(checkpoint_id, exist_ok=True)
        metadata_path = os.path.join(checkpoint_id, ".metadata")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            handle.write("ok\n")

    def async_save(self, state, checkpoint_id: str, no_dist: bool, **kwargs):
        self._materialize_checkpoint_dir(checkpoint_id)
        self._saved[checkpoint_id] = self._snapshot_state(state)
        upload_future: Future = Future()
        staging_future: Future | None = Future() if "async_stager" in kwargs else None
        self.async_calls.append(
            {
                "state": state,
                "checkpoint_id": checkpoint_id,
                "no_dist": no_dist,
                "future": upload_future,
                "staging_future": staging_future,
                "kwargs": kwargs,
            }
        )
        if staging_future is not None:
            return SimpleNamespace(upload_completion=upload_future, staging_completion=staging_future)
        return upload_future

    def save(self, state, checkpoint_id: str, no_dist: bool):
        self._materialize_checkpoint_dir(checkpoint_id)
        self._saved[checkpoint_id] = self._snapshot_state(state)
        self.save_calls.append({"state": state, "checkpoint_id": checkpoint_id, "no_dist": no_dist})

    def load(self, state, checkpoint_id: str, no_dist: bool):
        del no_dist
        saved = self._saved.get(checkpoint_id)
        if saved is None:
            return
        for key, value in state.items():
            if key not in saved:
                continue
            self._restore_value(value, saved[key])

    @classmethod
    def _snapshot_value(cls, value):
        state_dict_fn = getattr(value, "state_dict", None)
        load_state_dict_fn = getattr(value, "load_state_dict", None)
        if callable(state_dict_fn) and callable(load_state_dict_fn):
            return {"kind": "stateful", "value": copy.deepcopy(state_dict_fn())}
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "value": {key: cls._snapshot_value(nested) for key, nested in value.items()},
            }
        return {"kind": "value", "value": copy.deepcopy(value)}

    @classmethod
    def _snapshot_state(cls, state: Mapping[str, object]) -> dict[str, Any]:
        return {key: cls._snapshot_value(value) for key, value in state.items()}

    @classmethod
    def _restore_value(cls, target, snapshot) -> None:
        kind = snapshot["kind"]
        payload = snapshot["value"]
        if kind == "stateful":
            target.load_state_dict(copy.deepcopy(payload))
            return
        if kind == "mapping":
            if not isinstance(target, dict):
                return
            for key, nested_snapshot in payload.items():
                if key in target:
                    cls._restore_value(target[key], nested_snapshot)
                    continue
                target[key] = cls._materialize_snapshot(nested_snapshot)
            return

    @classmethod
    def _materialize_snapshot(cls, snapshot):
        kind = snapshot["kind"]
        payload = snapshot["value"]
        if kind == "mapping":
            return {key: cls._materialize_snapshot(value) for key, value in payload.items()}
        return copy.deepcopy(payload)


class _FakeRunner:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.train_state = SimpleNamespace(epoch=0, global_step=0)
        self.config = {"checkpoint.async_enabled": True, "checkpoint.interval": None}
        self.id = "test-runner"
        self.checkpoint_dir = str(checkpoint_dir)
        self.is_best = False
        self.distributed = False
        self.is_main_process = True
        self.is_step_mode = False

    def state_dict(self):
        return {"state": {"global_step": self.train_state.global_step}}

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


def _set_dist_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(dcp_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(dcp_module.dist, "is_initialized", lambda: False)


def test_dcp_async_save_does_not_wait_for_previous(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.config["checkpoint.interval"] = 1
    manager = TorchDistributedCheckpointManager(runner)

    def fail_wait(timeout=None):
        del timeout
        raise AssertionError("save_checkpoint should not call wait() on async path")

    monkeypatch.setattr(manager, "wait", fail_wait)

    runner.train_state.global_step = 1
    manager.save_checkpoint()
    runner.train_state.global_step = 2
    manager.save_checkpoint()

    # Second save should queue, not block and not launch immediately while inflight exists.
    assert len(fake_dcp.async_calls) == 1

    first_future = fake_dcp.async_calls[0]["future"]
    assert isinstance(first_future, Future)
    first_future.set_result(None)

    # Completion of first async save should schedule the queued latest snapshot.
    assert len(fake_dcp.async_calls) == 2
    second_future = fake_dcp.async_calls[1]["future"]
    assert isinstance(second_future, Future)
    second_future.set_result(None)

    latest_pointer = (tmp_path / "latest.pointer").read_text(encoding="utf-8").strip()
    assert latest_pointer


def test_dcp_save_checkpoint_has_no_global_barrier(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)

    monkeypatch.setattr(dcp_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(dcp_module.dist, "is_initialized", lambda: True)

    def fail_barrier(*args, **kwargs):
        del args, kwargs
        raise AssertionError("save_checkpoint should not call dist.barrier()")

    monkeypatch.setattr(dcp_module.dist, "barrier", fail_barrier)

    runner = _FakeRunner(tmp_path)
    runner.config["checkpoint.interval"] = 1
    runner.distributed = True
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint()

    assert len(fake_dcp.async_calls) == 1
    future = fake_dcp.async_calls[0]["future"]
    assert isinstance(future, Future)
    future.set_result(None)


def test_dcp_save_checkpoint_respects_disabled_async_mode(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "disabled", "checkpoint.interval": 1})
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint()

    assert len(fake_dcp.async_calls) == 0
    assert len(fake_dcp.save_calls) == 1


def test_dcp_async_with_pinned_mem_exposes_staging_wait_hook(monkeypatch, tmp_path: Path) -> None:
    class FakeStagingOptions:
        def __init__(self, *args):
            self.args = args

    class FakeDefaultStager:
        def __init__(self, options):
            self.options = options
            self.closed = False

        def close(self):
            self.closed = True

    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)
    monkeypatch.setattr(dcp_module, "AsyncCheckpointerType", SimpleNamespace(PROCESS="process"))
    monkeypatch.setattr(dcp_module, "StagingOptions", FakeStagingOptions)
    monkeypatch.setattr(dcp_module, "DefaultStager", FakeDefaultStager)

    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "async_with_pinned_mem", "checkpoint.interval": 1})
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint()

    assert len(fake_dcp.async_calls) == 1
    first_call = fake_dcp.async_calls[0]
    kwargs = first_call["kwargs"]
    assert kwargs["async_checkpointer_type"] == "process"
    assert isinstance(kwargs["async_stager"], FakeDefaultStager)
    assert manager.maybe_wait_for_staging(timeout=0.0) is False

    staging_future = first_call["staging_future"]
    assert isinstance(staging_future, Future)
    staging_future.set_result(None)
    assert manager.maybe_wait_for_staging(timeout=0.0) is True

    upload_future = first_call["future"]
    assert isinstance(upload_future, Future)
    upload_future.set_result(None)

    assert manager.close(timeout=1.0) is True
    assert kwargs["async_stager"].closed is True


def test_dcp_async_with_pinned_mem_requires_runtime_support(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)
    monkeypatch.setattr(dcp_module, "AsyncCheckpointerType", None)

    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "async_with_pinned_mem", "checkpoint.interval": 1})
    manager = TorchDistributedCheckpointManager(runner)
    runner.train_state.global_step = 1

    with pytest.raises((RuntimeError, AssertionError), match="async_with_pinned_mem|Non-blocking copy requires"):
        manager.save_checkpoint()


def test_dcp_resolve_checkpoint_id_supports_pointer_aliases(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    manager = TorchDistributedCheckpointManager(runner)

    target_name = "latest-g000000000001-q000001"
    target_dir = tmp_path / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / ".metadata").write_text("ok\n", encoding="utf-8")
    (tmp_path / "ckpt-s000000000001.pointer").write_text(f"{target_name}\n", encoding="utf-8")

    resolved = manager._resolve_checkpoint_id("ckpt-s000000000001")
    assert resolved == str(target_dir)


def test_dcp_latest_rotation_cleans_previous_non_best_target(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.config["checkpoint.interval"] = 100
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(last_step=True)
    first_call = fake_dcp.async_calls[0]
    first_checkpoint_id = Path(str(first_call["checkpoint_id"]))
    first_future = first_call["future"]
    assert isinstance(first_future, Future)
    first_future.set_result(None)
    assert first_checkpoint_id.exists()
    stale_archive_pointer = tmp_path / "ckpt-e000000.pointer"
    stale_archive_pointer.write_text(f"{first_checkpoint_id.name}\n", encoding="utf-8")

    runner.train_state.global_step = 2
    manager.save_checkpoint(last_step=True)
    second_call = fake_dcp.async_calls[1]
    second_future = second_call["future"]
    assert isinstance(second_future, Future)
    second_future.set_result(None)

    assert not first_checkpoint_id.exists()
    assert not stale_archive_pointer.exists()


def test_dcp_keep_latest_k_prunes_old_archives(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "disabled", "checkpoint.interval": 1, "checkpoint.keep_latest_k": 2})
    manager = TorchDistributedCheckpointManager(runner)

    for epoch in range(3):
        runner.train_state.epoch = epoch
        runner.train_state.global_step = epoch + 1
        manager.save_checkpoint(epochs=epoch)

    checkpoint_dirs = [Path(str(call["checkpoint_id"])) for call in fake_dcp.save_calls]
    assert len(checkpoint_dirs) == 3

    assert manager.close(timeout=1.0) is True
    assert checkpoint_dirs[0].exists() is False
    assert checkpoint_dirs[1].exists() is True
    assert checkpoint_dirs[2].exists() is True
    assert not (tmp_path / "ckpt-e000000.pointer").exists()
    assert (tmp_path / "ckpt-e000001.pointer").exists()
    assert (tmp_path / "ckpt-e000002.pointer").exists()


def test_dcp_save_checkpoint_respects_load_only(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "disabled", "checkpoint.load_only": True})
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint()
    manager.save_checkpoint(last_step=True)

    assert fake_dcp.save_calls == []
    assert fake_dcp.async_calls == []
    assert manager.close(timeout=1.0) is True


def test_dcp_load_only_disables_ft_dataloader_checkpoint_writes(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.load_only": True,
            "checkpoint.interval": None,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)

    assert fake_dcp.save_calls == []
    assert fake_dcp.async_calls == []
    assert not (tmp_path / "ft-replica-replica0").exists()
    assert manager.close(timeout=1.0) is True


def test_dcp_checkpoint_skips_periodic_writes_without_interval(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.config["checkpoint.async_mode"] = "disabled"
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)

    assert fake_dcp.save_calls == []
    assert fake_dcp.async_calls == []
    assert manager.close(timeout=1.0) is True


def test_dcp_save_interval_does_not_force_first_step(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.interval": 5,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    manager.save_checkpoint(epochs=1)  # skipped by interval
    manager.save_checkpoint(epochs=4)  # interval boundary

    assert len(fake_dcp.save_calls) == 1
    assert manager.close(timeout=1.0) is True


def test_dcp_last_step_forces_checkpoint_outside_interval(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.config.update({"checkpoint.async_mode": "disabled", "checkpoint.interval": 100})
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    manager.save_checkpoint(epochs=0, last_step=True)

    assert len(fake_dcp.save_calls) == 1
    assert manager.close(timeout=1.0) is True


def test_dcp_step_mode_save_interval_aligns_with_global_step(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunner(tmp_path)
    runner.is_step_mode = True
    runner.config.update({"checkpoint.async_mode": "disabled", "checkpoint.interval": 2})
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    assert fake_dcp.save_calls == []

    runner.train_state.global_step = 2
    manager.save_checkpoint(epochs=0)
    assert len(fake_dcp.save_calls) == 1
    checkpoint_id = Path(str(fake_dcp.save_calls[0]["checkpoint_id"]))
    assert checkpoint_id.name.startswith("ckpt-s000000000002-g000000000002-")
    assert manager.close(timeout=1.0) is True


def test_dcp_async_save_uses_dedicated_process_group(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    monkeypatch.setattr(dcp_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(dcp_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dcp_module.dist, "get_world_size", lambda *args, **kwargs: 2)

    created = {}

    def fake_new_group(*args, **kwargs):
        del args
        created["backend"] = kwargs.get("backend")
        created["group"] = object()
        return created["group"]

    monkeypatch.setattr(dcp_module.dist, "new_group", fake_new_group)
    monkeypatch.setattr(dcp_module.dist, "destroy_process_group", lambda *args, **kwargs: None)

    runner = _FakeRunner(tmp_path)
    runner.distributed = True
    runner.config.update(
        {
            "checkpoint.async_mode": "async",
            "checkpoint.dedicated_async_process_group": True,
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint()

    assert created["backend"] == "gloo"
    assert len(fake_dcp.async_calls) == 1
    kwargs = fake_dcp.async_calls[0]["kwargs"]
    assert kwargs["process_group"] is created["group"]

    future = fake_dcp.async_calls[0]["future"]
    assert isinstance(future, Future)
    future.set_result(None)
    assert manager.close(timeout=1.0) is True


class _StatefulLoader:
    def __init__(self, position: int = 0) -> None:
        self.position = position

    def state_dict(self) -> dict[str, int]:
        return {"position": int(self.position)}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.position = int(state["position"])


class _FakeRunnerWithFT(_FakeRunner):
    def __init__(self, checkpoint_dir: Path) -> None:
        super().__init__(checkpoint_dir)
        self.rank = 0
        self.dataloaders = DataLoaderDict({"train": _StatefulLoader(position=7)})


def test_dcp_ft_dataloader_checkpoint_save_and_restore(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    runner.dataloaders["train"].position = 0
    manager.load_checkpoint("latest")

    assert runner.dataloaders["train"].position == 7
    ft_root = tmp_path / "ft-replica-replica0"
    assert ft_root.exists()
    assert any((path / ".metadata").exists() for path in ft_root.iterdir())
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_restore_tracks_requested_checkpoint_target(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 10
    runner.dataloaders["train"].position = 10
    manager.save_checkpoint(epochs=0)

    runner.train_state.global_step = 20
    runner.dataloaders["train"].position = 20
    manager.save_checkpoint(epochs=1)

    model_checkpoint_ids = [
        str(call["checkpoint_id"])
        for call in fake_dcp.save_calls
        if Path(str(call["checkpoint_id"])).parent == tmp_path and "ckpt-e" in Path(str(call["checkpoint_id"])).name
    ]
    assert len(model_checkpoint_ids) == 2
    first_checkpoint_id = model_checkpoint_ids[0]

    runner.dataloaders["train"].position = 0
    manager.load_checkpoint(first_checkpoint_id)

    assert runner.dataloaders["train"].position == 10
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_restore_is_sequence_bounded(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    # Save two checkpoints at the same global step but different checkpoint sequence.
    runner.train_state.global_step = 10
    runner.dataloaders["train"].position = 10
    manager.save_checkpoint(epochs=0)

    runner.train_state.global_step = 10
    runner.dataloaders["train"].position = 11
    manager.save_checkpoint(epochs=1)

    model_checkpoint_ids = [
        str(call["checkpoint_id"])
        for call in fake_dcp.save_calls
        if Path(str(call["checkpoint_id"])).parent == tmp_path and "ckpt-e" in Path(str(call["checkpoint_id"])).name
    ]
    assert len(model_checkpoint_ids) == 2
    first_checkpoint_id = model_checkpoint_ids[0]
    second_checkpoint_id = model_checkpoint_ids[1]
    assert first_checkpoint_id != second_checkpoint_id

    runner.dataloaders["train"].position = 0
    manager.load_checkpoint(first_checkpoint_id)
    assert runner.dataloaders["train"].position == 10

    runner.dataloaders["train"].position = 0
    manager.load_checkpoint(second_checkpoint_id)
    assert runner.dataloaders["train"].position == 11
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_disabled_async_uses_background_thread(monkeypatch, tmp_path: Path) -> None:
    class _FakeDCPThreadCheck(_FakeDCP):
        def __init__(self) -> None:
            super().__init__()
            self.ft_save_threads: list[str] = []

        def save(self, state, checkpoint_id: str, no_dist: bool):
            if "ft-replica-" in checkpoint_id:
                self.ft_save_threads.append(threading.current_thread().name)
            return super().save(state, checkpoint_id=checkpoint_id, no_dist=no_dist)

    fake_dcp = _FakeDCPThreadCheck()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    assert manager.wait(timeout=1.0) is True

    assert fake_dcp.ft_save_threads
    assert all(name != threading.main_thread().name for name in fake_dcp.ft_save_threads)
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_disabled_async_waits_for_ft_durability(monkeypatch, tmp_path: Path) -> None:
    class _BlockingFTSaveDCP(_FakeDCP):
        def __init__(self) -> None:
            super().__init__()
            self.ft_started = threading.Event()
            self.ft_release = threading.Event()

        def save(self, state, checkpoint_id: str, no_dist: bool):
            if "ft-replica-" in checkpoint_id:
                self.ft_started.set()
                if not self.ft_release.wait(timeout=1.0):
                    raise RuntimeError("timed out waiting for FT checkpoint release")
            return super().save(state, checkpoint_id=checkpoint_id, no_dist=no_dist)

    fake_dcp = _BlockingFTSaveDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    save_done = threading.Event()

    def run_save() -> None:
        manager.save_checkpoint(epochs=0)
        save_done.set()

    save_thread = threading.Thread(target=run_save, daemon=True)
    save_thread.start()

    assert fake_dcp.ft_started.wait(timeout=1.0) is True
    save_thread.join(timeout=0.05)
    assert save_done.is_set() is False

    fake_dcp.ft_release.set()
    save_thread.join(timeout=1.0)
    assert save_done.is_set() is True
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_disabled_async_gates_pointer_publish(monkeypatch, tmp_path: Path) -> None:
    class _FailingFTSaveDCP(_FakeDCP):
        def save(self, state, checkpoint_id: str, no_dist: bool):
            if "ft-replica-" in checkpoint_id:
                raise RuntimeError("ft save failed")
            return super().save(state, checkpoint_id=checkpoint_id, no_dist=no_dist)

    fake_dcp = _FailingFTSaveDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        runner.train_state.global_step = 1
        manager.save_checkpoint(epochs=0)

    assert any("ft dataloader checkpoint save failed" in str(item.message) for item in captured)
    assert (tmp_path / "latest.pointer").exists() is False
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_async_save_does_not_wait_on_inflight(monkeypatch, tmp_path: Path) -> None:
    class _PoisonFuture(Future):
        def result(self, timeout=None):  # pylint: disable=unused-argument
            raise AssertionError("FT save path must not wait on inflight future")

    class _FakeDCPWithPoisonFT(_FakeDCP):
        def __init__(self) -> None:
            super().__init__()
            self._ft_async_calls = 0
            self._ft_called = threading.Event()

        def async_save(self, state, checkpoint_id: str, no_dist: bool, **kwargs):
            if no_dist and "dataloader" in state and "ft-replica-" in checkpoint_id:
                self._materialize_checkpoint_dir(checkpoint_id)
                self._saved[checkpoint_id] = self._snapshot_state(state)
                future = _PoisonFuture()
                self._ft_async_calls += 1
                self._ft_called.set()
                self.async_calls.append(
                    {
                        "state": state,
                        "checkpoint_id": checkpoint_id,
                        "no_dist": no_dist,
                        "future": future,
                        "staging_future": None,
                        "kwargs": kwargs,
                    }
                )
                return future
            return super().async_save(state, checkpoint_id=checkpoint_id, no_dist=no_dist, **kwargs)

    fake_dcp = _FakeDCPWithPoisonFT()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "async",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    # Keep global_step stable so this test isolates non-blocking pending behavior.
    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=1)

    # Second call should enqueue pending FT save instead of blocking on `future.result()`.
    assert fake_dcp._ft_called.wait(timeout=1.0) is True
    assert fake_dcp._ft_async_calls == 1


def test_dcp_ft_dataloader_async_gates_pointer_publish_until_ft_upload(monkeypatch, tmp_path: Path) -> None:
    class _FakeDCPWithBlockingFT(_FakeDCP):
        def __init__(self) -> None:
            super().__init__()
            self.ft_called = threading.Event()

        def async_save(self, state, checkpoint_id: str, no_dist: bool, **kwargs):
            if no_dist and "dataloader" in state and "ft-replica-" in checkpoint_id:
                self._materialize_checkpoint_dir(checkpoint_id)
                self._saved[checkpoint_id] = self._snapshot_state(state)
                upload_future: Future = Future()
                self.ft_called.set()
                self.async_calls.append(
                    {
                        "state": state,
                        "checkpoint_id": checkpoint_id,
                        "no_dist": no_dist,
                        "future": upload_future,
                        "staging_future": None,
                        "kwargs": kwargs,
                    }
                )
                return upload_future
            return super().async_save(state, checkpoint_id=checkpoint_id, no_dist=no_dist, **kwargs)

    fake_dcp = _FakeDCPWithBlockingFT()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "async",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    assert fake_dcp.ft_called.wait(timeout=1.0) is True

    main_future = None
    ft_future = None
    for call in fake_dcp.async_calls:
        checkpoint_id = str(call["checkpoint_id"])
        future = call["future"]
        assert isinstance(future, Future)
        if "ft-replica-" in checkpoint_id:
            ft_future = future
        else:
            main_future = future

    assert main_future is not None
    assert ft_future is not None

    main_future.set_result(None)
    assert (tmp_path / "latest.pointer").exists() is False

    ft_future.set_result(None)
    assert manager.close(timeout=5.0) is True
    assert (tmp_path / "latest.pointer").exists() is True


def test_dcp_ft_dataloader_async_pending_queue_is_bounded(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "async",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)
    collect_started = threading.Event()
    collect_release = threading.Event()
    original_collect = manager._collect_dataloader_state

    def blocked_collect():
        collect_started.set()
        assert collect_release.wait(timeout=1.0) is True
        return original_collect()

    monkeypatch.setattr(manager, "_collect_dataloader_state", blocked_collect)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    assert collect_started.wait(timeout=1.0) is True
    runner.train_state.global_step = 2
    manager.save_checkpoint(epochs=1)
    runner.train_state.global_step = 3
    manager.save_checkpoint(epochs=2)

    assert len(manager._ft_pending) == 1
    pending_name = manager._ft_pending[0].checkpoint_name
    assert "-g000000000003-" in pending_name

    collect_release.set()
    for call in fake_dcp.async_calls:
        future = call.get("future")
        if isinstance(future, Future) and not future.done():
            future.set_result(None)
    manager.close(timeout=1.0)


def test_dcp_ft_dataloader_progress_during_capture_does_not_drop_save(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    original_collect = manager._collect_dataloader_state

    def force_progress_during_capture():
        # Simulate training thread progression while FT state is being captured.
        runner.train_state.global_step += 1
        return original_collect()

    monkeypatch.setattr(manager, "_collect_dataloader_state", force_progress_during_capture)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        runner.train_state.global_step = 1
        manager.save_checkpoint(epochs=0)
        assert manager.wait(timeout=1.0) is True

    assert not any("ft dataloader checkpoint skipped" in str(item.message) for item in captured)
    ft_saved = [
        saved
        for checkpoint_id, saved in fake_dcp._saved.items()
        if "ft-replica-" in checkpoint_id and str(tmp_path) in checkpoint_id
    ]
    assert ft_saved
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_state_is_snapshot_consistent_with_checkpoint(monkeypatch, tmp_path: Path) -> None:
    class _TracingLoader(_StatefulLoader):
        def __init__(self, position: int = 0) -> None:
            super().__init__(position=position)
            self.capture_threads: list[str] = []

        def state_dict(self) -> dict[str, int]:
            self.capture_threads.append(threading.current_thread().name)
            return super().state_dict()

    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    tracing_loader = _TracingLoader(position=7)
    runner.dataloaders = DataLoaderDict({"train": tracing_loader})
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    tracing_loader.position = 99
    assert manager.wait(timeout=1.0) is True
    assert tracing_loader.capture_threads
    assert all(name != threading.main_thread().name for name in tracing_loader.capture_threads)

    ft_saved = [
        saved
        for checkpoint_id, saved in fake_dcp._saved.items()
        if "ft-replica-" in checkpoint_id and str(tmp_path) in checkpoint_id
    ]
    assert ft_saved
    assert ft_saved[0]["dataloader"]["value"]["train"]["position"] == 7
    assert manager.close(timeout=1.0) is True


def test_dcp_load_does_not_restore_ft_dataloader_before_main_load(monkeypatch, tmp_path: Path) -> None:
    fake_dcp = _FakeDCP()
    monkeypatch.setattr(dcp_module, "dcp", fake_dcp)
    _set_dist_unavailable(monkeypatch)

    runner = _FakeRunnerWithFT(tmp_path)
    runner.config.update(
        {
            "checkpoint.async_mode": "disabled",
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
        }
    )
    manager = TorchDistributedCheckpointManager(runner)

    monkeypatch.setattr(manager, "_resolve_checkpoint_id", lambda _: "fake-checkpoint-id")

    called = {"ft_restore": False}

    def mark_ft_restore(*args, **kwargs):
        del args, kwargs
        called["ft_restore"] = True

    monkeypatch.setattr(manager, "_load_ft_dataloader_checkpoint", mark_ft_restore)

    def fail_main_load(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("main dcp load failed")

    monkeypatch.setattr(dcp_module.dcp, "load", fail_main_load)

    try:
        manager.load_checkpoint("latest")
    except RuntimeError as exc:
        assert "main dcp load failed" in str(exc)
    else:
        raise AssertionError("manager.load_checkpoint should surface main dcp load failures")

    assert called["ft_restore"] is False
