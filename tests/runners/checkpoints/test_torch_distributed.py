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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from danling.data import DataLoaderDict
from danling.runners.checkpoints.torch_distributed import TorchDistributedCheckpointManager
from danling.runners.config import RunnerConfig

pytestmark = pytest.mark.filterwarnings("ignore:torch.distributed is disabled.*:UserWarning")


@dataclass
class _TrainState:
    epoch: int = 0
    global_step: int = 0


@dataclass
class _CheckpointWorkspace:
    id: str
    checkpoint_dir: str


class _NonParticipatingFaultTolerance:
    enabled = True
    replica_id = 3

    @staticmethod
    def participating_rank() -> int:
        return 1


class _CheckpointRunner:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        config: Mapping[str, Any] | None = None,
        dataloaders: bool = False,
    ) -> None:
        runner_config: dict[str, Any] = {
            "log": False,
            "checkpoint.backend": "dcp",
            "checkpoint.async_mode": "disabled",
            "checkpoint.interval": None,
        }
        if config is not None:
            runner_config.update(config)
        self.train_state = _TrainState()
        self.config = RunnerConfig(runner_config)
        self.id = "test-runner"
        self.workspace = _CheckpointWorkspace(id=self.id, checkpoint_dir=str(checkpoint_dir))
        self.is_best = False
        self.distributed = False
        self.is_main_process = True
        self.is_step_mode = False
        self.ft = None
        self.rank = 0
        self.dataloaders = DataLoaderDict()
        if dataloaders:
            self.dataloaders["train"] = _make_stateful_dataloader(position=7)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "runner": self.config.dict(),
            "state": {
                "epoch": self.train_state.epoch,
                "global_step": self.train_state.global_step,
            },
        }
        if self.dataloaders:
            state["dataloaders"] = self.dataloaders.state_dict()
        return state

    @property
    def checkpoint_interval(self) -> int:
        interval = self.config.get("checkpoint.interval")
        return int(interval) if interval is not None else -1


class _FailingRunnerConfig(RunnerConfig):
    def yaml(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("runner yaml write failed")


def _make_stateful_dataloader(position: int = 0) -> StatefulDataLoader:
    loader = StatefulDataLoader(list(range(128)), batch_size=1, shuffle=False)
    _restore_stateful_dataloader_position(loader, position)
    return loader


def _restore_stateful_dataloader_position(loader: StatefulDataLoader, position: int) -> None:
    reference = StatefulDataLoader(list(range(128)), batch_size=1, shuffle=False)
    iterator = iter(reference)
    for _ in range(position):
        next(iterator)
    loader.load_state_dict(reference.state_dict())


def _stateful_dataloader_position(loader: StatefulDataLoader) -> int:
    return _dataloader_state_position(loader.state_dict())


def _dataloader_state_position(state: Mapping[str, Any]) -> int:
    return int(state["_num_yielded"])


def _checkpoint_targets(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / ".metadata").exists())


def _pointer_target(root: Path, alias: str) -> str:
    return (root / f"{alias}.pointer").read_text(encoding="utf-8").strip()


def _target_path(root: Path, alias: str) -> Path:
    return root / _pointer_target(root, alias)


def _retaining_dcp_manager(tmp_path: Path) -> tuple[_CheckpointRunner, TorchDistributedCheckpointManager]:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "checkpoint.interval": 2,
            "checkpoint.keep_latest_k": 1,
        },
    )
    return runner, TorchDistributedCheckpointManager(runner)


def _save_best_dcp_checkpoint(tmp_path: Path) -> TorchDistributedCheckpointManager:
    runner, manager = _retaining_dcp_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    return manager


def test_dcp_async_save_writes_checkpoint(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.async_mode": "async", "checkpoint.interval": 1})
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint()
    assert manager.close(timeout=10.0) is True
    target = _target_path(tmp_path, "latest")
    assert (target / ".metadata").exists()
    assert (target / "runner.yaml").exists()


def test_dcp_writes_runner_config_sidecar_for_read_config(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "name": "dcp-sidecar-test",
            "checkpoint.interval": 1,
        },
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)

    target = _target_path(tmp_path, "latest")
    assert (target / "runner.yaml").exists()
    config = TorchDistributedCheckpointManager.read_config(tmp_path / "latest")
    assert config.name == "dcp-sidecar-test"
    assert config.get("checkpoint.backend") == "dcp"
    assert manager.close(timeout=1.0) is True


def test_dcp_sidecar_failure_does_not_publish_orphan_checkpoint(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.interval": 1})
    runner.config = _FailingRunnerConfig(runner.config)
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    with pytest.raises(RuntimeError, match="runner yaml write failed"):
        manager.save_checkpoint(epochs=0)

    assert not (tmp_path / "latest.pointer").exists()
    assert _checkpoint_targets(tmp_path) == []
    assert [path for path in tmp_path.iterdir() if path.is_dir()] == []
    assert manager.close(timeout=1.0) is True


def test_dcp_async_sidecar_failure_does_not_publish_orphan_checkpoint(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.async_mode": "async", "checkpoint.interval": 1})
    runner.config = _FailingRunnerConfig(runner.config)
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    with (
        pytest.warns(RuntimeWarning, match="dcp checkpoint save failed"),
        pytest.raises(RuntimeError, match="runner yaml write failed"),
    ):
        manager.close(timeout=10.0)
    assert not (tmp_path / "latest.pointer").exists()
    assert _checkpoint_targets(tmp_path) == []
    assert [path for path in tmp_path.iterdir() if path.is_dir()] == []


def test_dcp_skips_full_save_for_non_participating_ft_replica(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.interval": 1}, dataloaders=True)
    runner.ft = _NonParticipatingFaultTolerance()
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)

    assert manager.close(timeout=1.0) is True
    assert not (tmp_path / "latest.pointer").exists()
    ft_root = tmp_path / "ft-replica-3"
    assert ft_root.exists()
    assert any((path / ".metadata").exists() for path in ft_root.iterdir())


def test_dcp_force_checkpoint_bypasses_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0, force=True)

    assert (tmp_path / "latest.pointer").exists()
    assert len(_checkpoint_targets(tmp_path)) == 1
    assert manager.close(timeout=1.0) is True


def test_dcp_resolve_checkpoint_id_supports_pointer_aliases(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = TorchDistributedCheckpointManager(runner)

    target_name = "latest-g000000000001-q000001"
    target_dir = tmp_path / target_name
    target_dir.mkdir(parents=True)
    (target_dir / ".metadata").write_text("ok\n", encoding="utf-8")
    (tmp_path / "ckpt-s000000000001.pointer").write_text(f"{target_name}\n", encoding="utf-8")

    assert manager._resolve_checkpoint_id("ckpt-s000000000001") == str(target_dir)
    assert manager.close(timeout=1.0) is True


def test_dcp_latest_rotation_keeps_retained_history_target(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.interval": 100})
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(last_step=True)
    first_checkpoint = _target_path(tmp_path, "latest")
    stale_history_pointer = tmp_path / "ckpt-e000000.pointer"
    stale_history_pointer.write_text(f"{first_checkpoint.name}\n", encoding="utf-8")

    runner.train_state.global_step = 2
    manager.save_checkpoint(last_step=True)

    assert first_checkpoint.exists()
    assert stale_history_pointer.exists()
    assert manager.close(timeout=1.0) is True


def test_dcp_keep_latest_k_prunes_old_latest_only_targets(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "checkpoint.interval": 100,
            "checkpoint.keep_latest_k": 1,
        },
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(last_step=True)
    first_checkpoint = _target_path(tmp_path, "latest")
    stale_history_pointer = tmp_path / "ckpt-e000000.pointer"
    stale_history_pointer.write_text(f"{first_checkpoint.name}\n", encoding="utf-8")

    runner.train_state.global_step = 2
    manager.save_checkpoint(last_step=True)
    second_checkpoint = _target_path(tmp_path, "latest")

    assert manager.close(timeout=1.0) is True
    assert not first_checkpoint.exists()
    assert not stale_history_pointer.exists()
    assert second_checkpoint.exists()
    assert _pointer_target(tmp_path, "latest") == second_checkpoint.name


def test_dcp_keep_latest_k_prunes_old_history_checkpoints(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "checkpoint.interval": 1,
            "checkpoint.keep_latest_k": 2,
        },
    )
    manager = TorchDistributedCheckpointManager(runner)

    first_checkpoint: Path | None = None
    for epoch in range(3):
        runner.train_state.epoch = epoch
        runner.train_state.global_step = epoch + 1
        manager.save_checkpoint(epochs=epoch)
        if epoch == 0:
            first_checkpoint = _target_path(tmp_path, "ckpt-e000001")

    assert first_checkpoint is not None
    assert manager.close(timeout=1.0) is True
    assert not first_checkpoint.exists()
    assert _target_path(tmp_path, "ckpt-e000002").exists()
    assert _target_path(tmp_path, "ckpt-e000003").exists()
    assert not (tmp_path / "ckpt-e000001.pointer").exists()
    assert (tmp_path / "ckpt-e000002.pointer").exists()
    assert (tmp_path / "ckpt-e000003.pointer").exists()


def test_dcp_checkpoint_waits_for_interval(tmp_path: Path) -> None:
    runner, manager = _retaining_dcp_manager(tmp_path)
    runner.train_state.global_step = 10

    manager.save_checkpoint(epochs=0)

    assert manager.close(timeout=1.0) is True
    assert not (tmp_path / "latest.pointer").exists()


def test_dcp_checkpoint_writes_best_pointer(tmp_path: Path) -> None:
    manager = _save_best_dcp_checkpoint(tmp_path)
    best_target = _pointer_target(tmp_path, "best")

    assert (tmp_path / best_target).is_dir()
    assert manager.close(timeout=1.0) is True


def test_dcp_checkpoint_writes_history_pointer(tmp_path: Path) -> None:
    manager = _save_best_dcp_checkpoint(tmp_path)

    assert _pointer_target(tmp_path, "ckpt-e000002") == _pointer_target(tmp_path, "best")
    assert manager.close(timeout=1.0) is True


def test_dcp_force_checkpoint_updates_latest_without_replacing_best(tmp_path: Path) -> None:
    runner, manager = _retaining_dcp_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    best_target = _pointer_target(tmp_path, "best")

    runner.is_best = False
    runner.train_state.global_step = 30
    manager.save_checkpoint(epochs=2, force=True)
    forced_latest_target = _pointer_target(tmp_path, "latest")

    assert forced_latest_target.startswith("latest-g000000000030")
    assert _pointer_target(tmp_path, "best") == best_target
    assert manager.close(timeout=1.0) is True


def test_dcp_checkpoint_retention_prunes_forced_latest_target(tmp_path: Path) -> None:
    runner, manager = _retaining_dcp_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    best_target = _pointer_target(tmp_path, "best")
    runner.is_best = False
    runner.train_state.global_step = 30
    manager.save_checkpoint(epochs=2, force=True)
    forced_latest_target = _pointer_target(tmp_path, "latest")
    runner.train_state.global_step = 40
    manager.save_checkpoint(epochs=3)
    final_latest_target = _pointer_target(tmp_path, "latest")

    assert manager.close(timeout=1.0) is True
    assert final_latest_target.startswith("ckpt-e000004-g000000000040")
    assert _pointer_target(tmp_path, "best") == best_target
    assert (tmp_path / best_target).is_dir()
    assert not (tmp_path / forced_latest_target).exists()
    assert (tmp_path / final_latest_target).is_dir()


def test_dcp_save_checkpoint_respects_load_only(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.interval": 1, "checkpoint.load_only": True})
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint()
    manager.save_checkpoint(last_step=True)

    assert not (tmp_path / "latest.pointer").exists()
    assert _checkpoint_targets(tmp_path) == []
    assert manager.close(timeout=1.0) is True


def test_dcp_load_only_disables_ft_dataloader_checkpoint_writes(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.load_only": True,
            "checkpoint.interval": 1,
        },
        dataloaders=True,
    )
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)

    assert _checkpoint_targets(tmp_path) == []
    assert not (tmp_path / "ft-replica-replica0").exists()
    assert manager.close(timeout=1.0) is True


def test_dcp_checkpoint_skips_periodic_writes_without_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)

    assert not (tmp_path / "latest.pointer").exists()
    assert _checkpoint_targets(tmp_path) == []
    assert manager.close(timeout=1.0) is True


def test_dcp_save_interval_does_not_force_first_step(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.interval": 5})
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    manager.save_checkpoint(epochs=1)
    manager.save_checkpoint(epochs=4)

    assert len(_checkpoint_targets(tmp_path)) == 1
    assert (tmp_path / "ckpt-e000005.pointer").exists()
    assert manager.close(timeout=1.0) is True


def test_dcp_last_step_forces_checkpoint_outside_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.interval": 100})
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    manager.save_checkpoint(epochs=0, last_step=True)

    assert (tmp_path / "latest.pointer").exists()
    assert len(_checkpoint_targets(tmp_path)) == 1
    assert manager.close(timeout=1.0) is True


def test_dcp_step_mode_save_interval_aligns_with_global_step(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.interval": 2})
    runner.is_step_mode = True
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    assert _checkpoint_targets(tmp_path) == []

    runner.train_state.global_step = 2
    manager.save_checkpoint(epochs=0)
    checkpoint = _target_path(tmp_path, "ckpt-s000000000002")

    assert checkpoint.name.startswith("ckpt-s000000000002-g000000000002-")
    assert manager.close(timeout=1.0) is True


def test_dcp_main_checkpoint_returns_dataloader_state_for_standard_resume(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"checkpoint.interval": 1}, dataloaders=True)
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    _restore_stateful_dataloader_position(runner.dataloaders["train"], 0)
    checkpoint = manager.load_checkpoint("latest")

    assert _dataloader_state_position(checkpoint["dataloaders"]["train"]) == 7
    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 0
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_checkpoint_save_and_restore(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        },
        dataloaders=True,
    )
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    _restore_stateful_dataloader_position(runner.dataloaders["train"], 0)
    manager.load_checkpoint("latest")

    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 7
    ft_root = tmp_path / "ft-replica-replica0"
    assert ft_root.exists()
    assert any((path / ".metadata").exists() for path in ft_root.iterdir())
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_restore_tracks_requested_checkpoint_target(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        },
        dataloaders=True,
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 10
    _restore_stateful_dataloader_position(runner.dataloaders["train"], 10)
    manager.save_checkpoint(epochs=0)
    first_checkpoint = _pointer_target(tmp_path, "ckpt-e000001")

    runner.train_state.global_step = 20
    _restore_stateful_dataloader_position(runner.dataloaders["train"], 20)
    manager.save_checkpoint(epochs=1)

    _restore_stateful_dataloader_position(runner.dataloaders["train"], 0)
    manager.load_checkpoint(first_checkpoint)

    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 10
    assert manager.close(timeout=1.0) is True


def test_dcp_ft_dataloader_restore_is_sequence_bounded(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "checkpoint.enable_ft_dataloader_checkpoints": True,
            "checkpoint.ft_replica_id": "replica0",
            "checkpoint.interval": 1,
        },
        dataloaders=True,
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 10
    _restore_stateful_dataloader_position(runner.dataloaders["train"], 10)
    manager.save_checkpoint(epochs=0)
    first_checkpoint = _pointer_target(tmp_path, "ckpt-e000001")

    runner.train_state.global_step = 10
    _restore_stateful_dataloader_position(runner.dataloaders["train"], 11)
    manager.save_checkpoint(epochs=1)
    second_checkpoint = _pointer_target(tmp_path, "ckpt-e000002")
    assert first_checkpoint != second_checkpoint

    _restore_stateful_dataloader_position(runner.dataloaders["train"], 0)
    manager.load_checkpoint(first_checkpoint)
    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 10

    _restore_stateful_dataloader_position(runner.dataloaders["train"], 0)
    manager.load_checkpoint(second_checkpoint)
    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 11
    assert manager.close(timeout=1.0) is True
