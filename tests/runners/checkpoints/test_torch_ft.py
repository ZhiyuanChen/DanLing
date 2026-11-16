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

import random
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from danling.runners.checkpoints.torch_ft import TorchFTCheckpointManager
from tests.runners.checkpoints.utils import (
    _BlockingAsyncDcpManager,
    _checkpoint_targets,
    _CheckpointRunner,
    _dcp_async_dataloader_checkpoint_requires_dedicated_process_group_worker,
    _make_stateful_dataloader,
    _NonParticipatingFaultTolerance,
    _NoStateDataloader,
    _pointer_target,
    _rank_local_resume_worker,
    _stateful_dataloader_position,
    _StaticStateDataloader,
    _target_path,
)
from tests.runners.distributed import require_gloo, run_distributed

pytestmark = pytest.mark.filterwarnings("ignore:torch.distributed is disabled.*:UserWarning")


def test_rank_rng_and_loader_resume(tmp_path: Path) -> None:
    require_gloo()
    run_distributed(_rank_local_resume_worker, worker_args=(str(tmp_path), True))
    manager = TorchFTCheckpointManager(_CheckpointRunner(tmp_path))
    try:
        with pytest.raises(ValueError, match="rank-local RNG/dataloader state.*world size 1"):
            manager.load_checkpoint("latest")
    finally:
        manager.close()


def test_loader_tracks_partial_publication(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
            "ckpt.interval": 1,
        },
    )
    runner.is_best = True
    runner.dataloaders["train"] = _StaticStateDataloader()
    manager = TorchFTCheckpointManager(runner)
    (tmp_path / "best.pointer").mkdir()

    runner.train_state.global_step = 1
    with pytest.warns(RuntimeWarning, match="checkpoint failed"):
        manager.save_checkpoint(epochs=0)

    target = "ckpt-e000001-g000000000001-q000001"
    assert _pointer_target(tmp_path, "latest") == target
    assert _pointer_target(tmp_path, "ckpt-e000001") == target
    assert (tmp_path / target).is_dir()
    assert (tmp_path / "dataloader-replica-replica0" / target / ".metadata").exists()
    assert manager.close(timeout=1.0) is True


def test_async_loader_tracks_publication(
    tmp_path: Path,
) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.async_mode": "async",
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
            "ckpt.interval": 1,
        },
    )
    runner.is_best = True
    runner.dataloaders["train"] = _StaticStateDataloader()
    manager = TorchFTCheckpointManager(runner)
    (tmp_path / "best.pointer").mkdir()

    runner.train_state.global_step = 1
    with pytest.warns(RuntimeWarning, match="checkpoint failed"):
        manager.save_checkpoint(epochs=0)
        assert manager.close(timeout=10.0) is True

    target = "ckpt-e000001-g000000000001-q000001"
    assert _pointer_target(tmp_path, "latest") == target
    assert _pointer_target(tmp_path, "ckpt-e000001") == target
    assert (tmp_path / target).is_dir()
    assert (tmp_path / "dataloader-replica-replica0" / target / ".metadata").exists()


def test_async_sidecar_waits_for_checkpoint(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.async_mode": "async",
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
        },
    )
    runner.dataloaders["train"] = _StaticStateDataloader()
    manager = _BlockingAsyncDcpManager(runner)

    try:
        runner.train_state.global_step = 1
        manager.save_checkpoint(epochs=0, force=True)
        runner.train_state.global_step = 2
        manager.save_checkpoint(epochs=0, force=True)
        runner.train_state.global_step = 3
        manager.save_checkpoint(epochs=0, force=True)

        assert manager.close(timeout=0.0) is False
        assert not (tmp_path / "latest.pointer").exists()
        assert not (tmp_path / "dataloader-replica-replica0").exists()
    finally:
        if not manager.first_future.done():
            manager.first_future.set_result(None)
        manager.close(timeout=10.0)


def test_empty_loader_allows_publication(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.async_mode": "async",
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
            "ckpt.interval": 1,
        },
    )
    runner.dataloaders["train"] = _NoStateDataloader()
    manager = TorchFTCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    assert manager.close(timeout=10.0) is True

    target = _target_path(tmp_path, "latest")
    assert (target / ".metadata").exists()
    assert (target / "runner.yaml").exists()


def test_async_loader_publishes_checkpoint(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.async_mode": "async",
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
            "ckpt.interval": 1,
        },
    )
    runner.dataloaders["train"] = _StaticStateDataloader()
    manager = TorchFTCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)

    assert manager.close(timeout=10.0) is True
    target = _target_path(tmp_path, "latest")
    assert (target / ".metadata").exists()
    assert (target / "runner.yaml").exists()
    dataloader_root = tmp_path / "dataloader-replica-replica0"
    assert dataloader_root.exists()
    assert any((path / ".metadata").exists() for path in dataloader_root.iterdir())
    assert manager.checkpoint_health.error_count == 0


def test_async_loader_requires_group(
    tmp_path: Path,
) -> None:
    require_gloo()
    run_distributed(
        _dcp_async_dataloader_checkpoint_requires_dedicated_process_group_worker,
        world_size=2,
        worker_args=(str(tmp_path),),
    )


def test_nonparticipating_replica(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 1}, dataloaders=True)
    runner.fault_tolerance = _NonParticipatingFaultTolerance()
    manager = TorchFTCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)

    assert manager.close(timeout=1.0) is True
    assert not (tmp_path / "latest.pointer").exists()
    dataloader_root = tmp_path / "dataloader-replica-3"
    assert dataloader_root.exists()
    assert any((path / ".metadata").exists() for path in dataloader_root.iterdir())


def test_disabled_loader_checkpoint(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
            "ckpt.enabled": False,
            "ckpt.interval": 1,
        },
        dataloaders=True,
    )
    manager = TorchFTCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)

    assert _checkpoint_targets(tmp_path) == []
    assert not (tmp_path / "dataloader-replica-replica0").exists()
    assert manager.close(timeout=1.0) is True


def test_model_export_and_replica_resume(tmp_path: Path) -> None:
    class Replica:
        enabled = True

        def __init__(self, replica_id):
            self.replica_id = replica_id

        def participating_rank(self):
            return self.replica_id

    class ReplicaRunner(_CheckpointRunner):
        def __init__(self, replica):
            super().__init__(tmp_path, config={"ckpt.dataloader_checkpoint.enabled": True}, dataloaders=True)
            self.fault_tolerance = Replica(replica)
            self.replica = replica
            self.rng = {
                "python": random.Random(100 + replica).getstate(),
                "torch_cpu": torch.Generator().manual_seed(200 + replica).get_state(),
            }

        def state_dict(self):
            state = super().state_dict()
            state["model"]["weight"].fill_(self.replica + 2)
            state["state"]["rng"] = deepcopy(self.rng)
            state["fault_tolerance"] = {"step": self.replica + 7}
            return state

        def _export_checkpoint_local_state(self):
            return {
                key: value
                for key, value in self.state_dict().items()
                if key in {"dataloaders", "state", "fault_tolerance"}
            }

    runners, managers, expected_rng = [], [], []
    for replica in range(2):
        runner = ReplicaRunner(replica)
        expected_rng.append(deepcopy(runner.rng))
        runners.append(runner)
        managers.append(TorchFTCheckpointManager(runner))

    try:
        for manager in managers:
            manager.save_model_checkpoint()
        model = managers[0].load_model_checkpoint("model")
        torch.testing.assert_close(model["model"]["weight"], torch.full((2, 2), 2.0))
        for replica, (runner, manager) in enumerate(zip(runners, managers)):
            runner.train_state.global_step = 1
            runner.dataloaders["train"] = _make_stateful_dataloader(replica + 3)
            manager.save_checkpoint(force=True)
        for replica, (runner, manager) in enumerate(zip(runners, managers)):
            runner.dataloaders["train"] = _make_stateful_dataloader(0)
            checkpoint = manager.load_checkpoint("latest")
            assert _stateful_dataloader_position(runner.dataloaders["train"]) == replica + 3
            assert checkpoint["state"]["rng"]["python"] == expected_rng[replica]["python"]
            torch.testing.assert_close(checkpoint["state"]["rng"]["torch_cpu"], expected_rng[replica]["torch_cpu"])
            assert checkpoint["fault_tolerance"] == {"step": replica + 7}
    finally:
        for manager in managers:
            manager.close()


def test_dataloader_checkpoint_save_and_restore(tmp_path: Path) -> None:
    class LocalRunner(_CheckpointRunner):
        def state_dict(self):
            state = super().state_dict()
            state["state"]["rng"] = {"python": "full-writer-replica"}
            state["fault_tolerance"] = {"step": 1}
            return state

        def _export_checkpoint_local_state(self):
            return {
                "dataloaders": self.dataloaders.state_dict(),
                "state": {"rng": {"python": "own-replica"}},
                "fault_tolerance": {"step": 7},
            }

    runner = LocalRunner(
        tmp_path,
        config={
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
            "ckpt.interval": 1,
        },
        dataloaders=True,
    )
    manager = TorchFTCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    runner.dataloaders["train"] = _make_stateful_dataloader(0)
    checkpoint = manager.load_checkpoint("latest")

    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 7
    assert checkpoint["state"]["rng"] == {"python": "own-replica"}
    assert checkpoint["fault_tolerance"] == {"step": 7}
    dataloader_root = tmp_path / "dataloader-replica-replica0"
    assert dataloader_root.exists()
    assert any((path / ".metadata").exists() for path in dataloader_root.iterdir())
    assert manager.close(timeout=1.0) is True


def test_restore_requested_loader_target(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
            "ckpt.interval": 1,
        },
        dataloaders=True,
    )
    manager = TorchFTCheckpointManager(runner)

    runner.train_state.global_step = 10
    runner.dataloaders["train"] = _make_stateful_dataloader(10)
    manager.save_checkpoint(epochs=0)
    first_checkpoint = _pointer_target(tmp_path, "ckpt-e000001")

    runner.train_state.global_step = 20
    runner.dataloaders["train"] = _make_stateful_dataloader(20)
    manager.save_checkpoint(epochs=1)

    runner.dataloaders["train"] = _make_stateful_dataloader(0)
    manager.load_checkpoint(first_checkpoint)

    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 10
    assert manager.close(timeout=1.0) is True


def test_restore_loader_sequence_bound(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.dataloader_checkpoint.enabled": True,
            "ckpt.dataloader_checkpoint.replica_id": "replica0",
            "ckpt.interval": 1,
        },
        dataloaders=True,
    )
    manager = TorchFTCheckpointManager(runner)

    runner.train_state.global_step = 10
    runner.dataloaders["train"] = _make_stateful_dataloader(10)
    manager.save_checkpoint(epochs=0)
    first_checkpoint = _pointer_target(tmp_path, "ckpt-e000001")

    runner.train_state.global_step = 10
    runner.dataloaders["train"] = _make_stateful_dataloader(11)
    manager.save_checkpoint(epochs=1)
    second_checkpoint = _pointer_target(tmp_path, "ckpt-e000002")
    assert first_checkpoint != second_checkpoint

    runner.dataloaders["train"] = _make_stateful_dataloader(0)
    manager.load_checkpoint(first_checkpoint)
    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 10

    runner.dataloaders["train"] = _make_stateful_dataloader(0)
    manager.load_checkpoint(second_checkpoint)
    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 11
    assert manager.close(timeout=1.0) is True
