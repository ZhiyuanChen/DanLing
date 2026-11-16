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
from collections.abc import Mapping
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from danling.data import DataLoaderDict
from danling.runners.checkpoints.torch_distributed import TorchDistributedCheckpointManager
from danling.runners.checkpoints.torch_ft import TorchFTCheckpointManager
from danling.runners.config import RunnerConfig
from tests.runners.distributed import process_group


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
    def participating_rank() -> int | None:
        return None


class _NoStateDataloader:
    @staticmethod
    def state_dict() -> None:
        return None


class _StaticStateDataloader:
    @staticmethod
    def state_dict() -> dict[str, int]:
        return {"position": 7}


class _CheckpointRunner:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        config: Mapping[str, Any] | None = None,
        dataloaders: bool = False,
    ) -> None:
        runner_config: dict[str, Any] = {
            "logging.enabled": False,
            "ckpt.backend": "dcp",
            "ckpt.async_mode": "disabled",
            "ckpt.interval": None,
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
        self.fault_tolerance = None
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
            "model": {"weight": torch.ones(2, 2, dtype=torch.float32)},
        }
        if self.dataloaders:
            state["dataloaders"] = self.dataloaders.state_dict()
        return state

    @property
    def checkpoint_interval(self) -> int:
        interval = self.config.get("ckpt.interval")
        return int(interval) if interval is not None else -1


class _BlockingAsyncDcpManager(TorchFTCheckpointManager):
    def __init__(self, runner: _CheckpointRunner) -> None:
        super().__init__(runner)
        self.first_future: Future = Future()

    def _start_async_task_locked(self, task: Any) -> Future | None:
        if not self.first_future.done():
            return self.first_future
        future: Future = Future()
        future.set_result(None)
        return future


def _make_stateful_dataloader(position: int = 0) -> StatefulDataLoader:
    loader = StatefulDataLoader(list(range(128)), batch_size=1, shuffle=False)
    iterator = iter(loader)
    for _ in range(position):
        next(iterator)
    return loader


def _stateful_dataloader_position(loader: StatefulDataLoader) -> int:
    return int(next(iter(loader)).item())


def _dataloader_state_position(state: Mapping[str, Any]) -> int:
    # The test loader has monotonically numbered samples. Restore into an
    # independent loader and observe its next value instead of inspecting
    # TorchData's private checkpoint layout.
    loader = StatefulDataLoader(list(range(128)), batch_size=1, shuffle=False)
    loader.load_state_dict(state)
    return int(next(iter(loader)).item())


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
            "ckpt.interval": 2,
            "ckpt.keep_latest_k": 1,
        },
    )
    return runner, TorchDistributedCheckpointManager(runner)


def _configure_distributed_runner(runner: _CheckpointRunner, rank: int) -> None:
    runner.distributed = True
    runner.is_main_process = rank == 0
    runner.rank = rank


def _rank_local_resume_worker(rank: int, world_size: int, checkpoint_dir: str, use_ft: bool) -> None:
    with process_group("gloo", rank, world_size):
        captured = {}
        history = {2: {"val": {"loss": 0.1}}, 9: {"val": {"loss": 0.5, "accuracy": 0.7}}}
        saved_history = deepcopy(history)

        class RankRunner(_CheckpointRunner):
            def state_dict(self):
                rng = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
                state = super().state_dict()
                state["results"] = history
                state["state"]["rng"] = rng
                state["fault_tolerance"] = {"step": rank + 4}
                if not captured:
                    captured.update(rng)
                return state

            def _export_checkpoint_local_state(self):
                return {
                    key: value
                    for key, value in self.state_dict().items()
                    if key in ("state", "dataloaders", "fault_tolerance")
                }

        runner = RankRunner(
            Path(checkpoint_dir),
            config={
                "ckpt.interval": 1,
                "ckpt.dataloader_checkpoint.enabled": use_ft,
                "ckpt.dataloader_checkpoint.replica_id": "replica0",
            },
            dataloaders=True,
        )
        _configure_distributed_runner(runner, rank)
        runner.dataloaders["train"] = StatefulDataLoader(list(range(128)), batch_size=1, shuffle=False)
        iterator = iter(runner.dataloaders["train"])
        for _ in range(rank + 2):
            next(iterator)
        manager_cls = TorchFTCheckpointManager if use_ft else TorchDistributedCheckpointManager
        manager = manager_cls(runner)
        try:
            random.seed(100 + rank)
            torch.manual_seed(200 + rank)
            manager.save_checkpoint(force=True)
            history.clear()
            runner.dataloaders["train"] = StatefulDataLoader(list(range(128)), batch_size=1, shuffle=False)
            random.seed(300)
            torch.manual_seed(400)
            checkpoint = manager.load_checkpoint("latest")
            assert checkpoint["results"] == saved_history
            assert checkpoint["state"]["rng"]["python"] == captured["python"]
            torch.testing.assert_close(checkpoint["state"]["rng"]["torch_cpu"], captured["torch_cpu"])
            assert checkpoint["fault_tolerance"] == {"step": rank + 4}
            position = _dataloader_state_position(checkpoint["dataloaders"]["train"])
            assert position == rank + 2, (rank, position, checkpoint["dataloaders"]["train"])
            runner.dataloaders.load_state_dict(checkpoint["dataloaders"])
            value = next(iter(runner.dataloaders["train"])).item()
            assert value == rank + 2, (rank, value, position, checkpoint["dataloaders"]["train"])
        finally:
            manager.close()


def _assert_checkpoint_failure(
    manager: TorchDistributedCheckpointManager,
    exc_type: type[BaseException],
    target: str,
    alias: str | None = None,
) -> None:
    health = manager.checkpoint_health
    assert isinstance(health.first_error, exc_type)
    assert isinstance(health.last_error, exc_type)
    assert health.error_count == 1
    assert health.last_failed_target == target
    assert health.last_failed_alias == alias


def _dcp_async_dedicated_process_group_failure_worker(rank: int, world_size: int, checkpoint_dir: str) -> None:
    with process_group("gloo", rank, world_size):
        root = Path(checkpoint_dir)
        runner = _CheckpointRunner(
            root,
            config={
                "ckpt.async_mode": "async",
                "ckpt.async_process_group_backend": "invalid-backend",
                "ckpt.dedicated_async_process_group": True,
                "ckpt.interval": 1,
            },
        )
        _configure_distributed_runner(runner, rank)
        manager = TorchDistributedCheckpointManager(runner)

        runner.train_state.global_step = 1
        with pytest.warns(RuntimeWarning, match="async DCP checkpoints require"):
            manager.save_checkpoint(epochs=0)

        _assert_checkpoint_failure(manager, RuntimeError, "ckpt-e000001-g000000000001-q000001")
        assert manager.close(timeout=1.0) is True
        if rank == 0:
            assert not (root / "latest.pointer").exists()
            assert _checkpoint_targets(root) == []


def _dcp_async_dataloader_checkpoint_requires_dedicated_process_group_worker(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> None:
    with process_group("gloo", rank, world_size):
        root = Path(checkpoint_dir)
        runner = _CheckpointRunner(
            root,
            config={
                "ckpt.async_mode": "async",
                "ckpt.dedicated_async_process_group": False,
                "ckpt.dataloader_checkpoint.enabled": True,
                "ckpt.dataloader_checkpoint.replica_id": "replica0",
                "ckpt.interval": 1,
            },
        )
        _configure_distributed_runner(runner, rank)
        runner.dataloaders["train"] = _StaticStateDataloader()
        manager = TorchFTCheckpointManager(runner)

        runner.train_state.global_step = 1
        with pytest.warns(RuntimeWarning, match="async dataloader checkpoints require"):
            manager.save_checkpoint(epochs=0)

        _assert_checkpoint_failure(manager, RuntimeError, "ckpt-e000001-g000000000001-q000001")
        assert manager.close(timeout=1.0) is True
        if rank == 0:
            assert not (root / "latest.pointer").exists()
            assert _checkpoint_targets(root) == []
            assert not (root / "dataloader-replica-replica0").exists()


def _save_best_dcp_checkpoint(tmp_path: Path) -> TorchDistributedCheckpointManager:
    runner, manager = _retaining_dcp_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    return manager
