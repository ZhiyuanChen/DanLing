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

import importlib.util
import math
import os
import signal
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn, optim
from torchdata.stateful_dataloader import StatefulDataLoader

import danling.runners.torch_runner as torch_runner_module
from danling.runners import TorchRunner
from danling.runners.checkpoints import TorchDistributedCheckpointManager, TorchFTCheckpointManager
from danling.runners.compile import Compiler
from danling.runners.config import CompileConfig
from danling.tensors import NestedTensor
from tests.runners.distributed import configure_distributed_env, process_group, require_gloo, run_distributed


class TinyTorchRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class NoOptimizerTorchRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()


class TrainDispatchRunner(TinyTorchRunner):
    def __init__(self, config):
        self.dispatched_train_splits: list[str] | None = None
        self.dispatched_evaluate_splits: list[str] | None = None
        super().__init__(config)

    def train_steps(self, train_splits=None, evaluate_splits=None):
        self.dispatched_train_splits = list(train_splits or [])
        self.dispatched_evaluate_splits = list(evaluate_splits or [])
        return {}


class RecordingRestoreRunner(TinyTorchRunner):
    def __init__(self, config):
        self.restore_calls: list[tuple[str, object]] = []
        super().__init__(config)

    def load_checkpoint(self, checkpoint, *args, **kwargs) -> None:
        del args, kwargs
        self.restore_calls.append(("checkpoint", checkpoint))

    def load_pretrained(self, checkpoint, *args, **kwargs) -> None:
        del args, kwargs
        self.restore_calls.append(("pretrained", checkpoint))


class TinyStepLRTorchRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)


class TinyPlateauTorchRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=0, factor=0.5)


class StatefulDatasetTorchRunner(TinyTorchRunner):
    def __init__(self, config):
        super().__init__(config)
        self.datasets["train"] = list(range(8))


class StreamingLoader:
    def __init__(self, *values: float) -> None:
        self._values = values

    def __iter__(self):
        return iter(self._values)


class StreamingEpochRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Identity()
        self._optimizer_param = nn.Parameter(torch.zeros(()))
        self.optimizer = optim.SGD([self._optimizer_param], lr=0.1)

    def train_step(self, data):
        loss = torch.tensor(float(data))
        self.step()
        return None, loss

    def evaluate_step(self, data):
        return None, torch.tensor(float(data))


class RecordingStepLogRunner(StreamingEpochRunner):
    def __init__(self, config):
        self.step_log_calls: list[tuple[str, int, int | str | None, dict[str, object]]] = []
        super().__init__(config)

    def step_log(self, split, iteration, length=None, result=None):
        self.step_log_calls.append((split, iteration, length, dict(result or {})))
        return result


class EpochSchedulerRunner(TinyTorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.val_losses = tuple(float(value) for value in self.config.get("val_losses", [1.0, 2.0]))

    def train_epoch(self, split: str = "train"):
        del split
        return {"loss": 1.0}

    def evaluate_epoch(self, split: str = "val"):
        del split
        return {"loss": self.val_losses[self.train_state.epoch]}


class TelemetryRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Identity()
        self.param = nn.Parameter(torch.tensor(1.0))
        self.optimizer = optim.SGD([self.param], lr=0.1)

    def train_step(self, data):
        del data
        loss = self.param.square()
        self.backward(loss)
        self.step()
        return None, loss.detach()

    def evaluate_step(self, data):
        del data
        return None, self.param.square().detach()


class WeightedLossRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class MaskWeightedLossRunner(WeightedLossRunner):
    def __init__(self, config):
        super().__init__(config)

        class TokenMeanModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                del attention_mask
                return self.weight * tokens.float().mean()

        self.model = TokenMeanModel()
        self.criterion = lambda pred, target: pred.square()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class ContextRecordingRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        self.infer_context_entries = 0
        super().__init__(config)
        self.model = nn.Linear(4, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

    def infer_context(self):
        runner = self

        class RecordingContext:
            def __enter__(self):
                runner.infer_context_entries += 1

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                return False

        return RecordingContext()


class TimeoutRecordingRunner(TinyTorchRunner):
    def __init__(self, config):
        self.timeout_calls: list[timedelta] = []
        super().__init__(config)

    def _set_process_group_timeout(self, timeout: timedelta) -> None:
        self.timeout_calls.append(timeout)


class DistributedTinyTorchRunner(TorchRunner):
    @property
    def device(self):
        return torch.device("cpu")

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class DcpConfigTorchRunner(TorchRunner):
    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(1, 1)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class _FakeGradScaler:
    def __init__(self) -> None:
        self.unscale_calls: list[optim.Optimizer] = []
        self.step_calls: list[optim.Optimizer] = []
        self.update_calls = 0
        self.loaded_state: dict[str, object] | None = None

    def unscale_(self, optimizer: optim.Optimizer) -> None:
        self.unscale_calls.append(optimizer)

    def step(self, optimizer: optim.Optimizer) -> None:
        self.step_calls.append(optimizer)
        optimizer.step()

    def update(self) -> None:
        self.update_calls += 1

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.loaded_state = state_dict


class _FakePlacement:
    def __init__(self, *, replicate: bool = False, partial: bool = False) -> None:
        self._replicate = replicate
        self._partial = partial

    def is_replicate(self) -> bool:
        return self._replicate

    def is_partial(self) -> bool:
        return self._partial


class _FakeReplicate:
    def is_replicate(self) -> bool:
        return True

    def is_partial(self) -> bool:
        return False


class _FakeDTensor:
    def __init__(self, local: torch.Tensor, placements: list[_FakePlacement]) -> None:
        self._local = local
        self.placements = placements
        self.redistributed_to: list[object] | None = None

    def to_local(self) -> torch.Tensor:
        return self._local

    def detach(self) -> "_FakeDTensor":
        return self

    def redistribute(self, *, placements: list[object]) -> "_FakeDTensor":
        output = _FakeDTensor(self._local, self.placements)
        output.redistributed_to = placements
        return output


def _ddp_compile_wrap_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner(
        {"logging.enabled": False, "dist.backend": "gloo", "compile": {"enabled": True}}
    )
    try:
        assert isinstance(runner.model, nn.parallel.DistributedDataParallel)
        assert runner.model.module.__class__.__name__ == "OptimizedModule"
        assert hasattr(runner.model.module, "_orig_mod")
    finally:
        runner.close()


def _ddp_no_sync_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"logging.enabled": False, "dist.backend": "gloo", "accum_steps": 2})
    try:
        assert isinstance(runner.model, nn.parallel.DistributedDataParallel)
        runner.optimizer.zero_grad(set_to_none=True)

        runner.train_state.micro_step = 0
        first_input = torch.full((2, 4), float(rank + 1))
        target = torch.zeros(2, 2)
        with runner.train_context():
            runner.backward(runner.criterion(runner.model(first_input), target))

        first_grad = next(runner.model.parameters()).grad.detach().clone()
        first_gathered = [torch.zeros_like(first_grad) for _ in range(world_size)]
        dist.all_gather(first_gathered, first_grad)
        assert not torch.allclose(first_gathered[0], first_gathered[1])

        runner.train_state.micro_step = 1
        second_input = torch.ones(2, 4)
        with runner.train_context():
            runner.backward(runner.criterion(runner.model(second_input), target))

        second_grad = next(runner.model.parameters()).grad.detach().clone()
        second_gathered = [torch.zeros_like(second_grad) for _ in range(world_size)]
        dist.all_gather(second_gathered, second_grad)
        assert torch.allclose(second_gathered[0], second_gathered[1], atol=1e-6, rtol=1e-5)
    finally:
        runner.close()


def _torch_runner_reduce_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"logging.enabled": False, "dist.backend": "gloo", "ckpt": {"backend": "file"}})
    try:
        reduced = runner.reduce(torch.tensor(float(rank + 1)))
        assert reduced.item() == pytest.approx(1.5)
    finally:
        runner.close()


def _torch_runner_weighted_loss_logging_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"logging.enabled": False, "dist.backend": "gloo"})
    try:
        reduced = runner.reduce_loss_for_logging(torch.tensor(float(1 + rank * 2)), [1, 3][rank])
        assert reduced is not None
        assert reduced.item() == pytest.approx(2.5)
    finally:
        runner.close()


def _torch_runner_nonfinite_skip_sync_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner(
        {
            "logging.enabled": False,
            "dist.backend": "gloo",
            "ckpt": {"backend": "file"},
            "skip_nonfinite_grad": True,
        }
    )
    try:
        assert runner.model is not None
        parameters = list(runner.model.parameters())
        initial_parameters = [parameter.detach().clone() for parameter in parameters]
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        if rank == world_size - 1:
            parameters[0].grad.fill_(float("inf"))

        assert runner.optimizer_step() is False
        assert runner.train_state.global_step == 0
        for parameter, initial in zip(parameters, initial_parameters):
            torch.testing.assert_close(parameter, initial)
    finally:
        runner.close()


def _torch_runner_init_timeout_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner(
        {
            "logging.enabled": False,
            "dist": {"backend": "gloo", "init_timeout_seconds": 42},
            "ckpt": {"backend": "file"},
        }
    )
    try:
        assert dist.is_initialized()
        assert isinstance(runner.model, nn.parallel.DistributedDataParallel)
    finally:
        runner.close()


def _torch_runner_preinitialized_pg_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    with process_group("gloo", rank, world_size), pytest.raises(RuntimeError, match="already initialized"):
        DistributedTinyTorchRunner({"logging.enabled": False, "dist.backend": "gloo", "ckpt": {"backend": "file"}})


# ---------------------------------------------------------------------------
# Construction & Restore
# ---------------------------------------------------------------------------


class TestTorchRunnerBootstrap:

    def test_runner_accepts_explicit_training_components(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        assert runner.model is not None
        assert runner.criterion is not None
        assert runner.optimizer is not None
        assert runner.optimizer_container is not None

    def test_fault_tolerance_requires_torchft_package(self) -> None:
        if importlib.util.find_spec("torchft") is not None:
            pytest.skip("torchft is installed")
        with pytest.raises(ImportError, match="torchft"):
            TinyTorchRunner({"logging.enabled": False, "ft": {"enabled": True}})

    def test_declares_torchft_runtime_supported(self) -> None:
        assert TorchRunner._supports_torchft_runtime is True

    def test_train_uses_stable_split_order(self) -> None:
        runner = TrainDispatchRunner(
            {
                "logging.enabled": False,
                "steps": 1,
                "train_splits": ["b", "a"],
                "evaluate_splits": ["v2", "v1"],
            }
        )
        try:
            runner.train(train_splits=["b", "a", "b"], evaluate_splits=["v2", "v1", "v2"])

            assert runner.dispatched_train_splits == ["a", "b"]
            assert runner.dispatched_evaluate_splits == ["v1", "v2"]
        finally:
            runner.close()

    def test_train_rejects_unknown_splits(self) -> None:
        runner = TrainDispatchRunner(
            {
                "logging.enabled": False,
                "steps": 1,
                "train_splits": ["a"],
                "evaluate_splits": ["v1"],
            }
        )
        try:
            with pytest.raises(ValueError, match="unknown training split"):
                runner.train(train_splits=["a", "missing"], evaluate_splits=[])
            with pytest.raises(ValueError, match="unknown evaluation split"):
                runner.train(train_splits=["a"], evaluate_splits=["v1", "missing"])

            assert runner.dispatched_train_splits is None
            assert runner.dispatched_evaluate_splits is None
        finally:
            runner.close()

    def test_evaluate_rejects_unknown_requested_split(self) -> None:
        runner = TrainDispatchRunner(
            {
                "logging.enabled": False,
                "steps": 1,
                "train_splits": ["a"],
                "evaluate_splits": ["v1"],
            }
        )
        try:
            with pytest.raises(ValueError, match="unknown evaluation split"):
                runner.evaluate(evaluate_splits=["missing"])
        finally:
            runner.close()

    def test_close_timeout_does_not_destroy_process_group(self) -> None:
        class ToggleCloseCheckpointManager:
            drained = False

            def close(self, timeout=None):
                del timeout
                return self.drained

        runner = TinyTorchRunner({"logging.enabled": False})
        manager = ToggleCloseCheckpointManager()
        destroyed: list[bool] = []
        try:
            runner.checkpoint_manager = manager  # type: ignore[assignment]
            runner.destroy_process_group = lambda: destroyed.append(True)  # type: ignore[method-assign]

            with pytest.warns(RuntimeWarning, match="timed out while draining async checkpoints"):
                assert runner.close(timeout=0.0) is False

            assert destroyed == []

            manager.drained = True
            assert runner.close(timeout=1.0) is True
            assert destroyed == [True]
        finally:
            if not manager.drained:
                manager.drained = True
                runner.close(timeout=1.0)

    def test_runner_builds_optimizer_from_model_parameters(self) -> None:
        class AutoOptimRunner(TorchRunner):
            def init_distributed(self) -> None:
                return

            model = nn.Linear(4, 2)

        runner = AutoOptimRunner({"logging.enabled": False, "optim": {"type": "sgd", "lr": 0.1}})
        try:
            assert runner.optimizer is not None
        finally:
            runner.close()

    def test_optimizer_config_applies_parameter_groups(self) -> None:
        class AutoOptimRunner(TorchRunner):
            def init_distributed(self) -> None:
                return

            model = nn.Sequential(
                OrderedDict(
                    (
                        ("stem", nn.Linear(4, 4)),
                        ("head", nn.Linear(4, 2)),
                    )
                )
            )

        runner = AutoOptimRunner(
            {
                "logging.enabled": False,
                "optim": {
                    "type": "sgd",
                    "lr": 1.0,
                    "weight_decay": 0.1,
                    "param_groups": [
                        {
                            "pattern": r"^head\.",
                            "lr_multiplier": 0.25,
                            "weight_decay_multiplier": 0.0,
                        }
                    ],
                },
            }
        )
        try:
            assert runner.optimizer is not None
            assert len(runner.optimizer.param_groups) == 2
            explicit_group, default_group = runner.optimizer.param_groups
            assert explicit_group["lr"] == pytest.approx(0.25)
            assert explicit_group["weight_decay"] == pytest.approx(0.0)
            assert default_group["lr"] == pytest.approx(1.0)
            assert default_group["weight_decay"] == pytest.approx(0.1)
            assert set(explicit_group["params"]) == set(runner.model.head.parameters())
            assert set(default_group["params"]) == set(runner.model.stem.parameters())
        finally:
            runner.close()

    def test_optimizer_config_warns_when_parameter_group_matches_nothing(self) -> None:
        class AutoOptimRunner(TorchRunner):
            def init_distributed(self) -> None:
                return

            model = nn.Linear(4, 2)

        with pytest.warns(RuntimeWarning, match="matched no parameters"):
            runner = AutoOptimRunner(
                {
                    "logging.enabled": False,
                    "optim": {
                        "type": "sgd",
                        "lr": 0.1,
                        "param_groups": [{"pattern": "missing"}],
                    },
                }
            )
        try:
            assert runner.optimizer is not None
            assert len(runner.optimizer.param_groups) == 1
        finally:
            runner.close()

    def test_builds_stateful_dataloaders_by_default(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "dataloader": {"batch_size": 2}})
        try:
            runner.datasets["train"] = list(range(8))
            runner.build_dataloaders()

            assert isinstance(runner.dataloaders["train"], StatefulDataLoader)
        finally:
            runner.close()

    def test_dataloader_config_forwards_sampler_and_collate_fn(self) -> None:
        class ReverseSampler:
            def __iter__(self):
                return iter((3, 2, 1, 0))

            def __len__(self):
                return 4

        def collate_fn(batch):
            return tuple(batch)

        runner = TinyTorchRunner({"logging.enabled": False})
        try:
            runner.config.dataloader.batch_size = 2
            runner.config.dataloader.sampler = ReverseSampler()
            runner.config.dataloader.collate_fn = collate_fn
            runner.datasets["train"] = list(range(4))
            runner.build_dataloaders()

            assert list(runner.dataloaders["train"]) == [(3, 2), (1, 0)]
        finally:
            runner.close()

    def test_dataloader_config_forwards_batch_sampler(self) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "dataloader": {
                    "batch_size": 99,
                    "batch_sampler": [[2, 0], [3, 1]],
                    "drop_last": True,
                },
            }
        )
        try:
            runner.datasets["train"] = list(range(4))
            runner.build_dataloaders()

            assert [batch.tolist() for batch in runner.dataloaders["train"]] == [[2, 0], [3, 1]]
        finally:
            runner.close()

    def test_deterministic_dataloader_binds_seed_controls(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "deterministic": True, "dataloader": {"batch_size": 2}})
        try:
            runner.datasets["train"] = list(range(4))
            runner.build_dataloaders()

            loader = runner.dataloaders["train"]
            assert getattr(loader, "worker_init_fn") is not None
            assert isinstance(getattr(loader, "generator"), torch.Generator)
        finally:
            runner.close()

    def test_dataloader_reports_invalid_worker_options(self) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "dataloader": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "persistent_workers": True,
                },
            }
        )
        try:
            runner.datasets["train"] = list(range(8))
            with pytest.raises(ValueError, match="persistent_workers"):
                runner.build_dataloaders()
        finally:
            runner.close()

    def test_runner_restores_from_configured_checkpoint(self) -> None:
        runner = RecordingRestoreRunner({"logging.enabled": False, "checkpoint": "checkpoint-latest"})
        assert runner.restore_calls == [("checkpoint", "checkpoint-latest")]
        runner.close()

    def test_runner_loads_configured_pretrained_weights(self) -> None:
        runner = RecordingRestoreRunner({"logging.enabled": False, "pretrained": "checkpoint-best"})
        assert runner.restore_calls == [("pretrained", "checkpoint-best")]
        runner.close()

    def test_all_reduce_group_uses_fault_tolerance_replicate_group(self) -> None:
        class AllReduceRunner(TinyTorchRunner):
            def materialize_model(self) -> None:
                return

        class TorchFTRuntime:
            replicate_process_group = "ft_group"

            def close(self) -> None:
                return

        runner = AllReduceRunner({"logging.enabled": False})
        try:
            runner.fault_tolerance = TorchFTRuntime()
            assert runner.all_reduce_group() == "ft_group"
        finally:
            runner.fault_tolerance = None
            runner.close()

    def test_evaluate_runs_under_inference_context(self) -> None:
        runner = ContextRecordingRunner({"logging.enabled": False})
        try:
            runner.evaluate_step((torch.ones(2, 4), torch.zeros(2, 1)))
            assert runner.infer_context_entries == 1
        finally:
            runner.close()

    def test_infer_runs_under_inference_context(self) -> None:
        runner = ContextRecordingRunner({"logging.enabled": False})
        try:
            runner.infer_step(torch.ones(2, 4))
            assert runner.infer_context_entries == 1
        finally:
            runner.close()

    def test_auto_restore_prefers_checkpoint_over_pretrained(self) -> None:
        runner = RecordingRestoreRunner(
            {"logging.enabled": False, "checkpoint": "ckpt-latest", "pretrained": "ckpt-best"}
        )
        assert runner.restore_calls == [("checkpoint", "ckpt-latest")]
        runner.close()

    def test_auto_restore_prefers_resume_latest_over_pretrained(self) -> None:
        runner = RecordingRestoreRunner({"logging.enabled": False, "resume": True, "pretrained": "ckpt-best"})
        assert runner.restore_calls == [("checkpoint", os.path.join(runner.workspace.checkpoint_dir, "latest.pth"))]
        runner.close()

    def test_runner_warns_when_restore_sources_compete(self) -> None:
        with pytest.warns(
            RuntimeWarning,
            match="precedence is `checkpoint` > `resume` > `pretrained`",
        ):
            runner = RecordingRestoreRunner(
                {"logging.enabled": False, "checkpoint": "ckpt-latest", "resume": True, "pretrained": "ckpt-best"}
            )
        assert runner.restore_calls == [("checkpoint", "ckpt-latest")]
        runner.close()

    def test_auto_resume_uses_latest_checkpoint_path(self) -> None:
        runner = RecordingRestoreRunner({"logging.enabled": False, "resume": True})
        assert runner.restore_calls == [("checkpoint", os.path.join(runner.workspace.checkpoint_dir, "latest.pth"))]
        runner.close()

    def test_load_checkpoint_updates_config_source(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"logging.enabled": False})
        checkpoint_path = tmp_path / "checkpoint-latest.pth"
        try:
            torch.save(source.state_dict(), checkpoint_path)
        finally:
            source.close()

        runner = TinyTorchRunner({"logging.enabled": False})
        try:
            runner.load_checkpoint(checkpoint_path)
            assert runner.config.checkpoint == str(checkpoint_path)
        finally:
            runner.close()

    def test_load_pretrained_updates_config_source(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"logging.enabled": False})
        checkpoint_path = tmp_path / "checkpoint-best.pth"
        try:
            assert source.model is not None
            torch.save({"model": source.unwrap(source.model).state_dict()}, checkpoint_path)
        finally:
            source.close()

        runner = TinyTorchRunner({"logging.enabled": False})
        try:
            runner.load_pretrained(checkpoint_path)
            assert runner.config.pretrained == str(checkpoint_path)
        finally:
            runner.close()

    def test_from_pretrained_accepts_mapping(self) -> None:
        source = TinyTorchRunner({"logging.enabled": False})
        try:
            assert source.model is not None
            checkpoint = {"model": source.unwrap(source.model).state_dict()}
        finally:
            source.close()

        runner = TinyTorchRunner.from_pretrained({"logging.enabled": False}, checkpoint)
        try:
            assert runner.config.pretrained is None
        finally:
            runner.close()

    def test_ddp_training_uses_compiled_model(self) -> None:
        require_gloo()
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch build.")
        run_distributed(_ddp_compile_wrap_worker, world_size=2)

    def test_gradient_accumulation_uses_ddp_no_sync(self) -> None:
        require_gloo()
        run_distributed(_ddp_no_sync_worker, world_size=2)


class TestTorchRunnerCheckpointInterop:

    def test_tensorboard_uses_configured_writer_options(self, tmp_path: Path, monkeypatch) -> None:
        writer_calls: list[dict[str, object]] = []

        class RecordingSummaryWriter:
            def __init__(self, *args, **kwargs) -> None:
                del args
                writer_calls.append(kwargs)

            def add_scalar(self, *args, **kwargs) -> None:
                return

            def flush(self) -> None:
                return

            def close(self) -> None:
                return

        import torch.utils.tensorboard.writer as tensorboard_writer  # pylint: disable=import-outside-toplevel

        monkeypatch.setattr(tensorboard_writer, "SummaryWriter", RecordingSummaryWriter)
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "tensorboard": {
                    "enabled": True,
                    "comment": "debug",
                    "max_queue": 3,
                    "flush_secs": 7,
                    "filename_suffix": ".tb",
                },
            }
        )
        try:
            assert len(writer_calls) == 1
            kwargs = writer_calls[0]
            assert str(kwargs["log_dir"]).startswith(str(tmp_path))
            assert kwargs["comment"] == "debug"
            assert kwargs["max_queue"] == 3
            assert kwargs["flush_secs"] == 7
            assert kwargs["filename_suffix"] == ".tb"
        finally:
            runner.close()

    def test_dcp_backend_uses_plain_checkpoint_manager_by_default(self, tmp_path: Path) -> None:
        runner = DcpConfigTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "ckpt": {"backend": "dcp", "async_mode": "disabled"},
            }
        )
        try:
            assert isinstance(runner.checkpoint_manager, TorchDistributedCheckpointManager)
            assert not isinstance(runner.checkpoint_manager, TorchFTCheckpointManager)
        finally:
            runner.close()

    def test_dcp_backend_uses_torchft_checkpoint_manager_for_dataloader_checkpoints(self, tmp_path: Path) -> None:
        runner = DcpConfigTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "ckpt": {
                    "backend": "dcp",
                    "async_mode": "disabled",
                    "dataloader_checkpoint": {"enabled": True},
                },
            }
        )
        try:
            assert isinstance(runner.checkpoint_manager, TorchFTCheckpointManager)
        finally:
            runner.close()

    def test_read_config_accepts_dcp_directory(self, tmp_path: Path) -> None:
        runner = DcpConfigTorchRunner(
            {
                "logging.enabled": False,
                "name": "dcp-config-test",
                "workspace.root": str(tmp_path),
                "ckpt": {"backend": "dcp", "async_mode": "disabled", "interval": 1},
            }
        )
        try:
            runner.train_state.global_step = 1
            runner.save_checkpoint(force=True)
            config = DcpConfigTorchRunner.read_config(Path(runner.workspace.checkpoint_dir) / "latest")
        finally:
            runner.close()

        assert config["name"] == "dcp-config-test"
        assert config.get("ckpt.backend") == "dcp"

    def test_from_checkpoint_accepts_in_memory_checkpoint(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        checkpoint = runner.state_dict()
        runner.close()

        restored = TinyTorchRunner.from_checkpoint(checkpoint)
        try:
            assert isinstance(restored, TinyTorchRunner)
        finally:
            restored.close()

    def test_from_checkpoint_uses_requested_path_over_auto_restore(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"logging.enabled": False})
        checkpoint_path = tmp_path / "torch-runner.pth"
        try:
            checkpoint = dict(source.state_dict())
            checkpoint["runner"]["resume"] = True
            checkpoint["runner"]["pretrained"] = "stale-pretrained"
            torch.save(checkpoint, checkpoint_path)
        finally:
            source.close()

        restored = TinyTorchRunner.from_checkpoint(checkpoint_path)
        try:
            assert restored.config.checkpoint == str(checkpoint_path)
            assert restored.config.resume is False
            assert restored.config.pretrained is None
        finally:
            restored.close()

    def test_load_checkpoint_restores_stateful_dataloader_progress(self) -> None:
        source = StatefulDatasetTorchRunner(
            {"logging.enabled": False, "dataloader": {"batch_size": 2, "shuffle": False}}
        )
        try:
            assert next(iter(source.dataloaders["train"])).tolist() == [0, 1]
            checkpoint = source.state_dict()
        finally:
            source.close()

        restored = StatefulDatasetTorchRunner(
            {"logging.enabled": False, "dataloader": {"batch_size": 2, "shuffle": False}}
        )
        try:
            restored.load_checkpoint(checkpoint)
            remaining_batches = [batch.tolist() for batch in restored.dataloaders["train"]]
            assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
        finally:
            restored.close()

    def test_from_checkpoint_path_restores_stateful_dataloader_progress(self, tmp_path: Path) -> None:
        source = StatefulDatasetTorchRunner(
            {"logging.enabled": False, "dataloader": {"batch_size": 2, "shuffle": False}}
        )
        checkpoint_path = tmp_path / "torch-runner-stateful.pth"
        try:
            assert next(iter(source.dataloaders["train"])).tolist() == [0, 1]
            torch.save(source.state_dict(), checkpoint_path)
        finally:
            source.close()

        restored = StatefulDatasetTorchRunner.from_checkpoint(checkpoint_path)
        try:
            remaining_batches = [batch.tolist() for batch in restored.dataloaders["train"]]
            assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
        finally:
            restored.close()

    def test_auto_resume_restores_stateful_dataloader_progress(self, tmp_path: Path) -> None:
        config = {
            "logging.enabled": False,
            "workspace.root": str(tmp_path),
            "dataloader": {"batch_size": 2, "shuffle": False},
            "ckpt": {"async_mode": "disabled", "interval": 1},
        }
        source = StatefulDatasetTorchRunner(config)
        try:
            assert next(iter(source.dataloaders["train"])).tolist() == [0, 1]
            source.save_checkpoint(force=True)
        finally:
            source.close()

        restored_config = dict(config)
        restored_config["resume"] = True
        restored = StatefulDatasetTorchRunner(restored_config)
        try:
            remaining_batches = [batch.tolist() for batch in restored.dataloaders["train"]]
            assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
            assert restored.config.checkpoint == os.path.join(restored.workspace.checkpoint_dir, "latest.pth")
        finally:
            restored.close()


class TestTorchRunnerProfiling:

    @staticmethod
    def _profiling_runner(tmp_path: Path) -> TinyTorchRunner:
        return TinyTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "profiling": {"enabled": True, "wait": 1, "warmup": 1, "active": 1, "trace_dir": "trace-output"},
            }
        )

    def test_initializes_profiler_when_enabled(self, tmp_path: Path) -> None:
        runner = self._profiling_runner(tmp_path)
        try:
            trace_dir = Path(runner.workspace.dir) / "trace-output" / runner.timestamp / f"rank-{runner.rank:05d}"
            assert trace_dir.is_dir()
            assert runner._profiler is not None
            assert runner._profiler.step_num == 0
        finally:
            runner.close()

    def test_closes_profiler(self, tmp_path: Path) -> None:
        runner = self._profiling_runner(tmp_path)
        runner.close()

        assert runner._profiler is None
        assert runner._profiler_context is None

    def test_profiler_close_is_idempotent(self, tmp_path: Path) -> None:
        runner = self._profiling_runner(tmp_path)
        runner.close()
        runner.close()

        assert runner._profiler is None
        assert runner._profiler_context is None

    def test_profiler_close_reports_artifacts(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "profiling": {"enabled": True, "wait": 0, "warmup": 0, "active": 1, "trace_dir": "trace-output"},
            }
        )
        trace_dir = Path(runner.workspace.dir) / "trace-output" / runner.timestamp / f"rank-{runner.rank:05d}"
        try:
            runner.model(torch.ones(1, 4, device=runner.device))
            runner._step_profiler()
        finally:
            runner.close()

        output = capsys.readouterr().out
        assert "profiler artifacts:" in output
        assert f"trace_dir={trace_dir}" in output
        assert "operator_table=" in output


class TestTorchRunnerDistributedRuntime:

    def test_init_distributed_accepts_configured_timeout(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_init_timeout_worker, world_size=2)

    def test_init_distributed_rejects_preinitialized_process_group(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_preinitialized_pg_worker, world_size=2)

    def test_nonfinite_skip_decision_is_collective(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_nonfinite_skip_sync_worker, world_size=2)


# ---------------------------------------------------------------------------
# Runtime Mechanics
# ---------------------------------------------------------------------------


class TestTorchRunnerOptimization:

    def test_optimizer_step_requires_optimizer(self) -> None:
        runner = NoOptimizerTorchRunner({"logging.enabled": False})
        try:
            assert runner.optimizer is None
            with pytest.raises(ValueError, match="no optimizer"):
                runner.optimizer_step()
            assert runner.train_state.global_step == 0
        finally:
            runner.close()

    def test_runner_owned_fp16_requires_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="fp16 precision requires a CUDA device"):
            TinyTorchRunner({"logging.enabled": False, "precision": "fp16"})

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 grad scaler requires CUDA")
    def test_runner_owned_fp16_binds_grad_scaler(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "precision": "fp16"})
        try:
            assert runner.grad_scaler is not None
        finally:
            runner.close()

    def test_step_skips_optimizer_update_on_nonfinite_grad(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "skip_nonfinite_grad": True})
        assert runner.optimizer is not None
        assert runner.model is not None

        initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
        try:
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)
            next(runner.model.parameters()).grad.fill_(float("inf"))

            runner.step()

            for parameter, initial in zip(runner.model.parameters(), initial_parameters):
                torch.testing.assert_close(parameter, initial)
            assert runner.train_state.global_step == 0
        finally:
            runner.close()

    def test_optimizer_step_updates_ema_after_successful_step(self) -> None:
        class UpdatingEma(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.updated_model: nn.Module | None = None

            def update(self, model: nn.Module) -> None:
                self.updated_model = model

        runner = TinyTorchRunner({"logging.enabled": False})
        ema = UpdatingEma()
        runner.ema = ema
        try:
            assert runner.model is not None
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True
            assert ema.updated_model is runner.unwrap(runner.model)
        finally:
            runner.close()

    def test_optimizer_step_allows_eval_only_ema_without_update(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        runner.ema = nn.Identity()
        try:
            assert runner.model is not None
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True

            assert runner.train_state.global_step == 1
        finally:
            runner.close()

    def test_optimizer_step_rejects_noncallable_ema_update_before_mutation(self) -> None:
        class BadEma(nn.Module):
            update = "not callable"

        runner = TinyTorchRunner({"logging.enabled": False})
        runner.ema = BadEma()
        try:
            assert runner.model is not None
            initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            with pytest.raises(TypeError, match="not callable"):
                runner.optimizer_step()

            assert runner.train_state.global_step == 0
            for parameter, initial in zip(runner.model.parameters(), initial_parameters):
                torch.testing.assert_close(parameter, initial)
        finally:
            runner.close()

    def test_fp16_optimizer_step_uses_grad_scaler(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        scaler = _FakeGradScaler()
        runner.grad_scaler = scaler
        try:
            assert runner.optimizer is not None
            assert runner.model is not None
            initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True

            assert scaler.unscale_calls == [runner.optimizer]
            assert scaler.step_calls == [runner.optimizer]
            assert scaler.update_calls == 1
            for parameter, initial in zip(runner.model.parameters(), initial_parameters):
                assert not torch.equal(parameter, initial)
        finally:
            runner.close()

    def test_optimizer_step_updates_fake_grad_scaler_on_skip(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        scaler = _FakeGradScaler()
        runner.grad_scaler = scaler
        try:
            assert runner.optimizer is not None
            assert runner.model is not None
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)
            next(runner.model.parameters()).grad.fill_(float("inf"))

            assert runner.optimizer_step() is False

            assert scaler.unscale_calls == [runner.optimizer]
            assert scaler.step_calls == []
            assert scaler.update_calls == 1
            assert runner.train_state.global_step == 0
        finally:
            runner.close()

    def test_resume_without_grad_scaler_keeps_current_scaler(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        state = runner.state_dict()["state"]
        scaler = _FakeGradScaler()
        runner.grad_scaler = scaler
        try:
            runner.load_state_dict({"state": state})

            assert scaler.loaded_state is None
        finally:
            runner.close()

    def test_optimizer_step_records_grad_norm_when_clipping(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "max_grad_norm": 1.0})
        try:
            assert runner.model is not None
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True
            result = runner.get_step_result()
            assert "grad_norm" in result
            assert result["grad_norm"] > 0
        finally:
            runner.close()

    def test_steps_standard_pytorch_scheduler_after_optimizer_step(self) -> None:
        runner = TinyStepLRTorchRunner({"logging.enabled": False})
        try:
            assert runner.scheduler is not None
            assert runner.model is not None
            initial_lr = runner.scheduler.get_last_lr()[0]

            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            runner.step()

            assert runner.scheduler.get_last_lr()[0] < initial_lr
        finally:
            runner.close()

    def test_reduces_train_timeout_once_after_first_successful_step(self) -> None:
        runner = TimeoutRecordingRunner({"logging.enabled": False, "dist": {"train_timeout_seconds": 17}})
        try:
            assert runner.optimizer_step() is True
            assert runner.train_state.global_step == 1
            assert runner.timeout_calls == [timedelta(seconds=17)]

            assert runner.optimizer_step() is True
            assert runner.train_state.global_step == 2
            assert runner.timeout_calls == [timedelta(seconds=17)]
        finally:
            runner.close()

    def test_steps_profiler_after_successful_optimizer_step(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})

        class RecordingProfiler:
            def __init__(self) -> None:
                self.steps = 0

            def step(self) -> None:
                self.steps += 1

        profiler = RecordingProfiler()
        runner._profiler = profiler

        try:
            assert runner.optimizer_step() is True
            assert profiler.steps == 1
        finally:
            runner.close()

    def test_waits_for_checkpoint_staging_before_optimizer_mutation(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})

        class RecordingCheckpointManager:
            def __init__(self) -> None:
                self.weight_before_step: torch.Tensor | None = None

            def maybe_wait_for_staging(self) -> bool:
                assert runner.model is not None
                self.weight_before_step = runner.model.weight.detach().clone()
                return True

            def close(self, timeout=None) -> bool:
                del timeout
                return True

        manager = RecordingCheckpointManager()
        runner.checkpoint_manager = manager  # type: ignore[assignment]
        try:
            assert runner.model is not None
            initial_weight = runner.model.weight.detach().clone()
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True

            assert manager.weight_before_step is not None
            torch.testing.assert_close(manager.weight_before_step, initial_weight)
            assert not torch.allclose(runner.model.weight.detach(), initial_weight)
        finally:
            runner.close()

    def test_collects_gc_on_optimizer_step_interval(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "gc": {"interval": 2, "disable_automatic": False}})
        try:
            assert runner.optimizer_step() is True
            assert runner.optimizer_step() is True
            assert runner.optimizer_step() is True
            assert runner.supervisor._gc_last_collection == {"train": 2}
        finally:
            runner.close()


class TestTorchRunnerScheduling:

    def test_steps_danling_scheduler_after_optimizer_step(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "sched": {"type": "linear", "total_steps": 8}})
        try:
            assert runner.scheduler is not None
            assert runner.model is not None
            initial_step_count = runner.scheduler._step_count

            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            runner.step()

            assert runner.scheduler._step_count == initial_step_count + 1
        finally:
            runner.close()

    def test_defaults_metric_scheduler_to_epoch_interval(self) -> None:
        runner = TinyPlateauTorchRunner({"logging.enabled": False})
        try:
            assert runner.scheduler_interval == "epoch"
            assert runner.scheduler is not None
            assert runner.model is not None
            initial_lr = runner.scheduler.get_last_lr()[0]

            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            runner.step()

            assert runner.scheduler.get_last_lr()[0] == pytest.approx(initial_lr)
        finally:
            runner.close()

    def test_rejects_step_interval_for_metric_scheduler(self) -> None:
        with pytest.raises(ValueError, match="metric-based schedulers require `sched.interval='epoch'`"):
            TinyPlateauTorchRunner({"logging.enabled": False, "sched": {"interval": "step"}})

    def test_steps_reduce_on_plateau_after_epoch_result(self, tmp_path: Path) -> None:
        runner = EpochSchedulerRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "epochs": 2,
                "train_splits": ["train"],
                "evaluate_splits": ["val"],
                "sched": {"type": "reduce_on_plateau", "patience": 0, "factor": 0.5},
                "val_losses": [1.0, 2.0],
            }
        )
        try:
            assert runner.optimizer is not None
            initial_lr = runner.optimizer.param_groups[0]["lr"]

            runner.train()

            assert runner.optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.5)
        finally:
            runner.close()

    def test_respects_epoch_interval_for_standard_pytorch_scheduler(self) -> None:
        runner = TinyStepLRTorchRunner({"logging.enabled": False, "sched": {"interval": "epoch"}})
        try:
            assert runner.scheduler is not None
            assert runner.model is not None
            initial_lr = runner.scheduler.get_last_lr()[0]

            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            runner.step()

            assert runner.scheduler.get_last_lr()[0] == pytest.approx(initial_lr)

            runner._step_epoch_scheduler({"train": {"loss": 1.0}})

            assert runner.scheduler.get_last_lr()[0] < initial_lr
        finally:
            runner.close()

    def test_steps_epoch_interval_scheduler_once_per_step_mode_train_round(self, tmp_path: Path) -> None:
        runner = TinyStepLRTorchRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 4, "sched": {"interval": "epoch"}}
        )
        try:
            runner.dataloaders["train"] = [
                (torch.ones(4), torch.zeros(2)),
                (torch.ones(4), torch.zeros(2)),
            ]

            runner.train_steps(train_splits=["train"], evaluate_splits=[])

            assert runner.scheduler is not None
            assert runner.scheduler.get_last_lr()[0] == pytest.approx(0.001)
        finally:
            runner.close()

    def test_raises_for_missing_scheduler_monitor(self) -> None:
        runner = TinyPlateauTorchRunner({"logging.enabled": False, "sched": {"monitor": "val.accuracy"}})
        try:
            with pytest.raises(ValueError, match="sched.monitor"):
                runner._step_epoch_scheduler({"val": {"loss": 1.0}})
        finally:
            runner.close()

    def test_replicated_dtensor_loss_can_be_logged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_runner_module, "TorchDTensor", _FakeDTensor)
        runner = TinyTorchRunner({"logging.enabled": False})
        local = torch.tensor(3.0)
        tensor = _FakeDTensor(
            local,
            [_FakePlacement(replicate=True)],
        )

        try:
            loss = runner.reduce_loss_for_logging(tensor, loss_n=1)
            assert loss is not None
            torch.testing.assert_close(loss, local.to(dtype=torch.float64))
        finally:
            runner.close()

    def test_partial_dtensor_loss_can_be_logged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_runner_module, "TorchDTensor", _FakeDTensor)
        monkeypatch.setattr(torch_runner_module, "TorchReplicate", _FakeReplicate)
        runner = TinyTorchRunner({"logging.enabled": False})
        local = torch.tensor(3.0)
        tensor = _FakeDTensor(local, [_FakePlacement(partial=True)])

        try:
            loss = runner.reduce_loss_for_logging(tensor, loss_n=1)
            assert loss is not None
            torch.testing.assert_close(loss, local.to(dtype=torch.float64))
        finally:
            runner.close()

    def test_sharded_dtensor_loss_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_runner_module, "TorchDTensor", _FakeDTensor)
        runner = TinyTorchRunner({"logging.enabled": False})
        tensor = _FakeDTensor(torch.tensor(3.0), [_FakePlacement()])

        try:
            with pytest.raises(ValueError, match="Cannot reduce DTensor"):
                runner.reduce_loss_for_logging(tensor, loss_n=1)
        finally:
            runner.close()

    def test_reduce_returns_world_mean(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_reduce_worker, world_size=2)

    def test_loss_logging_reports_weighted_mean(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_weighted_loss_logging_worker, world_size=2)


# ---------------------------------------------------------------------------
# Execution Loops
# ---------------------------------------------------------------------------


class TestTorchRunnerEpochExecution:

    def test_train_epoch_supports_unsized_loader(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 1024})
        try:
            runner.dataloaders["train"] = StreamingLoader(1.0, 2.0)
            result = runner.train_epoch("train")
            assert math.isfinite(result["loss"])
            assert runner.train_state.global_step == 2
        finally:
            runner.close()

    def test_variable_length_batches_use_token_weighted_loss(self) -> None:
        runner = WeightedLossRunner({"logging.enabled": False, "accum_steps": 2, "logging.interval": 0})
        runner.dataloaders["train"] = [
            (torch.ones((1, 1)), torch.zeros((1, 1))),
            (torch.full((3, 1), 3.0), torch.zeros((3, 1))),
        ]

        try:
            result = runner.train_epoch("train")
            assert runner.model is not None
            assert runner.unwrap(runner.model).weight.detach().item() == pytest.approx(-0.4)
            assert result["loss"] == pytest.approx(7.0)
            assert runner.train_state.global_step == 1
        finally:
            runner.close()

    def test_partial_accumulation_window_scales_by_window_size(self) -> None:
        runner = WeightedLossRunner({"logging.enabled": False, "accum_steps": 4, "logging.interval": 0})
        runner.dataloaders["train"] = [
            (torch.ones((1, 1)), torch.zeros((1, 1))),
            (torch.full((3, 1), 3.0), torch.zeros((3, 1))),
        ]

        try:
            runner.train_epoch("train")
            assert runner.model is not None
            assert runner.unwrap(runner.model).weight.detach().item() == pytest.approx(-0.4)
            assert runner.train_state.global_step == 1
            assert runner.train_state.micro_step == 4
        finally:
            runner.close()

    def test_loss_weighting_uses_distributed_normalizer(self) -> None:
        class GradientScaleRunner(WeightedLossRunner):
            def _loss_normalizer_sync_divisor(self) -> int:
                return 4

            def _reduce_loss_normalizer_total(self, local_total: float) -> float:
                assert local_total == pytest.approx(4.0)
                return 10.0

        runner = GradientScaleRunner({"logging.enabled": False, "accum_steps": 2})
        runner.dataloaders["train"] = [
            (torch.ones((1, 1)), torch.zeros((1, 1))),
            (torch.full((3, 1), 3.0), torch.zeros((3, 1))),
        ]

        try:
            runner.train_epoch("train")
            assert runner.model is not None
            assert runner.unwrap(runner.model).weight.detach().item() == pytest.approx(-1.24)
        finally:
            runner.close()

    def test_explicit_loss_normalizer_controls_loss_weighting(self) -> None:
        runner = WeightedLossRunner({"logging.enabled": False, "accum_steps": 2})
        runner.dataloaders["train"] = [
            {"input": torch.ones((1, 1)), "target": torch.zeros((1, 1)), "loss_normalizer": 9},
            {"input": torch.full((1, 1), 3.0), "target": torch.zeros((1, 1)), "loss_normalizer": 1},
        ]

        try:
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(1.8)
            assert runner.model is not None
            assert runner.unwrap(runner.model).weight.detach().item() == pytest.approx(0.64)
        finally:
            runner.close()

    def test_mean_loss_uses_target_size_as_normalizer(self) -> None:
        runner = WeightedLossRunner({"logging.enabled": False, "accum_steps": 2})
        runner.dataloaders["train"] = [
            (torch.ones((1, 1)), torch.zeros((1, 1))),
            (torch.full((3, 1), 3.0), torch.zeros((3, 1))),
        ]

        try:
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(7.0)
        finally:
            runner.close()

    def test_attention_mask_controls_loss_weighting_without_targets(self) -> None:
        runner = MaskWeightedLossRunner({"logging.enabled": False, "accum_steps": 2})
        runner.dataloaders["train"] = [
            {
                "input": {
                    "tokens": torch.ones((1, 3), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }
            },
            {
                "input": {
                    "tokens": torch.full((1, 3), 3, dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 0, 0]], dtype=torch.long),
                }
            },
        ]

        try:
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(3.0)
        finally:
            runner.close()

    def test_custom_batch_schema_uses_unweighted_loss(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False})
        runner.dataloaders["train"] = [1.0, 9.0]

        try:
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(5.0)
        finally:
            runner.close()

    def test_train_epoch_skips_peak_memory_sampling_without_interval_logs(self) -> None:
        runner = TelemetryRunner({"logging.enabled": False, "logging.interval": 0})
        runner.dataloaders["train"] = [
            {
                "input": {
                    "input_ids": torch.ones((2, 3), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
                },
                "target": torch.zeros((2, 1)),
            },
            {
                "input": {
                    "input_ids": torch.ones((2, 3), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.long),
                },
                "target": torch.zeros((2, 1)),
            },
        ]

        try:
            result = runner.train_epoch("train")
            assert result["time"] > 0
            assert result["samples_per_s"] > 0
            assert "mem_alloc_mb" not in result
            assert "mem_reserved_mb" not in result
            assert "flops_per_s" not in result
            assert "mfu" not in result
        finally:
            runner.close()

    def test_train_epoch_reports_nested_tensor_tokens_per_s(self) -> None:
        runner = TelemetryRunner({"logging.enabled": False, "logging.interval": 0})
        runner.dataloaders["train"] = [
            {
                "input": {
                    "text": NestedTensor(
                        [torch.ones(3, dtype=torch.long), torch.ones(2, dtype=torch.long)],
                        batch_first=True,
                    )
                },
                "target": torch.zeros((2, 1)),
            },
            {
                "input": {
                    "text": NestedTensor(
                        [torch.ones(1, dtype=torch.long), torch.ones(4, dtype=torch.long)],
                        batch_first=True,
                    )
                },
                "target": torch.zeros((2, 1)),
            },
        ]

        try:
            result = runner.train_epoch("train")
            assert result["time"] > 0
            assert result["tokens_per_s"] > 0
            assert "samples_per_s" not in result
            assert "mem_alloc_mb" not in result
            assert "mem_reserved_mb" not in result
        finally:
            runner.close()

    def test_repeated_mode_assignment_does_not_retoggle_models(self) -> None:
        class TrackingModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[bool] = []

            def train(self, mode: bool = True):
                self.calls.append(bool(mode))
                return super().train(mode)

        runner = TinyTorchRunner({"logging.enabled": False})
        model = TrackingModule()
        ema = TrackingModule()
        runner.model = model
        runner.ema = ema

        try:
            runner.mode = "train"
            runner.mode = "evaluate"
            runner.mode = "evaluate"
            runner.mode = "train"

            assert model.calls == [False, True]
            assert ema.calls == [False, True]
        finally:
            runner.close()

    def test_evaluate_epoch_supports_unsized_loader(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 1024})
        try:
            runner.dataloaders["val"] = StreamingLoader(1.0, 2.0)
            result = runner.evaluate_epoch("val")
            assert math.isfinite(result["loss"])
        finally:
            runner.close()

    def test_evaluate_epoch_collects_gc_on_iteration_interval(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "gc": {"interval": 2, "disable_automatic": False}})
        try:
            runner.dataloaders["val"] = [1.0, 2.0, 3.0]
            runner.evaluate_epoch("val")
            assert runner.supervisor._gc_last_collection == {"evaluate:val": 2}
        finally:
            runner.close()


class TestTorchRunnerLoopResultStability:

    def test_train_epoch_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 2})
        try:
            runner.dataloaders["train"] = [1.0, 2.0, 3.0]
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(2.0)
        finally:
            runner.close()

    def test_evaluate_epoch_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 2})
        try:
            runner.dataloaders["val"] = [1.0, 2.0, 3.0]
            result = runner.evaluate_epoch("val")
            assert result["loss"] == pytest.approx(2.0)
        finally:
            runner.close()


class TestTorchRunnerStepExecution:

    def test_train_steps_result_is_independent_of_log_interval(self, tmp_path: Path) -> None:
        runner = StreamingEpochRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 3, "logging.interval": 2}
        )
        try:
            runner.dataloaders["train"] = [1.0, 2.0, 3.0]
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            assert runner.results[runner.train_state.global_step]["train"]["loss"] == pytest.approx(2.0)
        finally:
            runner.close()

    def test_train_steps_writes_full_latest_and_model_checkpoint(self, tmp_path: Path) -> None:
        runner = StreamingEpochRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "steps": 1,
                "ckpt.async_mode": "disabled",
                "ckpt.export_dtype": "fp16",
            }
        )
        try:
            runner.model = nn.Linear(1, 1)
            runner.dataloaders["train"] = [1.0]
            runner.train_steps(train_splits=["train"], evaluate_splits=[])

            checkpoint_dir = Path(runner.workspace.checkpoint_dir)
            latest_payload = torch.load(checkpoint_dir / "latest.pth", map_location="cpu", weights_only=False)
            assert "runner" in latest_payload
            assert "optimizer" in latest_payload
            assert latest_payload["model"]["weight"].dtype == torch.float32

            model_payload = torch.load(checkpoint_dir / "model.pth", map_location="cpu", weights_only=False)
            assert list(model_payload) == ["model"]
            assert model_payload["model"]["weight"].dtype == torch.float16
        finally:
            runner.close()

    def test_train_steps_logs_once_per_optimizer_step_under_accumulation(self, tmp_path: Path) -> None:
        runner = RecordingStepLogRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "steps": 2,
                "accum_steps": 2,
                "logging.interval": 2,
            }
        )
        runner.dataloaders["train"] = [1.0, 2.0, 3.0, 4.0]

        try:
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            assert [(iteration, length) for _, iteration, length, _ in runner.step_log_calls] == [(2, 2)]
        finally:
            runner.close()

    def test_train_steps_shares_budget_across_splits(self, tmp_path: Path) -> None:
        runner = StreamingEpochRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 2, "logging.interval": 0}
        )
        try:
            runner.dataloaders["a"] = [1.0, 2.0, 3.0]
            runner.dataloaders["b"] = [10.0, 20.0, 30.0]

            runner.train_steps(train_splits=["a", "b"], evaluate_splits=[])

            result = runner.results[runner.train_state.global_step]
            assert runner.train_state.global_step == 2
            assert result["a"]["loss"] == pytest.approx(1.0)
            assert result["b"]["loss"] == pytest.approx(10.0)
        finally:
            runner.close()

    def test_train_steps_rolls_loader_epoch_after_exhaustion(self, tmp_path: Path) -> None:
        class EpochAwareLoader:
            batch_sampler = None

            def __init__(self) -> None:
                self.epoch = 0
                self.sampler = self
                self.epochs: list[int] = []

            def set_epoch(self, epoch: int) -> None:
                self.epoch = int(epoch)
                self.epochs.append(self.epoch)

            def __iter__(self):
                base = self.epoch * 10
                return iter([float(base + 1), float(base + 2), float(base + 3)])

            def __len__(self) -> int:
                return 3

        runner = StreamingEpochRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 4, "logging.interval": 0}
        )
        try:
            loader = EpochAwareLoader()
            runner.dataloaders["train"] = loader
            runner.train_steps(train_splits=["train"], evaluate_splits=[])

            result = runner.results[runner.train_state.global_step]["train"]
            assert runner.train_state.global_step == 4
            assert loader.epochs == [0, 1]
            assert result["loss"] == pytest.approx(11.0)
        finally:
            runner.close()

    def test_train_steps_uses_monotonic_progress_across_loader_rollover(self, tmp_path: Path) -> None:
        runner = RecordingStepLogRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 3, "logging.interval": 1}
        )
        runner.dataloaders["train"] = StreamingLoader(1.0, 2.0)

        try:
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            assert [(iteration, length) for _, iteration, length, _ in runner.step_log_calls] == [
                (1, 3),
                (2, 3),
                (3, 3),
            ]
        finally:
            runner.close()


class TestTorchRunnerCompileRuntime:

    def test_train_compile_context_restores_ddp_optimizer_setting(self) -> None:
        dynamo = getattr(torch, "_dynamo", None)
        dynamo_config = getattr(dynamo, "config", None)
        if dynamo_config is None or not hasattr(dynamo_config, "optimize_ddp"):
            pytest.skip("torch._dynamo optimize_ddp is unavailable")

        previous = dynamo_config.optimize_ddp
        target = "ddp_optimizer" if previous != "ddp_optimizer" else False

        try:
            compiler = Compiler(CompileConfig({"enabled": True, "optimize_ddp": target}))
            with compiler.ddp_optimizer():
                assert dynamo_config.optimize_ddp == target
            assert dynamo_config.optimize_ddp == previous
        finally:
            dynamo_config.optimize_ddp = previous


class TestTorchRunnerStepEvaluation:

    def test_evaluate_steps_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 2})
        try:
            runner.dataloaders["val"] = [1.0, 2.0, 3.0]
            result = runner.evaluate_steps("val", steps=3)
            assert result["loss"] == pytest.approx(2.0)
        finally:
            runner.close()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestTorchRunnerSignalsAndShutdown:

    def test_sigterm_discards_unsynced_partial_accumulation(self, tmp_path: Path) -> None:
        class DistributedAccumRunner(TinyTorchRunner):
            @property
            def world_size(self) -> int:
                return 2

            def materialize_model(self) -> None:
                if self.model is None:
                    raise ValueError("cannot materialize model: model is not initialized")
                self.model = self.model.to(self.device)

            def _train_no_sync_targets(self):
                return (nn.Identity(),)

        runner = DistributedAccumRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "accum_steps": 2,
                "ckpt": {"async_mode": "disabled"},
            }
        )
        runner.train_state.micro_step = 1
        assert runner.model is not None
        initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
        for parameter in runner.model.parameters():
            parameter.grad = torch.ones_like(parameter)

        runner.supervisor.request_shutdown(signal.SIGTERM, None)

        with pytest.raises(SystemExit):
            runner.supervisor.maybe_handle_termination_signal()

        for parameter, initial in zip(runner.model.parameters(), initial_parameters):
            torch.testing.assert_close(parameter, initial)
            if parameter.grad is not None:
                assert torch.count_nonzero(parameter.grad).item() == 0
        assert runner.train_state.micro_step == 0
