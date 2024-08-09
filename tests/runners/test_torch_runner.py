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
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn, optim
from torchdata.stateful_dataloader import StatefulDataLoader

from danling.runners import TorchRunner
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

    @staticmethod
    def loss_fn(pred: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        return nn.functional.mse_loss(pred, target["value"])

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.criterion = self.loss_fn
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class ContextRecordingRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        self.forward_context_entries = 0
        super().__init__(config)
        self.model = nn.Linear(4, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

    def forward_context(self):
        runner = self

        class RecordingContext:
            def __enter__(self):
                runner.forward_context_entries += 1

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


def _ddp_compile_wrap_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"log": False, "backend": "gloo", "compile": {"enable": True}})
    try:
        assert isinstance(runner.model, nn.parallel.DistributedDataParallel)
        assert runner.model.module.__class__.__name__ == "OptimizedModule"
        assert hasattr(runner.model.module, "_orig_mod")
    finally:
        runner.close()


def _ddp_no_sync_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"log": False, "backend": "gloo", "accum_steps": 2})
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
    runner = DistributedTinyTorchRunner({"log": False, "backend": "gloo", "checkpoint": {"backend": "file"}})
    try:
        reduced = runner.reduce(torch.tensor(float(rank + 1)))
        assert reduced.item() == pytest.approx(1.5)
    finally:
        runner.close()


def _torch_runner_weighted_loss_logging_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"log": False, "backend": "gloo"})
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
            "log": False,
            "backend": "gloo",
            "checkpoint": {"backend": "file"},
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
        {"log": False, "backend": "gloo", "checkpoint": {"backend": "file"}, "comm": {"init_timeout_seconds": 42}}
    )
    try:
        assert dist.is_initialized()
        assert isinstance(runner.model, nn.parallel.DistributedDataParallel)
    finally:
        runner.close()


def _torch_runner_preinitialized_pg_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    with process_group("gloo", rank, world_size), pytest.raises(RuntimeError, match="already initialized"):
        DistributedTinyTorchRunner({"log": False, "backend": "gloo", "checkpoint": {"backend": "file"}})


# ---------------------------------------------------------------------------
# Construction & Restore
# ---------------------------------------------------------------------------


class TestTorchRunnerBootstrap:

    def test_supports_explicit_components_without_build_hooks(self) -> None:
        runner = TinyTorchRunner({"log": False})
        assert runner.model is not None
        assert runner.criterion is not None
        assert runner.optimizer is not None
        assert runner.optimizer_container is not None

    def test_ft_requires_torchft_package(self) -> None:
        if importlib.util.find_spec("torchft") is not None:
            pytest.skip("torchft is installed")
        with pytest.raises(ImportError, match="torchft"):
            TinyTorchRunner({"log": False, "ft": {"enabled": True}})

    def test_declares_torchft_runtime_supported(self) -> None:
        assert TorchRunner._supports_torchft_runtime is True

    def test_builds_optimizer_from_iter_optimizer_parameters(self) -> None:
        class AutoOptimRunner(TorchRunner):
            def init_distributed(self) -> None:
                return

            model = nn.Linear(4, 2)

        runner = AutoOptimRunner({"log": False, "optim": {"type": "sgd", "lr": 0.1}})
        try:
            assert runner.optimizer is not None
        finally:
            runner.close()

    def test_builds_stateful_dataloaders_by_default(self) -> None:
        runner = TinyTorchRunner({"log": False, "dataloader": {"batch_size": 2}})
        try:
            runner.datasets["train"] = list(range(8))
            runner.build_dataloaders()

            assert isinstance(runner.dataloaders["train"], StatefulDataLoader)
        finally:
            runner.close()

    def test_surfaces_stateful_dataloader_argument_errors(self) -> None:
        runner = TinyTorchRunner(
            {
                "log": False,
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

    def test_auto_resume_uses_config_source(self) -> None:
        runner = RecordingRestoreRunner({"log": False, "resume": "checkpoint-latest"})
        assert runner.restore_calls == [("checkpoint", "checkpoint-latest")]
        runner.close()

    def test_auto_pretrained_uses_config_source(self) -> None:
        runner = RecordingRestoreRunner({"log": False, "pretrained": "checkpoint-best"})
        assert runner.restore_calls == [("pretrained", "checkpoint-best")]
        runner.close()

    def test_all_reduce_group_uses_ft_replicate_group(self) -> None:
        class AllReduceRunner(TinyTorchRunner):
            def materialize_model(self) -> None:
                return

        class TorchFTRuntime:
            replicate_process_group = "ft_group"

            def close(self) -> None:
                return

        runner = AllReduceRunner({"log": False})
        try:
            runner.ft = TorchFTRuntime()
            assert runner.all_reduce_group() == "ft_group"
        finally:
            runner.ft = None
            runner.close()

    def test_evaluate_uses_forward_context(self) -> None:
        runner = ContextRecordingRunner({"log": False})
        try:
            runner.evaluate_step((torch.ones(2, 4), torch.zeros(2, 1)))
            assert runner.forward_context_entries == 1
        finally:
            runner.close()

    def test_infer_uses_forward_context(self) -> None:
        runner = ContextRecordingRunner({"log": False})
        try:
            runner.infer_step(torch.ones(2, 4))
            assert runner.forward_context_entries == 1
        finally:
            runner.close()

    def test_auto_restore_prefers_resume_over_pretrained(self) -> None:
        runner = RecordingRestoreRunner({"log": False, "resume": "ckpt-latest", "pretrained": "ckpt-best"})
        assert runner.restore_calls == [("checkpoint", "ckpt-latest")]
        runner.close()

    def test_auto_restore_prefers_auto_resume_over_pretrained(self) -> None:
        runner = RecordingRestoreRunner({"log": False, "auto_resume": True, "pretrained": "ckpt-best"})
        assert runner.restore_calls == [("checkpoint", os.path.join(runner.workspace.checkpoint_dir, "latest.pth"))]
        runner.close()

    def test_auto_restore_warns_when_all_sources_are_set(self) -> None:
        with pytest.warns(RuntimeWarning, match="precedence is `resume` > `auto_resume` > `pretrained`"):
            runner = RecordingRestoreRunner(
                {"log": False, "resume": "ckpt-latest", "auto_resume": True, "pretrained": "ckpt-best"}
            )
        assert runner.restore_calls == [("checkpoint", "ckpt-latest")]
        runner.close()

    def test_auto_resume_uses_latest_checkpoint_path(self) -> None:
        runner = RecordingRestoreRunner({"log": False, "auto_resume": True})
        assert runner.restore_calls == [("checkpoint", os.path.join(runner.workspace.checkpoint_dir, "latest.pth"))]
        runner.close()

    def test_manual_load_checkpoint_tracks_resume_source(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"log": False})
        checkpoint_path = tmp_path / "checkpoint-latest.pth"
        try:
            torch.save(source.state_dict(), checkpoint_path)
        finally:
            source.close()

        runner = TinyTorchRunner({"log": False})
        try:
            runner.load_checkpoint(checkpoint_path)
            assert runner.config.resume == str(checkpoint_path)
        finally:
            runner.close()

    def test_manual_load_pretrained_tracks_source(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"log": False})
        checkpoint_path = tmp_path / "checkpoint-best.pth"
        try:
            assert source.model is not None
            torch.save({"model": source.unwrap(source.model).state_dict()}, checkpoint_path)
        finally:
            source.close()

        runner = TinyTorchRunner({"log": False})
        try:
            runner.load_pretrained(checkpoint_path)
            assert runner.config.pretrained == str(checkpoint_path)
        finally:
            runner.close()

    def test_from_pretrained_accepts_mapping(self) -> None:
        source = TinyTorchRunner({"log": False})
        try:
            assert source.model is not None
            checkpoint = {"model": source.unwrap(source.model).state_dict()}
        finally:
            source.close()

        runner = TinyTorchRunner.from_pretrained({"log": False}, checkpoint)
        try:
            assert runner.config.pretrained is None
        finally:
            runner.close()

    def test_compile_happens_before_ddp_wrap(self) -> None:
        require_gloo()
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch build.")
        run_distributed(_ddp_compile_wrap_worker, world_size=2)

    def test_train_context_uses_ddp_no_sync_during_accumulation(self) -> None:
        require_gloo()
        run_distributed(_ddp_no_sync_worker, world_size=2)


class TestTorchRunnerCheckpointInterop:

    def test_read_config_accepts_dcp_directory(self, tmp_path: Path) -> None:
        runner = DcpConfigTorchRunner(
            {
                "log": False,
                "name": "dcp-config-test",
                "workspace_root": str(tmp_path),
                "checkpoint": {"backend": "dcp", "async_mode": "disabled", "interval": 1},
            }
        )
        try:
            runner.train_state.global_step = 1
            runner.save_checkpoint(force=True)
            config = DcpConfigTorchRunner.read_config(Path(runner.workspace.checkpoint_dir) / "latest")
        finally:
            runner.close()

        assert config["name"] == "dcp-config-test"
        assert config["checkpoint"]["backend"] == "dcp"

    def test_state_dict_refreshes_torch_rng_state(self) -> None:
        original_rng = torch.get_rng_state()
        runner = TinyTorchRunner({"log": False})
        try:
            torch.manual_seed(1234)
            torch.rand(4)
            expected = torch.get_rng_state().clone()

            checkpoint = runner.state_dict()

            torch.testing.assert_close(checkpoint["state"]["rng"]["torch_cpu"], expected)
        finally:
            runner.close()
            torch.set_rng_state(original_rng)

    def test_load_state_dict_restores_torch_rng_state(self) -> None:
        original_rng = torch.get_rng_state()
        runner = TinyTorchRunner({"log": False})
        try:
            torch.manual_seed(1234)
            torch.rand(4)
            checkpoint = runner.state_dict()
            expected = checkpoint["state"]["rng"]["torch_cpu"].clone()

            torch.manual_seed(5678)
            torch.rand(4)
            runner.load_state_dict(checkpoint)

            torch.testing.assert_close(torch.get_rng_state(), expected)
        finally:
            runner.close()
            torch.set_rng_state(original_rng)

    def test_from_checkpoint_accepts_mapping_payload(self) -> None:
        runner = TinyTorchRunner({"log": False})
        checkpoint = runner.state_dict()
        runner.close()

        restored = TinyTorchRunner.from_checkpoint(checkpoint)
        try:
            assert isinstance(restored, TinyTorchRunner)
        finally:
            restored.close()

    def test_from_checkpoint_path_bypasses_auto_restore_sources(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"log": False})
        checkpoint_path = tmp_path / "torch-runner.pth"
        try:
            checkpoint = dict(source.state_dict())
            checkpoint["runner"]["auto_resume"] = True
            checkpoint["runner"]["pretrained"] = "stale-pretrained"
            torch.save(checkpoint, checkpoint_path)
        finally:
            source.close()

        restored = TinyTorchRunner.from_checkpoint(checkpoint_path)
        try:
            assert restored.config.resume == str(checkpoint_path)
            assert restored.config.auto_resume is False
            assert restored.config.pretrained is None
        finally:
            restored.close()

    def test_load_checkpoint_restores_stateful_dataloader_progress(self) -> None:
        source = StatefulDatasetTorchRunner({"log": False, "dataloader": {"batch_size": 2, "shuffle": False}})
        try:
            assert next(iter(source.dataloaders["train"])).tolist() == [0, 1]
            checkpoint = source.state_dict()
        finally:
            source.close()

        restored = StatefulDatasetTorchRunner({"log": False, "dataloader": {"batch_size": 2, "shuffle": False}})
        try:
            restored.load_checkpoint(checkpoint)
            remaining_batches = [batch.tolist() for batch in restored.dataloaders["train"]]
            assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
        finally:
            restored.close()

    def test_from_checkpoint_path_restores_stateful_dataloader_progress(self, tmp_path: Path) -> None:
        source = StatefulDatasetTorchRunner({"log": False, "dataloader": {"batch_size": 2, "shuffle": False}})
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
            "log": False,
            "workspace_root": str(tmp_path),
            "dataloader": {"batch_size": 2, "shuffle": False},
            "checkpoint": {"async_mode": "disabled", "interval": 1},
        }
        source = StatefulDatasetTorchRunner(config)
        try:
            assert next(iter(source.dataloaders["train"])).tolist() == [0, 1]
            source.save_checkpoint(force=True)
        finally:
            source.close()

        restored_config = dict(config)
        restored_config["auto_resume"] = True
        restored = StatefulDatasetTorchRunner(restored_config)
        try:
            remaining_batches = [batch.tolist() for batch in restored.dataloaders["train"]]
            assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
            assert restored.config.resume == os.path.join(restored.workspace.checkpoint_dir, "latest.pth")
        finally:
            restored.close()


class TestTorchRunnerProfiling:

    @staticmethod
    def _profiling_runner(tmp_path: Path) -> TinyTorchRunner:
        return TinyTorchRunner(
            {
                "log": False,
                "workspace_root": str(tmp_path),
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

    def test_step_skips_optimizer_update_on_nonfinite_grad(self) -> None:
        runner = TinyTorchRunner({"log": False, "skip_nonfinite_grad": True})
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

    def test_steps_standard_pytorch_scheduler_after_optimizer_step(self) -> None:
        runner = TinyStepLRTorchRunner({"log": False})
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
        runner = TimeoutRecordingRunner({"log": False, "comm": {"train_timeout_seconds": 17}})
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
        runner = TinyTorchRunner({"log": False})

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
        runner = TinyTorchRunner({"log": False})

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
        runner = TinyTorchRunner({"log": False, "gc": {"interval": 2, "disable_automatic": False}})
        try:
            assert runner.optimizer_step() is True
            assert runner.optimizer_step() is True
            assert runner.optimizer_step() is True
            assert runner.supervisor._gc_last_collection == {"train": 2}
        finally:
            runner.close()


class TestTorchRunnerScheduling:

    def test_steps_danling_scheduler_after_optimizer_step(self) -> None:
        runner = TinyTorchRunner({"log": False, "scheduler": {"type": "linear", "total_steps": 8}})
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
        runner = TinyPlateauTorchRunner({"log": False})
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
        with pytest.raises(ValueError, match="metric-based schedulers require `scheduler.interval='epoch'`"):
            TinyPlateauTorchRunner({"log": False, "scheduler": {"interval": "step"}})

    def test_steps_reduce_on_plateau_after_epoch_result(self, tmp_path: Path) -> None:
        runner = EpochSchedulerRunner(
            {
                "log": False,
                "workspace_root": str(tmp_path),
                "epochs": 2,
                "train_splits": ["train"],
                "evaluate_splits": ["val"],
                "scheduler": {"type": "reduce_on_plateau", "patience": 0, "factor": 0.5},
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
        runner = TinyStepLRTorchRunner({"log": False, "scheduler": {"interval": "epoch"}})
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
            {"log": False, "workspace_root": str(tmp_path), "steps": 4, "scheduler": {"interval": "epoch"}}
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
        runner = TinyPlateauTorchRunner({"log": False, "scheduler": {"monitor": "val.accuracy"}})
        try:
            with pytest.raises(ValueError, match="scheduler.monitor"):
                runner._step_epoch_scheduler({"val": {"loss": 1.0}})
        finally:
            runner.close()

    def test_reduce_returns_world_mean(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_reduce_worker, world_size=2)

    def test_reduce_loss_for_logging_uses_weighted_normalizer(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_weighted_loss_logging_worker, world_size=2)


# ---------------------------------------------------------------------------
# Execution Loops
# ---------------------------------------------------------------------------


class TestTorchRunnerEpochExecution:

    def test_train_epoch_supports_unsized_loader(self) -> None:
        runner = StreamingEpochRunner({"log": False, "log_interval": 1024})
        try:
            runner.dataloaders["train"] = StreamingLoader(1.0, 2.0)
            result = runner.train_epoch("train")
            assert math.isfinite(result["loss"])
            assert runner.train_state.global_step == 2
        finally:
            runner.close()

    def test_variable_length_loss_uses_weighted_normalizer(self) -> None:
        runner = WeightedLossRunner({"log": False, "accum_steps": 2, "log_interval": 0})
        runner.dataloaders["train"] = [
            {
                "input": torch.ones((1, 1)),
                "target": {"value": torch.zeros((1, 1)), "tokens": NestedTensor([torch.zeros(1)])},
            },
            {
                "input": torch.full((3, 1), 3.0),
                "target": {"value": torch.zeros((3, 1)), "tokens": NestedTensor([torch.zeros(3)])},
            },
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
        runner = WeightedLossRunner({"log": False, "accum_steps": 4, "log_interval": 0})
        runner.dataloaders["train"] = [
            {
                "input": torch.ones((1, 1)),
                "target": {"value": torch.zeros((1, 1)), "tokens": NestedTensor([torch.zeros(1)])},
            },
            {
                "input": torch.full((3, 1), 3.0),
                "target": {"value": torch.zeros((3, 1)), "tokens": NestedTensor([torch.zeros(3)])},
            },
        ]

        try:
            runner.train_epoch("train")
            assert runner.model is not None
            assert runner.unwrap(runner.model).weight.detach().item() == pytest.approx(-0.4)
            assert runner.train_state.global_step == 1
            assert runner.train_state.micro_step == 4
        finally:
            runner.close()

    def test_mixed_nested_target_normalizer_raises(self) -> None:
        runner = WeightedLossRunner({"log": False, "accum_steps": 2, "log_interval": 0})
        runner.dataloaders["train"] = [
            {"input": torch.ones((1, 1)), "target": {"value": torch.zeros((1, 1))}},
            {
                "input": torch.full((3, 1), 3.0),
                "target": {"value": torch.zeros((3, 1)), "tokens": NestedTensor([torch.zeros(3)])},
            },
        ]

        try:
            with pytest.raises(ValueError, match="cannot mix weighted and uniform loss normalization"):
                runner.train_epoch("train")
        finally:
            runner.close()

    def test_gradient_scale_for_step_uses_sync_divisor(self) -> None:
        class GradientScaleRunner(WeightedLossRunner):
            def _loss_normalizer_sync_divisor(self) -> int:
                return 4

            def _reduce_loss_normalizer_total(self, local_total: float) -> float:
                assert local_total == pytest.approx(4.0)
                return 10.0

        runner = GradientScaleRunner({"log": False, "accum_steps": 2})
        runner._accumulation_normalizer_local = 4.0

        try:
            assert runner._gradient_scale_for_step() == pytest.approx(0.4)
        finally:
            runner.close()

    def test_loss_normalizer_reads_nested_target_numel(self) -> None:
        runner = WeightedLossRunner({"log": False})
        batch = {
            "input": torch.ones((3, 1)),
            "target": {"value": torch.zeros((3, 1)), "tokens": NestedTensor([torch.ones(2), torch.ones(3)])},
        }

        try:
            assert runner.loss_normalizer(batch) == 5
        finally:
            runner.close()

    def test_loss_normalizer_does_not_infer_from_dense_target_shape(self) -> None:
        runner = WeightedLossRunner({"log": False})
        batch = (torch.ones((3, 1)), torch.zeros((3, 1)))

        try:
            assert runner.loss_normalizer(batch) is None
        finally:
            runner.close()

    def test_loss_normalizer_does_not_infer_from_input_nested_tensor(self) -> None:
        runner = WeightedLossRunner({"log": False})
        batch = {
            "input": {"tokens": NestedTensor([torch.ones(2), torch.ones(3)])},
            "target": {"value": torch.zeros((2, 1))},
        }

        try:
            assert runner.loss_normalizer(batch) is None
        finally:
            runner.close()

    def test_loss_normalizer_does_not_infer_from_attention_mask(self) -> None:
        runner = WeightedLossRunner({"log": False})
        batch = {
            "input": {
                "tokens": torch.ones((2, 3), dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
            }
        }

        try:
            assert runner.loss_normalizer(batch) is None
        finally:
            runner.close()

    def test_train_epoch_skips_peak_memory_sampling_without_interval_logs(self) -> None:
        runner = TelemetryRunner({"log": False, "log_interval": 0})
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
            assert "max_memory_allocated_mb" not in result
            assert "max_memory_reserved_mb" not in result
            assert "flops_per_s" not in result
            assert "mfu" not in result
        finally:
            runner.close()

    def test_train_epoch_reports_nested_tensor_tokens_per_s(self) -> None:
        runner = TelemetryRunner({"log": False, "log_interval": 0})
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
            assert "max_memory_allocated_mb" not in result
            assert "max_memory_reserved_mb" not in result
        finally:
            runner.close()

    def test_mode_setter_skips_redundant_model_train_toggle(self) -> None:
        class TrackingModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[bool] = []

            def train(self, mode: bool = True):
                self.calls.append(bool(mode))
                return super().train(mode)

        runner = TinyTorchRunner({"log": False})
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
        runner = StreamingEpochRunner({"log": False, "log_interval": 1024})
        try:
            runner.dataloaders["val"] = StreamingLoader(1.0, 2.0)
            result = runner.evaluate_epoch("val")
            assert math.isfinite(result["loss"])
        finally:
            runner.close()

    def test_evaluate_epoch_collects_gc_on_iteration_interval(self) -> None:
        runner = StreamingEpochRunner({"log": False, "gc": {"interval": 2, "disable_automatic": False}})
        try:
            runner.dataloaders["val"] = [1.0, 2.0, 3.0]
            runner.evaluate_epoch("val")
            assert runner.supervisor._gc_last_collection == {"evaluate:val": 2}
        finally:
            runner.close()


class TestTorchRunnerLoopResultStability:

    def test_train_epoch_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"log": False, "log_interval": 2})
        try:
            runner.dataloaders["train"] = [1.0, 2.0, 3.0]
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(2.0)
        finally:
            runner.close()

    def test_evaluate_epoch_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"log": False, "log_interval": 2})
        try:
            runner.dataloaders["val"] = [1.0, 2.0, 3.0]
            result = runner.evaluate_epoch("val")
            assert result["loss"] == pytest.approx(2.0)
        finally:
            runner.close()


class TestTorchRunnerStepExecution:

    def test_train_steps_result_is_independent_of_log_interval(self, tmp_path: Path) -> None:
        runner = StreamingEpochRunner({"log": False, "workspace_root": str(tmp_path), "steps": 3, "log_interval": 2})
        try:
            runner.dataloaders["train"] = [1.0, 2.0, 3.0]
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            assert runner.results[runner.train_state.global_step]["train"]["loss"] == pytest.approx(2.0)
        finally:
            runner.close()

    def test_train_steps_logs_once_per_optimizer_step_under_accumulation(self, tmp_path: Path) -> None:
        runner = RecordingStepLogRunner(
            {"log": False, "workspace_root": str(tmp_path), "steps": 2, "accum_steps": 2, "log_interval": 2}
        )
        runner.dataloaders["train"] = [1.0, 2.0, 3.0, 4.0]

        try:
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            assert [(iteration, length) for _, iteration, length, _ in runner.step_log_calls] == [(1, 1)]
        finally:
            runner.close()

    def test_train_steps_shares_budget_across_splits(self, tmp_path: Path) -> None:
        runner = StreamingEpochRunner({"log": False, "workspace_root": str(tmp_path), "steps": 2, "log_interval": 0})
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

        runner = StreamingEpochRunner({"log": False, "workspace_root": str(tmp_path), "steps": 4, "log_interval": 0})
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
        runner = RecordingStepLogRunner({"log": False, "workspace_root": str(tmp_path), "steps": 3, "log_interval": 1})
        runner.dataloaders["train"] = StreamingLoader(1.0, 2.0)

        try:
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            assert [iteration for _, iteration, _, _ in runner.step_log_calls] == [1, 2]
        finally:
            runner.close()


class TestTorchRunnerCompileRuntime:

    def test_compile_guard_restores_dynamo_optimize_ddp(self) -> None:
        dynamo = getattr(torch, "_dynamo", None)
        dynamo_config = getattr(dynamo, "config", None)
        if dynamo_config is None or not hasattr(dynamo_config, "optimize_ddp"):
            pytest.skip("torch._dynamo optimize_ddp is unavailable")

        previous = dynamo_config.optimize_ddp
        target = "ddp_optimizer" if previous != "ddp_optimizer" else False

        try:
            compiler = Compiler(CompileConfig({"enable": True, "optimize_ddp": target}))
            with compiler.ddp_optimizer():
                assert dynamo_config.optimize_ddp == target
            assert dynamo_config.optimize_ddp == previous
        finally:
            dynamo_config.optimize_ddp = previous


class TestTorchRunnerStepEvaluation:

    def test_evaluate_steps_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"log": False, "log_interval": 2})
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
                "log": False,
                "workspace_root": str(tmp_path),
                "accum_steps": 2,
                "checkpoint": {"async_mode": "disabled"},
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
