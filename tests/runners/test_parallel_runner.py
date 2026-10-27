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

import os
from collections import OrderedDict
from contextlib import contextmanager
from itertools import chain

import pytest
import torch
from torch import distributed as dist
from torch import nn, optim
from torchdata.stateful_dataloader import StatefulDataLoader

import danling.runners.parallel_runner as parallel_runner_module
from danling.data import StepProxyLoader
from danling.runners import ParallelRunner
from danling.runners.config import RunnerConfig
from danling.runners.telemetry import LoopTelemetry
from danling.runners.topology import ParallelContext
from danling.tensors import NestedTensor
from tests.runners.distributed import configure_distributed_env, require_gloo, run_distributed


def _init_parallel_topology(runner: ParallelRunner) -> None:
    runner.topology = runner.build_topology()
    runner.parallel = ParallelContext(runner.topology)


def _parallel_config(
    *,
    replicate: int = 1,
    shard: int = 1,
    context: int = 1,
    pipeline: int = 1,
    tensor: int = 1,
    expert: int = 1,
    expert_tensor: int = 1,
    **kwargs,
):
    parallel = {
        "axes": {
            "replicate": replicate,
            "shard": shard,
            "context": context,
            "pipeline": pipeline,
            "tensor": tensor,
            "expert": expert,
            "expert_tensor": expert_tensor,
        }
    }
    parallel.update(kwargs)
    return {"parallel": parallel}


class CpuParallelRunner(ParallelRunner):
    @property
    def device(self):
        return torch.device("cpu")


class CudaParallelRunner(ParallelRunner):
    @property
    def device(self):
        return torch.device("cuda")


class TinyParallelRunner(CpuParallelRunner):
    def init_distributed(self) -> None:
        _init_parallel_topology(self)

    @property
    def world_size(self) -> int:
        return 16

    @property
    def rank(self) -> int:
        return 9

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class FakeShardedGradScaler:
    def __init__(self) -> None:
        self.unscale_calls: list[optim.Optimizer] = []
        self.step_calls: list[optim.Optimizer] = []
        self.update_calls = 0

    def unscale_(self, optimizer: optim.Optimizer) -> None:
        self.unscale_calls.append(optimizer)

    def step(self, optimizer: optim.Optimizer) -> None:
        self.step_calls.append(optimizer)
        optimizer.step()

    def update(self) -> None:
        self.update_calls += 1


def _pipeline_scaled_backward_worker(rank: int, world_size: int, device_type: str = "cpu") -> None:
    class PipelineRunner(ParallelRunner):
        def __init__(self, config):
            super().__init__(config)
            self.model = nn.Linear(2, 2, bias=False, dtype=torch.float64 if device_type == "cpu" else torch.float32)
            with torch.no_grad():
                self.model.weight.fill_((rank + 1) / 8)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        def setup_grad_scaler(self):
            self.grad_scaler = torch.amp.GradScaler(device_type, init_scale=128.0, growth_interval=2)

        def materialize_model(self):
            self.model = self.model.to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
            self.model_parts = [self.model]
            self.pipeline_has_first_stage = self.pipeline_rank == 0
            self.pipeline_has_last_stage = self.pipeline_rank == self.pipeline_degree - 1
            self.pipeline_schedule = self.build_pipeline_schedule(self.model)

    configure_distributed_env(rank, world_size)
    runner = PipelineRunner(
        {
            "device": f"cuda:{rank}" if device_type == "cuda" else "cpu",
            "dist.backend": "nccl" if device_type == "cuda" else "gloo",
            "precision": "fp16" if device_type == "cuda" else None,
            "logging.enabled": False,
            "logging.interval": 0,
            "max_grad_norm": 0.025,
            **_parallel_config(pipeline=world_size, pipeline_microbatches=2),
        }
    )
    try:
        weights = [
            torch.full(
                (2, 2), (index + 1) / 8, dtype=runner.model.weight.dtype, device=runner.device, requires_grad=True
            )
            for index in range(world_size)
        ]
        inputs = torch.arange(8, dtype=runner.model.weight.dtype, device=runner.device).view(4, 2) / 4
        targets = torch.zeros_like(inputs)
        with runner.infer_context():
            loss = nn.functional.mse_loss(inputs @ weights[0].T @ weights[1].T, targets)
        gradients = torch.autograd.grad(loss, weights)
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients]))
        coefficient = (0.025 / (norm + 1e-6)).clamp(max=1)
        runner.dataloaders["train"] = [(inputs, targets)]
        reported = runner.train_epoch()["loss"]
        torch.testing.assert_close(
            runner.model.weight,
            weights[rank].detach() - 0.1 * coefficient * gradients[rank],
            atol=1e-12 if device_type == "cpu" else 1e-4,
            rtol=1e-12 if device_type == "cpu" else 1e-3,
        )
        assert reported == pytest.approx(loss.detach().item(), rel=1e-3 if device_type == "cuda" else 1e-12)
        checkpoint = runner.state_dict()
        assert checkpoint["grad_scaler"]["scale"] == 128.0
        assert checkpoint["grad_scaler"]["_growth_tracker"] == 1
        modified = dict(checkpoint["grad_scaler"], scale=32.0, _growth_tracker=0)
        runner.grad_scaler.load_state_dict(modified)
        runner.load_state_dict(checkpoint)
        assert runner.grad_scaler.state_dict() == checkpoint["grad_scaler"]
        initial = runner.model.weight.detach().clone()
        runner.model.weight.grad = torch.full_like(initial, float("inf") if rank else 128.0)
        assert runner.optimizer_step() is False
        assert runner.train_state.global_step == 1
        torch.testing.assert_close(runner.model.weight, initial)
        assert runner.grad_scaler.get_scale() == 64.0
        assert runner.grad_scaler.state_dict()["_growth_tracker"] == 0
    finally:
        runner.close()


class ParallelizableLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 2)
        self.parallel_context = None

    def parallelize(self, parallel):
        self.parallel_context = parallel


class TensorParallelRunner(CpuParallelRunner):
    def init_distributed(self) -> None:
        _init_parallel_topology(self)

    @property
    def world_size(self) -> int:
        return 2

    @property
    def rank(self) -> int:
        return 0

    def __init__(self, config):
        super().__init__(config)
        self.model = ParallelizableLinear()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class RecordingPipelineSchedule:
    def __init__(self, module: nn.Module | None = None, *, loss: torch.Tensor | None = None) -> None:
        self.module = module
        self.loss = loss
        self.called = False
        self.last_kwargs: dict[str, object] = {}

    def step(self, *args, **kwargs):
        del args
        self.last_kwargs = dict(kwargs)
        losses = kwargs.get("losses")
        if self.loss is not None and losses is not None:
            losses.append(self.loss)
        self.called = True
        return None

    def eval(self, *args, **kwargs):
        del args, kwargs
        return None


class RecordingPipelineStage:
    def __init__(self, module: nn.Module) -> None:
        self.module = module


class RecordingMultiPipelineSchedule:
    def __init__(self, modules) -> None:
        self.stages = [RecordingPipelineStage(module) for module in modules]


class DistributedSmokeParallelRunner(CpuParallelRunner):
    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

    def materialize_model(self) -> None:
        if self.model is None:
            raise ValueError("model is not initialized")
        self.model = self.model.to(self.device)
        self.model_parts = [self.model]


class DistributedCheckpointPartsRunner(CpuParallelRunner):
    def __init__(self, config):
        super().__init__(config)
        self.model_parts = [nn.Linear(2, 2, bias=False) for _ in self.pipeline_stage_indices()]
        self.model = self.model_parts[0]
        self.optimizer = optim.Adam(chain.from_iterable(part.parameters() for part in self.model_parts), lr=0.01)

    def materialize_model(self) -> None:
        self.model_parts = [part.to(self.device) for part in self.model_parts]


def _pipeline_checkpoint_identity_worker(rank: int, world_size: int, checkpoint_dir: str, parts: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedCheckpointPartsRunner(
        {
            "logging.enabled": False,
            "workspace.dir": checkpoint_dir,
            "dist.backend": "gloo",
            "ckpt": {"backend": "dcp", "async_mode": "disabled", "interval": 1},
            **_parallel_config(
                pipeline=world_size,
                mesh_device_type="cpu",
                pipeline_partitions=[[f"stage{index}"] for index in range(world_size * parts)],
            ),
        }
    )
    try:
        expected = {}
        for index, part in zip(runner.pipeline_stage_indices(), runner.model_parts):
            with torch.no_grad():
                part.weight.fill_(index + 1)
            (part(torch.ones(1, 2)) * (index + 2)).sum().backward()
        runner.optimizer.step()
        runner.optimizer.zero_grad()
        for index, part in zip(runner.pipeline_stage_indices(), runner.model_parts):
            expected[index] = (
                part.weight.detach().clone(),
                {key: value.clone() for key, value in runner.optimizer.state[part.weight].items()},
            )
        runner.save_checkpoint(force=True)
        with torch.no_grad():
            for part in runner.model_parts:
                part.weight.zero_()
        # Numeric optimizer parameter IDs depend on local parameter order.
        # FQN checkpoint identity must survive an equivalent reordered optimizer.
        runner.optimizer = optim.Adam(list(reversed(list(runner.iter_optimizer_parameters()))), lr=0.01)
        runner.load_checkpoint("latest")
        for index, part in zip(runner.pipeline_stage_indices(), runner.model_parts):
            weight, moments = expected[index]
            torch.testing.assert_close(part.weight, weight)
            for key, value in moments.items():
                torch.testing.assert_close(runner.optimizer.state[part.weight][key], value)
    finally:
        runner.close()


@pytest.mark.parametrize("parts", [1, 2])
def test_pipeline_checkpoint_restores_global_stage_parameter_identity(tmp_path, parts: int) -> None:
    require_gloo()
    run_distributed(_pipeline_checkpoint_identity_worker, worker_args=(str(tmp_path), parts))


def test_pipeline_step_proxy_restores_its_consumed_ticks_without_loading_samples() -> None:
    loader = StatefulDataLoader(list(range(8)), batch_size=2, shuffle=False)
    proxy = StepProxyLoader(loader)
    iterator = iter(proxy)
    assert next(iterator) is None
    assert next(iterator) is None
    state = proxy.state_dict()
    assert loader.state_dict()["_num_yielded"] == 0
    restored_loader = StatefulDataLoader(list(range(8)), batch_size=2, shuffle=False)
    restored = StepProxyLoader(restored_loader)
    restored.load_state_dict(state)
    assert list(restored) == [None, None]
    assert restored_loader.state_dict()["_num_yielded"] == 0
    assert list(restored) == [None] * 4


def _parallel_gloo_smoke_worker(rank: int, world_size: int, workspace_root: str) -> None:
    configure_distributed_env(rank, world_size)
    os.environ["BACKEND"] = "gloo"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        runner = DistributedSmokeParallelRunner(
            {
                "logging.enabled": False,
                "workspace.root": workspace_root,
                "dataloader": {"batch_size": 2, "shuffle": False},
                "ckpt": {"async_mode": "disabled", "interval": 1},
                **_parallel_config(shard=1, pipeline=world_size, tensor=1),
            }
        )
    try:
        assert runner.world_size == world_size
        assert runner.tensor_degree == 1
        assert runner.pipeline_degree == world_size
        assert runner.data_degree == 1
        assert runner.tensor_rank == 0
        assert runner.pipeline_rank == rank
        assert runner.data_rank == 0
        assert runner.config.get("ckpt.backend") == "dcp"
        assert runner.device_mesh is not None
        assert runner.device_mesh.device_type == "cpu"
        assert runner.tensor_group is not None
        assert runner.pipeline_group is not None
        assert runner.shard_group is not None

        with torch.no_grad():
            for index, parameter in enumerate(runner.model.parameters(), start=1):
                parameter.fill_(float(index))

        runner.datasets["train"] = list(range(8))
        runner.build_dataloaders()
        assert isinstance(runner.dataloaders["train"], StatefulDataLoader)
        assert runner.state_dict()["parallel"] == {"axes": runner.parallel_axes_state(dict)}
        assert "train" in runner.dataloaders.state_dict()

        assert next(iter(runner.dataloaders["train"])).tolist() == [0, 1]
        saved_model_state = {key: value.detach().clone() for key, value in runner.model.state_dict().items()}
        saved_lr = runner.optimizer.param_groups[0]["lr"]
        runner.save_checkpoint(force=True)
        next(iter(runner.dataloaders["train"]))
        with torch.no_grad():
            for parameter in runner.model.parameters():
                parameter.add_(100.0)
        runner.optimizer.param_groups[0]["lr"] = 7.0
        runner.load_checkpoint("latest")
        for key, value in runner.model.state_dict().items():
            torch.testing.assert_close(value, saved_model_state[key])
        assert runner.optimizer.param_groups[0]["lr"] == saved_lr
        remaining_batches = [batch.tolist() for batch in runner.dataloaders["train"]]
        assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
    finally:
        runner.close()


def _parallel_gloo_middle_stage_loader_state_worker(rank: int, world_size: int, workspace_root: str) -> None:
    configure_distributed_env(rank, world_size)
    os.environ["BACKEND"] = "gloo"
    runner = DistributedSmokeParallelRunner(
        {
            "logging.enabled": False,
            "workspace.root": workspace_root,
            "dataloader": {"batch_size": 2, "shuffle": False},
            "ckpt": {"async_mode": "disabled", "interval": 1},
            **_parallel_config(shard=1, pipeline=world_size, tensor=1, mesh_device_type="cpu"),
        }
    )
    try:
        runner.pipeline_schedule = object()
        runner.pipeline_has_first_stage = runner.pipeline_rank == 0
        runner.pipeline_has_last_stage = runner.pipeline_rank == runner.pipeline_degree - 1
        runner.datasets["train"] = list(range(8))
        runner.build_dataloaders()

        raw_loader = runner.dataloaders.get("train")
        assert isinstance(raw_loader, StatefulDataLoader)
        loader = runner.dataloaders["train"]
        if runner.pipeline_rank == 1:
            assert isinstance(loader, StepProxyLoader)
            assert list(loader) == [None] * len(raw_loader)
        else:
            assert isinstance(loader, StatefulDataLoader)

        assert next(iter(raw_loader)).tolist() == [0, 1]
        runner.save_checkpoint(force=True)
        next(iter(raw_loader))

        runner.load_checkpoint("latest")
        restored_loader = runner.dataloaders.get("train")
        remaining_batches = [batch.tolist() for batch in restored_loader]
        assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
    finally:
        runner.close()


def _parallel_gloo_all_reduce_worker(rank: int, world_size: int, workspace_root: str) -> None:
    configure_distributed_env(rank, world_size)
    os.environ["BACKEND"] = "gloo"
    runner = DistributedSmokeParallelRunner(
        {
            "logging.enabled": False,
            "workspace.root": workspace_root,
            "ckpt": {"async_mode": "disabled", "interval": 1},
            **_parallel_config(shard=2, pipeline=2, tensor=1, mesh_device_type="cpu"),
        }
    )
    try:
        assert runner.data_degree == 2
        value = torch.tensor(float(rank + 1))
        runner.all_reduce(value, op=dist.ReduceOp.SUM)
        assert value.item() == pytest.approx(4.0 + 2.0 * runner.pipeline_rank)

        max_value = torch.tensor(float(rank + 1))
        runner.all_reduce(max_value, op=dist.ReduceOp.MAX)
        assert max_value.item() == pytest.approx(3.0 + runner.pipeline_rank)

        reduced = runner.reduce(torch.tensor(float(rank + 1)))
        assert reduced.item() == pytest.approx(2.0 + runner.pipeline_rank)

    finally:
        runner.close()


def _parallel_gloo_optimizer_skip_worker(rank: int, world_size: int, workspace_root: str) -> None:
    configure_distributed_env(rank, world_size)
    os.environ["BACKEND"] = "gloo"
    runner = DistributedSmokeParallelRunner(
        {
            "logging.enabled": False,
            "workspace.root": workspace_root,
            "ckpt": {"async_mode": "disabled", "interval": 1},
            **_parallel_config(shard=2, pipeline=2, tensor=1, mesh_device_type="cpu"),
        }
    )
    try:
        should_skip = runner._sync_optimizer_skip_decision(rank == 1)
        assert should_skip is True
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# Topology & Data Distribution
# ---------------------------------------------------------------------------


class TestParallelRunnerTopology:

    def test_topology_properties(self) -> None:
        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=8, tensor=1)})
        try:
            assert runner.tensor_degree == 1
            assert runner.pipeline_degree == 8
            assert runner.data_degree == 2
            assert runner.topology.domain_axes("optimizer") == runner.topology.axis_names
            assert runner.tensor_rank == 0
            assert runner.pipeline_rank == 1
            assert runner.data_rank == 1
        finally:
            runner.close()

    def test_rejects_world_size_mismatch(self) -> None:
        class BadRunner(TinyParallelRunner):
            @property
            def world_size(self) -> int:
                return 10

            @property
            def rank(self) -> int:
                return 0

        with pytest.raises(ValueError, match="WORLD_SIZE"):
            BadRunner({"logging.enabled": False, **_parallel_config(shard=1, pipeline=3, tensor=4)})

    def test_dataloader_shards_by_data_domain(self) -> None:
        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=8, tensor=1)})
        try:
            runner.datasets["train"] = list(range(32))
            runner.build_dataloaders()
            loader = runner.dataloaders.get("train")

            assert isinstance(loader, StatefulDataLoader)
            assert isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler)
            assert loader.sampler.num_replicas == 2
            assert loader.sampler.rank == 1
        finally:
            runner.close()

    def test_batch_telemetry_reports_from_first_stage_tensor0(self) -> None:
        class NoOpParallelLinear(nn.Linear):
            def parallelize(self, parallel):
                del parallel
                return None

        class TelemetryParallelRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 16

            @property
            def rank(self) -> int:
                return 9

            def __init__(self, config):
                super().__init__(config)
                self.model = NoOpParallelLinear(4, 2)
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        class ReporterParallelRunner(TelemetryParallelRunner):
            @property
            def rank(self) -> int:
                return 8

        batch = {
            "text": NestedTensor(
                [torch.ones(3, dtype=torch.long), torch.ones(2, dtype=torch.long)],
                batch_first=True,
            )
        }

        runner = TelemetryParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=4, tensor=2)})
        reporter = ReporterParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=4, tensor=2)})
        try:
            telemetry = LoopTelemetry(runner, start_time=0.0)
            assert runner.reports_batch_telemetry is False
            assert telemetry.infer_batch_counts(batch) == (0, False, 0, False)

            telemetry = LoopTelemetry(reporter, start_time=0.0)
            assert reporter.reports_batch_telemetry is True
            assert telemetry.infer_batch_counts(batch) == (0, False, 5, True)
        finally:
            reporter.close()
            runner.close()

    def test_state_dict_serializes_parallel_axes(self) -> None:
        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=8, tensor=1)})
        try:
            state = runner.state_dict()
            assert state["parallel"] == {"axes": runner.parallel_axes_state(dict)}
        finally:
            runner.close()

    def test_parallel_topology_fills_auto_shard_axis(self) -> None:
        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(replicate=2, shard=-1, pipeline=4)})
        try:
            assert runner.topology.axis_degree("shard") == 2
            assert runner.parallel_axes_state(dict)["shard"] == 2
            assert runner.config.parallel.axes.shard == -1
        finally:
            runner.close()

    def test_parallel_runner_initializes_gloo_groups(self, tmp_path) -> None:
        require_gloo()
        run_distributed(_parallel_gloo_smoke_worker, world_size=2, worker_args=(str(tmp_path),))

    def test_parallel_runner_restores_middle_stage_loader_state(self, tmp_path) -> None:
        require_gloo()
        run_distributed(_parallel_gloo_middle_stage_loader_state_worker, world_size=3, worker_args=(str(tmp_path),))

    def test_parallel_runner_reduces_metrics_across_data_domain(self, tmp_path) -> None:
        require_gloo()
        run_distributed(_parallel_gloo_all_reduce_worker, world_size=4, worker_args=(str(tmp_path),))

    def test_parallel_runner_skips_optimizer_step_across_optimizer_domain(self, tmp_path) -> None:
        require_gloo()
        run_distributed(_parallel_gloo_optimizer_skip_worker, world_size=4, worker_args=(str(tmp_path),))


# ---------------------------------------------------------------------------
# Model Parallelization
# ---------------------------------------------------------------------------


class TestParallelRunnerParallelization:

    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_meta_cpu_offload_preserves_cpu_parameters_and_runtime_buffer_aliases(self, device, pin_memory):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(2, 3, dtype=torch.float64))
                self.tied_weight = self.weight
                self.register_buffer("scale", torch.empty(3, dtype=torch.float32))
                self.register_buffer("scale_alias", self.scale, persistent=False)

            def init_weights(self):
                self.weight.fill_(1)
                self.scale = torch.full((3,), 2, dtype=torch.float32, device=self.weight.device)
                self.scale_alias = self.scale

        runner = object.__new__(ParallelRunner)
        runner.config = RunnerConfig(
            {
                "device": str(device),
                "model_init.meta": True,
                "fsdp": {"enabled": True, "offload_policy": {"pin_memory": pin_memory}},
            }
        )
        with torch.device("meta"):
            model = Model()
        model = runner.materialize_model_values(model)
        assert model.weight.device.type == "cpu"
        assert model.scale.device == torch.empty(0, device=device).device
        assert model.weight is model.tied_weight
        assert model.scale is model.scale_alias
        assert model.weight.dtype == torch.float64 and model.scale.dtype == torch.float32
        torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
        torch.testing.assert_close(model.scale, torch.full_like(model.scale, 2))

    @pytest.mark.parametrize("compile_selected", [False, True])
    def test_named_selective_checkpoint_preserves_dropout_rng_and_vjp(self, device, compile_selected):
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils.checkpoint import CheckpointPolicy

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 5, dtype=torch.float64))

            def forward(self, x):
                return torch.nn.functional.dropout(torch.mm(x, self.weight), p=0.4).sin()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()

            def activation_checkpoint_regions(self):
                return {"features": "block"}

            def compile_regions(self):
                return {"features": (self.block,)}

            def activation_checkpoint_policy(self, name):
                assert name == "features"
                return lambda ctx, op, *args, **kwargs: (
                    CheckpointPolicy.MUST_SAVE if op == torch.ops.aten.mm.default else CheckpointPolicy.PREFER_RECOMPUTE
                )

            def forward(self, x):
                return self.block(x)

        class CountForwardMM(TorchDispatchMode):
            calls = 0

            def __torch_dispatch__(self, op, types, args=(), kwargs=None):
                if op == torch.ops.aten.mm.default and args[0].shape == (3, 4) and args[1].shape == (4, 5):
                    self.calls += 1
                return op(*args, **(kwargs or {}))

        class RegionRunner(ParallelRunner):
            def init_distributed(self):
                _init_parallel_topology(self)

            @property
            def world_size(self):
                return 2

            def __init__(self, config):
                super().__init__(config)
                self.model = Model()

        runner = RegionRunner(
            {
                "logging.enabled": False,
                "device": str(device),
                "activation_checkpoint": {"enabled": True, "regions": ["features"]},
                "compile": {
                    "enabled": compile_selected,
                    "regions": ["features"],
                    "backend": "aot_eager",
                    "fullgraph": True,
                },
                **_parallel_config(shard=2),
            }
        )
        try:
            reference = Model().to(device)
            reference.load_state_dict(runner.model.state_dict())
            x = torch.randn(3, 4, dtype=torch.float64, device=device, requires_grad=True)
            reference_x = x.detach().clone().requires_grad_()
            torch.manual_seed(77)
            if compile_selected:
                actual = runner.model(x)
                forward_rng = torch.get_rng_state()
                forward_cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                gradients = torch.autograd.grad(actual.square().sum(), (x, *runner.model.parameters()))
            else:
                with CountForwardMM() as counter:
                    actual = runner.model(x)
                    forward_rng = torch.get_rng_state()
                    forward_cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                    gradients = torch.autograd.grad(actual.square().sum(), (x, *runner.model.parameters()))
                assert counter.calls == 1
            torch.testing.assert_close(torch.get_rng_state(), forward_rng)
            if forward_cuda_rng is not None:
                torch.testing.assert_close(torch.cuda.get_rng_state(device), forward_cuda_rng)
            if compile_selected:
                # Compiled CUDA dropout can use a different random kernel from
                # eager. Control its VJP with the mask realized in the forward;
                # replay must use that same mask without advancing RNG state.
                mask = actual.detach().ne(0).to(x.dtype) / 0.6
                expected = (torch.mm(reference_x, reference.block.weight) * mask).sin()
            else:
                torch.manual_seed(77)
                expected = reference(reference_x)
            expected_gradients = torch.autograd.grad(expected.square().sum(), (reference_x, *reference.parameters()))
            torch.testing.assert_close(actual, expected)
            for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
                torch.testing.assert_close(actual_gradient, expected_gradient)
        finally:
            runner.close()

    def test_meta_model_is_parallelized_and_sharded_before_allocation(self, monkeypatch):
        events = []

        class Model(nn.Linear):
            def __init__(self):
                super().__init__(3, 2, dtype=torch.float64)

            def parallelize(self, parallel):
                events.append(("parallel", self.weight.is_meta))

            def init_weights(self):
                events.append(("init", self.weight.is_meta))
                self.weight.fill_(1)
                self.bias.zero_()

        class MetaRunner(CpuParallelRunner):
            def init_distributed(self):
                _init_parallel_topology(self)

            @property
            def world_size(self):
                return 2

            def build_model(self):
                return Model()

            def _check_fsdp_prerequisites(self):
                return

            def fsdp_kwargs(self):
                return {}

            def apply_fsdp(self, model, kwargs):
                events.append(("shard", model.weight.is_meta))
                return model

        monkeypatch.setattr(parallel_runner_module.parallel_fsdp, "check", lambda: None)
        runner = MetaRunner(
            {"logging.enabled": False, "model_init.meta": True, "fsdp.enabled": True, **_parallel_config(shard=2)}
        )
        try:
            assert events == [("parallel", True), ("shard", True), ("init", False)]
            torch.testing.assert_close(
                runner.model(torch.ones(1, 3, dtype=torch.float64)), torch.full((1, 2), 3.0, dtype=torch.float64)
            )
        finally:
            runner.close()

    def test_requires_model_parallelization_for_model_axes(self) -> None:
        with pytest.raises(NotImplementedError, match="model-specific parallelization"):
            TinyParallelRunner({"logging.enabled": False, **_parallel_config(context=2, pipeline=4, tensor=2)})

    def test_fsdp_fp16_uses_sharded_grad_scaler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = object.__new__(CudaParallelRunner)
        runner.config = RunnerConfig(
            {
                "logging.enabled": False,
                "precision": "fp16",
                "fsdp": {"enabled": True},
                **_parallel_config(shard=2),
            }
        )
        runner._fp8_enabled = False
        monkeypatch.setattr(parallel_runner_module.parallel_fsdp, "check", lambda: None)
        monkeypatch.setattr(parallel_runner_module, "ShardedGradScaler", FakeShardedGradScaler)

        runner.setup_grad_scaler()

        assert isinstance(runner.grad_scaler, FakeShardedGradScaler)

    def test_fp16_training_uses_bound_sharded_grad_scaler(self) -> None:
        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=8)})
        grad_scaler = FakeShardedGradScaler()
        runner.grad_scaler = grad_scaler
        try:
            assert runner.model is not None
            assert runner.optimizer is not None
            initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True

            assert grad_scaler.unscale_calls == [runner.optimizer]
            assert grad_scaler.step_calls == [runner.optimizer]
            assert grad_scaler.update_calls == 1
            for parameter, initial in zip(runner.model.parameters(), initial_parameters):
                assert not torch.equal(parameter, initial)
        finally:
            runner.close()

    def test_loss_parallel_auto_follows_tensor_parallel_axis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events: list[str] = []

        @contextmanager
        def fake_loss_parallel():
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        monkeypatch.setattr(parallel_runner_module, "torch_loss_parallel", fake_loss_parallel)
        runner = TensorParallelRunner({"logging.enabled": False, **_parallel_config(tensor=2)})
        try:
            assert runner.loss_parallel_enabled is True
            with runner.infer_context():
                events.append("body")
            assert events == ["enter", "body", "exit"]
        finally:
            runner.close()

    def test_loss_parallel_can_be_disabled_for_tensor_parallel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            parallel_runner_module,
            "torch_loss_parallel",
            lambda: pytest.fail("loss_parallel should not be entered"),
        )
        runner = TensorParallelRunner({"logging.enabled": False, **_parallel_config(tensor=2, loss_parallel=False)})
        try:
            assert runner.loss_parallel_enabled is False
            with runner.infer_context():
                pass
        finally:
            runner.close()

    def test_loss_parallel_true_requires_tensor_parallel_axis(self) -> None:
        with pytest.raises(ValueError, match="parallel.loss_parallel=True requires parallel.axes.tensor > 1"):
            TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=16, loss_parallel=True)})

    def test_activation_checkpointing_requires_target_modules(self) -> None:
        with pytest.raises(ValueError, match="activation_checkpoint.module_classes"):
            TinyParallelRunner(
                {
                    "logging.enabled": False,
                    "activation_checkpoint": {"enabled": True},
                    **_parallel_config(shard=2, pipeline=8, tensor=1),
                }
            )

    def test_fsdp_wraps_target_modules_before_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(4, 4)

            def forward(self, x):
                return self.proj(x)

        class BlockModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([Block(), Block()])
                self.head = nn.Linear(4, 2)

            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return self.head(x)

        class FsdpPolicyRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 2

            @property
            def rank(self) -> int:
                return 0

            def _check_fsdp_prerequisites(self) -> None:
                return

            def __init__(self, config):
                super().__init__(config)
                self.model = BlockModel()
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        calls: list[tuple[nn.Module, dict[str, object]]] = []

        def fake_fully_shard(module, **kwargs):
            calls.append((module, dict(kwargs)))
            return module

        monkeypatch.setattr(parallel_runner_module, "fully_shard", fake_fully_shard)
        runner = FsdpPolicyRunner(
            {
                "logging.enabled": False,
                "fsdp": {
                    "enabled": True,
                    "mesh": "mesh",
                    "module_classes": ["Block"],
                    "reshard_after_forward": "default",
                    "root_reshard_after_forward": False,
                },
                **_parallel_config(shard=2, pipeline=1, tensor=1),
            }
        )
        try:
            assert [module for module, _kwargs in calls] == [
                runner.model.blocks[0],
                runner.model.blocks[1],
                runner.model,
            ]
            assert calls[0][1]["reshard_after_forward"] is True
            assert calls[1][1]["reshard_after_forward"] is True
            assert calls[2][1]["reshard_after_forward"] is False
        finally:
            runner.close()

    def test_fsdp_rejects_unmatched_target_modules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FsdpPolicyRunner(TinyParallelRunner):
            def _check_fsdp_prerequisites(self) -> None:
                return

        monkeypatch.setattr(parallel_runner_module, "fully_shard", lambda module, **kwargs: module)
        with pytest.raises(ValueError, match="fsdp.module_classes matched no modules"):
            FsdpPolicyRunner(
                {
                    "logging.enabled": False,
                    "fsdp": {"enabled": True, "mesh": "mesh", "module_classes": ["MissingBlock"]},
                    **_parallel_config(shard=2, pipeline=8, tensor=1),
                }
            )


# ---------------------------------------------------------------------------
# Pipeline Binding & Materialization
# ---------------------------------------------------------------------------


class TestParallelRunnerPipelineBinding:

    def test_model_owned_parallelize_runs_for_model_axes(self) -> None:
        class ParallelizableLinear(nn.Linear):
            def __init__(self) -> None:
                super().__init__(4, 2)
                self.parallel_context = None

            def parallelize(self, parallel):
                self.parallel_context = parallel

        class ModelParallelRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 16

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                self.model = ParallelizableLinear()
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        runner = ModelParallelRunner(
            {"logging.enabled": False, **_parallel_config(context=2, pipeline=1, tensor=2, expert=2, expert_tensor=2)}
        )
        try:
            assert runner.model_parallel_axes == ("tensor", "context", "expert", "expert_tensor")
            assert runner.model_parallel_degree == 16
            assert isinstance(runner.model, ParallelizableLinear)
            assert runner.model.parallel_context is runner.parallel
        finally:
            runner.close()

    def test_model_parallelization_keeps_pipeline_schedule_current(self) -> None:
        class ParallelizedModule(nn.Module):
            def __init__(self, module: nn.Module) -> None:
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        class ParallelizeRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 1

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                stage = nn.Linear(4, 2)
                self.pipeline_schedule = RecordingPipelineSchedule(stage)
                self.model_parts = [stage]
                self.model = stage
                self.parallelized_inputs: list[nn.Module] = []
                self.pipeline_has_first_stage = True
                self.pipeline_has_last_stage = True
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(stage.parameters(), lr=0.1)

            def parallelize_model(self, model: nn.Module) -> nn.Module:
                self.parallelized_inputs.append(model)
                return ParallelizedModule(model)

        runner = ParallelizeRunner({"logging.enabled": False, **_parallel_config(shard=1, pipeline=1, tensor=1)})
        try:
            assert len(runner.parallelized_inputs) == 1
            assert isinstance(runner.model_parts[0], ParallelizedModule)
            assert runner.model is runner.model_parts[0]
            assert runner.pipeline_schedule.module is runner.model_parts[0]
        finally:
            runner.close()

    def test_compile_keeps_pipeline_schedule_current(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch build.")

        class CompileRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 1

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                stage = nn.Linear(4, 2)
                schedule = RecordingPipelineSchedule(stage)
                self.pipeline_schedule = schedule
                self.model_parts = [stage]
                self.model = stage
                self.pipeline_has_first_stage = True
                self.pipeline_has_last_stage = True
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(stage.parameters(), lr=0.1)

        runner = CompileRunner(
            {
                "logging.enabled": False,
                **_parallel_config(shard=1, pipeline=1, tensor=1),
                "compile": {"enabled": True},
            }
        )
        try:
            assert hasattr(runner.model_parts[0], "_orig_mod")
            assert runner.pipeline_schedule.module is runner.model_parts[0]
        finally:
            runner.close()

    def test_fp8_policy_keeps_pipeline_schedule_current(self) -> None:
        class Fp8WrappedModule(nn.Module):
            def __init__(self, module: nn.Module) -> None:
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        class Fp8Runner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 1

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                stage = nn.Linear(4, 2)
                self.pipeline_schedule = RecordingPipelineSchedule(stage)
                self.model_parts = [stage]
                self.model = stage
                self.pipeline_has_first_stage = True
                self.pipeline_has_last_stage = True
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(stage.parameters(), lr=0.1)

            def apply_fp8_module_policy(self, module: nn.Module, *, recipe=None) -> nn.Module:
                del recipe
                return Fp8WrappedModule(module)

        runner = Fp8Runner(
            {"logging.enabled": False, "precision": "fp8", **_parallel_config(shard=1, pipeline=1, tensor=1)}
        )
        try:
            assert isinstance(runner.model_parts[0], Fp8WrappedModule)
            assert runner.pipeline_schedule.module is runner.model_parts[0]
            assert runner.model is runner.model_parts[0]
        finally:
            runner.close()

    def test_compile_allows_opaque_schedule(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch build.")

        class OpaqueSchedule:
            def step(self, *args, **kwargs):
                del args, kwargs
                return None

            def eval(self, *args, **kwargs):
                del args, kwargs
                return None

        class CompileRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 1

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                stage = nn.Linear(4, 2)
                self.pipeline_schedule = OpaqueSchedule()
                self.model_parts = [stage]
                self.model = stage
                self.pipeline_has_first_stage = True
                self.pipeline_has_last_stage = True
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(stage.parameters(), lr=0.1)

        runner = CompileRunner(
            {
                "logging.enabled": False,
                **_parallel_config(shard=1, pipeline=1, tensor=1),
                "compile": {"enabled": True},
            }
        )
        try:
            assert hasattr(runner.model_parts[0], "_orig_mod")
        finally:
            runner.close()

    def test_pipeline_builds_schedule_for_local_stage(self) -> None:
        captured = {}
        sentinel_schedule = object()

        class PartitionRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)
                self.pipeline_group = object()

            @property
            def world_size(self) -> int:
                return 2

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Linear(4, 2)
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

            def build_pipeline_schedule(self, stage_model):
                captured["stage_model"] = stage_model
                return sentinel_schedule

        runner = PartitionRunner({"logging.enabled": False, **_parallel_config(shard=1, pipeline=2, tensor=1)})
        try:
            assert captured["stage_model"] is runner.model_parts[0]
            assert runner.pipeline_schedule is sentinel_schedule
            assert runner.pipeline_has_first_stage is True
            assert runner.pipeline_has_last_stage is False
        finally:
            runner.close()

    def test_pipeline_uses_configured_local_stage(self) -> None:
        captured = {}

        class PartitionRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)
                self.pipeline_group = object()

            @property
            def world_size(self) -> int:
                return 2

            @property
            def rank(self) -> int:
                return 1

            def __init__(self, config):
                super().__init__(config)
                self.root_model = nn.Sequential(
                    OrderedDict(
                        (
                            ("stem", nn.Linear(4, 4)),
                            ("head", nn.Linear(4, 2)),
                        )
                    )
                )
                self.model = self.root_model
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

            def build_pipeline_schedule(self, stage_model):
                captured["stage_model"] = stage_model
                return RecordingPipelineSchedule(stage_model)

        runner = PartitionRunner(
            {
                "logging.enabled": False,
                **_parallel_config(
                    shard=1,
                    pipeline=2,
                    tensor=1,
                    pipeline_partitions=[["stem"], ["head"]],
                ),
            }
        )
        try:
            assert runner.pipeline_rank == 1
            assert runner.model is runner.root_model.head
            assert runner.model_parts == [runner.root_model.head]
            assert captured["stage_model"] is runner.root_model.head
            assert runner.pipeline_schedule.module is runner.root_model.head
        finally:
            runner.close()

    def test_pipeline_uses_model_owned_partition(self) -> None:
        captured = {}

        class PartitionableModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 2)])
                self.partition_calls = []

            def build_pipeline_model_part(self, *, stage_index, num_stages, module_fqns, parallel):
                self.partition_calls.append((stage_index, num_stages, module_fqns, parallel))
                return self.parts[stage_index]

        class PartitionRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)
                self.pipeline_group = object()

            @property
            def world_size(self) -> int:
                return 2

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                self.root_model = PartitionableModel()
                self.model = self.root_model
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

            def build_pipeline_schedule(self, stage_model):
                captured["stage_model"] = stage_model
                return RecordingPipelineSchedule(stage_model)

        runner = PartitionRunner({"logging.enabled": False, **_parallel_config(shard=1, pipeline=2, tensor=1)})
        try:
            assert runner.root_model.partition_calls == [(0, 2, None, runner.parallel)]
            assert runner.model is runner.root_model.parts[0]
            assert captured["stage_model"] is runner.root_model.parts[0]
        finally:
            runner.close()

    def test_pipeline_stage_indices_support_looped_virtual_stages(self) -> None:
        class PartitionRunner(TinyParallelRunner):
            @property
            def world_size(self) -> int:
                return 2

            @property
            def rank(self) -> int:
                return 1

        runner = PartitionRunner(
            {
                "logging.enabled": False,
                **_parallel_config(
                    shard=1,
                    pipeline=2,
                    tensor=1,
                    pipeline_partitions=[["s0"], ["s1"], ["s2"], ["s3"]],
                ),
            }
        )
        try:
            assert runner.pipeline_stage_indices() == (1, 3)
        finally:
            runner.close()

    def test_pipeline_stage_indices_reject_non_multiple_virtual_stages(self) -> None:
        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=4, pipeline=4, tensor=1)})
        try:
            with pytest.raises(ValueError, match="divisible"):
                runner.pipeline_stage_indices(6)
        finally:
            runner.close()

    def test_pipeline_uses_configured_virtual_stages(self) -> None:
        captured = {}

        class PartitionRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)
                self.pipeline_group = object()

            @property
            def world_size(self) -> int:
                return 2

            @property
            def rank(self) -> int:
                return 1

            def __init__(self, config):
                super().__init__(config)
                self.root_model = nn.Sequential(
                    OrderedDict(
                        (
                            ("s0", nn.Linear(4, 4)),
                            ("s1", nn.Linear(4, 4)),
                            ("s2", nn.Linear(4, 4)),
                            ("s3", nn.Linear(4, 2)),
                        )
                    )
                )
                self.model = self.root_model
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

            def build_pipeline_schedule(self, stage_model):
                captured["stage_model"] = stage_model
                return RecordingMultiPipelineSchedule(stage_model)

        runner = PartitionRunner(
            {
                "logging.enabled": False,
                **_parallel_config(
                    shard=1,
                    pipeline=2,
                    tensor=1,
                    pipeline_partitions=[["s0"], ["s1"], ["s2"], ["s3"]],
                ),
            }
        )
        try:
            assert runner.pipeline_stage_indices() == (1, 3)
            assert runner.model_parts == [runner.root_model.s1, runner.root_model.s3]
            assert captured["stage_model"] == [runner.root_model.s1, runner.root_model.s3]
            assert [stage.module for stage in runner.pipeline_schedule.stages] == runner.model_parts
            assert runner.pipeline_has_first_stage is False
            assert runner.pipeline_has_last_stage is True
        finally:
            runner.close()

    def test_pipeline_rejects_multiple_local_parts_without_schedule(self) -> None:
        class PartitionRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)
                self.pipeline_group = object()

            @property
            def world_size(self) -> int:
                return 2

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Linear(4, 2)
                self.model_parts = [nn.Linear(4, 2), nn.Linear(2, 2)]
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        with pytest.raises(ValueError, match="multiple local model_parts"):
            PartitionRunner({"logging.enabled": False, **_parallel_config(shard=1, pipeline=2, tensor=1)})

    def test_pipeline_compile_keeps_schedule_current(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch build.")

        class PartitionRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)
                self.pipeline_group = object()

            @property
            def world_size(self) -> int:
                return 2

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Linear(4, 2)
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

            def build_pipeline_schedule(self, stage_model):
                return RecordingPipelineSchedule(stage_model)

        runner = PartitionRunner(
            {
                "logging.enabled": False,
                **_parallel_config(shard=1, pipeline=2, tensor=1),
                "compile": {"enabled": True},
            }
        )

        try:
            assert hasattr(runner.model_parts[0], "_orig_mod")
            assert runner.pipeline_schedule.module is runner.model_parts[0]
        finally:
            runner.close()


# ---------------------------------------------------------------------------
# Runtime Behavior
# ---------------------------------------------------------------------------


class TestParallelRunnerRuntimeBehavior:

    @pytest.mark.parametrize("device_type", ["cpu", "cuda"])
    def test_native_pipeline_scaling_clipping_and_scaler_resume(self, device_type) -> None:
        require_gloo()
        if device_type == "cuda" and torch.cuda.device_count() < 2:
            pytest.skip("two CUDA devices are required")
        run_distributed(_pipeline_scaled_backward_worker, worker_args=(device_type,))

    @pytest.mark.parametrize("device_type", ["cpu", "cuda"])
    def test_pipeline_scaled_microbatches_match_native_update(self, device_type) -> None:
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")

        class Schedule:
            def __init__(self, runner):
                self.runner = runner

            def step(self, inputs, *, target, losses):
                for part, labels in zip(inputs.chunk(2), target.chunk(2)):
                    loss = self.runner._pipeline_loss(self.runner.model(part), labels)
                    losses.append(loss.detach())
                    loss.backward()

        class LocalRunner(ParallelRunner):
            @property
            def world_size(self):
                return 1

            @property
            def rank(self):
                return 0

            def init_distributed(self):
                _init_parallel_topology(self)

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Linear(2, 2, bias=False, dtype=torch.float64 if device_type == "cpu" else torch.float32)
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
                self.pipeline_schedule = Schedule(self)

            def materialize_model(self):
                self.model = self.model.to(self.device)
                self.model_parts = [self.model]

        runner = LocalRunner(
            {"device": device_type, "logging.enabled": False, "precision": "fp16" if device_type == "cuda" else None}
        )
        try:
            runner.grad_scaler = torch.amp.GradScaler(device_type, init_scale=128.0, growth_interval=2)
            inputs = torch.arange(8, dtype=runner.model.weight.dtype, device=device_type).view(4, 2) / 4
            targets = torch.zeros_like(inputs)
            initial = runner.model.weight.detach().clone()
            with runner.infer_context():
                reference = runner.criterion(runner.model(inputs), targets)
            gradient = torch.autograd.grad(reference, runner.model.weight)[0]
            _, loss = runner.train_step((inputs, targets))
            torch.testing.assert_close(runner.model.weight, initial - 0.1 * gradient, atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(loss, reference, atol=1e-4, rtol=1e-3)
            state = runner.state_dict()
            assert state["grad_scaler"] == runner.grad_scaler.state_dict()
            runner.grad_scaler.load_state_dict(dict(state["grad_scaler"], scale=32.0))
            runner.load_state_dict(state)
            assert runner.grad_scaler.state_dict() == state["grad_scaler"]
        finally:
            runner.close()

    def test_runner_mode_updates_all_local_model_parts(self) -> None:
        class OpaqueSchedule:
            def step(self, *args, **kwargs):
                del args, kwargs
                return None

            def eval(self, *args, **kwargs):
                del args, kwargs
                return None

        class TrackingModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[bool] = []

            def train(self, mode: bool = True):
                self.calls.append(bool(mode))
                return super().train(mode)

        class ModeRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 1

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                parts = [TrackingModule(), TrackingModule()]
                self.pipeline_schedule = OpaqueSchedule()
                self.model_parts = parts
                self.model = parts[0]

        runner = ModeRunner({"logging.enabled": False, **_parallel_config(shard=1, pipeline=1, tensor=1)})
        try:
            first_part, second_part = runner.model_parts
            runner.mode = "evaluate"
            runner.mode = "train"

            assert first_part.calls == [False, True]
            assert second_part.calls == [False, True]
        finally:
            runner.close()

    def test_middle_pipeline_stage_restores_loader_progress(self) -> None:
        class RestoreRunner(TinyParallelRunner):
            def __init__(self, config):
                super().__init__(config)
                self.pipeline_schedule = object()
                self.pipeline_has_first_stage = False
                self.pipeline_has_last_stage = False

        runner = RestoreRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=8, tensor=1)})
        try:
            loader = StatefulDataLoader(list(range(8)), batch_size=2, shuffle=False)
            target_loader = StatefulDataLoader(list(range(8)), batch_size=2, shuffle=False)
            target_iterator = iter(target_loader)
            next(target_iterator)
            next(target_iterator)
            runner.dataloaders["train"] = loader

            runner.dataloaders.load_state_dict({"train": target_loader.state_dict()})

            assert isinstance(runner.dataloaders["train"], StepProxyLoader)
            assert loader.state_dict()["_num_yielded"] == 2
            assert [batch.tolist() for batch in loader] == [[4, 5], [6, 7]]
        finally:
            runner.close()

    def test_timeout_process_groups_include_parallel_subgroups(self) -> None:
        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=8, tensor=1)})
        try:
            runner.parallel.groups = {
                "replicate": "replicate_group",
                "shard": "shard_group",
                "pipeline": "pipeline_group",
                "tensor": "tensor_group",
            }

            assert runner._timeout_process_groups() == (
                None,
                "replicate_group",
                "shard_group",
                "pipeline_group",
                "tensor_group",
            )
        finally:
            runner.close()

    def test_middle_pipeline_stage_trains_without_loading_batches(self) -> None:
        class NoIterLoader:
            batch_sampler = None
            sampler = None

            def __len__(self) -> int:
                return 3

            def __iter__(self):
                raise AssertionError("middle pipeline stage should not iterate loader payload")

        class MiddleStageRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 3

            @property
            def rank(self) -> int:
                return 1

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Linear(4, 2)
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
                self.pipeline_schedule = RecordingPipelineSchedule()
                self.model_parts = [self.model]
                self.pipeline_has_first_stage = False
                self.pipeline_has_last_stage = False

        runner = MiddleStageRunner(
            {"logging.enabled": False, "logging.interval": 0, **_parallel_config(shard=1, pipeline=3, tensor=1)}
        )
        runner.dataloaders["train"] = NoIterLoader()

        runner.train_epoch(split="train")
        assert runner.train_state.global_step == 3

    def test_load_optimizer_requires_state_dict(self) -> None:
        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=8, tensor=1)})
        with pytest.raises(ValueError, match="checkpoint has no optimizer state"):
            runner.load_optimizer(None)

    def test_optimizer_parameters_include_shared_parameter_once(self) -> None:
        class SharedPart(nn.Module):
            def __init__(self, parameter: nn.Parameter) -> None:
                super().__init__()
                self.weight = parameter

            def forward(self, data):
                return data

        runner = TinyParallelRunner({"logging.enabled": False, **_parallel_config(shard=2, pipeline=8, tensor=1)})
        try:
            shared = nn.Parameter(torch.ones(1))
            runner.model_parts = [SharedPart(shared), SharedPart(shared)]
            parameters = list(runner.iter_optimizer_parameters())
            assert parameters == [shared]
        finally:
            runner.close()

    def test_pipeline_loss_uses_token_weighted_sum(self) -> None:
        runner = TinyParallelRunner(
            {"logging.enabled": False, "accum_steps": 16, **_parallel_config(shard=2, pipeline=8, tensor=1)}
        )
        try:
            runner._pipeline_loss_weighting = "train"
            runner._pipeline_loss_divisor_local = 0.0
            runner._accumulation_divisor_local = 0.0
            runner._prepare_pipeline_loss_normalization(None, torch.zeros(4, 1))

            loss_a = runner._pipeline_loss(torch.ones(1, 1), torch.zeros(1, 1))
            loss_b = runner._pipeline_loss(torch.full((3, 1), 3.0), torch.zeros(3, 1))
            runner._finish_pipeline_loss_normalization([loss_a, loss_b])
            weighted = runner._pipeline_loss_value([loss_a, loss_b])

            assert weighted is not None
            assert weighted.item() == pytest.approx(7.0)
        finally:
            runner._pipeline_loss_weighting = None
            runner.close()

    def test_pipeline_train_step_omits_return_outputs(self) -> None:
        class StrictRunner(CpuParallelRunner):
            def init_distributed(self) -> None:
                _init_parallel_topology(self)

            @property
            def world_size(self) -> int:
                return 1

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Linear(4, 2)
                self.criterion = nn.MSELoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
                self.schedule = RecordingPipelineSchedule(loss=torch.tensor(4.0))
                self.pipeline_schedule = self.schedule
                self.model_parts = [self.model]
                self.pipeline_has_first_stage = True
                self.pipeline_has_last_stage = True

        runner = StrictRunner({"logging.enabled": False, **_parallel_config(shard=1, pipeline=1, tensor=1)})
        _, loss = runner.train_step((torch.randn(2, 4), torch.randn(2, 2)))
        assert runner.schedule.called is True
        assert "return_outputs" not in runner.schedule.last_kwargs
        assert loss is not None
        assert loss.item() == 1.0

    def test_pipeline_loss_logging_uses_local_loss(self) -> None:
        class LocalPipelineRunner(TinyParallelRunner):
            @property
            def world_size(self) -> int:
                return 1

            @property
            def rank(self) -> int:
                return 0

            def __init__(self, config):
                super().__init__(config)
                self.pipeline_schedule = object()

        runner = LocalPipelineRunner({"logging.enabled": False, **_parallel_config(shard=1, pipeline=1, tensor=1)})
        try:
            reduced = runner.reduce_loss_for_logging(torch.tensor(2.0), 4)
            assert reduced is not None
            assert reduced.item() == pytest.approx(2.0)
        finally:
            runner.close()
