from __future__ import annotations

import pytest
import torch
from torch import nn, optim

import danling.runners.fsdp_runner as fsdp_runner_module
import danling.runners.torch_runner as torch_runner_module
from danling.runners import FsdpRunner


class LocalTinyFsdpRunner(FsdpRunner):
    def init_distributed(self) -> None:
        return

    @property
    def distributed(self) -> bool:
        return True

    @property
    def device(self):
        return torch.device("cpu")

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


def test_fsdp_runner_requires_distributed_mode(monkeypatch) -> None:
    class DistCheckRunner(FsdpRunner):
        model = nn.Linear(4, 2)

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch_runner_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(torch_runner_module.dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="WORLD_SIZE > 1"):
        DistCheckRunner({"log": False})


def test_fsdp_runner_forces_dcp_backend(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch_runner_module.dist, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_runner_module.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_runner_module.torch.cuda, "set_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_runner_module.torch.cuda, "is_available", lambda: True)

    class FakeFSDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            del kwargs
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(fsdp_runner_module, "FSDP", FakeFSDP)

    class DistTinyFsdpRunner(FsdpRunner):
        def __init__(self, config):
            super().__init__(config)
            self.model = nn.Linear(4, 2)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

    runner = DistTinyFsdpRunner({"log": False, "checkpoint": {"backend": "file"}})
    assert runner.config.checkpoint.backend == "dcp"


def test_fsdp_runner_materialize_wraps_model(monkeypatch) -> None:
    class FakeFSDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            self.kwargs = kwargs
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(fsdp_runner_module, "FSDP", FakeFSDP)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    runner = LocalTinyFsdpRunner({"log": False, "fsdp": {"use_orig_params": True}})
    assert isinstance(runner.model, FakeFSDP)
    assert runner.model.kwargs["use_orig_params"] is True


def test_fsdp_runner_hsdp_validates_world_size(monkeypatch) -> None:
    class FakeFSDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            del kwargs
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class FakeRunner(LocalTinyFsdpRunner):
        @property
        def world_size(self) -> int:
            return 4

    monkeypatch.setattr(fsdp_runner_module, "FSDP", FakeFSDP)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="world size mismatch"):
        FakeRunner({"log": False, "fsdp": {"mode": "hybrid_shard", "replicate_degree": 2, "shard_degree": 1}})


def test_fsdp_runner_hsdp_builds_device_mesh(monkeypatch) -> None:
    captured = {}

    class FakeFSDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            self.kwargs = kwargs
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class FakeShardingStrategy:
        HYBRID_SHARD = object()

    class FakeRunner(LocalTinyFsdpRunner):
        @property
        def world_size(self) -> int:
            return 4

    def fake_init_device_mesh(device_type, mesh_shape, mesh_dim_names=None):
        captured["device_type"] = device_type
        captured["mesh_shape"] = mesh_shape
        captured["mesh_dim_names"] = mesh_dim_names
        return "mesh"

    monkeypatch.setattr(fsdp_runner_module, "FSDP", FakeFSDP)
    monkeypatch.setattr(fsdp_runner_module, "ShardingStrategy", FakeShardingStrategy)
    monkeypatch.setattr(fsdp_runner_module, "init_device_mesh", fake_init_device_mesh)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    runner = FakeRunner({"log": False, "fsdp": {"mode": "hybrid_shard", "replicate_degree": 2, "shard_degree": 2}})
    assert runner.model.kwargs["device_mesh"] == "mesh"
    assert runner.model.kwargs["sharding_strategy"] is FakeShardingStrategy.HYBRID_SHARD
    assert captured["mesh_shape"] == (2, 2)


def test_fsdp_runner_load_optimizer_requires_state_dict(monkeypatch) -> None:
    class FakeFSDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            del kwargs
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fsdp_runner_module, "FSDP", FakeFSDP)
    runner = LocalTinyFsdpRunner({"log": False})
    with pytest.raises(ValueError, match="checkpoint has no optimizer state"):
        runner.load_optimizer(None)
