from __future__ import annotations

import pytest
import torch
from torch import nn, optim

import danling.runners.dtpp_runner as dtpp_runner_module
from danling.runners import DtppRunner


class TinyDtppRunner(DtppRunner):
    def init_distributed(self) -> None:
        DtppRunner.init_distributed(self)

    @property
    def world_size(self) -> int:
        return 8

    @property
    def rank(self) -> int:
        return 5

    @property
    def device(self):
        return torch.device("cpu")

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


def test_dtpp_runner_computes_topology(monkeypatch) -> None:
    class FakeMesh:
        def get_group(self, dim):
            return f"{dim}_group"

    monkeypatch.setattr(dtpp_runner_module, "init_device_mesh", lambda *args, **kwargs: FakeMesh())
    runner = TinyDtppRunner(
        {
            "log": False,
            "tppp.tp_degree": 2,
            "tppp.pp_degree": 2,
            "dtpp.mode": "hybrid_shard",
            "dtpp.replicate_degree": 2,
            "dtpp.shard_degree": 1,
        }
    )
    assert runner.tp_degree == 2
    assert runner.pp_degree == 2
    assert runner.dp_degree == 2
    assert runner.replicate_degree == 2
    assert runner.shard_degree == 1


def test_dtpp_runner_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="invalid dtpp mode"):
        TinyDtppRunner(
            {
                "log": False,
                "tppp.tp_degree": 2,
                "tppp.pp_degree": 2,
                "dtpp.mode": "bad_mode",
                "dtpp.replicate_degree": 1,
                "dtpp.shard_degree": 2,
            }
        )


def test_dtpp_runner_hybrid_shard_builds_expected_fsdp_kwargs(monkeypatch) -> None:
    captured = {}

    class FakeFSDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            captured.update(kwargs)
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class FakeShardingStrategy:
        HYBRID_SHARD = object()

    class FakeMesh:
        def get_group(self, dim):
            return f"{dim}_group"

    monkeypatch.setattr(dtpp_runner_module, "FSDP", FakeFSDP)
    monkeypatch.setattr(dtpp_runner_module, "ShardingStrategy", FakeShardingStrategy)
    monkeypatch.setattr(dtpp_runner_module, "init_device_mesh", lambda *args, **kwargs: FakeMesh())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class WrappedRunner(DtppRunner):
        def init_distributed(self) -> None:
            DtppRunner.init_distributed(self)

        @property
        def world_size(self) -> int:
            return 8

        @property
        def rank(self) -> int:
            return 5

        @property
        def device(self):
            return torch.device("cpu")

        def __init__(self, config):
            super().__init__(config)
            self.model = nn.Linear(4, 2)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

    runner = WrappedRunner(
        {
            "log": False,
            "tppp.tp_degree": 2,
            "tppp.pp_degree": 2,
            "dtpp.mode": "hybrid_shard",
            "dtpp.replicate_degree": 2,
            "dtpp.shard_degree": 1,
            "compile": {"enable": False},
        }
    )
    assert isinstance(runner.model, FakeFSDP)
    assert captured["process_group"] == ("shard_group", "replicate_group")
    assert captured["sharding_strategy"] is FakeShardingStrategy.HYBRID_SHARD


def test_dtpp_runner_reduce_divides_by_dp_degree(monkeypatch) -> None:
    class FakeMesh:
        def get_group(self, dim):
            return f"{dim}_group"

    monkeypatch.setattr(dtpp_runner_module, "init_device_mesh", lambda *args, **kwargs: FakeMesh())

    class ReduceRunner(TinyDtppRunner):
        def _all_reduce_data_parallel(self, tensor: torch.Tensor) -> torch.Tensor:
            tensor.mul_(self.dp_degree)
            return tensor

    runner = ReduceRunner(
        {
            "log": False,
            "tppp.tp_degree": 2,
            "tppp.pp_degree": 2,
            "dtpp.mode": "full_shard",
            "dtpp.replicate_degree": 1,
            "dtpp.shard_degree": 2,
        }
    )

    value = torch.tensor(2.5)
    reduced = runner.reduce(value.clone())
    assert reduced.item() == 2.5
