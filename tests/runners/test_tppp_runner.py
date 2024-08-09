from __future__ import annotations

import pytest
import torch
from torch import nn, optim

from danling.runners import TpppRunner
from danling.runners.tppp_runner import TpppTopology


def _init_tppp_topology(runner: TpppRunner) -> None:
    runner.topology = TpppTopology(
        world_size=runner.world_size,
        rank=runner.rank,
        tp_degree=runner.config.tppp.tp_degree,
        pp_degree=runner.config.tppp.pp_degree,
    )


class TinyTpppRunner(TpppRunner):
    def init_distributed(self) -> None:
        _init_tppp_topology(self)

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


def test_tppp_runner_computes_topology() -> None:
    runner = TinyTpppRunner({"log": False, "tppp.tp_degree": 2, "tppp.pp_degree": 4})
    assert runner.tp_degree == 2
    assert runner.pp_degree == 4
    assert runner.dp_degree == 2
    assert runner.tp_rank == 1
    assert runner.pp_rank == 0
    assert runner.dp_rank == 1


def test_tppp_runner_rejects_invalid_topology() -> None:
    class BadRunner(TinyTpppRunner):
        @property
        def world_size(self) -> int:
            return 10

        @property
        def rank(self) -> int:
            return 0

    with pytest.raises(ValueError, match="WORLD_SIZE"):
        BadRunner({"log": False, "tppp.tp_degree": 4, "tppp.pp_degree": 3})


def test_tppp_runner_dataloader_shards_by_dp_only(monkeypatch) -> None:
    captured = {}

    class CaptureSampler:
        def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
            captured["num_replicas"] = num_replicas
            captured["rank"] = rank
            del shuffle
            self.dataset = dataset

        def __iter__(self):
            return iter(range(len(self.dataset)))

        def __len__(self):
            return len(self.dataset)

    monkeypatch.setattr(torch.utils.data.distributed, "DistributedSampler", CaptureSampler)
    runner = TinyTpppRunner({"log": False, "tppp.tp_degree": 2, "tppp.pp_degree": 4})
    runner.datasets["train"] = list(range(32))
    runner.build_dataloaders()

    assert captured["num_replicas"] == 2
    assert captured["rank"] == 1


def test_tppp_runner_compile_rebinds_single_stage_schedule(monkeypatch) -> None:
    class CompiledWrapper(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class DummySchedule:
        def __init__(self, module: nn.Module) -> None:
            self.module = module

        def step(self, *args, **kwargs):
            del args, kwargs
            return None

        def eval(self, *args, **kwargs):
            del args, kwargs
            return None

    class CompileRunner(TpppRunner):
        def init_distributed(self) -> None:
            _init_tppp_topology(self)

        @property
        def world_size(self) -> int:
            return 1

        @property
        def rank(self) -> int:
            return 0

        def __init__(self, config):
            super().__init__(config)
            stage = nn.Linear(4, 2)
            schedule = DummySchedule(stage)
            self.pipeline_schedule = schedule
            self.model_parts = [stage]
            self.model = stage
            self.pp_has_first_stage = True
            self.pp_has_last_stage = True
            self.criterion = nn.MSELoss()
            self.optimizer = optim.SGD(stage.parameters(), lr=0.1)

    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: CompiledWrapper(module))

    runner = CompileRunner(
        {"log": False, "tppp.tp_degree": 1, "tppp.pp_degree": 1, "compile": {"enable": True, "components": ["model"]}}
    )
    assert isinstance(runner.model_parts[0], CompiledWrapper)
    assert runner.pipeline_schedule.module is runner.model_parts[0]


def test_tppp_runner_compile_allows_opaque_schedule(monkeypatch) -> None:
    class CompiledWrapper(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class OpaqueSchedule:
        def step(self, *args, **kwargs):
            del args, kwargs
            return None

        def eval(self, *args, **kwargs):
            del args, kwargs
            return None

    class CompileRunner(TpppRunner):
        def init_distributed(self) -> None:
            _init_tppp_topology(self)

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
            self.pp_has_first_stage = True
            self.pp_has_last_stage = True
            self.criterion = nn.MSELoss()
            self.optimizer = optim.SGD(stage.parameters(), lr=0.1)

    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: CompiledWrapper(module))

    runner = CompileRunner(
        {
            "log": False,
            "tppp.tp_degree": 1,
            "tppp.pp_degree": 1,
            "compile": {"enable": True, "components": ["model"]},
        }
    )
    assert isinstance(runner.model_parts[0], CompiledWrapper)


def test_tppp_runner_middle_stage_train_epoch_skips_loader_payload_iteration() -> None:
    class NoIterLoader:
        batch_sampler = None
        sampler = None

        def __len__(self) -> int:
            return 3

        def __iter__(self):
            raise AssertionError("middle PP stage should not iterate loader payload")

    class DummySchedule:
        def step(self, *args, **kwargs):
            del args, kwargs
            return None

        def eval(self, *args, **kwargs):
            del args, kwargs
            return None

    class MiddleStageRunner(TpppRunner):
        def init_distributed(self) -> None:
            _init_tppp_topology(self)

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
            self.pipeline_schedule = DummySchedule()
            self.model_parts = [self.model]
            self.pp_has_first_stage = False
            self.pp_has_last_stage = False

    runner = MiddleStageRunner({"log": False, "log_interval": 0, "tppp.tp_degree": 1, "tppp.pp_degree": 3})
    runner.dataloaders["train"] = NoIterLoader()

    runner.train_epoch(split="train")
    assert runner.train_state.global_step == 3


def test_tppp_runner_load_optimizer_requires_state_dict_when_optimizer_exists() -> None:
    runner = TinyTpppRunner({"log": False, "tppp.tp_degree": 2, "tppp.pp_degree": 4})
    with pytest.raises(ValueError, match="checkpoint has no optimizer state"):
        runner.load_optimizer(None)


def test_tppp_runner_materialize_model_builds_default_schedule(monkeypatch) -> None:
    class PartitionRunner(TpppRunner):
        def init_distributed(self) -> None:
            _init_tppp_topology(self)
            self.pp_group = object()

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

    captured = {}
    sentinel_schedule = object()

    def fake_build_schedule(self, stage_model):
        captured["stage_model"] = stage_model
        return sentinel_schedule

    monkeypatch.setattr(TpppRunner, "build_pipeline_schedule", fake_build_schedule)

    runner = PartitionRunner({"log": False, "tppp.tp_degree": 1, "tppp.pp_degree": 2})

    assert captured["stage_model"] is runner.model_parts[0]
    assert runner.pipeline_schedule is sentinel_schedule
    assert runner.pp_has_first_stage is True
    assert runner.pp_has_last_stage is False


def test_tppp_runner_materialize_model_auto_pipeline_compiles_and_rebinds(monkeypatch) -> None:
    class CompiledWrapper(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class DummySchedule:
        def __init__(self, module: nn.Module) -> None:
            self.module = module

        def step(self, *args, **kwargs):
            del args, kwargs
            return None

        def eval(self, *args, **kwargs):
            del args, kwargs
            return None

    class PartitionRunner(TpppRunner):
        def init_distributed(self) -> None:
            _init_tppp_topology(self)
            self.pp_group = object()

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

    def fake_build_schedule(self, stage_model):
        return DummySchedule(stage_model)

    monkeypatch.setattr(TpppRunner, "build_pipeline_schedule", fake_build_schedule)
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: CompiledWrapper(module))

    runner = PartitionRunner(
        {
            "log": False,
            "tppp.tp_degree": 1,
            "tppp.pp_degree": 2,
            "compile": {"enable": True, "components": ["model"]},
        }
    )

    assert isinstance(runner.model_parts[0], CompiledWrapper)
    assert runner.pipeline_schedule.module is runner.model_parts[0]


def test_tppp_runner_train_step_does_not_pass_return_outputs() -> None:
    class RecordingSchedule:
        def __init__(self) -> None:
            self.called = False
            self.last_kwargs = {}

        def step(self, *args, **kwargs):
            del args
            self.last_kwargs = dict(kwargs)
            losses = kwargs.get("losses")
            if losses is not None:
                losses.append(torch.tensor(1.0))
            self.called = True
            return None

        def eval(self, *args, **kwargs):
            del args, kwargs
            return None

    class StrictRunner(TpppRunner):
        def init_distributed(self) -> None:
            _init_tppp_topology(self)

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
            self.schedule = RecordingSchedule()
            self.pipeline_schedule = self.schedule
            self.model_parts = [self.model]
            self.pp_has_first_stage = True
            self.pp_has_last_stage = True

    runner = StrictRunner({"log": False, "tppp.tp_degree": 1, "tppp.pp_degree": 1})
    _, loss = runner.train_step((torch.randn(2, 4), torch.randn(2, 2)))
    assert runner.schedule.called is True
    assert "return_outputs" not in runner.schedule.last_kwargs
    assert loss is not None
    assert loss.item() == 1.0
