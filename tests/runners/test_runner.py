from __future__ import annotations

import pytest
import torch
from torch import nn, optim

import danling.runners.deepspeed_runner as deepspeed_runner_module
import danling.runners.dtpp_runner as dtpp_runner_module
import danling.runners.fsdp_runner as fsdp_runner_module
import danling.runners.torch_runner as torch_runner_module
import danling.runners.tppp_runner as tppp_runner_module
from danling.runners import DeepSpeedRunner, DtppRunner, FsdpRunner, Runner, TorchRunner, TpppRunner


class _TinyRunnerMixin:
    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class TinyRunner(_TinyRunnerMixin, Runner):
    pass


def test_runner_defaults_to_torch_stack() -> None:
    runner = TinyRunner({"log": False})
    assert isinstance(runner, TorchRunner)
    assert runner.config.stack == "ddp"


def test_runner_unknown_stack_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stack"):
        TinyRunner({"log": False, "stack": "unknown_stack"})


def test_runner_selects_fsdp_stack(monkeypatch) -> None:
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

    runner = TinyRunner({"log": False, "stack": "fsdp"})
    assert isinstance(runner, FsdpRunner)
    assert runner.config.stack == "fsdp"


def test_runner_selects_tppp_stack(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch_runner_module.dist, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_runner_module.dist, "broadcast_object_list", lambda *args, **kwargs: None)

    class FakeMesh:
        def get_group(self, name):
            del name
            return object()

    monkeypatch.setattr(tppp_runner_module, "init_device_mesh", lambda *args, **kwargs: FakeMesh())

    runner = TinyRunner({"log": False, "stack": "tppp"})
    assert isinstance(runner, TpppRunner)
    assert runner.config.stack == "tppp"


def test_runner_selects_dtpp_stack(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch_runner_module.dist, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_runner_module.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_runner_module.torch.cuda, "set_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_runner_module.torch.cuda, "is_available", lambda: True)

    class FakeMesh:
        def get_group(self, name):
            del name
            return object()

    class FakeFSDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            del kwargs
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(dtpp_runner_module, "init_device_mesh", lambda *args, **kwargs: FakeMesh())
    monkeypatch.setattr(dtpp_runner_module, "FSDP", FakeFSDP)
    monkeypatch.setattr(dtpp_runner_module.DtppRunner, "device", property(lambda self: torch.device("cpu")))

    runner = TinyRunner({"log": False, "stack": "dtpp"})
    assert isinstance(runner, DtppRunner)
    assert runner.config.stack == "dtpp"


def test_runner_selects_deepspeed_stack(monkeypatch) -> None:
    monkeypatch.setattr(deepspeed_runner_module.ds, "check", lambda: None)

    class FakeEngine(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module
            self.global_steps = 0

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        def backward(self, loss):
            loss.backward()

        def step(self):
            self.global_steps += 1

        def save_checkpoint(self, save_dir, tag, client_state=None, save_latest=False):
            del client_state, save_latest
            return (save_dir, tag)

        def load_checkpoint(self, load_dir, tag=None, load_module_only=False):
            del load_module_only
            if tag is None:
                return None, None
            return (load_dir, tag), {}

    class FakeDeepSpeed:
        DeepSpeedEngine = FakeEngine

        @staticmethod
        def initialize(model, optimizer=None, lr_scheduler=None, config=None):
            del config
            return FakeEngine(model), optimizer, None, lr_scheduler

    monkeypatch.setattr(deepspeed_runner_module, "deepspeed", FakeDeepSpeed)

    runner = TinyRunner({"log": False, "stack": "deepspeed"})
    assert isinstance(runner, DeepSpeedRunner)
    assert runner.config.stack == "deepspeed"
