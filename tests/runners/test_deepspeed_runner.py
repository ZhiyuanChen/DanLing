from __future__ import annotations

import torch
from torch import nn, optim

import danling.runners.deepspeed_runner as deepspeed_runner_module
from danling.runners import DeepSpeedRunner


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


class TinyDeepSpeedRunner(DeepSpeedRunner):
    def init_distributed(self) -> None:
        return

    @property
    def device(self):
        return torch.device("cpu")

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


def test_deepspeed_runner_normalizes_deepspeed_config(monkeypatch) -> None:
    monkeypatch.setattr(deepspeed_runner_module.ds, "check", lambda: None)
    monkeypatch.setattr(deepspeed_runner_module, "deepspeed", FakeDeepSpeed)

    runner = TinyDeepSpeedRunner(
        {
            "log": False,
            "precision": "bf16",
            "accum_steps": 4,
            "dataloader": {"batch_size": 8},
            "deepspeed": {"gradient_accumulation_steps": 7},
        }
    )

    ds_cfg = runner.get_deepspeed_config()
    assert ds_cfg["gradient_accumulation_steps"] == 1
    assert ds_cfg["train_micro_batch_size_per_gpu"] == 8
    assert ds_cfg["bf16"]["enabled"] is True


def test_deepspeed_runner_optimizer_step_uses_engine_step(monkeypatch) -> None:
    monkeypatch.setattr(deepspeed_runner_module.ds, "check", lambda: None)
    monkeypatch.setattr(deepspeed_runner_module, "deepspeed", FakeDeepSpeed)

    runner = TinyDeepSpeedRunner({"log": False})
    before = runner.train_state.global_step
    stepped = runner._optimizer_step()

    assert stepped is True
    assert runner.train_state.global_step == before + 1


def test_deepspeed_runner_forces_file_checkpoint_backend(monkeypatch) -> None:
    monkeypatch.setattr(deepspeed_runner_module.ds, "check", lambda: None)
    monkeypatch.setattr(deepspeed_runner_module, "deepspeed", FakeDeepSpeed)

    runner = TinyDeepSpeedRunner({"log": False, "checkpoint": {"backend": "dcp"}})
    DeepSpeedRunner.init_distributed(runner)
    assert runner.config.checkpoint.backend == "file"
