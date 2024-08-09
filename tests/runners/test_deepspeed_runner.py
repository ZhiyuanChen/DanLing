from __future__ import annotations

from pathlib import Path

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


def test_deepspeed_runner_load_checkpoint_tracks_resume_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(deepspeed_runner_module.ds, "check", lambda: None)
    monkeypatch.setattr(deepspeed_runner_module, "deepspeed", FakeDeepSpeed)

    checkpoint_dir = tmp_path / "resume_ckpt"
    checkpoint_tag = "latest-tag"
    (checkpoint_dir / checkpoint_tag).mkdir(parents=True)
    (checkpoint_dir / "latest").write_text(checkpoint_tag, encoding="utf-8")

    runner = TinyDeepSpeedRunner({"log": False})
    runner.load_checkpoint(checkpoint_dir)
    assert runner.config.resume == str(checkpoint_dir)


def test_deepspeed_runner_load_pretrained_tracks_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(deepspeed_runner_module.ds, "check", lambda: None)
    monkeypatch.setattr(deepspeed_runner_module, "deepspeed", FakeDeepSpeed)

    checkpoint_dir = tmp_path / "pretrained_ckpt"
    checkpoint_tag = "latest-tag"
    (checkpoint_dir / checkpoint_tag).mkdir(parents=True)
    (checkpoint_dir / "latest").write_text(checkpoint_tag, encoding="utf-8")

    runner = TinyDeepSpeedRunner({"log": False})
    runner.load_pretrained(checkpoint_dir)
    assert runner.config.pretrained == str(checkpoint_dir)


def test_deepspeed_runner_auto_resume_uses_config_source(monkeypatch) -> None:
    monkeypatch.setattr(deepspeed_runner_module.ds, "check", lambda: None)
    monkeypatch.setattr(deepspeed_runner_module, "deepspeed", FakeDeepSpeed)

    calls: list[object] = []

    def fake_load_checkpoint(self, checkpoint, *args, **kwargs):
        del args, kwargs
        calls.append(checkpoint)

    monkeypatch.setattr(DeepSpeedRunner, "load_checkpoint", fake_load_checkpoint)
    runner = TinyDeepSpeedRunner({"log": False, "resume": "checkpoint-latest"})
    assert calls == ["checkpoint-latest"]
    runner.close()


def test_deepspeed_runner_auto_resume_uses_backend_latest_source(monkeypatch) -> None:
    monkeypatch.setattr(deepspeed_runner_module.ds, "check", lambda: None)
    monkeypatch.setattr(deepspeed_runner_module, "deepspeed", FakeDeepSpeed)

    calls: list[object] = []

    def fake_load_checkpoint(self, checkpoint, *args, **kwargs):
        del args, kwargs
        calls.append(checkpoint)

    monkeypatch.setattr(DeepSpeedRunner, "load_checkpoint", fake_load_checkpoint)
    runner = TinyDeepSpeedRunner({"log": False, "auto_resume": True})
    assert calls == [runner.checkpoint_dir]
    runner.close()
