from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn, optim

import danling.optim.optimizer as optimizer_module
import danling.runners.torch_runner as torch_runner_module
from danling.runners import TorchRunner


class TinyTorchRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


def test_torch_runner_supports_explicit_components_without_build_hooks() -> None:
    runner = TinyTorchRunner({"log": False})
    assert runner.model is not None
    assert runner.criterion is not None
    assert runner.optimizer is not None
    assert runner.optimizer_container is not None


def test_torch_runner_compile_happens_before_ddp_wrap(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")

    compile_calls = {"count": 0}

    class FakeDDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            del kwargs
            super().__init__()
            self.module = module
            self.was_compiled = bool(getattr(module, "_compiled_marker", False))

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    def fake_compile(module, **kwargs):
        del kwargs
        compile_calls["count"] += 1
        module._compiled_marker = True
        return module

    monkeypatch.setattr(torch_runner_module.torch, "compile", fake_compile)
    monkeypatch.setattr(torch_runner_module.nn.parallel, "DistributedDataParallel", FakeDDP)

    runner = TinyTorchRunner({"log": False, "compile": {"enable": True, "components": ["model"]}})
    assert isinstance(runner.model, FakeDDP)
    assert runner.model.was_compiled is True
    assert compile_calls["count"] == 1


def test_torch_runner_train_context_uses_ddp_no_sync_during_accumulation(monkeypatch) -> None:
    class FakeDDP(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module
            self.no_sync_calls = 0

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        @contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            yield

    monkeypatch.setattr(torch_runner_module.nn.parallel, "DistributedDataParallel", FakeDDP)

    runner = TinyTorchRunner({"log": False, "accum_steps": 2})
    runner.model = FakeDDP(runner.model)

    runner.train_state.micro_step = 0
    with runner.train_context():
        pass

    assert runner.model.no_sync_calls == 1


def test_torch_runner_read_config_accepts_dcp_directory(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".metadata").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(
        torch_runner_module.TorchDistributedCheckpointManager,
        "read_config",
        staticmethod(lambda checkpoint: {"log": False, "stack": "ddp", "checkpoint": str(checkpoint)}),
    )

    config = TinyTorchRunner.read_config(checkpoint_dir)
    assert config["stack"] == "ddp"


def test_torch_runner_from_checkpoint_accepts_mapping_payload() -> None:
    runner = TinyTorchRunner({"log": False})
    checkpoint = runner.state_dict()
    runner.close()

    restored = TinyTorchRunner.from_checkpoint(checkpoint)
    try:
        assert isinstance(restored, TinyTorchRunner)
    finally:
        restored.close()


def test_torch_runner_step_skips_optimizer_update_on_nonfinite_grad(monkeypatch) -> None:
    runner = TinyTorchRunner({"log": False, "skip_nonfinite_grad": True})
    assert runner.optimizer is not None
    assert runner.model is not None

    step_calls = {"count": 0}
    original_step = runner.optimizer.step

    def counted_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(runner.optimizer, "step", counted_step)

    for parameter in runner.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    next(runner.model.parameters()).grad.fill_(float("inf"))

    runner.step()

    assert step_calls["count"] == 0
    assert runner.train_state.global_step == 0


def test_torch_runner_grad_clipping_scans_optimizer_parameters_once(monkeypatch) -> None:
    scans = {"count": 0}

    runner = TinyTorchRunner({"log": False, "max_grad_norm": 1.0})

    monkeypatch.setattr(optimizer_module, "clip_grad_norm_", lambda parameters, max_norm: torch.tensor(0.0))
    original_iter = optimizer_module.OptimizerParameterCache.iter_unique_parameters

    def counted_iter(optimizer):
        scans["count"] += 1
        yield from original_iter(optimizer)

    monkeypatch.setattr(optimizer_module.OptimizerParameterCache, "iter_unique_parameters", staticmethod(counted_iter))
    runner.step()
    runner.step()
    assert scans["count"] == 1


def test_torch_runner_reduce_returns_world_mean(monkeypatch) -> None:
    tensor = torch.tensor(3.0)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch_runner_module.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch_runner_module.dist, "all_reduce", lambda t: t.mul_(4))

    reduced = TorchRunner.reduce(tensor.clone())
    assert reduced.item() == 3.0
