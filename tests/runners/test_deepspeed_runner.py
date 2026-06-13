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
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from danling.data import DataLoaderDict
from danling.runners import DeepSpeedRunner
from danling.runners.checkpoints import FileCheckpointManager
from danling.runners.config import RunnerConfig
from danling.runners.state import RunnerState

HAS_DEEPSPEED = importlib.util.find_spec("deepspeed") is not None


def _bare_deepspeed_runner(
    tmp_path: Path,
    *,
    model: object,
    fail_on_error: bool = False,
    config: Mapping[str, object] | None = None,
) -> DeepSpeedRunner:
    """Build a DeepSpeedRunner skeleton wired to a real FileCheckpointManager.

    The runner bypasses `__init__` (no DeepSpeed engine), but uses the real
    checkpoint manager so checkpoint_health, fail_on_error, and the failure
    hooks exercise production code rather than a hand-mirrored fake.
    """
    config_payload: dict[str, object] = {"logging.enabled": False}
    if config is not None:
        config_payload.update(config)
    config = RunnerConfig(config_payload)
    config["ckpt"].fail_on_error = fail_on_error

    runner = object.__new__(DeepSpeedRunner)
    state = RunnerState(config=config)
    runner.config = config
    runner.state = state
    runner.train_state = state.train
    runner.elastic_state = state.elastic
    runner.rng_state = state.rng
    runner.dataloaders = DataLoaderDict()
    runner.workspace = SimpleNamespace(checkpoint_dir=str(tmp_path))
    runner.ema = None
    runner.optimizer = None
    runner.scheduler = None
    runner.model = model
    runner.checkpoint_manager = FileCheckpointManager(runner)
    return runner


class TestDeepSpeedRunnerCheckpointPaths:

    def test_deepspeed_runner_resolves_pointer_file_and_checkpoint_dir(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_tag = "ckpt-s000000000001"
        (checkpoint_dir / checkpoint_tag).mkdir(parents=True)
        latest_pointer = checkpoint_dir / "latest.pointer"
        best_pointer = checkpoint_dir / "best.pointer"
        latest_pointer.write_text(checkpoint_tag, encoding="utf-8")
        best_pointer.write_text(checkpoint_tag, encoding="utf-8")

        assert DeepSpeedRunner._resolve_deepspeed_checkpoint(checkpoint_dir) == (str(checkpoint_dir), checkpoint_tag)
        assert DeepSpeedRunner._resolve_deepspeed_checkpoint(best_pointer) == (str(checkpoint_dir), checkpoint_tag)
        assert DeepSpeedRunner._resolve_deepspeed_checkpoint(checkpoint_dir / checkpoint_tag) == (
            str(checkpoint_dir),
            checkpoint_tag,
        )

    def test_deepspeed_runner_read_config_accepts_pointer_aliases(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_tag = "ckpt-s000000000001"
        tag_dir = checkpoint_dir / checkpoint_tag
        tag_dir.mkdir(parents=True)
        RunnerConfig({"logging.enabled": False, "seed": 123}).yaml(tag_dir / "runner.yaml")
        (checkpoint_dir / "latest.pointer").write_text(checkpoint_tag, encoding="utf-8")

        config = DeepSpeedRunner.read_config(checkpoint_dir)

        assert config["seed"] == 123
        assert config.get("logging.enabled") is False

    def test_deepspeed_checkpoint_failure_is_recorded_without_interrupting(self, tmp_path: Path) -> None:
        class FailingModel:
            @staticmethod
            def save_checkpoint(*args, **kwargs) -> None:
                del args, kwargs
                raise OSError("disk full")

        runner = _bare_deepspeed_runner(tmp_path, model=FailingModel())

        with pytest.warns(RuntimeWarning, match="checkpoint failed"):
            runner.save_checkpoint(save_best=False, force=True)

        health = runner.checkpoint_manager.checkpoint_health
        assert health.error_count == 1
        assert isinstance(health.first_error, OSError)
        assert health.last_failed_target == "latest"
        assert health.last_successful_target is None

    def test_deepspeed_model_checkpoint_uses_model_only_file_export(self, tmp_path: Path) -> None:
        class PublishingModel:
            module = nn.Linear(2, 1)

            @staticmethod
            def save_checkpoint(*args, **kwargs) -> None:
                del args, kwargs
                raise AssertionError("engine checkpoint should not be used for model export")

        runner = _bare_deepspeed_runner(
            tmp_path,
            model=PublishingModel(),
            config={"ckpt.export_dtype": "fp16"},
        )

        runner.save_model_checkpoint()
        assert runner.checkpoint_manager.close(timeout=1.0) is True

        payload = torch.load(tmp_path / "model.pth", map_location="cpu", weights_only=False)
        assert list(payload) == ["model"]
        assert payload["model"]["weight"].dtype == torch.float16

    def test_deepspeed_checkpoint_keep_latest_k_prunes_history_tags(self, tmp_path: Path) -> None:
        class SavingModel:
            def __init__(self) -> None:
                self.tags: list[str] = []

            def save_checkpoint(self, checkpoint_dir, tag, client_state, save_latest) -> None:
                del client_state, save_latest
                self.tags.append(tag)
                tag_dir = Path(checkpoint_dir) / tag
                tag_dir.mkdir(parents=True)
                (tag_dir / "marker.txt").write_text(tag, encoding="utf-8")

        runner = _bare_deepspeed_runner(
            tmp_path,
            model=SavingModel(),
            config={"epochs": 3, "ckpt.interval": 1, "ckpt.keep_latest_k": 1},
        )

        try:
            runner.save_checkpoint(epochs=0, save_best=False)
            runner.save_checkpoint(epochs=1, save_best=False)
            runner.save_checkpoint(epochs=2, save_best=False)
            assert runner.checkpoint_manager.close()
        finally:
            runner.checkpoint_manager.close()

        assert not (tmp_path / "ckpt-e000001").exists()
        assert not (tmp_path / "ckpt-e000002").exists()
        assert (tmp_path / "ckpt-e000003").is_dir()
        assert (tmp_path / "latest.pointer").read_text(encoding="utf-8") == "ckpt-e000003"

    def test_deepspeed_checkpoint_failure_raises_when_configured(self, tmp_path: Path) -> None:
        class FailingModel:
            @staticmethod
            def save_checkpoint(*args, **kwargs) -> None:
                del args, kwargs
                raise OSError("disk full")

        runner = _bare_deepspeed_runner(tmp_path, model=FailingModel(), fail_on_error=True)

        with (
            pytest.warns(RuntimeWarning, match="checkpoint failed"),
            pytest.raises(OSError, match="disk full"),
        ):
            runner.save_checkpoint(save_best=False, force=True)

        health = runner.checkpoint_manager.checkpoint_health
        assert health.error_count == 1
        assert isinstance(health.first_error, OSError)


class TestDeepSpeedRunnerRuntimeBoundaries:

    def test_deepspeed_runner_declares_torchft_runtime_unsupported(self) -> None:
        assert DeepSpeedRunner._supports_torchft_runtime is False


class TestDeepSpeedRunnerRealBackend:

    @pytest.mark.skipif(not HAS_DEEPSPEED, reason="deepspeed is not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_deepspeed_runner_initializes_real_engine_on_cuda(self) -> None:
        class RealTinyDeepSpeedRunner(DeepSpeedRunner):
            def init_distributed(self) -> None:
                return

            @property
            def device(self):
                return torch.device("cuda", 0)

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Linear(4, 2)
                self.criterion = nn.MSELoss()

        runner = RealTinyDeepSpeedRunner(
            {
                "logging.enabled": False,
                "dataloader": {"batch_size": 2},
                "optim": {"type": "sgd", "lr": 0.1},
                "deepspeed": {"zero_optimization": {"stage": 0}},
            }
        )
        try:
            inputs = torch.randn(2, 4, device=runner.device)
            targets = torch.randn(2, 2, device=runner.device)
            loss = F.mse_loss(runner.model(inputs), targets)
            runner.backward(loss)
            assert runner.optimizer_step() is True
            assert runner.train_state.global_step == 1
        finally:
            runner.close()
