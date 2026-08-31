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
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn, optim

import danling.runners.torch_runner as torch_runner_module
from danling.runners import TorchRunner
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


class NoOptimizerTorchRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()


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
        self._optimizer_param = nn.Parameter(torch.zeros(()))
        self.optimizer = optim.SGD([self._optimizer_param], lr=0.1)

    def train_step(self, data):
        loss = torch.tensor(float(data))
        self.step()
        return None, loss

    def evaluate_step(self, data):
        return None, torch.tensor(float(data))


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
        skipped = self.backward(loss)
        self.step()
        return None, loss.detach().new_zeros(()) if skipped else loss.detach()

    def evaluate_step(self, data):
        del data
        return None, self.param.square().detach()


class WeightedLossRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class MaskWeightedLossRunner(WeightedLossRunner):
    def __init__(self, config):
        super().__init__(config)

        class TokenMeanModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                del attention_mask
                return self.weight * tokens.float().mean()

        self.model = TokenMeanModel()
        self.criterion = lambda pred, target: pred.square()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


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


class _FakePlacement:
    def __init__(self, *, replicate: bool = False, partial: bool = False) -> None:
        self._replicate = replicate
        self._partial = partial

    def is_replicate(self) -> bool:
        return self._replicate

    def is_partial(self) -> bool:
        return self._partial


class _FakeReplicate:
    def is_replicate(self) -> bool:
        return True

    def is_partial(self) -> bool:
        return False


class _FakeDTensor:
    def __init__(self, local: torch.Tensor, placements: list[_FakePlacement]) -> None:
        self._local = local
        self.placements = placements

    def to_local(self) -> torch.Tensor:
        return self._local

    def detach(self) -> _FakeDTensor:
        return self

    def redistribute(self, *, placements: list[object]) -> _FakeDTensor:
        return _FakeDTensor(self._local, placements)


def _ddp_compile_wrap_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner(
        {"logging.enabled": False, "dist.backend": "gloo", "compile": {"enabled": True}}
    )
    try:
        assert runner.model is not None
        initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
        _prediction, loss = runner.train_step((torch.ones(2, 4), torch.zeros(2, 2)))

        assert loss is not None and torch.isfinite(loss)
        assert runner.train_state.global_step == 1
        for parameter, initial in zip(runner.model.parameters(), initial_parameters):
            assert not torch.equal(parameter, initial)
            gathered = [torch.zeros_like(parameter) for _ in range(world_size)]
            dist.all_gather(gathered, parameter)
            torch.testing.assert_close(gathered[0], gathered[1])
    finally:
        runner.close()


def _ddp_no_sync_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"logging.enabled": False, "dist.backend": "gloo", "accum_steps": 2})
    try:
        assert runner.model is not None
        initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
        first_input = torch.full((2, 4), float(rank + 1))
        target = torch.zeros(2, 2)

        runner.train_step((first_input, target))
        assert runner.train_state.global_step == 0
        runner.train_step((torch.ones(2, 4), target))

        assert runner.train_state.global_step == 1
        for parameter, initial in zip(runner.model.parameters(), initial_parameters):
            assert not torch.equal(parameter, initial)
            gathered = [torch.zeros_like(parameter) for _ in range(world_size)]
            dist.all_gather(gathered, parameter)
            torch.testing.assert_close(gathered[0], gathered[1])
    finally:
        runner.close()


def _torch_runner_reduce_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"logging.enabled": False, "dist.backend": "gloo", "ckpt": {"backend": "file"}})
    try:
        reduced = runner.reduce(torch.tensor(float(rank + 1)))
        assert reduced.item() == pytest.approx(1.5)
    finally:
        runner.close()


def _torch_runner_weighted_loss_logging_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner({"logging.enabled": False, "dist.backend": "gloo"})
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
            "logging.enabled": False,
            "dist.backend": "gloo",
            "ckpt": {"backend": "file"},
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


def _torch_runner_nonfinite_loss_accumulation_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    runner = DistributedTinyTorchRunner(
        {
            "logging.enabled": False,
            "dist.backend": "gloo",
            "ckpt": {"backend": "file"},
            "accum_steps": 2,
            "skip_nonfinite_loss": True,
        }
    )
    try:
        assert runner.model is not None
        runner.optimizer.zero_grad(set_to_none=True)
        target = torch.zeros(2, 2)

        runner.train_step((torch.full((2, 4), float(rank + 1)), target))
        first_grad = next(runner.model.parameters()).grad.detach().clone()
        first_gathered = [torch.zeros_like(first_grad) for _ in range(world_size)]
        dist.all_gather(first_gathered, first_grad)
        assert not torch.allclose(first_gathered[0], first_gathered[1])

        invalid_input = torch.full((2, 4), float("inf")) if rank else torch.ones(2, 4)
        _prediction, loss = runner.train_step((invalid_input, target))
        assert loss is not None and loss.item() == 0
        assert runner.train_state.global_step == 1

        for parameter in runner.model.parameters():
            gathered = [torch.zeros_like(parameter) for _ in range(world_size)]
            dist.all_gather(gathered, parameter)
            torch.testing.assert_close(gathered[0], gathered[1])
            assert torch.isfinite(parameter).all()
    finally:
        runner.close()


def _torch_runner_preinitialized_pg_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    with process_group("gloo", rank, world_size), pytest.raises(RuntimeError, match="already initialized"):
        DistributedTinyTorchRunner({"logging.enabled": False, "dist.backend": "gloo", "ckpt": {"backend": "file"}})


# ---------------------------------------------------------------------------
# Construction & Restore
# ---------------------------------------------------------------------------


class TestTorchRunnerBootstrap:

    def test_fault_tolerance_requires_torchft_package(self) -> None:
        if importlib.util.find_spec("torchft") is not None:
            pytest.skip("torchft is installed")
        with pytest.raises(ImportError, match="torchft"):
            TinyTorchRunner({"logging.enabled": False, "ft": {"enabled": True}})

    def test_train_rejects_unknown_splits(self) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "steps": 1,
                "train_splits": ["a"],
                "evaluate_splits": ["v1"],
            }
        )
        try:
            with pytest.raises(ValueError, match="unknown training split"):
                runner.train(train_splits=["a", "missing"], evaluate_splits=[])
            with pytest.raises(ValueError, match="unknown evaluation split"):
                runner.train(train_splits=["a"], evaluate_splits=["v1", "missing"])
        finally:
            runner.close()

    def test_evaluate_rejects_unknown_requested_split(self) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "steps": 1,
                "train_splits": ["a"],
                "evaluate_splits": ["v1"],
            }
        )
        try:
            with pytest.raises(ValueError, match="unknown evaluation split"):
                runner.evaluate(evaluate_splits=["missing"])
        finally:
            runner.close()

    def test_runner_builds_optimizer_from_model_parameters(self) -> None:
        class AutoOptimRunner(TorchRunner):
            def init_distributed(self) -> None:
                return

            model = nn.Linear(4, 2)

        runner = AutoOptimRunner({"logging.enabled": False, "optim": {"type": "sgd", "lr": 0.1}})
        try:
            assert runner.model is not None
            assert runner.optimizer is not None
            initial = [parameter.detach().clone() for parameter in runner.model.parameters()]
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True

            assert runner.train_state.global_step == 1
            assert any(
                not torch.equal(parameter, before) for parameter, before in zip(runner.model.parameters(), initial)
            )
        finally:
            runner.close()

    def test_optimizer_config_applies_parameter_groups(self) -> None:
        class AutoOptimRunner(TorchRunner):
            def init_distributed(self) -> None:
                return

            model = nn.Sequential(
                OrderedDict(
                    (
                        ("stem", nn.Linear(4, 4)),
                        ("head", nn.Linear(4, 2)),
                    )
                )
            )

        runner = AutoOptimRunner(
            {
                "logging.enabled": False,
                "optim": {
                    "type": "sgd",
                    "lr": 1.0,
                    "weight_decay": 0.1,
                    "param_groups": [
                        {
                            "pattern": r"^head\.",
                            "lr_multiplier": 0.25,
                            "weight_decay_multiplier": 0.0,
                        }
                    ],
                },
            }
        )
        try:
            assert runner.model is not None
            assert runner.optimizer is not None
            with torch.no_grad():
                for parameter in runner.model.parameters():
                    parameter.zero_()
                    parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True

            for parameter in runner.model.head.parameters():
                torch.testing.assert_close(parameter, torch.full_like(parameter, -0.25))
            for parameter in runner.model.stem.parameters():
                torch.testing.assert_close(parameter, torch.full_like(parameter, -1.0))
        finally:
            runner.close()

    def test_optimizer_config_warns_when_parameter_group_matches_nothing(self) -> None:
        class AutoOptimRunner(TorchRunner):
            def init_distributed(self) -> None:
                return

            model = nn.Linear(4, 2)

        with pytest.warns(RuntimeWarning, match="matched no parameters"):
            runner = AutoOptimRunner(
                {
                    "logging.enabled": False,
                    "optim": {
                        "type": "sgd",
                        "lr": 0.1,
                        "param_groups": [{"pattern": "missing"}],
                    },
                }
            )
        try:
            assert runner.model is not None
            assert runner.optimizer is not None
            initial = [parameter.detach().clone() for parameter in runner.model.parameters()]
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True
            assert any(
                not torch.equal(parameter, before) for parameter, before in zip(runner.model.parameters(), initial)
            )
        finally:
            runner.close()

    def test_dataloader_config_forwards_sampler_and_collate_fn(self) -> None:
        class ReverseSampler:
            def __iter__(self):
                return iter((3, 2, 1, 0))

            def __len__(self):
                return 4

        def collate_fn(batch):
            return tuple(batch)

        runner = TinyTorchRunner({"logging.enabled": False})
        try:
            runner.config.dataloader.batch_size = 2
            runner.config.dataloader.sampler = ReverseSampler()
            runner.config.dataloader.collate_fn = collate_fn
            runner.datasets["train"] = list(range(4))
            runner.build_dataloaders()

            assert list(runner.dataloaders["train"]) == [(3, 2), (1, 0)]
        finally:
            runner.close()

    def test_dataloader_config_forwards_batch_sampler(self) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "dataloader": {
                    "batch_size": 99,
                    "batch_sampler": [[2, 0], [3, 1]],
                    "drop_last": True,
                },
            }
        )
        try:
            runner.datasets["train"] = list(range(4))
            runner.build_dataloaders()

            assert [batch.tolist() for batch in runner.dataloaders["train"]] == [[2, 0], [3, 1]]
        finally:
            runner.close()

    def test_deterministic_dataloader_repeats_shuffled_batches(self) -> None:
        deterministic_was_enabled = torch.are_deterministic_algorithms_enabled()

        def shuffled_batches() -> list[list[int]]:
            runner = TinyTorchRunner(
                {
                    "logging.enabled": False,
                    "seed": 1016,
                    "deterministic": True,
                    "dataloader": {"batch_size": 2, "shuffle": True},
                }
            )
            try:
                runner.datasets["train"] = list(range(8))
                runner.build_dataloaders()
                return [batch.tolist() for batch in runner.dataloaders["train"]]
            finally:
                runner.close()

        try:
            assert shuffled_batches() == shuffled_batches()
        finally:
            torch.use_deterministic_algorithms(deterministic_was_enabled, warn_only=True)

    def test_dataloader_reports_invalid_worker_options(self) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
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

    def test_evaluate_step_applies_configured_precision(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "precision": "bf16"})
        try:
            prediction, loss = runner.evaluate_step((torch.ones(2, 4), torch.zeros(2, 2)))
            assert prediction.dtype == torch.bfloat16
            assert loss is not None and torch.isfinite(loss)
        finally:
            runner.close()

    def test_gather_infer_predictions_restores_order_and_deduplicates_padding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Loader:
            sampler = [0, 2, 4]

            def __iter__(self):
                yield torch.tensor([[0.0], [2.0], [4.0]])

            def __len__(self):
                return 1

        def gather(gathered, local):
            gathered[:] = [local, [(1, 1.0), (3, 3.0), (4, 4.0)]]

        runner = TinyTorchRunner({"logging.enabled": False})
        runner.model = nn.Identity()
        runner.dataloaders["infer"] = Loader()
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        monkeypatch.setattr(dist, "all_gather_object", gather)
        monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
        try:
            assert runner.infer("infer") == [0.0, 1.0, 2.0, 3.0, 4.0]
        finally:
            monkeypatch.setenv("WORLD_SIZE", "1")
            runner.close()

    def test_gather_infer_predictions_rejects_output_without_sampler_indices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Loader:
            sampler = [0]

            def __iter__(self):
                yield torch.tensor([[0.0], [1.0]])

            def __len__(self):
                return 1

        runner = TinyTorchRunner({"logging.enabled": False})
        runner.model = nn.Identity()
        runner.dataloaders["infer"] = Loader()
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
        try:
            with pytest.raises(ValueError, match="more predictions"):
                runner.infer("infer")
        finally:
            monkeypatch.setenv("WORLD_SIZE", "1")
            runner.close()

    def test_load_checkpoint_updates_config_source(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"logging.enabled": False})
        checkpoint_path = tmp_path / "checkpoint-latest.pth"
        try:
            torch.save(source.state_dict(), checkpoint_path)
        finally:
            source.close()

        runner = TinyTorchRunner({"logging.enabled": False})
        try:
            runner.load_checkpoint(checkpoint_path)
            assert runner.config.checkpoint == str(checkpoint_path)
        finally:
            runner.close()

    def test_load_pretrained_updates_config_source(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"logging.enabled": False})
        checkpoint_path = tmp_path / "checkpoint-best.pth"
        try:
            assert source.model is not None
            torch.save({"model": source.unwrap(source.model).state_dict()}, checkpoint_path)
        finally:
            source.close()

        runner = TinyTorchRunner({"logging.enabled": False})
        try:
            runner.load_pretrained(checkpoint_path)
            assert runner.config.pretrained == str(checkpoint_path)
        finally:
            runner.close()

    def test_from_pretrained_accepts_mapping(self) -> None:
        source = TinyTorchRunner({"logging.enabled": False})
        try:
            assert source.model is not None
            with torch.no_grad():
                source.model.weight.fill_(2.0)
                source.model.bias.fill_(-1.0)
            checkpoint = {"model": source.unwrap(source.model).state_dict()}
        finally:
            source.close()

        runner = TinyTorchRunner.from_pretrained({"logging.enabled": False}, checkpoint)
        try:
            assert runner.model is not None
            assert torch.equal(runner.model.weight, torch.full_like(runner.model.weight, 2.0))
            assert torch.equal(runner.model.bias, torch.full_like(runner.model.bias, -1.0))
        finally:
            runner.close()

    def test_ddp_training_uses_compiled_model(self) -> None:
        require_gloo()
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch build.")
        run_distributed(_ddp_compile_wrap_worker, world_size=2)

    def test_gradient_accumulation_uses_ddp_no_sync(self) -> None:
        require_gloo()
        run_distributed(_ddp_no_sync_worker, world_size=2)


class TestTorchRunnerCheckpointInterop:

    @pytest.mark.skipif(
        importlib.util.find_spec("tensorboard") is not None,
        reason="requires an environment without the optional TensorBoard dependency",
    )
    def test_tensorboard_missing_dependency_has_actionable_error(self, tmp_path: Path) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "tensorboard.enabled": False,
            }
        )
        try:
            with pytest.raises(ImportError, match=r"danling\[tensorboard\]"):
                runner.init_tensorboard()
        finally:
            runner.close()

    def test_tensorboard_writes_events_to_configured_directory(self, tmp_path: Path) -> None:
        pytest.importorskip("tensorboard", reason="TensorBoard support is an optional dependency")
        log_dir = tmp_path / "events"
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "tensorboard": {
                    "enabled": True,
                    "log_dir": str(log_dir),
                    "filename_suffix": ".tb",
                },
            }
        )
        try:
            assert runner.writer is not None
            runner.writer.add_scalar("train/loss", 1.0, 1)
            runner.writer.flush()
        finally:
            runner.close()

        assert list(log_dir.glob("events.out.tfevents.*.tb"))

    def test_read_config_accepts_dcp_directory(self, tmp_path: Path) -> None:
        runner = DcpConfigTorchRunner(
            {
                "logging.enabled": False,
                "name": "dcp-config-test",
                "workspace.root": str(tmp_path),
                "ckpt": {"backend": "dcp", "async_mode": "disabled", "interval": 1},
            }
        )
        try:
            runner.train_state.global_step = 1
            runner.save_checkpoint(force=True)
            config = DcpConfigTorchRunner.read_config(Path(runner.workspace.checkpoint_dir) / "latest")
        finally:
            runner.close()

        assert config["name"] == "dcp-config-test"
        assert config.get("ckpt.backend") == "dcp"

    def test_from_checkpoint_accepts_in_memory_checkpoint(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        try:
            runner.train_state.global_step = 7
            checkpoint = runner.state_dict()
        finally:
            runner.close()

        restored = TinyTorchRunner.from_checkpoint(checkpoint)
        try:
            assert restored.train_state.global_step == 7
        finally:
            restored.close()

    def test_state_dict_restores_the_next_torch_random_draw(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        try:
            torch.manual_seed(1016)
            torch.rand(3)
            checkpoint = runner.state_dict()
            expected = torch.rand(3)
            torch.rand(3)

            runner.load_state_dict(checkpoint)

            torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)
        finally:
            runner.close()

    def test_from_checkpoint_uses_requested_path_over_auto_restore(self, tmp_path: Path) -> None:
        source = TinyTorchRunner({"logging.enabled": False})
        checkpoint_path = tmp_path / "torch-runner.pth"
        try:
            checkpoint = dict(source.state_dict())
            checkpoint["runner"]["resume"] = True
            checkpoint["runner"]["pretrained"] = "stale-pretrained"
            torch.save(checkpoint, checkpoint_path)
        finally:
            source.close()

        restored = TinyTorchRunner.from_checkpoint(checkpoint_path)
        try:
            assert restored.config.checkpoint == str(checkpoint_path)
            assert restored.config.resume is False
            assert restored.config.pretrained is None
        finally:
            restored.close()

    def test_load_checkpoint_restores_stateful_dataloader_progress(self) -> None:
        source = StatefulDatasetTorchRunner(
            {"logging.enabled": False, "dataloader": {"batch_size": 2, "shuffle": False}}
        )
        try:
            assert next(iter(source.dataloaders["train"])).tolist() == [0, 1]
            checkpoint = source.state_dict()
        finally:
            source.close()

        restored = StatefulDatasetTorchRunner(
            {"logging.enabled": False, "dataloader": {"batch_size": 2, "shuffle": False}}
        )
        try:
            restored.load_checkpoint(checkpoint)
            remaining_batches = [batch.tolist() for batch in restored.dataloaders["train"]]
            assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
        finally:
            restored.close()

    def test_from_checkpoint_path_restores_stateful_dataloader_progress(self, tmp_path: Path) -> None:
        source = StatefulDatasetTorchRunner(
            {"logging.enabled": False, "dataloader": {"batch_size": 2, "shuffle": False}}
        )
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
            "logging.enabled": False,
            "workspace.root": str(tmp_path),
            "dataloader": {"batch_size": 2, "shuffle": False},
            "ckpt": {"async_mode": "disabled", "interval": 1},
        }
        source = StatefulDatasetTorchRunner(config)
        try:
            assert next(iter(source.dataloaders["train"])).tolist() == [0, 1]
            source.save_checkpoint(force=True)
        finally:
            source.close()

        restored_config = dict(config)
        restored_config["resume"] = True
        restored = StatefulDatasetTorchRunner(restored_config)
        try:
            remaining_batches = [batch.tolist() for batch in restored.dataloaders["train"]]
            assert remaining_batches == [[2, 3], [4, 5], [6, 7]]
            assert restored.config.checkpoint == os.path.join(restored.workspace.checkpoint_dir, "latest.pth")
        finally:
            restored.close()


class TestTorchRunnerProfiling:
    def test_profiling_writes_trace_and_operator_table(self, tmp_path: Path) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "profiling": {"enabled": True, "wait": 0, "warmup": 0, "active": 1, "trace_dir": "trace-output"},
            }
        )
        trace_dir = Path(runner.workspace.dir) / "trace-output"
        try:
            _prediction, loss = runner.train_step((torch.ones(2, 4), torch.zeros(2, 2)))
            assert loss is not None and torch.isfinite(loss)
            assert runner.train_state.global_step == 1
        finally:
            runner.close()

        assert list(trace_dir.rglob("*.pt.trace.json"))
        assert list(trace_dir.rglob("operator_table.txt"))


class TestTorchRunnerDistributedRuntime:
    def test_init_distributed_rejects_preinitialized_process_group(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_preinitialized_pg_worker, world_size=2)

    def test_nonfinite_skip_decision_is_collective(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_nonfinite_skip_sync_worker, world_size=2)

    def test_nonfinite_loss_preserves_and_synchronizes_accumulated_gradients(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_nonfinite_loss_accumulation_worker, world_size=2)


# ---------------------------------------------------------------------------
# Runtime Mechanics
# ---------------------------------------------------------------------------


class TestTorchRunnerOptimization:
    def test_nonfinite_loss_is_replaced_with_zero_before_backward(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "skip_nonfinite_loss": True})
        assert runner.model is not None
        initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
        try:
            _prediction, loss = runner.train_step((torch.full((2, 4), float("inf")), torch.zeros(2, 2)))

            assert loss is not None and loss.item() == 0
            assert runner.train_state.global_step == 1
            for parameter, initial in zip(runner.model.parameters(), initial_parameters):
                torch.testing.assert_close(parameter, initial)
        finally:
            runner.close()

    def test_optimizer_step_requires_optimizer(self) -> None:
        runner = NoOptimizerTorchRunner({"logging.enabled": False})
        try:
            assert runner.optimizer is None
            with pytest.raises(ValueError, match="no optimizer"):
                runner.optimizer_step()
            assert runner.train_state.global_step == 0
        finally:
            runner.close()

    def test_runner_owned_fp16_requires_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="fp16 precision requires a CUDA device"):
            TinyTorchRunner({"logging.enabled": False, "precision": "fp16"})

    def test_step_skips_optimizer_update_on_nonfinite_grad(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "skip_nonfinite_grad": True})
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

    def test_optimizer_step_allows_eval_only_ema_without_update(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        runner.ema = nn.Identity()
        try:
            assert runner.model is not None
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True

            assert runner.train_state.global_step == 1
        finally:
            runner.close()

    def test_optimizer_step_rejects_noncallable_ema_update_before_mutation(self) -> None:
        class BadEma(nn.Module):
            update = "not callable"

        runner = TinyTorchRunner({"logging.enabled": False})
        runner.ema = BadEma()
        try:
            assert runner.model is not None
            initial_parameters = [parameter.detach().clone() for parameter in runner.model.parameters()]
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            with pytest.raises(TypeError, match="not callable"):
                runner.optimizer_step()

            assert runner.train_state.global_step == 0
            for parameter, initial in zip(runner.model.parameters(), initial_parameters):
                torch.testing.assert_close(parameter, initial)
        finally:
            runner.close()

    def test_optimizer_step_records_grad_norm_when_clipping(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "max_grad_norm": 1.0})
        try:
            assert runner.model is not None
            for parameter in runner.model.parameters():
                parameter.grad = torch.ones_like(parameter)

            assert runner.optimizer_step() is True
            result = runner.get_step_result()
            assert "grad_norm" in result
            assert result["grad_norm"] > 0
        finally:
            runner.close()

    def test_steps_standard_pytorch_scheduler_after_optimizer_step(self) -> None:
        runner = TinyStepLRTorchRunner({"logging.enabled": False})
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


class TestTorchRunnerScheduling:

    def test_steps_danling_scheduler_after_optimizer_step(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False, "sched": {"type": "linear", "total_steps": 8}})
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

    def test_defaults_metric_scheduler_to_epoch_interval(self) -> None:
        runner = TinyPlateauTorchRunner({"logging.enabled": False})
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
        with pytest.raises(ValueError, match="metric-based schedulers require `sched.interval='epoch'`"):
            TinyPlateauTorchRunner({"logging.enabled": False, "sched": {"interval": "step"}})

    def test_steps_reduce_on_plateau_after_epoch_result(self, tmp_path: Path) -> None:
        runner = EpochSchedulerRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "epochs": 2,
                "train_splits": ["train"],
                "evaluate_splits": ["val"],
                "sched": {"type": "reduce_on_plateau", "patience": 0, "factor": 0.5},
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

    def test_steps_epoch_interval_scheduler_once_per_step_mode_train_round(self, tmp_path: Path) -> None:
        runner = TinyStepLRTorchRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 4, "sched": {"interval": "epoch"}}
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

    def test_raises_for_missing_scheduler_monitor(self, tmp_path: Path) -> None:
        runner = EpochSchedulerRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "epochs": 1,
                "train_splits": ["train"],
                "evaluate_splits": ["val"],
                "sched": {"type": "reduce_on_plateau", "monitor": "val.accuracy"},
            }
        )
        try:
            with pytest.raises(ValueError, match="sched.monitor"):
                runner.train()
        finally:
            runner.close()

    def test_replicated_dtensor_loss_can_be_logged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_runner_module, "TorchDTensor", _FakeDTensor)
        runner = TinyTorchRunner({"logging.enabled": False})
        local = torch.tensor(3.0)
        tensor = _FakeDTensor(
            local,
            [_FakePlacement(replicate=True)],
        )

        try:
            loss = runner.reduce_loss_for_logging(tensor, loss_n=1)
            assert loss is not None
            torch.testing.assert_close(loss, local.to(dtype=torch.float64))
        finally:
            runner.close()

    def test_partial_dtensor_loss_can_be_logged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_runner_module, "TorchDTensor", _FakeDTensor)
        monkeypatch.setattr(torch_runner_module, "TorchReplicate", _FakeReplicate)
        runner = TinyTorchRunner({"logging.enabled": False})
        local = torch.tensor(3.0)
        tensor = _FakeDTensor(local, [_FakePlacement(partial=True)])

        try:
            loss = runner.reduce_loss_for_logging(tensor, loss_n=1)
            assert loss is not None
            torch.testing.assert_close(loss, local.to(dtype=torch.float64))
        finally:
            runner.close()

    def test_sharded_dtensor_loss_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_runner_module, "TorchDTensor", _FakeDTensor)
        runner = TinyTorchRunner({"logging.enabled": False})
        tensor = _FakeDTensor(torch.tensor(3.0), [_FakePlacement()])

        try:
            with pytest.raises(ValueError, match="Cannot reduce DTensor"):
                runner.reduce_loss_for_logging(tensor, loss_n=1)
        finally:
            runner.close()

    def test_reduce_returns_world_mean(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_reduce_worker, world_size=2)

    def test_loss_logging_reports_weighted_mean(self) -> None:
        require_gloo()
        run_distributed(_torch_runner_weighted_loss_logging_worker, world_size=2)


# ---------------------------------------------------------------------------
# Execution Loops
# ---------------------------------------------------------------------------


class TestTorchRunnerEpochExecution:

    def test_train_epoch_supports_unsized_loader(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 1024})
        try:
            runner.dataloaders["train"] = StreamingLoader(1.0, 2.0)
            result = runner.train_epoch("train")
            assert math.isfinite(result["loss"])
            assert runner.train_state.global_step == 2
        finally:
            runner.close()

    def test_variable_length_batches_use_token_weighted_loss(self) -> None:
        runner = WeightedLossRunner({"logging.enabled": False, "accum_steps": 2, "logging.interval": 0})
        runner.dataloaders["train"] = [
            (torch.ones((1, 1)), torch.zeros((1, 1))),
            (torch.full((3, 1), 3.0), torch.zeros((3, 1))),
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
        runner = WeightedLossRunner({"logging.enabled": False, "accum_steps": 4, "logging.interval": 0})
        runner.dataloaders["train"] = [
            (torch.ones((1, 1)), torch.zeros((1, 1))),
            (torch.full((3, 1), 3.0), torch.zeros((3, 1))),
        ]

        try:
            runner.train_epoch("train")
            assert runner.model is not None
            assert runner.unwrap(runner.model).weight.detach().item() == pytest.approx(-0.4)
            assert runner.train_state.global_step == 1
            assert runner.train_state.micro_step == 4
        finally:
            runner.close()

    def test_loss_weighting_uses_distributed_normalizer(self) -> None:
        class GradientScaleRunner(WeightedLossRunner):
            def _loss_normalizer_sync_divisor(self) -> int:
                return 4

            def _reduce_loss_normalizer_total(self, local_total: float) -> float:
                del local_total
                return 10.0

        runner = GradientScaleRunner({"logging.enabled": False, "accum_steps": 2})
        runner.dataloaders["train"] = [
            (torch.ones((1, 1)), torch.zeros((1, 1))),
            (torch.full((3, 1), 3.0), torch.zeros((3, 1))),
        ]

        try:
            runner.train_epoch("train")
            assert runner.model is not None
            assert runner.unwrap(runner.model).weight.detach().item() == pytest.approx(-1.24)
        finally:
            runner.close()

    def test_explicit_loss_normalizer_controls_loss_weighting(self) -> None:
        runner = WeightedLossRunner({"logging.enabled": False, "accum_steps": 2})
        runner.dataloaders["train"] = [
            {"input": torch.ones((1, 1)), "target": torch.zeros((1, 1)), "loss_normalizer": 9},
            {"input": torch.full((1, 1), 3.0), "target": torch.zeros((1, 1)), "loss_normalizer": 1},
        ]

        try:
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(1.8)
            assert runner.model is not None
            assert runner.unwrap(runner.model).weight.detach().item() == pytest.approx(0.64)
        finally:
            runner.close()

    def test_mean_loss_uses_target_size_as_normalizer(self) -> None:
        runner = WeightedLossRunner({"logging.enabled": False, "accum_steps": 2})
        runner.dataloaders["train"] = [
            (torch.ones((1, 1)), torch.zeros((1, 1))),
            (torch.full((3, 1), 3.0), torch.zeros((3, 1))),
        ]

        try:
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(7.0)
        finally:
            runner.close()

    def test_attention_mask_controls_loss_weighting_without_targets(self) -> None:
        runner = MaskWeightedLossRunner({"logging.enabled": False, "accum_steps": 2})
        runner.dataloaders["train"] = [
            {
                "input": {
                    "tokens": torch.ones((1, 3), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }
            },
            {
                "input": {
                    "tokens": torch.full((1, 3), 3, dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 0, 0]], dtype=torch.long),
                }
            },
        ]

        try:
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(3.0)
        finally:
            runner.close()

    def test_custom_batch_schema_uses_unweighted_loss(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False})
        runner.dataloaders["train"] = [1.0, 9.0]

        try:
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(5.0)
        finally:
            runner.close()

    def test_train_epoch_skips_peak_memory_sampling_without_interval_logs(self) -> None:
        runner = TelemetryRunner({"logging.enabled": False, "logging.interval": 0})
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
            assert "mem_alloc_mb" not in result
            assert "mem_reserved_mb" not in result
            assert "flops_per_s" not in result
            assert "mfu" not in result
        finally:
            runner.close()

    def test_train_epoch_reports_nested_tensor_tokens_per_s(self) -> None:
        runner = TelemetryRunner({"logging.enabled": False, "logging.interval": 0})
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
            assert "mem_alloc_mb" not in result
            assert "mem_reserved_mb" not in result
        finally:
            runner.close()

    def test_mode_updates_model_and_ema_training_state(self) -> None:
        runner = TinyTorchRunner({"logging.enabled": False})
        model = nn.Identity()
        ema = nn.Identity()
        runner.model = model
        runner.ema = ema

        try:
            runner.mode = "evaluate"
            assert model.training is False
            assert ema.training is False

            runner.mode = "train"
            assert model.training is True
            assert ema.training is True
        finally:
            runner.close()

    def test_evaluate_epoch_supports_unsized_loader(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 1024})
        try:
            runner.dataloaders["val"] = StreamingLoader(1.0, 2.0)
            result = runner.evaluate_epoch("val")
            assert math.isfinite(result["loss"])
        finally:
            runner.close()


class TestTorchRunnerLoopResultStability:

    def test_train_epoch_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 2})
        try:
            runner.dataloaders["train"] = [1.0, 2.0, 3.0]
            result = runner.train_epoch("train")
            assert result["loss"] == pytest.approx(2.0)
        finally:
            runner.close()

    def test_evaluate_epoch_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 2})
        try:
            runner.dataloaders["val"] = [1.0, 2.0, 3.0]
            result = runner.evaluate_epoch("val")
            assert result["loss"] == pytest.approx(2.0)
        finally:
            runner.close()

    def test_evaluate_epoch_allows_zero_weight_padding(self) -> None:
        class ZeroWeightedEvaluationRunner(StreamingEpochRunner):
            def _get_loss_normalizer(self, data):
                return int(data[1])

            def evaluate_step(self, data):
                return None, torch.tensor(float(data[0]))

        runner = ZeroWeightedEvaluationRunner({"logging.enabled": False, "logging.interval": 0})
        try:
            runner.dataloaders["val"] = [(1.0, 1), (100.0, 0)]
            result = runner.evaluate_epoch("val")
            assert result["loss"] == pytest.approx(1.0)
        finally:
            runner.close()


class TestTorchRunnerStepExecution:

    def test_train_steps_result_is_independent_of_log_interval(self, tmp_path: Path) -> None:
        runner = StreamingEpochRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 3, "logging.interval": 2}
        )
        try:
            runner.dataloaders["train"] = [1.0, 2.0, 3.0]
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            assert runner.results[runner.train_state.global_step]["train"]["loss"] == pytest.approx(2.0)
        finally:
            runner.close()

    def test_train_steps_writes_full_latest_and_model_checkpoint(self, tmp_path: Path) -> None:
        runner = StreamingEpochRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "steps": 1,
                "ckpt.async_mode": "disabled",
                "ckpt.export_dtype": "fp16",
            }
        )
        try:
            runner.model = nn.Linear(1, 1)
            runner.dataloaders["train"] = [1.0]
            runner.train_steps(train_splits=["train"], evaluate_splits=[])

            checkpoint_dir = Path(runner.workspace.checkpoint_dir)
            latest_payload = torch.load(checkpoint_dir / "latest.pth", map_location="cpu", weights_only=False)
            assert "runner" in latest_payload
            assert "optimizer" in latest_payload
            assert latest_payload["model"]["weight"].dtype == torch.float32

            model_payload = torch.load(checkpoint_dir / "model.pth", map_location="cpu", weights_only=False)
            assert "model" in model_payload
            assert model_payload["model"]["weight"].dtype == torch.float16
        finally:
            runner.close()

    def test_train_steps_logs_once_per_optimizer_step_under_accumulation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runner = StreamingEpochRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "steps": 2,
                "accum_steps": 2,
                "logging.interval": 2,
            }
        )
        runner.dataloaders["train"] = [1.0, 2.0, 3.0, 4.0]

        try:
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            assert capsys.readouterr().out.count("training on train [2/2]") == 1
        finally:
            runner.close()

    def test_train_steps_shares_budget_across_splits(self, tmp_path: Path) -> None:
        runner = StreamingEpochRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 2, "logging.interval": 0}
        )
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

            def set_epoch(self, epoch: int) -> None:
                self.epoch = int(epoch)

            def __iter__(self):
                base = self.epoch * 10
                return iter([float(base + 1), float(base + 2), float(base + 3)])

            def __len__(self) -> int:
                return 3

        runner = StreamingEpochRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 4, "logging.interval": 0}
        )
        try:
            loader = EpochAwareLoader()
            runner.dataloaders["train"] = loader
            runner.train_steps(train_splits=["train"], evaluate_splits=[])

            result = runner.results[runner.train_state.global_step]["train"]
            assert runner.train_state.global_step == 4
            assert result["loss"] == pytest.approx(11.0)
        finally:
            runner.close()

    def test_train_steps_uses_monotonic_progress_across_loader_rollover(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runner = StreamingEpochRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "steps": 3, "logging.interval": 1}
        )
        runner.dataloaders["train"] = StreamingLoader(1.0, 2.0)

        try:
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
            output = capsys.readouterr().out
            positions = [output.index(f"training on train [{step}/3]") for step in (1, 2, 3)]
            assert positions == sorted(positions)
        finally:
            runner.close()


class TestTorchRunnerStepEvaluation:

    def test_evaluate_steps_result_is_independent_of_log_interval(self) -> None:
        runner = StreamingEpochRunner({"logging.enabled": False, "logging.interval": 2})
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
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "accum_steps": 2,
                "ckpt": {"async_mode": "disabled"},
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
