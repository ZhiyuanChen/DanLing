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

import copy
import importlib.util
import math
import os
import random
import signal
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn, optim

import danling.runners.torch_runner as torch_runner_module
from danling.metrics import GlobalMetrics, MultiTaskMetrics, StreamMetrics
from danling.metrics.functional import mse
from danling.runners import RunnerConfig, TorchRunner
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
        native = runner.unwrap(runner.model)
        assert isinstance(native, nn.Linear)
        assert set(runner.state_dict()["model"]) == set(native.state_dict())
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
        # A rank may have no local contribution while another still has one.
        if rank == 0:
            next(runner.model.parameters()).grad = None

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


def _torch_runner_empty_token_rank_worker(rank: int, world_size: int, reduction: str) -> None:
    class ClassificationRunner(TorchRunner):
        def __init__(self, config):
            super().__init__(config)
            self.model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
            self.criterion = (
                nn.MSELoss()
                if reduction == "nan"
                else nn.CrossEntropyLoss(weight=torch.tensor([0.25, 1.75], dtype=torch.float64), reduction=reduction)
            )
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.1)

    configure_distributed_env(rank, world_size)
    runner = ClassificationRunner(
        {
            "device": "cpu",
            "dist.backend": "gloo",
            "logging.enabled": False,
            "logging.interval": 0,
            "skip_nonfinite_loss": True,
        }
    )
    try:
        initial = runner.unwrap(runner.model).weight.detach().clone()
        inputs = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
        targets = torch.tensor([-100, 1])
        reference = (
            runner.criterion(runner.unwrap(runner.model)(inputs[1:]), torch.zeros(1, 2, dtype=torch.float64))
            if reduction == "nan"
            else runner.criterion(runner.unwrap(runner.model)(inputs), targets)
        )
        gradient = torch.autograd.grad(reference, runner.unwrap(runner.model).weight)[0]
        runner.dataloaders["train"] = [(inputs[rank : rank + 1], targets[rank : rank + 1])]
        if reduction == "nan":
            runner.dataloaders["train"] = [
                {
                    "input": inputs[rank : rank + 1],
                    "target": torch.full((1, 2), float("nan") if rank == 0 else 0.0, dtype=torch.float64),
                    "loss_normalizer": 0.0 if rank == 0 else 0.75,
                }
            ]
        runner.train_epoch()
        expected = initial - 0.1 * (gradient + 0.1 * initial)
        torch.testing.assert_close(runner.unwrap(runner.model).weight, expected)
        assert runner.train_state.global_step == 1
        runner.dataloaders["train"] = [(inputs[rank : rank + 1], torch.full((1,), -100))]
        if reduction == "nan":
            runner.dataloaders["train"] = [
                {
                    "input": inputs[rank : rank + 1],
                    "target": torch.full((1, 2), float("nan"), dtype=torch.float64),
                    "loss_normalizer": 0.0,
                }
            ]
        runner.train_epoch()
        torch.testing.assert_close(runner.unwrap(runner.model).weight, expected)
        assert runner.train_state.global_step == 1
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# Construction & Restore
# ---------------------------------------------------------------------------


class TestMetaModelLifecycle:
    @pytest.mark.parametrize("restore_pretrained", [False, True])
    def test_meta_build_preserves_ties_buffers_dtype_and_optimizer_ownership(self, tmp_path, restore_pretrained):
        events = []

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Linear(3, 3, bias=False, dtype=torch.float64)
                self.right = nn.Linear(3, 3, bias=False, dtype=torch.float64)
                self.right.weight = self.left.weight
                self.register_buffer("scale", torch.empty((), dtype=torch.float64))
                self.register_buffer("counter", torch.empty((), dtype=torch.int64), persistent=False)
                self.register_buffer("scale_alias", self.scale)

            def init_weights(self):
                self.left.weight.fill_(0.25)
                self.scale.fill_(2)
                self.counter.fill_(7)

            def forward(self, value):
                return self.right(self.left(value)) * self.scale

        class MetaRunner(TorchRunner):
            def init_distributed(self):
                return

            def build_model(self):
                model = Model()
                events.append(("build", model.left.weight.is_meta))
                return model

            def initialize_model(self, model):
                events.append(("init", model.left.weight.is_meta))
                super().initialize_model(model)

            def build_optimizer(self):
                events.append(("optim", self.model.left.weight.is_meta))
                super().build_optimizer()

        config = {
            "device": "cpu",
            "logging.enabled": False,
            "model_init.meta": True,
            "optim": {"type": "sgd", "lr": 0.1},
        }
        expected_weight = 0.25
        if restore_pretrained:
            checkpoint = tmp_path / "pretrained.pth"
            torch.save(
                {
                    "model": {
                        "left.weight": torch.full((3, 3), 0.75, dtype=torch.float64),
                        "right.weight": torch.full((3, 3), 0.75, dtype=torch.float64),
                        "scale": torch.tensor(3.0, dtype=torch.float64),
                        "scale_alias": torch.tensor(3.0, dtype=torch.float64),
                    }
                },
                checkpoint,
            )
            config["pretrained"] = str(checkpoint)
            expected_weight = 0.75
        runner = MetaRunner(config)
        try:
            model = runner.model
            assert events == [("build", True), ("init", False), ("optim", False)]
            assert model.left.weight is model.right.weight
            assert model.scale is model.scale_alias
            assert model.left.weight.dtype == model.scale.dtype == torch.float64
            assert model.counter.dtype == torch.int64 and model.counter.item() == 7
            torch.testing.assert_close(model.left.weight, torch.full_like(model.left.weight, expected_weight))
            assert model.scale.item() == (3 if restore_pretrained else 2)
            assert runner.optimizer.param_groups[0]["params"] == [model.left.weight]
            value = torch.ones(2, 3, dtype=torch.float64)
            loss = model(value).square().sum()
            loss.backward()
            assert model.left.weight.grad is not None
            runner.optimizer.step()
            assert not torch.equal(model.left.weight, torch.full_like(model.left.weight, expected_weight))
        finally:
            runner.close()

    def test_initialized_model_is_not_reinitialized_when_meta_construction_is_enabled(self):
        class ExistingRunner(TorchRunner):
            def init_distributed(self):
                return

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Linear(2, 2, dtype=torch.float64)
                with torch.no_grad():
                    self.model.weight.fill_(3)
                    self.model.bias.fill_(4)

            def initialize_model(self, model):
                raise AssertionError("initialized model must not be reset")

        runner = ExistingRunner({"device": "cpu", "logging.enabled": False, "model_init.meta": True})
        try:
            torch.testing.assert_close(
                runner.model(torch.ones(1, 2, dtype=torch.float64)), torch.full((1, 2), 10.0, dtype=torch.float64)
            )
        finally:
            runner.close()


class TestTorchRunnerBootstrap:

    def test_named_compile_groups_preserve_outputs_gradients_names_and_hooks(self, device):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(4, 4, dtype=torch.float64)
                self.second = nn.Linear(4, 4, dtype=torch.float64)
                self.skip = nn.Linear(4, 4, dtype=torch.float64)

            def compile_regions(self):
                return {"blocks": (self.first, self.second)}

            @torch.compiler.disable
            def eager_skip(self, x):
                return self.skip(x)

            def forward(self, x):
                return self.second(self.first(x).sin()) + self.eager_skip(x)

        class RegionRunner(TorchRunner):
            def init_distributed(self):
                return

            def __init__(self, config):
                super().__init__(config)
                self.model = Model()

        runner = RegionRunner(
            {
                "device": str(device),
                "logging.enabled": False,
                "compile": {"enabled": True, "regions": ["blocks"], "backend": "aot_eager", "fullgraph": True},
            }
        )
        try:
            reference = Model().to(device)
            reference.load_state_dict(runner.model.state_dict())
            assert tuple(runner.model.state_dict()) == tuple(reference.state_dict())
            for model in (runner.model, reference):
                model.first.register_forward_hook(lambda module, args, output: output + 0.5)
            x = torch.randn(3, 4, dtype=torch.float64, device=device, requires_grad=True)
            reference_x = x.detach().clone().requires_grad_()
            actual = runner.model(x)
            expected = reference(reference_x)
            torch.testing.assert_close(actual, expected)
            gradients = torch.autograd.grad(actual.square().sum(), (x, *runner.model.parameters()))
            expected_gradients = torch.autograd.grad(expected.square().sum(), (reference_x, *reference.parameters()))
            for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
                torch.testing.assert_close(actual_gradient, expected_gradient)
        finally:
            runner.close()

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

    def test_compiled_wrappers_export_native_model_state(self, tmp_path: Path) -> None:
        class NativeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.Linear(4, 2)
                self._orig_mod = nn.Linear(4, 2)

            def forward(self, values):
                return self.module(values) + self._orig_mod(values)

            def native_export(self):
                return self.state_dict()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original = NativeModel().to(device)
        compiled = torch.compile(original, backend="eager")
        runner = NoOptimizerTorchRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "ckpt.async_mode": "disabled"}
        )
        try:
            for wrapped in (
                original,
                compiled,
                nn.DataParallel(compiled),
                torch.compile(nn.DataParallel(original), backend="eager"),
            ):
                native = runner.unwrap(wrapped)
                assert native is original
                torch.testing.assert_close(native.native_export(), original.state_dict())
            runner.model = nn.DataParallel(compiled)
            runner.save_model_checkpoint()
            payload = torch.load(
                Path(runner.workspace.checkpoint_dir) / "model.pth", map_location="cpu", weights_only=False
            )
            restored = NativeModel().to(device)
            restored.load_state_dict(payload["model"], strict=True)
            torch.testing.assert_close(restored.state_dict(), original.state_dict())
        finally:
            runner.close()

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

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_seeded_loader_resume_preserves_model_random_stream(self, tmp_path: Path, shuffle: bool) -> None:
        config = {
            "logging.enabled": False,
            "workspace.dir": str(tmp_path),
            "seed": 42,
            "deterministic": False,
            "dataloader": {"batch_size": 2, "shuffle": shuffle},
        }
        source = StatefulDatasetTorchRunner(config)
        try:
            iterator = iter(source.dataloaders["train"])
            first = next(iterator)
            assert len(first) == 2
            checkpoint = source.state_dict()
            expected_batch = next(iterator)
            expected_noise = torch.rand(8)
        finally:
            source.close()

        restored = StatefulDatasetTorchRunner(config)
        try:
            restored.load_checkpoint(checkpoint)
            actual_batch = next(iter(restored.dataloaders["train"]))
            actual_noise = torch.rand(8)
            assert torch.equal(actual_batch, expected_batch)
            assert torch.equal(actual_noise, expected_noise)
        finally:
            restored.close()

    @pytest.mark.parametrize("accum_steps", [1, 2, 3])
    def test_training_boundary_does_not_checkpoint_an_unconsumed_batch(self, tmp_path: Path, accum_steps: int) -> None:
        config = {
            "logging.enabled": False,
            "workspace.dir": str(tmp_path),
            "seed": 42,
            "accum_steps": accum_steps,
            "dataloader": {"batch_size": 1, "shuffle": False},
        }
        source = StatefulDatasetTorchRunner(config)
        try:
            iterator = source._iter_train_batches(source.dataloaders["train"])
            for index in range(accum_steps):
                _, batch, flush = next(iterator)
                assert batch.item() == index
                assert flush == (index == accum_steps - 1)
                source.train_state.micro_step += 1
            checkpoint = source.state_dict()
            _, expected, _ = next(iterator)
        finally:
            source.close()
        restored = StatefulDatasetTorchRunner(config)
        try:
            restored.load_checkpoint(checkpoint)
            _, actual, _ = next(restored._iter_train_batches(restored.dataloaders["train"]))
            assert torch.equal(actual, expected)
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
    @pytest.mark.parametrize("operator_table_rows", [0, 80])
    def test_profiling_writes_trace_and_operator_table(self, tmp_path: Path, operator_table_rows: int) -> None:
        runner = TinyTorchRunner(
            {
                "logging.enabled": False,
                "workspace.root": str(tmp_path),
                "profiling": {
                    "enabled": True,
                    "wait": 0,
                    "warmup": 0,
                    "active": 1,
                    "trace_dir": "trace-output",
                    "operator_table_rows": operator_table_rows,
                },
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
        assert bool(list(trace_dir.rglob("operator_table.txt"))) == (operator_table_rows > 0)


class TestTorchRunnerDistributedRuntime:
    @pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
    def test_explicit_cpu_device_overrides_cuda_availability(self, tmp_path: Path, monkeypatch, device) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        runner = TinyTorchRunner(
            {"logging.enabled": False, "workspace.root": str(tmp_path), "seed": None, "device": device}
        )
        try:
            assert runner.device == torch.device("cpu")
            assert runner.config.get("device") == "cpu"
            assert runner.model is not None
            assert all(parameter.device.type == "cpu" for parameter in runner.model.parameters())
            prediction, loss = runner.train_step((torch.ones(2, 4), torch.zeros(2, 2)))
            assert prediction.device.type == "cpu"
            assert loss is not None and loss.device.type == "cpu"
            assert runner.train_state.global_step == 1
        finally:
            runner.close()

    @pytest.mark.parametrize("configured", ["cuda:3", torch.device("cuda:3"), "cuda", None])
    def test_device_preserves_explicit_index_or_uses_local_rank(self, monkeypatch, configured) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv("LOCAL_RANK", "2")
        runner = object.__new__(TorchRunner)
        runner.config = RunnerConfig({"device": configured})
        expected = torch.device("cuda", 3 if configured is not None and str(configured) == "cuda:3" else 2)
        assert runner.device == expected
        if configured is None:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
            assert runner.device == torch.device("cpu")

    @pytest.mark.parametrize(("device", "backend"), [("cpu", "gloo"), ("cuda:3", "nccl")])
    @pytest.mark.parametrize(("world_size", "ft_enabled"), [(2, False), (1, True)])
    def test_init_distributed_uses_selected_device(self, monkeypatch, device, backend, world_size, ft_enabled) -> None:
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.delenv("BACKEND", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RESTART_COUNT", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        selected_devices = []
        monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)
        initialized = False
        initialization = {}

        def init_process_group(**kwargs):
            nonlocal initialized
            initialization.update(kwargs)
            initialized = True

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: initialized)
        monkeypatch.setattr(dist, "init_process_group", init_process_group)
        monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
        monkeypatch.setattr(dist, "broadcast_object_list", lambda objects: None)
        runner = object.__new__(TorchRunner)
        runner.config = RunnerConfig({"device": device, "ft.enabled": ft_enabled})
        runner.timestamp = "test-device"
        runner.init_distributed()
        assert initialization["backend"] == backend
        if device == "cpu":
            assert not selected_devices
        else:
            assert selected_devices and all(index == 3 for index in selected_devices)

    @pytest.mark.parametrize("replica_id", [0, 1])
    def test_single_process_ft_replicas_use_distinct_data_shards(self, monkeypatch, replica_id) -> None:
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        runner = object.__new__(TorchRunner)
        runner.fault_tolerance = SimpleNamespace(
            enabled=True, group_size=2, data_parallel_info=lambda degree, rank: (degree * 2, replica_id + rank)
        )
        sampler = runner.build_datasampler(list(range(4)), split="train", shuffle=False)
        assert runner.world_size == 1 and runner.rank == 0
        assert runner.distributed
        assert list(sampler) == [replica_id, replica_id + 2]

    @pytest.mark.parametrize(
        ("backend", "degree", "expected"), [("nccl", 2, "cuda:0"), ("nccl", 1, "cuda:0"), ("gloo", 2, "cpu")]
    )
    def test_ft_collective_payload_supports_the_native_backend(self, monkeypatch, backend, degree, expected) -> None:
        replica_group = object()
        runner = object.__new__(TorchRunner)
        runner.config = RunnerConfig({"device": "cuda:0"})
        runner.fault_tolerance = SimpleNamespace(enabled=True, group_size=degree, replicate_process_group=replica_group)
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1 if group is None else degree)
        monkeypatch.setattr(dist, "get_backend", lambda group=None: backend if group is None else "gloo")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert runner.all_reduce_device() == torch.device(expected)

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

    @pytest.mark.parametrize("amp", [False, True])
    def test_discarded_loss_keeps_missing_adamw_gradients_and_amp_state(self, amp) -> None:
        runner = TinyTorchRunner({"device": "cpu", "logging.enabled": False, "skip_nonfinite_loss": True})
        try:
            runner.optimizer = optim.AdamW(runner.model.parameters(), lr=0.1, weight_decay=0.2)
            runner._bind_optimizer_container()
            if amp:
                runner.grad_scaler = torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=2)
                runner.grad_scaler.scale(torch.zeros(()))
                scale_state = runner.grad_scaler.state_dict()
            initial = [parameter.detach().clone() for parameter in runner.model.parameters()]
            runner.train_step((torch.full((1, 4), float("inf")), torch.zeros(1, 2)))
            assert runner.optimizer.state_dict()["state"] == {}
            assert runner.train_state.global_step == 1
            for parameter, expected in zip(runner.model.parameters(), initial):
                torch.testing.assert_close(parameter, expected)
            if amp:
                assert runner.grad_scaler.state_dict() == scale_state
            runner.train_step((torch.ones(1, 4), torch.zeros(1, 2)))
            assert runner.train_state.global_step == 2
            assert runner.optimizer.state_dict()["state"]
        finally:
            runner.close()

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

    @pytest.mark.parametrize("reduction", ["mean", "sum", "nan"])
    def test_empty_token_rank_participates_in_global_normalization(self, reduction) -> None:
        require_gloo()
        run_distributed(_torch_runner_empty_token_rank_worker, worker_args=(reduction,))

    @pytest.mark.parametrize("empty_first", [False, True])
    @pytest.mark.parametrize("invalid", ["empty", "nan"])
    def test_zero_weight_loss_preserves_other_microbatch_gradients(self, empty_first, invalid) -> None:
        runner = WeightedLossRunner(
            {"device": "cpu", "logging.enabled": False, "accum_steps": 2, "logging.interval": 0}
        )
        try:
            runner.model.double()
            valid = {
                "input": torch.ones(1, 1, dtype=torch.float64),
                "target": torch.zeros(1, 1, dtype=torch.float64),
                "loss_normalizer": 0.25,
            }
            empty = {
                "input": torch.empty(0, 1, dtype=torch.float64),
                "target": torch.empty(0, 1, dtype=torch.float64),
                "loss_normalizer": 0.0,
            }
            if invalid == "nan":
                empty = {
                    "input": torch.ones(1, 1, dtype=torch.float64),
                    "target": torch.full((1, 1), float("nan"), dtype=torch.float64),
                    "loss_normalizer": 0.0,
                }
            runner.dataloaders["train"] = [empty, valid] if empty_first else [valid, empty]
            result = runner.train_epoch()
            torch.testing.assert_close(runner.model.weight, torch.tensor([[0.8]], dtype=torch.float64))
            assert result["loss"] == pytest.approx(1.0)
            assert runner.train_state.global_step == 1
        finally:
            runner.close()

    @pytest.mark.parametrize("kind", ["ce", "nll", "mse"])
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_accumulated_criterion_matches_concatenated_batch(self, kind, reduction) -> None:
        runner = WeightedLossRunner(
            {"device": "cpu", "logging.enabled": False, "accum_steps": 4, "logging.interval": 0}
        )
        try:
            linear = nn.Linear(2, 2, bias=False, dtype=torch.float64)
            runner.model = nn.Sequential(linear, nn.LogSoftmax(dim=1)) if kind == "nll" else linear
            weight = torch.tensor([0.25, 1.75], dtype=torch.float64)
            runner.criterion = (
                nn.MSELoss(reduction=reduction)
                if kind == "mse"
                else (nn.NLLLoss if kind == "nll" else nn.CrossEntropyLoss)(weight=weight, reduction=reduction)
            )
            runner.optimizer = optim.SGD(runner.model.parameters(), lr=0.1)
            runner._bind_optimizer_container()
            inputs = torch.tensor([[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0], [0.5, -2.0]], dtype=torch.float64)
            target = torch.zeros_like(inputs) if kind == "mse" else torch.tensor([-100, 0, 1, 1])
            expected = [parameter.detach().clone() for parameter in runner.model.parameters()]
            loss = runner.criterion(runner.model(inputs), target)
            gradients = torch.autograd.grad(loss, tuple(runner.model.parameters()))
            runner.dataloaders["train"] = list(zip(inputs.split((1, 1, 2)), target.split((1, 1, 2))))
            runner.train_epoch()
            for parameter, initial, gradient in zip(runner.model.parameters(), expected, gradients):
                torch.testing.assert_close(parameter, initial - 0.1 * gradient)
            assert runner.train_state.global_step == 1
        finally:
            runner.close()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_real_loss_normalizers_preserve_fractional_weights(self, reduction) -> None:
        runner = WeightedLossRunner(
            {"device": "cpu", "logging.enabled": False, "accum_steps": 2, "logging.interval": 0}
        )
        try:
            runner.model.double()
            runner.criterion = nn.MSELoss(reduction=reduction)
            runner.dataloaders["train"] = [
                {
                    "input": torch.ones(1, 1, dtype=torch.float64),
                    "target": torch.zeros(1, 1, dtype=torch.float64),
                    "loss_normalizer": 0.25,
                },
                {
                    "input": torch.full((1, 1), 3.0, dtype=torch.float64),
                    "target": torch.zeros(1, 1, dtype=torch.float64),
                    "loss_normalizer": 0.75,
                },
            ]
            runner.train_epoch()
            gradient = 2 * (0.25 + 9 * 0.75) if reduction == "mean" else 2 * (1 + 9)
            torch.testing.assert_close(runner.model.weight, torch.tensor([[1 - 0.1 * gradient]], dtype=torch.float64))
        finally:
            runner.close()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_empty_loss_window_keeps_optimizer_and_scheduler_unchanged(self, reduction) -> None:
        runner = TinyStepLRTorchRunner(
            {"device": "cpu", "logging.enabled": False, "accum_steps": 2, "logging.interval": 0}
        )
        try:
            runner.criterion = nn.CrossEntropyLoss(reduction=reduction)
            runner.optimizer.param_groups[0]["weight_decay"] = 0.1
            initial = [parameter.detach().clone() for parameter in runner.model.parameters()]
            scheduler_state = runner.scheduler.state_dict()
            runner.dataloaders["train"] = [(torch.ones(1, 4), torch.full((1,), -100))] * 2
            runner.train_epoch()
            assert runner.train_state.global_step == 0
            assert runner.scheduler.state_dict() == scheduler_state
            for parameter, expected in zip(runner.model.parameters(), initial):
                torch.testing.assert_close(parameter, expected)
        finally:
            runner.close()

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

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("failure", [None, "factory", "evaluation"])
    def test_training_evaluation_restores_state_rng_and_shared_custom_metrics(self, device, failure) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")

        class Collector:
            def __init__(self):
                self.observations = ["training"]

            def __deepcopy__(self, memo):
                raise AssertionError("custom collector must use its factory")

            def reset(self):
                self.observations.clear()

            def average(self):
                return {"score": float(len(self.observations))}

        runner = TinyTorchRunner({"device": device, "logging.enabled": False, "logging.interval": 0})
        try:
            runner.model = nn.Sequential(runner.model, nn.Dropout())
            runner.model.train()
            runner.model[0].eval()
            runner.ema = copy.deepcopy(runner.model)
            runner.ema.eval()
            runner.ema[0].train()
            flags = [(module, module.training) for model in (runner.model, runner.ema) for module in model.modules()]
            collector = Collector()
            runner.metrics = runner.train_metrics = runner.evaluate_metrics = collector
            meters, loaders = runner.meters, runner.dataloaders
            meters.loss.update(3.0, n=2)
            runner.split = "train"
            runner._loss_reporting_reduction = "sum"
            runner.train_state.global_step = 7
            runner.dataloaders["val"] = [None]
            python_rng, cpu_rng = random.getstate(), torch.get_rng_state()
            numpy_rng = None if torch_runner_module.np_random is None else torch_runner_module.np_random.get_state()
            cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None
            writes, snapshots = [], []
            runner.write_result = lambda result, split, index: writes.append(index)

            def consume_rng():
                random.random()
                torch.rand(5)
                if torch_runner_module.np_random is not None:
                    torch_runner_module.np_random.rand(5)
                if cuda_rng is not None:
                    torch.rand(5, device=torch.cuda.current_device())

            def metrics_factory():
                consume_rng()
                if failure == "factory":
                    raise RuntimeError("validation factory failed")
                return Collector()

            def evaluate_step(_data):
                consume_rng()
                runner.metrics.observations.append("validation")
                snapshots.extend([runner.state_dict(), runner._export_checkpoint_local_state()])
                if failure == "evaluation":
                    raise RuntimeError("validation evaluation failed")
                return None, None

            runner.build_evaluation_metrics = metrics_factory
            runner.evaluate_step = evaluate_step
            if failure is None:
                with runner._evaluation_context(["val"]):
                    assert runner.evaluate_epoch("val")["score"] == 1
                assert writes == [7]
            else:
                with (
                    pytest.raises(RuntimeError, match=f"validation {failure} failed"),
                    runner._evaluation_context(["val"]),
                ):
                    runner.evaluate_epoch("val")
            assert runner.mode == "train" and runner.split == "train"
            assert runner.meters is meters and meters.loss.count == 2
            assert runner.dataloaders is loaders
            assert runner.metrics is runner.train_metrics is runner.evaluate_metrics is collector
            assert collector.observations == ["training"]
            assert runner._loss_reporting_reduction == "sum"
            assert all(module.training == flag for module, flag in flags)
            assert random.getstate() == python_rng
            torch.testing.assert_close(torch.get_rng_state(), cpu_rng)
            if numpy_rng is not None:
                actual = torch_runner_module.np_random.get_state()
                assert actual[0] == numpy_rng[0] and actual[2:] == numpy_rng[2:]
                assert actual[1].tolist() == numpy_rng[1].tolist()
            if cuda_rng is not None:
                torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)
            for snapshot in snapshots:
                rng = snapshot["state"]["rng"]
                assert rng["python"] == python_rng
                torch.testing.assert_close(rng["torch_cpu"], cpu_rng)
                if cuda_rng is not None:
                    torch.testing.assert_close(rng["torch_cuda"][torch.cuda.current_device()], cuda_rng)
        finally:
            runner.close()

    @pytest.mark.parametrize("kind", ["global", "stream", "multitask", "multitask_child"])
    def test_training_evaluation_preserves_shared_native_collector_state(self, kind) -> None:
        runner = TinyTorchRunner({"device": "cpu", "logging.enabled": False, "logging.interval": 0})
        try:
            metrics = (
                GlobalMetrics(mse(), distributed=False) if kind == "global" else StreamMetrics(mse(), distributed=False)
            )
            validation_metrics = metrics
            if kind in ("multitask", "multitask_child"):
                validation_metrics = MultiTaskMetrics(task=metrics, aggregate="macro")
                metrics = MultiTaskMetrics(task=metrics, aggregate="macro")
                if kind == "multitask":
                    validation_metrics = metrics

            def update(collector, value):
                args = (torch.tensor([float(value)]), torch.zeros(1))
                (
                    collector.update({"task": args})
                    if kind in ("multitask", "multitask_child")
                    else collector.update(*args)
                )

            update(metrics, 3)
            before = metrics.average().dict()
            runner.metrics = runner.train_metrics = metrics
            runner.evaluate_metrics = validation_metrics
            runner.dataloaders["val"] = [None]

            def evaluate_step(_data):
                update(runner.metrics, 1)
                return None, None

            runner.evaluate_step = evaluate_step
            with runner._evaluation_context(["val"]):
                result = runner.evaluate_epoch("val")
                assert result["task" if kind in ("multitask", "multitask_child") else "mse"] == pytest.approx(1)
            assert runner.metrics is runner.train_metrics is metrics
            assert runner.evaluate_metrics is validation_metrics
            assert metrics.average().dict() == before
        finally:
            runner.close()

    def test_training_evaluation_isolates_shared_stateful_metric_callable(self) -> None:
        from torchmetrics import MeanSquaredError

        metric = MeanSquaredError()
        training = StreamMetrics(error=metric, distributed=False)
        evaluation = StreamMetrics(error=metric, distributed=False)
        training.update(torch.tensor([3.0]), torch.zeros(1))
        runner = TinyTorchRunner({"device": "cpu", "logging.enabled": False, "logging.interval": 0})
        try:
            runner.metrics = runner.train_metrics = training
            runner.evaluate_metrics = evaluation
            runner.dataloaders["val"] = [None]

            def evaluate_step(_data):
                runner.metrics.update(torch.ones(1), torch.zeros(1))
                assert runner.metrics["error"].metric.compute().item() == pytest.approx(1)
                return None, None

            runner.evaluate_step = evaluate_step
            with runner._evaluation_context(["val"]):
                assert runner.evaluate_epoch("val")["error"] == pytest.approx(1)
            assert runner.metrics is runner.train_metrics is training
            assert runner.evaluate_metrics is evaluation
            assert training.average()["error"] == pytest.approx(9)
            assert metric.compute().item() == pytest.approx(9)
        finally:
            runner.close()

    def test_training_validation_replays_origin_and_preserves_checkpoint_loader_cursors(self) -> None:
        configuration = {
            "device": "cpu",
            "logging.enabled": False,
            "logging.interval": 0,
            "dataloader.batch_size": 1,
            "dataloader.shuffle": False,
        }
        runner = StatefulDatasetTorchRunner(configuration)
        restored = None
        try:
            runner.datasets["val"] = torch.utils.data.TensorDataset(
                torch.arange(4.0).view(4, 1).expand(4, 4), torch.zeros(4, 2)
            )
            runner.build_dataloaders()
            train, val = runner.dataloaders["train"], runner.dataloaders["val"]
            train_iterator, val_iterator = iter(train), iter(val)
            for _ in range(2):
                next(train_iterator)
                next(val_iterator)
            observations, checkpoints = [], []

            def evaluate_step(data):
                observations.append(data[0][0, 0].item())
                checkpoints.append(copy.deepcopy(runner.state_dict()))
                return None, None

            runner.evaluate_step = evaluate_step
            for _ in range(2):
                with runner._evaluation_context(["val"]):
                    runner.evaluate_steps("val", steps=2)
            assert observations == [0, 1, 0, 1]
            assert runner.dataloaders["train"] is train and runner.dataloaders["val"] is val
            assert next(train_iterator).item() == 2
            assert next(val_iterator)[0][0, 0].item() == 2
            assert all(set(checkpoint["dataloaders"]) == {"train", "val"} for checkpoint in checkpoints)
            restored = StatefulDatasetTorchRunner(configuration)
            restored.datasets["val"] = runner.datasets["val"]
            restored.build_dataloaders()
            restored.load_checkpoint(checkpoints[0])
            assert next(iter(restored.dataloaders["train"])).item() == 2
            assert next(iter(restored.dataloaders["val"]))[0][0, 0].item() == 2
        finally:
            runner.close()
            if restored is not None:
                restored.close()

    @pytest.mark.parametrize("factory", [False, True])
    def test_training_evaluation_requires_an_independent_shared_custom_collector(self, factory) -> None:
        runner = TinyTorchRunner({"device": "cpu", "logging.enabled": False})
        try:
            collector = object()
            runner.metrics = runner.train_metrics = runner.evaluate_metrics = collector
            runner.dataloaders["val"] = [None]
            if factory:
                runner.build_evaluation_metrics = lambda: collector
            with (
                pytest.raises(TypeError, match="fresh collector|training collector"),
                runner._evaluation_context(["val"]),
            ):
                pytest.fail("a shared collector must not be reset by validation")
            assert runner.metrics is runner.train_metrics is runner.evaluate_metrics is collector
            assert runner.mode == "train"
        finally:
            runner.close()

    def test_training_evaluation_custom_stream_uses_explicit_origin_factory(self) -> None:
        runner = StreamingEpochRunner({"device": "cpu", "logging.enabled": False, "logging.interval": 0})
        try:
            stream = iter([1.0, 2.0, 3.0])
            runner.dataloaders["val"] = stream
            assert next(stream) == 1
            with pytest.raises(TypeError, match="build_evaluation_dataloader"), runner._evaluation_context(["val"]):
                pytest.fail("one-shot streams need a fresh-origin factory")
            runner.build_evaluation_dataloader = lambda split: StreamingLoader(4.0, 6.0)
            for _ in range(2):
                with runner._evaluation_context(["val"]):
                    assert runner.evaluate_steps("val", steps=2)["loss"] == pytest.approx(5)
            assert runner.dataloaders["val"] is stream and next(stream) == 2
        finally:
            runner.close()

    @pytest.mark.parametrize("steps", [None, 0, 1])
    def test_evaluation_uses_fresh_meter_schema_and_keeps_results_readable(self, steps) -> None:
        class Collector:
            def __init__(self):
                self.observations = ["previous"]

            def reset(self):
                self.observations.clear()

            def average(self):
                return {"loss.avg": 7.0}

        runner = TinyTorchRunner({"logging.enabled": False, "logging.interval": 0})
        try:
            collector = Collector()
            runner.evaluate_metrics = collector
            runner.meters.loss.update(3.0)
            runner.meters.lr.update(0.1)
            runner.meters.grad_norm.update(1.0)
            runner.dataloaders["val"] = [None]

            def evaluate_step(_data):
                collector.observations.append("validation")
                return None, None

            runner.evaluate_step = evaluate_step
            result = runner.evaluate_epoch("val") if steps is None else runner.evaluate_steps("val", steps=steps)
            assert result["loss.avg"] == 7.0
            assert "lr" not in result and "grad_norm" not in result
            assert runner.mode == "evaluate" and runner.split == "val"
            assert runner.metrics is runner.evaluate_metrics is collector
            assert collector.observations == ([] if steps == 0 else ["validation"])
            assert runner.get_epoch_result()["loss.avg"] == 7.0
        finally:
            runner.close()

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

    def test_resume_after_split_last_update_continues_next_round(self, tmp_path: Path) -> None:
        class BoundaryRunner(TinyTorchRunner):
            def __init__(self, config):
                super().__init__(config)
                self.datasets["train"] = [(torch.full((4,), float(value)), torch.zeros(2)) for value in (1, 2)]
                self.consumed = []
                self.interrupt = False

            def train_step(self, data):
                self.consumed.append(int(data[0][0, 0].item()))
                return super().train_step(data)

            def save_checkpoint(self, *args, **kwargs):
                super().save_checkpoint(*args, **kwargs)
                if self.interrupt and self.train_state.global_step == 2:
                    raise SystemExit

        config = {
            "device": "cpu",
            "seed": 42,
            "steps": 4,
            "logging.enabled": False,
            "logging.interval": 0,
            "dataloader": {"batch_size": 1, "shuffle": False},
            "ckpt": {"backend": "file", "async_mode": "disabled", "interval": 2},
        }
        baseline = BoundaryRunner({**config, "workspace.dir": str(tmp_path / "baseline")})
        source = BoundaryRunner({**config, "workspace.dir": str(tmp_path / "resumed")})
        restored = None
        try:
            baseline.train_steps(["train"], [])
            source.interrupt = True
            with pytest.raises(SystemExit):
                source.train_steps(["train"], [])
            checkpoint = Path(source.workspace.checkpoint_dir) / "latest.pth"
            source.close()

            restored = BoundaryRunner({**config, "workspace.dir": str(tmp_path / "resumed")})
            restored.load_checkpoint(checkpoint)
            restored.train_steps(["train"], [])

            assert source.consumed + restored.consumed == baseline.consumed == [1, 2, 1, 2]
            assert restored.train_state.global_step == restored.train_state.micro_step == 4
            for actual, expected in zip(restored.model.parameters(), baseline.model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            baseline.close()
            source.close()
            if restored is not None:
                restored.close()

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


class TestPeriodicEvaluation:

    def test_validation_budget_works_without_an_interval(self, tmp_path):
        class RecordingRunner(StreamingEpochRunner):
            def __init__(self, config):
                super().__init__(config)
                self.validation = []

            def evaluate_step(self, data):
                self.validation.append(self.train_state.global_step)
                return super().evaluate_step(data)

        runner = RecordingRunner(
            {
                "device": "cpu",
                "workspace.dir": str(tmp_path),
                "logging.enabled": False,
                "steps": 2,
                "evaluation.steps": 1,
                "ckpt.enabled": False,
            }
        )
        try:
            runner.dataloaders["train"] = [1.0, 2.0]
            runner.dataloaders["val"] = [3.0, 4.0]
            runner.train_steps(["train"], ["val"])
            assert runner.validation == [2]
        finally:
            runner.close()

    def test_cadence_counts_updates_and_best_contains_complete_validation(self, tmp_path):
        class ScoredRunner(WeightedLossRunner):
            def __init__(self, config):
                super().__init__(config)
                self.validation = []
                self.validated_weights = {}

            def evaluate_step(self, data):
                del data
                step = self.train_state.global_step
                self.validation.append((self.split, step, self.train_state.micro_step))
                self.validated_weights[step] = self.model.weight.detach().clone()
                return None, torch.tensor({0: 0.5, 2: 0.1, 3: 0.9}[step])

        runner = ScoredRunner(
            {
                "device": "cpu",
                "workspace.dir": str(tmp_path),
                "logging.enabled": False,
                "logging.interval": 0,
                "steps": 3,
                "accum_steps": 2,
                "skip_nonfinite_grad": True,
                "evaluation": {"interval": 2, "steps": 1},
                "score": {"split": "val", "mode": "min"},
                "ckpt": {"backend": "file", "interval": 1, "async_mode": "disabled"},
            }
        )
        try:
            good = (torch.ones(1, 1), torch.zeros(1, 1))
            runner.dataloaders["train"] = [(torch.ones(1, 1), torch.full((1, 1), float("nan"))), *[good] * 7]
            runner.dataloaders["val"] = [good, good]
            runner.dataloaders["other"] = [good, good]
            runner.train_steps(["train"], ["val", "other"])
            assert runner.validation == [
                (split, step, micro) for step, micro in ((2, 6), (3, 8)) for split in ("val", "other")
            ]
            assert list(runner.results) == [2, 3]
            assert runner.best_index == 2
            checkpoint = torch.load(Path(runner.workspace.checkpoint_dir) / "best.pth", weights_only=False)
            assert checkpoint["state"]["train"]["global_step"] == 2
            assert checkpoint["state"]["train"]["last_evaluation_step"] == 2
            assert {"val", "other"} <= checkpoint["results"][2].keys()
            torch.testing.assert_close(checkpoint["model"]["weight"], runner.validated_weights[2])
        finally:
            runner.close()

    def test_epoch_boundaries_share_global_step_indices_and_do_not_repeat_validation(self, tmp_path):
        class EpochRunner(StreamingEpochRunner):
            def __init__(self, config):
                super().__init__(config)
                self.validated_steps = []

            def evaluate_step(self, data):
                self.validated_steps.append(self.train_state.global_step)
                return super().evaluate_step(data)

        runner = EpochRunner(
            {
                "device": "cpu",
                "workspace.dir": str(tmp_path),
                "logging.enabled": False,
                "logging.interval": 0,
                "epochs": 2,
                "evaluation.interval": 1,
                "ckpt.enabled": False,
            }
        )
        try:
            runner.dataloaders["train"] = [1.0, 2.0]
            runner.dataloaders["val"] = [3.0]
            runner.train_epochs(["train"], ["val"])
            assert runner.validated_steps == [1, 2, 3, 4]
            assert list(runner.results) == [1, 2, 3, 4]
            assert [step for step, result in runner.results.items() if "train" in result] == [2, 4]
        finally:
            runner.close()

    def test_failed_split_does_not_publish_partial_results(self, tmp_path):
        class FailingRunner(StreamingEpochRunner):
            def evaluate_step(self, data):
                if self.split == "broken":
                    raise RuntimeError("validation failed")
                return super().evaluate_step(data)

        runner = FailingRunner(
            {
                "device": "cpu",
                "workspace.dir": str(tmp_path),
                "logging.enabled": False,
                "steps": 1,
                "evaluation.interval": 1,
                "ckpt.enabled": False,
            }
        )
        try:
            runner.dataloaders["train"] = [1.0]
            runner.dataloaders["val"] = [2.0]
            runner.dataloaders["broken"] = [3.0]
            with pytest.raises(RuntimeError, match="validation failed"):
                runner.train_steps(["train"], ["val", "broken"])
            assert not runner.results
            assert runner.train_state.last_evaluation_step is None
            assert runner.mode == torch_runner_module.RunnerMode.train
        finally:
            runner.close()

    def test_mid_epoch_validation_keeps_checkpoint_history_on_epoch_boundaries(self, tmp_path):
        class ScoredRunner(StreamingEpochRunner):
            def evaluate_step(self, data):
                del data
                step = self.train_state.global_step
                if step == 4:
                    assert not (Path(self.workspace.checkpoint_dir) / "ckpt-e000002.pth").exists()
                return None, torch.tensor(0.1 if step == 1 else 0.9)

        runner = ScoredRunner(
            {
                "device": "cpu",
                "workspace.dir": str(tmp_path),
                "logging.enabled": False,
                "logging.interval": 0,
                "epochs": 2,
                "evaluation.interval": 1,
                "score.split": "val",
                "ckpt": {"backend": "file", "interval": 2, "async_mode": "disabled"},
            }
        )
        try:
            runner.dataloaders["train"] = [1.0, 2.0]
            runner.dataloaders["val"] = [3.0]
            runner.train_epochs(["train"], ["val"])
            checkpoints = Path(runner.workspace.checkpoint_dir)
            assert [path.name for path in checkpoints.glob("ckpt-e*.pth")] == ["ckpt-e000002.pth"]
            best = torch.load(checkpoints / "best.pth", weights_only=False)
            latest = torch.load(checkpoints / "latest.pth", weights_only=False)
            assert best["state"]["train"]["global_step"] == 1
            assert latest["state"]["train"]["global_step"] == 4
            assert latest["state"]["train"]["epoch"] == 2
        finally:
            runner.close()

    def test_validation_and_resume_preserve_stochastic_training(self, tmp_path):
        class StochasticRunner(TinyTorchRunner):
            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.4), nn.Linear(8, 2))
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
                self.datasets["train"] = [(torch.full((4,), index / 10), torch.zeros(2)) for index in range(8)]
                self.datasets["val"] = [(torch.ones(4), torch.zeros(2))] * 4
                self.seen = []
                self.validation = []
                self.interrupt = False

            def train_step(self, data):
                self.seen.extend(data[0][:, 0].tolist())
                noise = random.random() + float(torch_runner_module.np_random.random())
                return super().train_step((data[0] + noise, data[1]))

            def evaluate_step(self, data):
                self.validation.append(self.train_state.global_step)
                random.random()
                torch_runner_module.np_random.random()
                torch.rand(3)
                return super().evaluate_step(data)

            def save_checkpoint(self, *args, **kwargs):
                super().save_checkpoint(*args, **kwargs)
                if self.interrupt and self.train_state.global_step == 2:
                    raise RuntimeError("pause after saved validation")

        def build(path, interval):
            torch.manual_seed(42)
            return StochasticRunner(
                {
                    "device": "cpu",
                    "seed": 42,
                    "workspace.dir": str(path),
                    "logging.enabled": False,
                    "logging.interval": 0,
                    "steps": 4,
                    "accum_steps": 2,
                    "dataloader.batch_size": 2,
                    "evaluation": {"interval": interval, "steps": 1},
                    "ckpt": {"backend": "file", "interval": 0, "async_mode": "disabled"},
                }
            )

        def snapshot(runner):
            return (
                copy.deepcopy(runner.model.state_dict()),
                copy.deepcopy(runner.optimizer.state_dict()),
                torch.rand(3),
                random.random(),
                float(torch_runner_module.np_random.random()),
            )

        baseline = build(tmp_path / "baseline", 0)
        try:
            baseline.train_steps(["train"], [])
            expected, seen = snapshot(baseline), baseline.seen
        finally:
            baseline.close()
        uninterrupted = build(tmp_path / "uninterrupted", 2)
        try:
            uninterrupted.train_steps(["train"], ["val"])
            torch.testing.assert_close(snapshot(uninterrupted), expected, rtol=0, atol=0)
            assert uninterrupted.validation == [2, 4]
            assert uninterrupted.seen == seen
        finally:
            uninterrupted.close()
        source = build(tmp_path / "resumed", 2)
        checkpoint = Path(source.workspace.checkpoint_dir) / "latest.pth"
        try:
            source.interrupt = True
            with pytest.raises(RuntimeError, match="pause after saved validation"):
                source.train_steps(["train"], ["val"])
            consumed = source.seen
        finally:
            source.close()
        restored = build(tmp_path / "resumed", 2)
        try:
            restored.load_checkpoint(checkpoint)
            restored.train_steps(["train"], ["val"])
            assert restored.validation == [4]
            assert consumed + restored.seen == seen
            torch.testing.assert_close(snapshot(restored), expected, rtol=0, atol=0)
        finally:
            restored.close()

    @pytest.mark.parametrize("source_interval", [0, 1])
    def test_epoch_resume_rejects_switching_result_index_conventions(self, tmp_path, source_interval):
        config = {"device": "cpu", "logging.enabled": False, "epochs": 2, "ckpt.enabled": False}
        source = StreamingEpochRunner(
            {**config, "workspace.dir": str(tmp_path / "source"), "evaluation.interval": source_interval}
        )
        restored = StreamingEpochRunner(
            {**config, "workspace.dir": str(tmp_path / "restored"), "evaluation.interval": 1 - source_interval}
        )
        try:
            source.append_result({"val": {"loss": 0.1}}, index=0)
            with pytest.raises(ValueError, match="cannot mix epoch-indexed and global-step-indexed"):
                restored.load_state_dict(source.state_dict())
        finally:
            source.close()
            restored.close()


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

    @pytest.mark.parametrize("step_mode", [False, True])
    @pytest.mark.parametrize("before_loop", [False, True])
    def test_sigterm_resume_keeps_every_consumed_training_batch(self, tmp_path, step_mode, before_loop) -> None:
        class ShutdownRunner(StreamingEpochRunner):
            def __init__(self, config):
                super().__init__(config)
                self.datasets["train"] = list(range(8))
                self.consumed = []
                self.request_signal = not before_loop

            def train_step(self, data):
                self.consumed.append(int(data.item()))
                if self.request_signal:
                    self.request_signal = False
                    self.supervisor.request_shutdown(signal.SIGTERM, None)
                return super().train_step(data)

        config = {
            "device": "cpu",
            "logging.enabled": False,
            "logging.interval": 0,
            "workspace.dir": str(tmp_path),
            "accum_steps": 3,
            "steps": 3,
            "dataloader": {"batch_size": 1, "shuffle": False},
            "ckpt": {"backend": "file", "async_mode": "disabled"},
        }
        source = ShutdownRunner(config)
        checkpoint_path = Path(source.workspace.checkpoint_dir) / "latest.pth"
        try:
            if before_loop:
                source.supervisor.request_shutdown(signal.SIGTERM, None)
            with pytest.raises(SystemExit):
                source.train_steps(["train"], []) if step_mode else source.train_epoch()
            assert source.consumed == ([] if before_loop else [0, 1, 2])
            consumed = source.consumed
        finally:
            source.close()
        restored = ShutdownRunner(config)
        try:
            restored.request_signal = False
            restored.load_checkpoint(checkpoint_path)
            restored.train_steps(["train"], []) if step_mode else restored.train_epoch()
            assert consumed + restored.consumed == list(range(8))
            assert restored.train_state.global_step == 3
        finally:
            restored.close()

    def test_recovery_restores_loader_before_selecting_each_window(self, tmp_path) -> None:
        runner = StatefulDatasetTorchRunner(
            {
                "device": "cpu",
                "logging.enabled": False,
                "workspace.dir": str(tmp_path),
                "dataloader": {"batch_size": 1, "shuffle": False},
            }
        )
        try:
            loader = runner.dataloaders["train"]
            states = []
            iterator = iter(loader)
            for index in range(5):
                next(iterator)
                if index in (1, 4):
                    states.append(runner.dataloaders.state_dict())

            class Recovery:
                def begin_step(self):
                    runner.load_dataloaders(states.pop(0))

                def close(self):
                    pass

            runner.fault_tolerance = Recovery()
            batches = runner._iter_train_batches(loader)
            assert next(batches)[1].item() == 2
            runner.train_state.micro_step += 1
            assert next(batches)[1].item() == 5
        finally:
            runner.close()

    def test_step_mode_resume_preserves_split_target_and_sampler_epoch(self, tmp_path) -> None:
        class SplitRunner(StreamingEpochRunner):
            def __init__(self, config):
                super().__init__(config)
                self.consumed = []
                self.interrupt = False

            def train_step(self, data):
                self.consumed.append((self.split, int(data.item())))
                result = super().train_step(data)
                if self.interrupt and self.train_state.global_step == 6:
                    self.supervisor.request_shutdown(signal.SIGTERM, None)
                return result

        def build(path):
            runner = SplitRunner(
                {
                    "device": "cpu",
                    "seed": 42,
                    "logging.enabled": False,
                    "logging.interval": 0,
                    "workspace.dir": str(path),
                    "steps": 8,
                    "ckpt": {"backend": "file", "async_mode": "disabled"},
                }
            )
            for split, size in (("a", 10), ("b", 1)):
                sampler = torch.utils.data.DistributedSampler(
                    list(range(size)), num_replicas=1, rank=0, seed=42, shuffle=True
                )
                runner.dataloaders[split] = torch_runner_module.StatefulDataLoader(
                    list(range(size)), batch_size=1, sampler=sampler
                )
            return runner

        baseline = build(tmp_path / "baseline")
        try:
            baseline.train_steps(["a", "b"], [])
            expected = baseline.consumed
        finally:
            baseline.close()
        source = build(tmp_path / "resumed")
        checkpoint_path = Path(source.workspace.checkpoint_dir) / "latest.pth"
        try:
            source.interrupt = True
            with pytest.raises(SystemExit):
                source.train_steps(["a", "b"], [])
            consumed = source.consumed
        finally:
            source.close()
        restored = build(tmp_path / "resumed")
        try:
            restored.load_checkpoint(checkpoint_path)
            restored.train_steps(["a", "b"], [])
            assert consumed + restored.consumed == expected
            assert restored.train_state.global_step == 8
        finally:
            restored.close()

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


def test_bounded_inference_and_evaluation_do_not_consume_extra_batches(tmp_path):
    class CountingLoader:
        def __init__(self):
            self.reads = []

        def __len__(self):
            return 4

        def __iter__(self):
            for index in range(4):
                self.reads.append(index)
                yield torch.full((1, 4), float(index)), torch.zeros(1, 2)

    runner = TinyTorchRunner({"device": "cpu", "workspace.dir": str(tmp_path), "logging.enabled": False})
    try:
        for operation, steps in (("infer", 0), ("infer", 1), ("evaluate_steps", 1)):
            loader = CountingLoader()
            runner.dataloaders["val"] = loader
            getattr(runner, operation)(split="val", steps=steps)
            assert loader.reads == list(range(steps))
    finally:
        runner.close()
