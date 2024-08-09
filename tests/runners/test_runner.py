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

from pathlib import Path

import pytest
from torch import nn, optim

from danling.runners import BaseRunner, DeepSpeedRunner, ParallelRunner, Runner, TorchRunner
from danling.runners.config import RunnerConfig
from danling.runners.runner import RUNNER_REGISTRY


class _TinyRunnerMixin:
    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class TinyRunner(_TinyRunnerMixin, Runner):
    pass


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "ddp"),
        ({"stack": "auto"}, "ddp"),
        ({"stack": "parallel"}, "parallel"),
        ({"stack": "ds"}, "deepspeed"),
    ],
)
def test_runner_resolve_stack_normalizes_names(config, expected: str) -> None:
    assert Runner.resolve_stack(config) == expected


@pytest.mark.parametrize(
    ("stack", "expected_cls"),
    [
        ("ddp", TorchRunner),
        ("torch", TorchRunner),
        ("parallel", ParallelRunner),
        ("deepspeed", DeepSpeedRunner),
        ("ds", DeepSpeedRunner),
    ],
)
def test_runner_resolve_runner_class_maps_supported_stacks(stack: str, expected_cls: type[TorchRunner]) -> None:
    assert Runner.resolve_runner_class({"stack": stack}) is expected_cls


def test_runner_defaults_to_torch_stack() -> None:
    runner = TinyRunner({"log": False})
    assert isinstance(runner, TorchRunner)
    assert runner.config.stack == "ddp"


def test_runner_entrypoint_does_not_mutate_input_config() -> None:
    config = {"log": False}
    runner = TinyRunner(config)
    try:
        assert config == {"log": False}
        assert runner.config.stack == "ddp"
    finally:
        runner.close()


def test_runner_direct_construction_post_init_runs_once(tmp_path) -> None:
    events: list[tuple[str, str]] = []

    class ImplRunner(BaseRunner):
        def __post_init__(self) -> None:
            events.append(("impl", type(self).__name__))

    original_ddp = RUNNER_REGISTRY["ddp"]
    original_torch = RUNNER_REGISTRY["torch"]
    RUNNER_REGISTRY["ddp"] = ImplRunner
    RUNNER_REGISTRY["torch"] = ImplRunner
    runner = None
    try:
        runner = Runner({"stack": "ddp", "log": False, "workspace_root": str(tmp_path)})
        assert isinstance(runner, ImplRunner)
        assert events == [("impl", "ImplRunner")]
    finally:
        if runner is not None:
            runner.close()
        RUNNER_REGISTRY["ddp"] = original_ddp
        RUNNER_REGISTRY["torch"] = original_torch


def test_runner_unknown_stack_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stack"):
        Runner.resolve_runner_class({"stack": "unknown_stack"})


def test_runner_read_config_dispatches_dcp_directory(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "dcp-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".metadata").touch()
    RunnerConfig({"log": False, "stack": "parallel", "name": "dcp-test"}).yaml(checkpoint_dir / "runner.yaml")

    config = Runner.read_config(checkpoint_dir)

    assert config.stack == "parallel"
    assert config.name == "dcp-test"


def test_runner_read_config_dispatches_deepspeed_pointer(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_tag = "ckpt-g000000000001"
    tag_dir = checkpoint_dir / checkpoint_tag
    tag_dir.mkdir(parents=True)
    RunnerConfig({"log": False, "stack": "deepspeed", "name": "ds-test"}).yaml(tag_dir / "runner.yaml")
    (checkpoint_dir / "latest.pointer").write_text(checkpoint_tag, encoding="utf-8")

    config = Runner.read_config(checkpoint_dir)

    assert config.stack == "deepspeed"
    assert config.name == "ds-test"
