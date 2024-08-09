# DanLing
# Copyright (C) 2022-Present  DanLing
#
# This file is part of DanLing.
#
# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0
#
# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base_runner import BaseRunner
from .config import normalize_stack_name
from .deepspeed_runner import DeepSpeedRunner
from .parallel_runner import ParallelRunner
from .torch_runner import TorchRunner

RUNNER_REGISTRY = {
    "ddp": TorchRunner,
    "torch": TorchRunner,
    "deepspeed": DeepSpeedRunner,
    "ds": DeepSpeedRunner,
    "parallel": ParallelRunner,
}


class Runner(BaseRunner):
    """Dynamic runner entrypoint that selects stack-specific runner classes."""

    @staticmethod
    def resolve_stack(config: Mapping[str, Any]) -> str:
        return normalize_stack_name(config.get("stack", "auto"))

    @classmethod
    def resolve_runner_class(cls, config: Mapping[str, Any]) -> type[TorchRunner]:
        stack = cls.resolve_stack(config)
        if stack in RUNNER_REGISTRY:
            return RUNNER_REGISTRY[stack]
        valid = ", ".join(sorted(RUNNER_REGISTRY))
        raise ValueError(f"Unknown stack: {stack!r}. Valid options are: {valid}")

    def __new__(cls, config):
        runner_cls = cls.resolve_runner_class(config)
        if cls is Runner:
            return runner_cls(config)
        dynamic_cls = type(cls.__name__, (cls, runner_cls), {})
        return super().__new__(dynamic_cls)
