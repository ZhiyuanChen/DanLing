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
from contextlib import contextmanager
from typing import Any

import torch

from .config import CompileConfig


class Compiler:
    """Small policy object for runner-owned `torch.compile` decisions."""

    def __init__(self, config: CompileConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enable", False))

    def compile(self, obj: Any) -> Any:
        if obj is None:
            return None
        if not self.enabled:
            return obj
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        return torch.compile(obj, **self.kwargs)

    @contextmanager
    def ddp_optimizer(self):
        if not self.enabled:
            yield
            return

        optimize_ddp = self.config.get("optimize_ddp", "ddp_optimizer")
        if optimize_ddp is None:
            yield
            return

        dynamo = getattr(torch, "_dynamo", None)
        dynamo_config = getattr(dynamo, "config", None)
        if dynamo_config is None or not hasattr(dynamo_config, "optimize_ddp"):
            yield
            return

        previous = dynamo_config.optimize_ddp
        dynamo_config.optimize_ddp = optimize_ddp
        try:
            yield
        finally:
            dynamo_config.optimize_ddp = previous

    @property
    def kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key in ("backend", "mode"):
            value = self.config.get(key)
            if value is not None:
                kwargs[key] = value

        for key in ("fullgraph", "dynamic"):
            value = self.config.get(key)
            if value is not None:
                kwargs[key] = bool(value)

        options = self.config.get("options")
        if options is not None:
            if not isinstance(options, Mapping):
                raise ValueError(f"`compile.options` must be a mapping, got {type(options).__name__}")
            kwargs["options"] = dict(options)
        return kwargs
