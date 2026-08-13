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

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from types import FunctionType
from typing import Any

import torch

from .config import CompileConfig


def _call_module(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
    return module._call_impl(*args, **kwargs)


def _independent_module_group_call() -> Any:
    r"""Return one code-object-local module call while preserving ``nn.Module`` semantics."""

    return FunctionType(
        _call_module.__code__.replace(),
        _call_module.__globals__,
        _call_module.__name__,
        _call_module.__defaults__,
        _call_module.__closure__,
    )


class Compiler:
    """Small policy object for runner-owned `torch.compile` decisions."""

    def __init__(self, config: CompileConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def precompile_artifact_dir(self) -> str | None:
        artifact_dir = self.config.get("precompile_artifact_dir")
        return None if artifact_dir is None else str(artifact_dir)

    @property
    def memory_policy(self) -> str | None:
        policy = self.config.get("memory_policy")
        return None if policy is None else str(policy)

    def artifact_fingerprint(self, extra: Mapping[str, Any] | None = None) -> str:
        compile_config = dict(self.config)
        compile_config.pop("precompile_artifact_dir", None)
        payload: dict[str, Any] = {
            "compile": compile_config,
            "torch": torch.__version__,
        }
        if extra is not None:
            payload["extra"] = dict(extra)
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def compile(self, obj: Any) -> Any:
        if obj is None:
            return None
        if not self.enabled:
            return obj
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        compiled = torch.compile(obj, **self.kwargs)
        if isinstance(compiled, torch.nn.Module):
            compiled.forward = self.ddp_optimizer()(compiled.forward)  # type: ignore[method-assign]
            return compiled
        return self.ddp_optimizer()(compiled)

    def compile_modules(
        self,
        modules: Sequence[torch.nn.Module],
    ) -> tuple[torch.nn.Module, ...]:
        r"""Compile caller-grouped submodules through one shared graph cache."""

        modules = tuple(modules)
        if not self.enabled or not modules:
            return modules
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        if len({id(module) for module in modules}) != len(modules):
            raise ValueError("a compile group cannot contain the same module twice")
        if any(module._compiled_call_impl is not None for module in modules):
            raise ValueError("a compile group cannot contain an already compiled module")
        forward = type(modules[0]).forward
        if any(type(module).forward is not forward for module in modules[1:]):
            raise ValueError("a compile group must share one forward implementation")

        # Dynamo caches by Python code object. Give each caller-defined group one
        # independent frame without relying on version-specific compile kwargs.
        # Calling ``_call_impl`` (rather than ``forward``) preserves hooks and the
        # rest of ``nn.Module.__call__`` semantics.
        compiled_call = self.ddp_optimizer()(torch.compile(_independent_module_group_call(), **self.kwargs))
        for module in modules:
            module._compiled_call_impl = partial(compiled_call, module)
        return modules

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
