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

from collections.abc import Mapping, Sequence
from typing import Any

import torch

_SUPPORTED_COMPONENTS = frozenset({"model", "loss"})


def _compile_cfg(config: Any):
    compile_cfg = config.get("compile")
    if not compile_cfg:
        return None
    if not bool(compile_cfg.get("enable", False)):
        return None
    return compile_cfg


def _normalize_components(raw_components: Any) -> set[str]:
    if raw_components is None:
        return {"model"}
    if isinstance(raw_components, str):
        components = [raw_components]
    elif isinstance(raw_components, Sequence):
        components = list(raw_components)
    else:
        raise ValueError(
            "`compile.components` must be a string or a sequence of strings, " f"got {type(raw_components).__name__}"
        )

    normalized = {str(component).strip().lower() for component in components if str(component).strip()}
    if not normalized:
        return {"model"}

    unsupported = normalized - _SUPPORTED_COMPONENTS
    if unsupported:
        raise ValueError(
            f"Unknown compile components: {sorted(unsupported)}. "
            f"Supported components are {sorted(_SUPPORTED_COMPONENTS)}."
        )
    return normalized


def should_compile_component(config: Any, component: str) -> bool:
    compile_cfg = _compile_cfg(config)
    if compile_cfg is None:
        return False
    components = _normalize_components(compile_cfg.get("components"))
    return component.lower() in components


def maybe_compile_component(obj: Any, config: Any, *, component: str) -> Any:
    if obj is None:
        return None
    if not should_compile_component(config, component):
        return obj
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")

    compile_cfg = _compile_cfg(config)
    if compile_cfg is None:
        return obj

    compile_kwargs: dict[str, Any] = {}
    backend = compile_cfg.get("backend")
    if backend is not None:
        compile_kwargs["backend"] = backend
    mode = compile_cfg.get("mode")
    if mode is not None:
        compile_kwargs["mode"] = mode
    fullgraph = compile_cfg.get("fullgraph")
    if fullgraph is not None:
        compile_kwargs["fullgraph"] = bool(fullgraph)
    dynamic = compile_cfg.get("dynamic")
    if dynamic is not None:
        compile_kwargs["dynamic"] = bool(dynamic)
    options = compile_cfg.get("options")
    if isinstance(options, Mapping):
        compile_kwargs["options"] = dict(options)

    return torch.compile(obj, **compile_kwargs)


def maybe_compile_model(model: Any, config: Any) -> Any:
    return maybe_compile_component(model, config, component="model")


def maybe_compile_loss(loss_fn: Any, config: Any) -> Any:
    return maybe_compile_component(loss_fn, config, component="loss")


def maybe_enable_ddp_optimizer(config: Any) -> None:
    if not should_compile_component(config, "model"):
        return

    compile_cfg = _compile_cfg(config)
    if compile_cfg is None:
        return
    optimize_ddp = compile_cfg.get("optimize_ddp", "ddp_optimizer")
    if optimize_ddp is None:
        return

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None:
        return
    dynamo_config = getattr(dynamo, "config", None)
    if dynamo_config is None or not hasattr(dynamo_config, "optimize_ddp"):
        return

    dynamo_config.optimize_ddp = optimize_ddp
