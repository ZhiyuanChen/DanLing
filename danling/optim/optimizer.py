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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import distributed as dist
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_ as native_clip_grad_norm_
from torch.nn.utils import clip_grad_value_

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

SCHEDULER_METRIC_UNSET = object()
SchedulerInterval = Literal["step", "epoch"]


@torch.no_grad()
def clip_grad_norm(
    parameters: list[nn.Parameter], max_norm: float, *, pipeline_group: Any = None, device: torch.device | None = None
) -> torch.Tensor:
    """Clip one model norm across distinct DTensor meshes and pipeline stages.

    A DTensor norm reduces its own mesh. Pipeline ranks own disjoint model
    parameters, so their squared norms need a separate sum before every rank
    applies the same clipping coefficient. Scalar accumulation preserves
    Float64 gradients and keeps empty stages in the pipeline collective.
    """
    if pipeline_group is None and not any(isinstance(parameter.grad, DTensor) for parameter in parameters):
        return native_clip_grad_norm_(parameters, max_norm)
    by_mesh: dict[Any, list[torch.Tensor]] = {}
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad
            mesh = gradient.device_mesh if isinstance(gradient, DTensor) else None
            by_mesh.setdefault((mesh, gradient.device), []).append(gradient)
    squared_norms = []
    for gradients in by_mesh.values():
        norms = torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients])
        norm = torch.linalg.vector_norm(norms)
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        squared_norms.append(norm.to(dtype=torch.float64).square())
    if squared_norms:
        norm_device = device or squared_norms[0].device
        total_squared = torch.stack([norm.to(norm_device) for norm in squared_norms]).sum()
    else:
        total_squared = torch.zeros((), dtype=torch.float64, device=device)
    if pipeline_group is not None:
        dist.all_reduce(total_squared, op=dist.ReduceOp.SUM, group=pipeline_group)
    total_norm = total_squared.sqrt()
    coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for gradients in by_mesh.values():
        for gradient in gradients:
            gradient.mul_(coefficient.to(device=gradient.device))
    return total_norm


@dataclass(slots=True)
class OptimizerStepResult:
    stepped: bool
    grad_norm: float | None = None


def scheduler_requires_metric(scheduler: object | None) -> bool:
    plateau_cls = getattr(optim.lr_scheduler, "ReduceLROnPlateau", None)
    return plateau_cls is not None and isinstance(scheduler, plateau_cls)


def normalize_scheduler_interval(interval: str | None, scheduler: object | None) -> SchedulerInterval:
    aliases: dict[str, SchedulerInterval] = {
        "step": "step",
        "steps": "step",
        "update": "step",
        "updates": "step",
        "epoch": "epoch",
        "epochs": "epoch",
        "eval": "epoch",
        "evaluation": "epoch",
        "validation": "epoch",
    }
    if interval is None:
        return "epoch" if scheduler_requires_metric(scheduler) else "step"

    normalized = str(interval).strip().lower()
    if normalized not in aliases:
        choices = "', '".join(sorted(aliases))
        raise ValueError(f"invalid scheduler interval: {interval!r}. Expected one of '{choices}'.")

    resolved = aliases[normalized]
    if scheduler_requires_metric(scheduler) and resolved == "step":
        raise ValueError(
            "metric-based schedulers require `sched.interval='epoch'` or `'validation'` "
            "under the runner-owned scheduler contract."
        )
    return resolved


def step_scheduler(scheduler: object | None, *, scheduler_metric: Any = SCHEDULER_METRIC_UNSET) -> bool:
    if scheduler is None:
        return False

    step = getattr(scheduler, "step", None)
    if not callable(step):
        return False

    if scheduler_requires_metric(scheduler):
        if scheduler_metric is SCHEDULER_METRIC_UNSET:
            raise ValueError(
                "scheduler step requires an explicit metric, but none was provided. "
                "Set `sched.interval='epoch'` and expose a monitored metric."
            )
        step(scheduler_metric)
        return True

    step()
    return True


class OptimizerParameterCache:
    """Cache unique optimizer parameters for repeated clipping calls."""

    _CACHE_ATTR = "_danling_parameter_cache"
    _optimizer: optim.Optimizer | None
    _parameters: list[nn.Parameter] | None

    def __init__(self, optimizer: optim.Optimizer | None = None) -> None:
        self._optimizer = optimizer
        self._parameters = None
        self._group_signature: tuple[tuple[int, int], ...] | None = None

    def invalidate(self) -> None:
        self._parameters = None
        self._group_signature = None

    def bind(self, optimizer: optim.Optimizer | None) -> None:
        if optimizer is self._optimizer:
            return
        self._optimizer = optimizer
        self.invalidate()

    def get_parameters_for_clipping(self, optimizer: optim.Optimizer | None = None) -> list[nn.Parameter]:
        if optimizer is not None:
            self.bind(optimizer)
        if self._optimizer is None:
            return []

        group_signature = tuple((id(group["params"]), len(group["params"])) for group in self._optimizer.param_groups)
        if self._parameters is not None and self._group_signature == group_signature:
            return self._parameters

        parameters = list(self.iter_unique_parameters(self._optimizer))
        self._parameters = parameters
        self._group_signature = group_signature
        return parameters

    @staticmethod
    def iter_unique_parameters(optimizer: optim.Optimizer | None) -> Iterator[nn.Parameter]:
        if optimizer is None:
            return

        seen = set()
        for group in optimizer.param_groups:
            for parameter in group.get("params", ()):
                if parameter is None:
                    continue
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                yield parameter


class OptimizerContainer:
    """Container that owns optimizer-side step concerns."""

    _CONTAINER_ATTR = "_danling_optimizer_container"
    optimizer: optim.Optimizer
    parameter_cache: OptimizerParameterCache
    scheduler: object | None
    scheduler_interval: SchedulerInterval

    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler: object | None = None,
        scheduler_interval: str | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = normalize_scheduler_interval(scheduler_interval, scheduler)
        self.parameter_cache = OptimizerParameterCache(optimizer)

    def clip_gradients(
        self,
        *,
        max_grad_value: float | None = None,
        max_grad_norm: float | None = None,
        pipeline_group: Any = None,
        device: torch.device | None = None,
    ) -> float | None:
        if max_grad_value is None and max_grad_norm is None:
            return None

        parameters = self.parameter_cache.get_parameters_for_clipping(self.optimizer)
        if max_grad_value is not None:
            groups: dict[Any, list[nn.Parameter]] = {}
            for parameter in parameters:
                if parameter.grad is not None:
                    gradient = parameter.grad
                    mesh = gradient.device_mesh if isinstance(gradient, DTensor) else None
                    placements = gradient.placements if isinstance(gradient, DTensor) else None
                    groups.setdefault((mesh, placements, gradient.device), []).append(parameter)
            for group in groups.values():
                clip_grad_value_(group, max_grad_value)
        if max_grad_norm is not None:
            grad_norm = clip_grad_norm(parameters, max_grad_norm, pipeline_group=pipeline_group, device=device)
            return float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        return None

    def step(
        self,
        *,
        max_grad_value: float | None = None,
        max_grad_norm: float | None = None,
        zero_grad: bool = True,
        skip_nonfinite_grad: bool = False,
        scheduler_metric: Any = SCHEDULER_METRIC_UNSET,
    ) -> OptimizerStepResult:
        if skip_nonfinite_grad and self.has_nan_inf_grad():
            if zero_grad:
                self.optimizer.zero_grad()
            return OptimizerStepResult(stepped=False)

        grad_norm = self.clip_gradients(max_grad_value=max_grad_value, max_grad_norm=max_grad_norm)

        self.optimizer.step()
        if self.scheduler_interval == "step":
            self.step_scheduler(scheduler_metric=scheduler_metric)
        if zero_grad:
            self.optimizer.zero_grad()
        return OptimizerStepResult(stepped=True, grad_norm=grad_norm)

    def step_scheduler(self, *, scheduler_metric: Any = SCHEDULER_METRIC_UNSET) -> bool:
        return step_scheduler(self.scheduler, scheduler_metric=scheduler_metric)

    def has_nan_inf_grad(self) -> bool:
        parameters = self.parameter_cache.get_parameters_for_clipping(self.optimizer)
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce().values()
            elif grad.layout != torch.strided:
                grad = grad.values()
            if not torch.isfinite(grad).all():
                return True
        return False

    def zero_grad(self, *args, **kwargs) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def invalidate_cache(self) -> None:
        self.parameter_cache.invalidate()

    def __getattr__(self, name: str):
        return getattr(self.optimizer, name)
