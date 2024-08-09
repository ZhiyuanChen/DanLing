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
from typing import Any, Literal

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

SCHEDULER_METRIC_UNSET = object()
SchedulerInterval = Literal["step", "epoch"]


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
            "metric-based schedulers require `scheduler.interval='epoch'` or `'validation'` "
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
                "Set `scheduler.interval='epoch'` and expose a monitored metric."
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

    def invalidate(self) -> None:
        self._parameters = None

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

        if self._parameters is not None:
            return self._parameters

        parameters = list(self.iter_unique_parameters(self._optimizer))
        self._parameters = parameters
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
        if self.parameter_cache is None:
            raise ValueError("optimizer is required")

    def step(
        self,
        *,
        max_grad_value: float | None = None,
        max_grad_norm: float | None = None,
        zero_grad: bool = True,
        skip_nonfinite_grad: bool = False,
        scheduler_metric: Any = SCHEDULER_METRIC_UNSET,
    ) -> bool:
        if skip_nonfinite_grad and self.has_nan_inf_grad():
            if zero_grad:
                self.optimizer.zero_grad()
            return False

        if max_grad_value is not None or max_grad_norm is not None:
            parameters = self.parameter_cache.get_parameters_for_clipping(self.optimizer)
            if max_grad_value is not None:
                clip_grad_value_(parameters, max_grad_value)
            if max_grad_norm is not None:
                clip_grad_norm_(parameters, max_grad_norm)

        self.optimizer.step()
        if self.scheduler_interval == "step":
            self.step_scheduler(scheduler_metric=scheduler_metric)
        if zero_grad:
            self.optimizer.zero_grad()
        return True

    def step_scheduler(self, *, scheduler_metric: Any = SCHEDULER_METRIC_UNSET) -> bool:
        return step_scheduler(self.scheduler, scheduler_metric=scheduler_metric)

    def has_nan_inf_grad(self) -> bool:
        parameters = self.parameter_cache.get_parameters_for_clipping(self.optimizer)
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                return True
        return False

    def zero_grad(self, *args, **kwargs) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def invalidate_cache(self) -> None:
        self.parameter_cache.invalidate()

    def __getattr__(self, name: str):
        return getattr(self.optimizer, name)
