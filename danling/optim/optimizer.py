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

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


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

    def __init__(self, optimizer: optim.Optimizer, scheduler: object | None = None) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        if zero_grad:
            self.optimizer.zero_grad()
        return True

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
