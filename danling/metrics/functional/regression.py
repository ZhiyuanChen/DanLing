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

from typing import TYPE_CHECKING, Optional

import torch
from lazy_imports import try_import
from torch import Tensor

from .utils import MetricFunc

with try_import() as tm:
    from torchmetrics.functional import regression as tmreg

if TYPE_CHECKING:  # pragma: no cover
    from ..state import MetricState


def _reduce_correlation(
    values: Tensor,
    targets: Tensor,
    multioutput: str,
) -> Tensor:
    if multioutput == "raw_values":
        return values
    if multioutput == "uniform_average":
        return values.mean()
    if multioutput == "variance_weighted":
        if values.ndim == 0 or targets.ndim <= 1:
            return values
        weights = targets.to(dtype=values.dtype).var(dim=0, correction=0)
        total_weight = weights.sum()
        if total_weight <= 0:
            return values.mean()
        return (values * weights).sum() / total_weight
    raise ValueError(f"Invalid multioutput value: {multioutput}")


def _empty_regression_values(num_outputs: int) -> Tensor:
    if num_outputs <= 1:
        return torch.tensor(float("nan"))
    return torch.full((num_outputs,), float("nan"))


class pearson(MetricFunc):
    def __init__(self, multioutput: str = "uniform_average", *, name: Optional[str] = "pearson") -> None:
        self.multioutput = multioutput
        super().__init__(name=name, preds_targets=True, task="regression")

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        tm.check()
        pearson = tmreg.pearson_corrcoef(state.preds, state.targets)
        return _reduce_correlation(pearson, state.targets, self.multioutput)


class spearman(MetricFunc):
    def __init__(self, multioutput: str = "uniform_average", *, name: Optional[str] = "spearman") -> None:
        self.multioutput = multioutput
        super().__init__(name=name, preds_targets=True, task="regression")

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        tm.check()
        spearman = tmreg.spearman_corrcoef(state.preds, state.targets)
        return _reduce_correlation(spearman, state.targets, self.multioutput)


class r2_score(MetricFunc):
    def __init__(self, multioutput: str = "uniform_average", *, name: Optional[str] = "r2") -> None:
        self.multioutput = multioutput
        super().__init__(name=name, preds_targets=True, task="regression")

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        tm.check()
        return tmreg.r2_score(state.preds, state.targets, multioutput=self.multioutput)


class rmse(MetricFunc):
    def __init__(self, num_outputs: int = 1, *, name: Optional[str] = "rmse") -> None:
        self.num_outputs = num_outputs
        super().__init__(name=name, preds_targets=True, task="regression")

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return _empty_regression_values(self.num_outputs)
        tm.check()
        return tmreg.mean_squared_error(state.preds, state.targets, squared=False, num_outputs=self.num_outputs)


class mse(MetricFunc):
    def __init__(self, num_outputs: int = 1, *, name: Optional[str] = "mse") -> None:
        self.num_outputs = num_outputs
        super().__init__(name=name, preds_targets=True, task="regression")

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return _empty_regression_values(self.num_outputs)
        tm.check()
        return tmreg.mean_squared_error(state.preds, state.targets, squared=True, num_outputs=self.num_outputs)


class mae(MetricFunc):
    def __init__(self, num_outputs: int = 1, *, name: Optional[str] = "mae") -> None:
        self.num_outputs = num_outputs
        super().__init__(name=name, preds_targets=True, task="regression")

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return _empty_regression_values(self.num_outputs)
        tm.check()
        return tmreg.mean_absolute_error(state.preds, state.targets, num_outputs=self.num_outputs)
