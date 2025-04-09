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

# pylint: disable=redefined-builtin
# mypy: disable-error-code="arg-type"
from __future__ import annotations

from lazy_imports import try_import
from torch import Tensor

from danling.tensor import NestedTensor

with try_import() as tm:
    from torchmetrics.functional import regression as tmreg


def pearson(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    multioutput: str = "uniform_average",
    num_outputs: int = 1,
    **kwargs,
):
    tm.check()
    pearson = tmreg.pearson_corrcoef(input, target, **kwargs)
    if multioutput == "raw_values":
        return pearson
    if multioutput == "uniform_average":
        return pearson.mean()
    if multioutput == "variance_weighted":
        return pearson.mul(1 / pearson.var(dim=0)).sum(dim=0)
    raise ValueError(f"Invalid multioutput value: {multioutput}")


def spearman(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    multioutput: str = "uniform_average",
    num_outputs: int = 1,
    **kwargs,
):
    tm.check()
    spearman = tmreg.spearman_corrcoef(input, target, **kwargs)
    if multioutput == "raw_values":
        return spearman
    if multioutput == "uniform_average":
        return spearman.mean()
    if multioutput == "variance_weighted":
        return spearman.mul(1 / spearman.var(dim=0)).sum(dim=0)
    raise ValueError(f"Invalid multioutput value: {multioutput}")


def r2_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    multioutput: str = "uniform_average",
    num_outputs: int = 1,
    **kwargs,
):
    tm.check()
    return tmreg.r2_score(input, target, multioutput=multioutput, **kwargs)


def mae(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    **kwargs,
):
    tm.check()
    return tmreg.mean_absolute_error(input, target, **kwargs)


def mse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    **kwargs,
):
    tm.check()
    return tmreg.mean_squared_error(input, target, squared=True, num_outputs=num_outputs, **kwargs)


def rmse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    **kwargs,
):
    tm.check()
    return tmreg.mean_squared_error(input, target, squared=False, num_outputs=num_outputs, **kwargs)
