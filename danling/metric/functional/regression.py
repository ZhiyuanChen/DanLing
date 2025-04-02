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

import torch
from lazy_imports import try_import
from torch import Tensor

from danling.tensor import NestedTensor

with try_import() as te:
    from torcheval.metrics import functional as tef
with try_import() as tm:
    from torchmetrics import functional as tmf

from .preprocess import preprocess_regression, with_preprocess


@with_preprocess(preprocess_regression, num_outputs=1, ignore_nan=False)
def pearson(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    **kwargs,
):
    tm.check()
    try:
        ret = tmf.pearson_corrcoef(input, target, **kwargs)
        return ret.mean()
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


@with_preprocess(preprocess_regression, num_outputs=1, ignore_nan=False)
def spearman(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    **kwargs,
):
    tm.check()
    try:
        ret = tmf.spearman_corrcoef(input, target, **kwargs)
        return ret.mean()
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


@with_preprocess(preprocess_regression, num_outputs=1, ignore_nan=False)
def r2_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    multioutput: str = "raw_values",
    **kwargs,
):
    te.check()
    try:
        ret = tef.r2_score(input, target, multioutput=multioutput, **kwargs)
        if multioutput != "raw_values":
            return ret
        return ret.mean()
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


@with_preprocess(preprocess_regression, num_outputs=1, ignore_nan=False)
def mse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    multioutput: str = "raw_values",
    **kwargs,
):
    te.check()
    ret = tef.mean_squared_error(input, target, **kwargs)
    if multioutput != "raw_values":
        return ret
    return ret.mean()


@with_preprocess(preprocess_regression, num_outputs=1, ignore_nan=False)
def rmse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    multioutput: str = "raw_values",
    **kwargs,
):
    te.check()
    ret = tef.mean_squared_error(input, target, **kwargs).sqrt()
    if multioutput != "raw_values":
        return ret
    return ret.mean()
