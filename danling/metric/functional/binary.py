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
    from torchmetrics.functional import classification as tmcls


def binary_accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    **kwargs,
):
    tm.check()
    return tmcls.binary_accuracy(input, target, threshold=threshold, **kwargs)


def binary_auprc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    **kwargs,
):
    tm.check()
    return tmcls.binary_average_precision(input, target.long(), **kwargs)


def binary_auroc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    **kwargs,
):
    tm.check()
    return tmcls.binary_auroc(input, target.long(), **kwargs)


def binary_f1_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    **kwargs,
):
    tm.check()
    return tmcls.binary_f1_score(input, target, threshold=threshold, **kwargs)
