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

with try_import() as te:
    from torchmetrics.functional import classification as tmcls


def multiclass_accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_classes: int | None = None,
    **kwargs,
):
    te.check()
    return tmcls.multiclass_accuracy(input, target, num_classes=num_classes, average=average, **kwargs)


def multiclass_auprc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_classes: int | None = None,
    **kwargs,
):
    te.check()
    return tmcls.multiclass_average_precision(input, target, num_classes=num_classes, average=average, **kwargs)


def multiclass_auroc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_classes: int | None = None,
    **kwargs,
):
    te.check()
    return tmcls.multiclass_auroc(input, target, num_classes=num_classes, average=average, **kwargs)


def multiclass_f1_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_classes: int | None = None,
    **kwargs,
):
    te.check()
    return tmcls.multiclass_f1_score(input, target, num_classes=num_classes, average=average, **kwargs)
