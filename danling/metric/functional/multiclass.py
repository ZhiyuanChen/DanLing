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
    from torcheval.metrics import functional as tef

from .preprocess import preprocess_multiclass, with_preprocess


@with_preprocess(preprocess_multiclass, ignore_index=-100)
def multiclass_accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_classes: int | None = None,
    **kwargs,
):
    te.check()
    return tef.multiclass_accuracy(input=input, target=target, num_classes=num_classes, average=average, **kwargs)


@with_preprocess(preprocess_multiclass, ignore_index=-100)
def multiclass_auprc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_classes: int | None = None,
    **kwargs,
):
    te.check()
    return tef.multiclass_auprc(input=input, target=target, num_classes=num_classes, average=average, **kwargs)


@with_preprocess(preprocess_multiclass, ignore_index=-100)
def multiclass_auroc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_classes: int | None = None,
    **kwargs,
):
    te.check()
    return tef.multiclass_auroc(input=input, target=target, num_classes=num_classes, average=average, **kwargs)


@with_preprocess(preprocess_multiclass, ignore_index=-100)
def multiclass_f1_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_classes: int | None = None,
    **kwargs,
):
    te.check()
    return tef.multiclass_f1_score(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
