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
with try_import() as tm:
    from torchmetrics import functional as tmf

from .preprocess import preprocess_multilabel, with_preprocess


@with_preprocess(preprocess_multilabel, ignore_index=-100)
def multilabel_accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    num_labels: int | None = None,
    **kwargs,
):
    te.check()
    return tef.multilabel_accuracy(input=input, target=target, threshold=threshold, **kwargs)


@with_preprocess(preprocess_multilabel, ignore_index=-100)
def multilabel_auprc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_labels: int | None = None,
    **kwargs,
):
    te.check()
    return tef.multilabel_auprc(input=input, target=target, num_labels=num_labels, average=average, **kwargs)


@with_preprocess(preprocess_multilabel, ignore_index=-100)
def multilabel_auroc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_labels: int | None = None,
    **kwargs,
):
    te.check()
    if average == "macro":
        return tef.binary_auroc(input=input.T, target=target.T, num_tasks=num_labels, **kwargs).mean()
    if average == "macro":
        return tef.binary_auroc(input=input.flatten(), target=target.flatten(), **kwargs)
    raise ValueError(f"Invalid average: {average}")


@with_preprocess(preprocess_multilabel, ignore_index=-100)
def multilabel_f1_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    num_labels: int | None = None,
    **kwargs,
):
    tm.check()
    return tmf.classification.multilabel_f1_score(
        input, target, threshold=threshold, num_labels=num_labels, average=average, **kwargs
    )
