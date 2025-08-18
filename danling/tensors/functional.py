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

from typing import TYPE_CHECKING, Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .functions import NestedTensorFuncRegistry

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor


@NestedTensorFuncRegistry.implement(F.linear)
def linear(input: NestedTensor, weight: Tensor, bias: Tensor | None = None) -> NestedTensor:
    from .nested_tensor import NestedTensor

    concat, original_shapes = input.concatenate()
    output_shapes = [torch.Size([*i[:-1], weight.shape[0]]) for i in original_shapes]
    output = F.linear(concat, weight, bias)
    return NestedTensor.from_concatenated(output, output_shapes, **input._state)  # type: ignore[arg-type]


# Activations


@NestedTensorFuncRegistry.implement(F.elu)
def elu(input: NestedTensor, alpha: float = 1.0, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.elu(t, alpha, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.celu)
def celu(input: NestedTensor, alpha: float = 1.0, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.celu(t, alpha, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.selu)
def selu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.selu(t, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.gelu)
def gelu(input: NestedTensor, approximate: str = "none") -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.gelu(t, approximate) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.relu)
def relu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.relu(t, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.leaky_relu)
def leaky_relu(input: NestedTensor, negative_slope: float = 1e-2, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.leaky_relu(t, negative_slope, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.rrelu)
def rrelu(
    input: NestedTensor, lower: float = 1.0 / 8, upper: float = 1.0 / 3, training: bool = False, inplace: bool = False
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.rrelu(t, lower, upper, training, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.glu)
def glu(input: NestedTensor, dim: int = -1) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.glu(t, dim) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.silu)
def silu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.silu(t, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.hardtanh)
def hardtanh(input: NestedTensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.hardtanh(t, min_val, max_val, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.hardsigmoid)
def hardsigmoid(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.hardsigmoid(t, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.mish)
def mish(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.mish(t, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.hardswish)
def hardswish(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.hardswish(t, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.logsigmoid)
def logsigmoid(input: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.logsigmoid(t) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.softplus)
def softplus(input: NestedTensor, beta: float = 1.0, threshold: float = 20.0) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.softplus(t, beta, threshold) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.hardshrink)
def hardshrink(input: NestedTensor, lambd: float = 0.5) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.hardshrink(t, lambd) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.softshrink)
def softshrink(input: NestedTensor, lambd: float = 0.5) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.softshrink(t, lambd) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.threshold)
def threshold(input: NestedTensor, threshold: float, value: float, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.threshold(t, threshold, value, inplace) for t in input._storage)


# Convolutions


@NestedTensorFuncRegistry.implement(F.conv1d)
def conv1d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.conv1d(t, weight, bias, stride, padding, dilation, groups) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.conv2d)
def conv2d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.conv2d(t, weight, bias, stride, padding, dilation, groups) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.conv3d)
def conv3d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.conv3d(t, weight, bias, stride, padding, dilation, groups) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.conv_transpose1d)
def conv_transpose1d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.conv_transpose1d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.conv_transpose2d)
def conv_transpose2d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.conv_transpose2d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.conv_transpose3d)
def conv_transpose3d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.conv_transpose3d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input._storage
    )


# Criterions


@NestedTensorFuncRegistry.implement(F.ctc_loss)
def ctc_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.ctc_loss(
        input_concat,
        target_concat,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )


@NestedTensorFuncRegistry.implement(F.cosine_embedding_loss)
def cosine_embedding_loss(
    input1: NestedTensor,
    input2: NestedTensor,
    target: NestedTensor | Tensor,
    margin: float = 0.0,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input1_concat = input1.concat if isinstance(input1, NestedTensor) else input1.masked_select(target.mask)
    input2_concat = input2.concat if isinstance(input2, NestedTensor) else input2.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input1.mask)
    return F.cosine_embedding_loss(
        input1_concat,
        input2_concat,
        target_concat,
        margin=margin,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


@NestedTensorFuncRegistry.implement(F.hinge_embedding_loss)
def hinge_embedding_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    margin: float = 1.0,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.hinge_embedding_loss(
        input_concat, target_concat, margin=margin, size_average=size_average, reduce=reduce, reduction=reduction
    )


@NestedTensorFuncRegistry.implement(F.cross_entropy)
def cross_entropy(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.cross_entropy(
        input_concat,
        target_concat,
        weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


@NestedTensorFuncRegistry.implement(F.binary_cross_entropy)
def binary_cross_entropy(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.binary_cross_entropy(
        input_concat, target_concat, weight=weight, size_average=size_average, reduce=reduce, reduction=reduction
    )


@NestedTensorFuncRegistry.implement(F.binary_cross_entropy_with_logits)
def binary_cross_entropy_with_logits(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    pos_weight: Tensor | None = None,
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.binary_cross_entropy_with_logits(
        input_concat,
        target_concat,
        weight=weight,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        pos_weight=pos_weight,
    )


@NestedTensorFuncRegistry.implement(F.nll_loss)
def nll_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.nll_loss(
        input_concat,
        target_concat,
        weight=weight,
        ignore_index=ignore_index,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


@NestedTensorFuncRegistry.implement(F.gaussian_nll_loss)
def gaussian_nll_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    var: Tensor | None = None,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.gaussian_nll_loss(input_concat, target_concat, var=var, full=full, eps=eps, reduction=reduction)


@NestedTensorFuncRegistry.implement(F.poisson_nll_loss)
def poisson_nll_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    log_input: bool = False,
    full: bool = False,
    size_average: bool | None = None,
    eps: float = 1e-8,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.poisson_nll_loss(
        input_concat,
        target_concat,
        log_input=log_input,
        full=full,
        size_average=size_average,
        eps=eps,
        reduce=reduce,
        reduction=reduction,
    )


@NestedTensorFuncRegistry.implement(F.mse_loss)
def mse_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.mse_loss(input_concat, target_concat, size_average=size_average, reduce=reduce, reduction=reduction)


@NestedTensorFuncRegistry.implement(F.l1_loss)
def l1_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.l1_loss(input_concat, target_concat, size_average=size_average, reduce=reduce, reduction=reduction)


@NestedTensorFuncRegistry.implement(F.smooth_l1_loss)
def smooth_l1_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.smooth_l1_loss(
        input_concat, target_concat, size_average=size_average, reduce=reduce, reduction=reduction, beta=beta
    )


@NestedTensorFuncRegistry.implement(F.soft_margin_loss)
def soft_margin_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.soft_margin_loss(
        input_concat, target_concat, size_average=size_average, reduce=reduce, reduction=reduction
    )


@NestedTensorFuncRegistry.implement(F.multi_margin_loss)
def multi_margin_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.multi_margin_loss(
        input_concat,
        target_concat,
        p=p,
        margin=margin,
        weight=weight,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


@NestedTensorFuncRegistry.implement(F.multilabel_margin_loss)
def multilabel_margin_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.multilabel_margin_loss(
        input_concat, target_concat, size_average=size_average, reduce=reduce, reduction=reduction
    )


@NestedTensorFuncRegistry.implement(F.multilabel_soft_margin_loss)
def multilabel_soft_margin_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.multilabel_soft_margin_loss(
        input_concat, target_concat, weight=weight, size_average=size_average, reduce=reduce, reduction=reduction
    )


@NestedTensorFuncRegistry.implement(F.triplet_margin_loss)
def triplet_margin_loss(
    anchor: NestedTensor,
    positive: NestedTensor,
    negative: NestedTensor,
    margin: float = 1.0,
    p: int = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    if not isinstance(anchor, NestedTensor):
        positive = positive.masked_select(anchor.mask)
        negative = negative.masked_select(anchor.mask)
    anchor_concat = anchor.concat
    positive_concat = positive.concat
    negative_concat = negative.concat
    return F.triplet_margin_loss(
        anchor_concat,
        positive_concat,
        negative_concat,
        margin=margin,
        p=p,
        eps=eps,
        swap=swap,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


@NestedTensorFuncRegistry.implement(F.triplet_margin_with_distance_loss)
def triplet_margin_with_distance_loss(
    anchor: NestedTensor,
    positive: NestedTensor,
    negative: NestedTensor,
    distance_function: Callable[[Tensor, Tensor], Tensor] | None = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    anchor_concat = anchor.concat if isinstance(anchor, NestedTensor) else anchor.masked_select(positive.mask)
    positive_concat = positive.concat if isinstance(positive, NestedTensor) else positive.masked_select(anchor.mask)
    negative_concat = negative.concat if isinstance(negative, NestedTensor) else negative.masked_select(anchor.mask)
    return F.triplet_margin_with_distance_loss(
        anchor_concat,
        positive_concat,
        negative_concat,
        distance_function=distance_function,
        margin=margin,
        swap=swap,
        reduction=reduction,
    )


@NestedTensorFuncRegistry.implement(F.margin_ranking_loss)
def margin_ranking_loss(
    input1: NestedTensor,
    input2: NestedTensor,
    target: NestedTensor | Tensor,
    margin: float = 0.0,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input1_concat = input1.concat if isinstance(input1, NestedTensor) else input1.masked_select(target.mask)
    input2_concat = input2.concat if isinstance(input2, NestedTensor) else input2.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input1.mask)
    return F.margin_ranking_loss(
        input1_concat,
        input2_concat,
        target_concat,
        margin=margin,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


@NestedTensorFuncRegistry.implement(F.huber_loss)
def huber_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    reduction: str = "mean",
    delta: float = 1.0,
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.huber_loss(input_concat, target_concat, reduction, delta)


@NestedTensorFuncRegistry.implement(F.kl_div)
def kl_div(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input.masked_select(target.mask)
    target_concat = target.concat if isinstance(target, NestedTensor) else target.masked_select(input.mask)
    return F.kl_div(
        input_concat,
        target_concat,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        log_target=log_target,
    )


# Dropout


@NestedTensorFuncRegistry.implement(F.dropout)
def dropout(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.dropout(t, p, training, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.dropout1d)
def dropout1d(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.dropout1d(t, p, training, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.dropout2d)
def dropout2d(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.dropout2d(t, p, training, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.dropout3d)
def dropout3d(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.dropout3d(t, p, training, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.alpha_dropout)
def alpha_dropout(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.alpha_dropout(t, p, training, inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.feature_alpha_dropout)
def feature_alpha_dropout(
    input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.feature_alpha_dropout(t, p, training, inplace) for t in input._storage)


# Normalizations


@NestedTensorFuncRegistry.implement(F.batch_norm)
def batch_norm(
    input: NestedTensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    concat, original_shapes = input.concatenate()
    output = F.batch_norm(concat, running_mean, running_var, weight, bias, training, momentum, eps)
    return NestedTensor.from_concatenated(output, original_shapes, **input._state)


@NestedTensorFuncRegistry.implement(F.instance_norm)
def instance_norm(
    input: NestedTensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.instance_norm(
            t.unsqueeze(0), running_mean, running_var, weight, bias, use_input_stats, momentum, eps
        ).squeeze(0)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.group_norm)
def group_norm(
    input: NestedTensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.group_norm(t.unsqueeze(0), num_groups, weight, bias, eps).squeeze(0) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.layer_norm)
def layer_norm(
    input: NestedTensor,
    normalized_shape: Tuple,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.layer_norm(t.squeeze(0), normalized_shape, weight, bias, eps).squeeze(0) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.rms_norm)
def rms_norm(
    input: NestedTensor, normalized_shape: Tuple, weight: Tensor | None = None, eps: float = 1e-5
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.rms_norm(t.squeeze(0), normalized_shape, weight, eps).squeeze(0) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.local_response_norm)
def local_response_norm(
    input: NestedTensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.local_response_norm(t.unsqueeze(0), size, alpha, beta, k).squeeze(0) for t in input._storage)


# Utils


@NestedTensorFuncRegistry.implement(F.interpolate)
def interpolate(
    input: NestedTensor,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.interpolate(t, size, scale_factor, mode, align_corners, recompute_scale_factor) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.pad)
def pad(input: NestedTensor, pad: Tuple[int, ...], mode: str = "constant", value: float = 0.0) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.pad(t, pad, mode, value) for t in input._storage)
