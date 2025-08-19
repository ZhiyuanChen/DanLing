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

from typing import TYPE_CHECKING, Tuple

from torch import Tensor
from torch.nn import functional as F

from .functions import NestedTensorFuncRegistry

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor


@NestedTensorFuncRegistry.implement(F.linear)
def linear(input: NestedTensor, weight: Tensor, bias: Tensor | None = None) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.linear(t, weight, bias) for t in input._storage)


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


# Dropout


@NestedTensorFuncRegistry.implement(F.dropout)
def dropout(input: NestedTensor, p: float, training: bool = True) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.dropout(t, p, training) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.dropout1d)
def dropout1d(input: NestedTensor, p: float, training: bool = True) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.dropout1d(t, p, training) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.dropout2d)
def dropout2d(input: NestedTensor, p: float, training: bool = True) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.dropout2d(t, p, training) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.dropout3d)
def dropout3d(input: NestedTensor, p: float, training: bool = True) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.dropout3d(t, p, training) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.alpha_dropout)
def alpha_dropout(input: NestedTensor, p: float, training: bool = True) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.alpha_dropout(t, p, training) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.feature_alpha_dropout)
def feature_alpha_dropout(input: NestedTensor, p: float, training: bool = True) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.feature_alpha_dropout(t, p, training) for t in input._storage)


# Pooling


@NestedTensorFuncRegistry.implement(F.avg_pool1d)
def avg_pool1d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.avg_pool1d(t, kernel_size, stride, padding, ceil_mode, count_include_pad) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.avg_pool2d)
def avg_pool2d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.avg_pool2d(t, kernel_size, stride, padding, ceil_mode, count_include_pad) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.avg_pool3d)
def avg_pool3d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.avg_pool3d(t, kernel_size, stride, padding, ceil_mode, count_include_pad) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool1d)
def adaptive_avg_pool1d(input: NestedTensor, output_size: int | Tuple[int, ...]) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_avg_pool1d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool2d)
def adaptive_avg_pool2d(input: NestedTensor, output_size: int | Tuple[int, ...]) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_avg_pool2d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool3d)
def adaptive_avg_pool3d(input: NestedTensor, output_size: int | Tuple[int, ...]) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_avg_pool3d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.max_pool1d)
def max_pool1d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.max_pool1d(t, kernel_size, stride, padding, dilation, ceil_mode) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.max_pool2d)
def max_pool2d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.max_pool2d(t, kernel_size, stride, padding, dilation, ceil_mode) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.max_pool3d)
def max_pool3d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.max_pool3d(t, kernel_size, stride, padding, dilation, ceil_mode) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool1d)
def adaptive_max_pool1d(input: NestedTensor, output_size: int | Tuple[int, ...]) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_max_pool1d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool2d)
def adaptive_max_pool2d(input: NestedTensor, output_size: int | Tuple[int, ...]) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_max_pool2d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool3d)
def adaptive_max_pool3d(input: NestedTensor, output_size: int | Tuple[int, ...]) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_max_pool3d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.max_pool1d_with_indices)
def max_pool1d_with_indices(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> Tuple[NestedTensor, NestedTensor]:
    from .nested_tensor import NestedTensor

    pooled, indices = zip(
        *[F.max_pool1d_with_indices(t, kernel_size, stride, padding, dilation, ceil_mode) for t in input._storage]
    )
    return NestedTensor(pooled), NestedTensor(indices)


@NestedTensorFuncRegistry.implement(F.max_pool2d_with_indices)
def max_pool2d_with_indices(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> Tuple[NestedTensor, NestedTensor]:
    from .nested_tensor import NestedTensor

    pooled, indices = zip(
        *[F.max_pool2d_with_indices(t, kernel_size, stride, padding, dilation, ceil_mode) for t in input._storage]
    )
    return NestedTensor(pooled), NestedTensor(indices)


@NestedTensorFuncRegistry.implement(F.max_pool3d_with_indices)
def max_pool3d_with_indices(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> Tuple[NestedTensor, NestedTensor]:
    from .nested_tensor import NestedTensor

    pooled, indices = zip(
        *[F.max_pool3d_with_indices(t, kernel_size, stride, padding, dilation, ceil_mode) for t in input._storage]
    )
    return NestedTensor(pooled), NestedTensor(indices)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool1d_with_indices)
def adaptive_max_pool1d_with_indices(
    input: NestedTensor, output_size: int | Tuple[int, ...]
) -> Tuple[NestedTensor, NestedTensor]:
    from .nested_tensor import NestedTensor

    pooled, indices = zip(*[F.adaptive_max_pool1d_with_indices(t, output_size) for t in input._storage])
    return NestedTensor(pooled), NestedTensor(indices)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool2d_with_indices)
def adaptive_max_pool2d_with_indices(
    input: NestedTensor, output_size: int | Tuple[int, ...]
) -> Tuple[NestedTensor, NestedTensor]:
    from .nested_tensor import NestedTensor

    pooled, indices = zip(*[F.adaptive_max_pool2d_with_indices(t, output_size) for t in input._storage])
    return NestedTensor(pooled), NestedTensor(indices)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool3d_with_indices)
def adaptive_max_pool3d_with_indices(
    input: NestedTensor, output_size: int | Tuple[int, ...]
) -> Tuple[NestedTensor, NestedTensor]:
    from .nested_tensor import NestedTensor

    pooled, indices = zip(*[F.adaptive_max_pool3d_with_indices(t, output_size) for t in input._storage])
    return NestedTensor(pooled), NestedTensor(indices)


@NestedTensorFuncRegistry.implement(F.fractional_max_pool2d)
def fractional_max_pool2d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    output_size: int | Tuple[int, ...] | None = None,
    output_ratio: float | Tuple[float, float] | None = None,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.fractional_max_pool2d(t, kernel_size, output_size, output_ratio, return_indices, ceil_mode)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.fractional_max_pool3d)
def fractional_max_pool3d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    output_size: int | Tuple[int, ...] | None = None,
    output_ratio: float | Tuple[float, float, float] | None = None,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.fractional_max_pool3d(t, kernel_size, output_size, output_ratio, return_indices, ceil_mode)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.fractional_max_pool2d_with_indices)
def fractional_max_pool2d_with_indices(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    output_size: int | Tuple[int, ...] | None = None,
    output_ratio: float | Tuple[float, float] | None = None,
    return_indices: bool = True,
    ceil_mode: bool = False,
) -> Tuple[NestedTensor, NestedTensor]:
    from .nested_tensor import NestedTensor

    pooled, indices = zip(
        *[
            F.fractional_max_pool2d_with_indices(t, kernel_size, output_size, output_ratio, return_indices, ceil_mode)
            for t in input._storage
        ]
    )
    return NestedTensor(pooled), NestedTensor(indices)


@NestedTensorFuncRegistry.implement(F.fractional_max_pool3d_with_indices)
def fractional_max_pool3d_with_indices(
    input: NestedTensor,
    kernel_size: int | Tuple[int, ...],
    output_size: int | Tuple[int, ...] | None = None,
    output_ratio: float | Tuple[float, float, float] | None = None,
    return_indices: bool = True,
    ceil_mode: bool = False,
) -> Tuple[NestedTensor, NestedTensor]:
    from .nested_tensor import NestedTensor

    pooled, indices = zip(
        *[
            F.fractional_max_pool3d_with_indices(t, kernel_size, output_size, output_ratio, return_indices, ceil_mode)
            for t in input._storage
        ]
    )
    return NestedTensor(pooled), NestedTensor(indices)


@NestedTensorFuncRegistry.implement(F.lp_pool1d)
def lp_pool1d(
    input: NestedTensor,
    norm_type: float,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    ceil_mode: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.lp_pool1d(t, norm_type, kernel_size, stride, ceil_mode) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.lp_pool2d)
def lp_pool2d(
    input: NestedTensor,
    norm_type: float,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    ceil_mode: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.lp_pool2d(t, norm_type, kernel_size, stride, ceil_mode) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.lp_pool3d)
def lp_pool3d(
    input: NestedTensor,
    norm_type: float,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] | None = None,
    ceil_mode: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.lp_pool3d(t, norm_type, kernel_size, stride, ceil_mode) for t in input._storage)


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

    concat_tensor, original_shapes = input.concatenate()
    normalized = F.batch_norm(concat_tensor, running_mean, running_var, weight, bias, training, momentum, eps)
    return NestedTensor.from_concatenated(normalized, original_shapes, **input._state)


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
