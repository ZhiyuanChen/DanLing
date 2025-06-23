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


@NestedTensorFuncRegistry.implement(F.embedding)
def embedding(
    input: NestedTensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.embedding(t, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.embedding_bag)
def embedding_bag(
    input: NestedTensor,
    weight: Tensor,
    offsets: Tensor | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Tensor | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.embedding_bag(
            t,
            weight,
            offsets=offsets if offsets is not None else torch.tensor([0], device=t.device, dtype=torch.long),
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.linear)
def linear(input: NestedTensor, weight: Tensor, bias: Tensor | None = None) -> NestedTensor:
    from .nested_tensor import NestedTensor

    concat, original_shapes = input.concatenate()
    output_shapes = [torch.Size([*i[:-1], weight.shape[0]]) for i in original_shapes]
    output = F.linear(concat, weight, bias)
    return NestedTensor.from_concatenated(output, output_shapes, **input._state)  # type: ignore[arg-type]


# Normalization


@NestedTensorFuncRegistry.implement(F.normalize)
def normalize(
    input: NestedTensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
    out=None,  # noqa: ANN001
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    adjusted_dim = dim if dim < 0 else dim - 1
    return NestedTensor(F.normalize(t, p=p, dim=adjusted_dim, eps=eps, out=out) for t in input._storage)


# Activations


@NestedTensorFuncRegistry.implement(F.elu)
def elu(input: NestedTensor, alpha: float = 1.0, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.elu(t, alpha=alpha, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.celu)
def celu(input: NestedTensor, alpha: float = 1.0, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.celu(t, alpha=alpha, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.selu)
def selu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.selu(t, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.gelu)
def gelu(input: NestedTensor, approximate: str = "none") -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.gelu(t, approximate=approximate) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.relu)
def relu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.relu(t, inplace=inplace) for t in input._storage)


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

    return NestedTensor(F.glu(t, dim=dim) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.silu)
def silu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.silu(t, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.hardtanh)
def hardtanh(input: NestedTensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.hardtanh(t, min_val=min_val, max_val=max_val, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.hardsigmoid)
def hardsigmoid(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.hardsigmoid(t, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.mish)
def mish(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.mish(t, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.relu6)
def relu6(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.relu6(t, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.hardswish)
def hardswish(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.hardswish(t, inplace=inplace) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.logsigmoid)
def logsigmoid(input: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.logsigmoid(t) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.sigmoid)
def sigmoid(input: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.sigmoid(t) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.softplus)
def softplus(input: NestedTensor, beta: float = 1.0, threshold: float = 20.0) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.softplus(t, beta=beta, threshold=threshold) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.hardshrink)
def hardshrink(input: NestedTensor, lambd: float = 0.5) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.hardshrink(t, lambd=lambd) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.softsign)
def softsign(input: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.softsign(t) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.tanh)
def tanh(input: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.tanh(t) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.tanhshrink)
def tanhshrink(input: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.tanhshrink(t) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.softmax)
def softmax(input: NestedTensor, dim: int, _stacklevel: int = 3, dtype=None):
    from .nested_tensor import NestedTensor

    return NestedTensor(F.softmax(t, dim=dim, _stacklevel=_stacklevel, dtype=dtype) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.log_softmax)
def log_softmax(input: NestedTensor, dim: int, _stacklevel: int = 3, dtype=None):
    from .nested_tensor import NestedTensor

    return NestedTensor(F.log_softmax(t, dim=dim, _stacklevel=_stacklevel, dtype=dtype) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.softmin)
def softmin(input: NestedTensor, dim: int, _stacklevel: int = 3, dtype=None):
    from .nested_tensor import NestedTensor

    return NestedTensor(F.softmin(t, dim=dim, _stacklevel=_stacklevel, dtype=dtype) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.gumbel_softmax)
def gumbel_softmax(input: NestedTensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-10, dim: int = -1):
    from .nested_tensor import NestedTensor

    return NestedTensor(F.gumbel_softmax(t, tau=tau, hard=hard, eps=eps, dim=dim) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.softshrink)
def softshrink(input: NestedTensor, lambd: float = 0.5) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.softshrink(t, lambd) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.threshold)
def threshold(input: NestedTensor, threshold: float, value: float, inplace: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.threshold(t, threshold, value, inplace) for t in input._storage)


# Geometric


@NestedTensorFuncRegistry.implement(F.grid_sample)
def grid_sample(
    input: NestedTensor,
    grid: NestedTensor | Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool | None = None,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    if isinstance(grid, NestedTensor):
        return NestedTensor(
            F.grid_sample(i, g, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
            for i, g in zip(input._storage, grid._storage)
        )
    return NestedTensor(
        F.grid_sample(i, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        for i in input._storage
    )


@NestedTensorFuncRegistry.implement(F.affine_grid)
def affine_grid(theta: NestedTensor, size: Tensor, align_corners: bool | None = None) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.affine_grid(t, size, align_corners=align_corners) for t in theta._storage)


# Convolutions


@NestedTensorFuncRegistry.implement(F.conv1d)
def conv1d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.conv1d(t, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.conv2d)
def conv2d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.conv2d(t, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.conv3d)
def conv3d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.conv3d(t, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        for t in input._storage
    )


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
        F.conv_transpose1d(
            t,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )
        for t in input._storage
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
        F.conv_transpose2d(
            t,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )
        for t in input._storage
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
        F.conv_transpose3d(
            t,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )
        for t in input._storage
    )


# Pooling


@NestedTensorFuncRegistry.implement(F.avg_pool1d)
def avg_pool1d(
    input: NestedTensor, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.avg_pool1d(
            t,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.avg_pool2d)
def avg_pool2d(
    input: NestedTensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.avg_pool2d(
            t,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.avg_pool3d)
def avg_pool3d(
    input: NestedTensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.avg_pool3d(
            t,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.max_pool1d)
def max_pool1d(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False
):
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor max_pool1d.")
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.max_pool1d(t, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.max_pool2d)
def max_pool2d(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False
):
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor max_pool2d.")
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.max_pool2d(t, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.max_pool3d)
def max_pool3d(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False
):
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor max_pool3d.")
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.max_pool3d(t, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool1d)
def adaptive_avg_pool1d(input: NestedTensor, output_size) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_avg_pool1d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool2d)
def adaptive_avg_pool2d(input: NestedTensor, output_size) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_avg_pool2d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool3d)
def adaptive_avg_pool3d(input: NestedTensor, output_size) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_avg_pool3d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool1d)
def adaptive_max_pool1d(input: NestedTensor, output_size, return_indices: bool = False):
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor adaptive_max_pool1d.")
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_max_pool1d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool2d)
def adaptive_max_pool2d(input: NestedTensor, output_size, return_indices: bool = False):
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor adaptive_max_pool2d.")
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_max_pool2d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool3d)
def adaptive_max_pool3d(input: NestedTensor, output_size, return_indices: bool = False):
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor adaptive_max_pool3d.")
    from .nested_tensor import NestedTensor

    return NestedTensor(F.adaptive_max_pool3d(t, output_size) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.lp_pool1d)
def lp_pool1d(input: NestedTensor, norm_type: float, kernel_size, stride=None, ceil_mode: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.lp_pool1d(t, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.lp_pool2d)
def lp_pool2d(input: NestedTensor, norm_type: float, kernel_size, stride=None, ceil_mode: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.lp_pool2d(t, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.lp_pool3d)
def lp_pool3d(input: NestedTensor, norm_type: float, kernel_size, stride=None, ceil_mode: bool = False) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.lp_pool3d(t, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.pixel_shuffle)
def pixel_shuffle(input: NestedTensor, upscale_factor: int) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.pixel_shuffle(t, upscale_factor) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.pixel_unshuffle)
def pixel_unshuffle(input: NestedTensor, downscale_factor: int) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.pixel_unshuffle(t, downscale_factor) for t in input._storage)


@NestedTensorFuncRegistry.implement(F.max_unpool1d)
def max_unpool1d(
    input: NestedTensor, indices: NestedTensor | Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    from .nested_tensor import NestedTensor

    if isinstance(indices, NestedTensor):
        return NestedTensor(
            F.max_unpool1d(t, idx, kernel_size, stride=stride, padding=padding, output_size=output_size)
            for t, idx in zip(input._storage, indices._storage)
        )
    return NestedTensor(
        F.max_unpool1d(t, indices, kernel_size, stride=stride, padding=padding, output_size=output_size)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.max_unpool2d)
def max_unpool2d(
    input: NestedTensor, indices: NestedTensor | Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    from .nested_tensor import NestedTensor

    if isinstance(indices, NestedTensor):
        return NestedTensor(
            F.max_unpool2d(t, idx, kernel_size, stride=stride, padding=padding, output_size=output_size)
            for t, idx in zip(input._storage, indices._storage)
        )
    return NestedTensor(
        F.max_unpool2d(t, indices, kernel_size, stride=stride, padding=padding, output_size=output_size)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.max_unpool3d)
def max_unpool3d(
    input: NestedTensor, indices: NestedTensor | Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    from .nested_tensor import NestedTensor

    if isinstance(indices, NestedTensor):
        return NestedTensor(
            F.max_unpool3d(t, idx, kernel_size, stride=stride, padding=padding, output_size=output_size)
            for t, idx in zip(input._storage, indices._storage)
        )
    return NestedTensor(
        F.max_unpool3d(t, indices, kernel_size, stride=stride, padding=padding, output_size=output_size)
        for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.max_pool1d_with_indices)
def max_pool1d_with_indices(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=True
):
    from .nested_tensor import NestedTensor

    outputs = []
    indices = []
    for t in input._storage:
        out, idx = F.max_pool1d(
            t,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
        outputs.append(out)
        indices.append(idx)
    return NestedTensor(outputs, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(F.max_pool2d_with_indices)
def max_pool2d_with_indices(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=True
):
    from .nested_tensor import NestedTensor

    outputs = []
    indices = []
    for t in input._storage:
        out, idx = F.max_pool2d(
            t,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
        outputs.append(out)
        indices.append(idx)
    return NestedTensor(outputs, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(F.max_pool3d_with_indices)
def max_pool3d_with_indices(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=True
):
    from .nested_tensor import NestedTensor

    outputs = []
    indices = []
    for t in input._storage:
        out, idx = F.max_pool3d(
            t,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
        outputs.append(out)
        indices.append(idx)
    return NestedTensor(outputs, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool1d_with_indices)
def adaptive_max_pool1d_with_indices(input: NestedTensor, output_size, return_indices: bool = True):
    from .nested_tensor import NestedTensor

    outputs = []
    indices = []
    for t in input._storage:
        out, idx = F.adaptive_max_pool1d(t, output_size, return_indices=True)
        outputs.append(out)
        indices.append(idx)
    return NestedTensor(outputs, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool2d_with_indices)
def adaptive_max_pool2d_with_indices(input: NestedTensor, output_size, return_indices: bool = True):
    from .nested_tensor import NestedTensor

    outputs = []
    indices = []
    for t in input._storage:
        out, idx = F.adaptive_max_pool2d(t, output_size, return_indices=True)
        outputs.append(out)
        indices.append(idx)
    return NestedTensor(outputs, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool3d_with_indices)
def adaptive_max_pool3d_with_indices(input: NestedTensor, output_size, return_indices: bool = True):
    from .nested_tensor import NestedTensor

    outputs = []
    indices = []
    for t in input._storage:
        out, idx = F.adaptive_max_pool3d(t, output_size, return_indices=True)
        outputs.append(out)
        indices.append(idx)
    return NestedTensor(outputs, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(F.fractional_max_pool2d_with_indices)
def fractional_max_pool2d_with_indices(
    input: NestedTensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = True,
    _random_samples=None,
):
    from .nested_tensor import NestedTensor

    outputs = []
    indices = []
    for t in input._storage:
        out, idx = F.fractional_max_pool2d(
            t,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=True,
            _random_samples=_random_samples,
        )
        outputs.append(out)
        indices.append(idx)
    return NestedTensor(outputs, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(F.fractional_max_pool3d_with_indices)
def fractional_max_pool3d_with_indices(
    input: NestedTensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = True,
    _random_samples=None,
):
    from .nested_tensor import NestedTensor

    outputs = []
    indices = []
    for t in input._storage:
        out, idx = F.fractional_max_pool3d(
            t,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=True,
            _random_samples=_random_samples,
        )
        outputs.append(out)
        indices.append(idx)
    return NestedTensor(outputs, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(F.unfold)
def unfold(input: NestedTensor, kernel_size, dilation=1, padding=0, stride=1) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.unfold(t, kernel_size, dilation=dilation, padding=padding, stride=stride) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.fold)
def fold(input: NestedTensor, output_size, kernel_size, dilation=1, padding=0, stride=1) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.fold(t, output_size, kernel_size, dilation=dilation, padding=padding, stride=stride) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.multi_head_attention_forward)
def multi_head_attention_forward(
    query: NestedTensor,
    key: NestedTensor | Tensor,
    value: NestedTensor | Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor | None,
    bias_k: Tensor | None,
    bias_v: Tensor | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Tensor | None = None,
    need_weights: bool = True,
    attn_mask: Tensor | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Tensor | None = None,
    k_proj_weight: Tensor | None = None,
    v_proj_weight: Tensor | None = None,
    static_k: Tensor | None = None,
    static_v: Tensor | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
):
    from .nested_tensor import NestedTensor

    if not isinstance(query, NestedTensor):
        raise TypeError("query must be a NestedTensor")
    key_nt = key if isinstance(key, NestedTensor) else None
    value_nt = value if isinstance(value, NestedTensor) else None
    key_padding_mask = key_padding_mask if key_padding_mask is not None else ~query.mask

    q_t = query.tensor
    k_t = key_nt.tensor if key_nt is not None else key
    v_t = value_nt.tensor if value_nt is not None else value

    # multi_head_attention_forward expects (L, N, E)
    q_t = q_t.transpose(0, 1)
    k_t = k_t.transpose(0, 1) if k_t.dim() == 3 else k_t
    v_t = v_t.transpose(0, 1) if v_t.dim() == 3 else v_t
    kpm = key_padding_mask

    attn_output, attn_weights = F.multi_head_attention_forward(  # type: ignore[call-arg]
        q_t,
        k_t,
        v_t,
        embed_dim_to_check,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout_p,
        out_proj_weight,
        out_proj_bias,
        training=training,
        key_padding_mask=kpm,
        need_weights=need_weights,
        attn_mask=attn_mask,
        use_separate_proj_weight=use_separate_proj_weight,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        static_k=static_k,
        static_v=static_v,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(0, 1)
    nt_output = NestedTensor.from_tensor_mask(attn_output, query.mask, **query._state)
    if need_weights:
        return nt_output, attn_weights
    return nt_output


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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input1_concat = input1.concat if isinstance(input1, NestedTensor) else input1[target.mask.bool()]
    input2_concat = input2.concat if isinstance(input2, NestedTensor) else input2[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input1.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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
    weight: Tensor | None = None,
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
    return F.mse_loss(
        input_concat, target_concat, size_average=size_average, reduce=reduce, reduction=reduction, weight=weight
    )


@NestedTensorFuncRegistry.implement(F.l1_loss)
def l1_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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

    input1_concat = input1.concat if isinstance(input1, NestedTensor) else input1[target.mask.bool()]
    input2_concat = input2.concat if isinstance(input2, NestedTensor) else input2[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input1.mask.bool()]
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
    weight: Tensor | None = None,
    reduction: str = "mean",
    delta: float = 1.0,
) -> Tensor:
    from .nested_tensor import NestedTensor

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
    return F.huber_loss(input_concat, target_concat, reduction=reduction, delta=delta, weight=weight)


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

    input_concat = input.concat if isinstance(input, NestedTensor) else input[target.mask.bool()]
    target_concat = target.concat if isinstance(target, NestedTensor) else target[input.mask.bool()]
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
    antialias: bool = False,
) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.interpolate(t, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias=antialias)
        for t in input._storage
    )


if hasattr(F, "grouped_mm"):

    @NestedTensorFuncRegistry.implement(F.grouped_mm)
    def grouped_mm(
        mat_a: NestedTensor,
        mat_b: Tensor | NestedTensor,
        *,
        offs: Tensor | None = None,
        bias: Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> NestedTensor:
        from .nested_tensor import NestedTensor

        if isinstance(mat_b, NestedTensor):
            outputs = [
                F.grouped_mm(a, b, offs=offs, bias=bias, out_dtype=out_dtype)
                for a, b in zip(mat_a._storage, mat_b._storage)
            ]
        else:
            outputs = [F.grouped_mm(a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype) for a in mat_a._storage]
        return NestedTensor(outputs, **mat_a._state)


if hasattr(F, "scaled_mm"):

    @NestedTensorFuncRegistry.implement(F.scaled_mm)
    def scaled_mm(
        mat_a: NestedTensor,
        mat_b: Tensor | NestedTensor,
        scale_a,
        scale_recipe_a,
        scale_b,
        scale_recipe_b,
        swizzle_a=None,
        swizzle_b=None,
        bias: Tensor | None = None,
        output_dtype: torch.dtype | None = torch.bfloat16,
        contraction_dim=(),
        use_fast_accum: bool = False,
    ) -> NestedTensor:
        from .nested_tensor import NestedTensor

        if isinstance(mat_b, NestedTensor):
            outputs = [
                F.scaled_mm(
                    a,
                    b,
                    scale_a,
                    scale_recipe_a,
                    scale_b,
                    scale_recipe_b,
                    swizzle_a=swizzle_a,
                    swizzle_b=swizzle_b,
                    bias=bias,
                    output_dtype=output_dtype,
                    contraction_dim=contraction_dim,
                    use_fast_accum=use_fast_accum,
                )
                for a, b in zip(mat_a._storage, mat_b._storage)
            ]
        else:
            outputs = [
                F.scaled_mm(
                    a,
                    mat_b,
                    scale_a,
                    scale_recipe_a,
                    scale_b,
                    scale_recipe_b,
                    swizzle_a=swizzle_a,
                    swizzle_b=swizzle_b,
                    bias=bias,
                    output_dtype=output_dtype,
                    contraction_dim=contraction_dim,
                    use_fast_accum=use_fast_accum,
                )
                for a in mat_a._storage
            ]
        return NestedTensor(outputs, **mat_a._state)


if hasattr(F, "scaled_grouped_mm"):

    @NestedTensorFuncRegistry.implement(F.scaled_grouped_mm)
    def scaled_grouped_mm(
        mat_a: NestedTensor,
        mat_b: Tensor | NestedTensor,
        scale_a,
        scale_recipe_a,
        scale_b,
        scale_recipe_b,
        swizzle_a=None,
        swizzle_b=None,
        bias: Tensor | None = None,
        offs: Tensor | None = None,
        output_dtype: torch.dtype | None = torch.bfloat16,
        contraction_dim=(),
        use_fast_accum: bool = False,
    ) -> NestedTensor:
        from .nested_tensor import NestedTensor

        if isinstance(mat_b, NestedTensor):
            outputs = [
                F.scaled_grouped_mm(
                    a,
                    b,
                    scale_a,
                    scale_recipe_a,
                    scale_b,
                    scale_recipe_b,
                    swizzle_a=swizzle_a,
                    swizzle_b=swizzle_b,
                    bias=bias,
                    offs=offs,
                    output_dtype=output_dtype,
                    contraction_dim=contraction_dim,
                    use_fast_accum=use_fast_accum,
                )
                for a, b in zip(mat_a._storage, mat_b._storage)
            ]
        else:
            outputs = [
                F.scaled_grouped_mm(
                    a,
                    mat_b,
                    scale_a,
                    scale_recipe_a,
                    scale_b,
                    scale_recipe_b,
                    swizzle_a=swizzle_a,
                    swizzle_b=swizzle_b,
                    bias=bias,
                    offs=offs,
                    output_dtype=output_dtype,
                    contraction_dim=contraction_dim,
                    use_fast_accum=use_fast_accum,
                )
                for a in mat_a._storage
            ]
        return NestedTensor(outputs, **mat_a._state)


@NestedTensorFuncRegistry.implement(F.pad)
def pad(input: NestedTensor, pad: Tuple[int, ...], mode: str = "constant", value: float = 0.0) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(F.pad(t, pad, mode, value) for t in input._storage)
