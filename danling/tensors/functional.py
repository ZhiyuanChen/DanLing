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
from .ops import (
    _concat_apply,
    _concat_apply_same_shape,
    _concat_dim_for_tensor_dim,
    _map_storage,
    _map_storage_pair,
    _translate_non_batch_dim,
)

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor


def concat_tensors(*values: NestedTensor | Tensor) -> tuple[Tensor, ...]:
    r"""
    Return tensors for each input, expanding NestedTensor values to their concatenated representation.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> a, b = concat_tensors(nt, nt)
        >>> torch.equal(a, nt.concat) and torch.equal(b, nt.concat)
        True
    """
    from .nested_tensor import NestedTensor

    mask: Tensor | None = None
    for v in values:
        if isinstance(v, NestedTensor):
            mask = v.mask if not v.mask_value else ~v.mask
            break

    if mask is None:
        return tuple(values)
    return tuple(v.concat if isinstance(v, NestedTensor) else v[mask] for v in values)


def _apply(input: NestedTensor, op: Callable, *args, **kwargs) -> NestedTensor:
    r"""
    Applies an operator to each tensor in a NestedTensor.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import _apply
        >>> nt = NestedTensor(torch.tensor([1.0, -2.0]), torch.tensor([3.0, -4.0, 5.0]))
        >>> out = _apply(nt, torch.abs)
        >>> ref = torch.abs(nt.tensor)
        >>> torch.equal(out, ref)
        True
    """
    return _map_storage(input, lambda t: op(t, *args, **kwargs))


def _apply_pair(input: NestedTensor, other: NestedTensor | Tensor, op: Callable, *args, **kwargs):
    r"""
    Applies a binary operator to a NestedTensor and another operand.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import _apply_pair
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> other = torch.tensor([1.0, 1.0])
        >>> out = _apply_pair(nt, other, torch.add)
        >>> ref = torch.add(nt.tensor, other)
        >>> torch.equal(out, ref)
        True
    """
    return _map_storage_pair(input, other, lambda x, y: op(x, y, *args, **kwargs))


def _apply_concat(input: NestedTensor, op: Callable, *args, **kwargs) -> NestedTensor:
    r"""
    Applies an operator to a NestedTensor and return a NestedTensor result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import _apply_concat
        >>> nt = NestedTensor(torch.tensor([1.0, -2.0]), torch.tensor([3.0, -4.0, 5.0]))
        >>> out = _apply_concat(nt, torch.abs)
        >>> ref = torch.abs(nt.tensor)
        >>> torch.equal(out, ref)
        True
    """
    return _concat_apply_same_shape(input, lambda t: op(t, *args, **kwargs))


def _apply_elementwise(input: NestedTensor, op: Callable, *args, inplace: bool | None = None, **kwargs) -> NestedTensor:
    r"""
    Applies an elementwise operator to a NestedTensor, optionally in-place.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import _apply_elementwise
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0, -5.0]))
        >>> out = _apply_elementwise(nt, F.relu)
        >>> ref = F.relu(nt.tensor)
        >>> torch.equal(out, ref)
        True
    """
    if inplace:
        return _apply(input, op, *args, inplace=inplace, **kwargs)
    return _apply_concat(input, op, *args, **kwargs)


def _can_concat_normalize(input: NestedTensor, normalized_shape: Tuple[int, ...]) -> bool:
    r"""
    Return whether a normalized_shape is compatible with all tensors in a NestedTensor.

    Examples:
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import _can_concat_normalize
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> _can_concat_normalize(nt, (2,))
        True
    """
    if not input._storage:
        return True
    if not normalized_shape:
        return True
    for t in input._storage:
        if t.dim() < len(normalized_shape):
            return False
        if tuple(t.shape[-len(normalized_shape) :]) != normalized_shape:  # noqa: E203
            return False
    return True


# Embeddings


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
    r"""
    Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.
    See also [torch.nn.functional.embedding][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> weight = torch.arange(6.0).reshape(3, 2)
        >>> nt = NestedTensor(torch.tensor([0, 1]), torch.tensor([1, 2, 0]))
        >>> out = F.embedding(nt, weight)
        >>> ref = F.embedding(nt.tensor, weight)
        >>> torch.allclose(out, ref)
        True
    """
    return _concat_apply(
        input,
        lambda t: F.embedding(
            t,
            weight,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        ),
        lambda shape: torch.Size([*shape, weight.shape[1]]),
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
    r"""
    Compute sums, means or maxes of `bags` of embeddings.
    See also [torch.nn.functional.embedding_bag][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> weight = torch.arange(6.0).reshape(3, 2)
        >>> offsets = torch.tensor([0])
        >>> a = torch.tensor([0, 1])
        >>> b = torch.tensor([1, 2, 0])
        >>> nt = NestedTensor(a, b)
        >>> out = F.embedding_bag(nt, weight, offsets=offsets)
        >>> ref = NestedTensor(
        ...     F.embedding_bag(a, weight, offsets=offsets),
        ...     F.embedding_bag(b, weight, offsets=offsets),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _map_storage(
        input,
        lambda t: F.embedding_bag(
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
        ),
    )


@NestedTensorFuncRegistry.implement(F.linear)
def linear(input: NestedTensor, weight: Tensor, bias: Tensor | None = None) -> NestedTensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    See also [torch.nn.functional.linear][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> bias = torch.tensor([0.5, -0.5])
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.linear(nt, weight, bias)
        >>> ref = F.linear(nt.tensor, weight, bias)
        >>> torch.allclose(out, ref)
        True
    """
    if all(t.dim() == 1 for t in input._storage):
        return F.linear(torch.stack(input._storage, dim=0), weight=weight, bias=bias)
    return _concat_apply(
        input,
        lambda t: F.linear(t, weight, bias),
        lambda shape: torch.Size([*shape[:-1], weight.shape[0]]),
    )


# Normalization


@NestedTensorFuncRegistry.implement(F.normalize)
def normalize(
    input: NestedTensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
    out=None,  # noqa: ANN001
) -> NestedTensor:
    r"""
    Performs L_p normalization of inputs over the specified dimension. See also [torch.nn.functional.normalize][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.normalize(nt, dim=-1)
        >>> ref = F.normalize(nt.tensor, dim=-1)
        >>> torch.allclose(out, ref)
        True
    """
    target_dim = _translate_non_batch_dim(input, dim, name="normalize")
    concat_dim = _concat_dim_for_tensor_dim(input, target_dim)
    if concat_dim is None or out is not None:
        return _apply(input, F.normalize, p=p, dim=target_dim, eps=eps, out=out)
    return _apply_concat(input, F.normalize, p=p, dim=concat_dim, eps=eps, out=out)


# Activations


@NestedTensorFuncRegistry.implement(F.elu)
def elu(input: NestedTensor, alpha: float = 1.0, inplace: bool = False) -> NestedTensor:
    r"""
    Applies the Exponential Linear Unit (ELU) function element-wise.
    See also [torch.nn.functional.elu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.elu(nt), F.elu(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.elu, alpha=alpha, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.celu)
def celu(input: NestedTensor, alpha: float = 1.0, inplace: bool = False) -> NestedTensor:
    r"""
    Applies element-wise, :math:`\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))`.
    See also [torch.nn.functional.celu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.celu(nt), F.celu(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.celu, alpha=alpha, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.selu)
def selu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    r"""
    Applies element-wise, :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`, with
    :math:`\alpha=1.6732632423543772848170429916717` and :math:`scale=1.0507009873554804934193349852946`.
    See also [torch.nn.functional.selu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.selu(nt), F.selu(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.selu, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.gelu)
def gelu(input: NestedTensor, approximate: str = "none") -> NestedTensor:
    r"""
    When the approximate argument is 'none', it applies element-wise the function :math:`\text{GELU}(x) = x * \Phi(x)`
    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.
    See also [torch.nn.functional.gelu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.gelu(nt), F.gelu(nt.tensor))
        True
    """
    return _apply_concat(input, F.gelu, approximate=approximate)


@NestedTensorFuncRegistry.implement(F.relu)
def relu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    r"""
    Applies the rectified linear unit function element-wise. See :class:`~torch.nn.ReLU` for more details.
    See also [torch.nn.functional.relu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.relu(nt), F.relu(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.relu, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.leaky_relu)
def leaky_relu(input: NestedTensor, negative_slope: float = 1e-2, inplace: bool = False) -> NestedTensor:
    r"""
    Applies element-wise, :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)` See
    :class:`~torch.nn.LeakyReLU` for more details.
    See also [torch.nn.functional.leaky_relu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.leaky_relu(nt), F.leaky_relu(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.leaky_relu, negative_slope, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.rrelu)
def rrelu(
    input: NestedTensor, lower: float = 1.0 / 8, upper: float = 1.0 / 3, training: bool = False, inplace: bool = False
) -> NestedTensor:
    r"""
    Randomized leaky ReLU.
    See also [torch.nn.functional.rrelu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.rrelu(nt, training=False), F.rrelu(nt.tensor, training=False))
        True
    """
    return _apply_elementwise(input, F.rrelu, lower, upper, training, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.glu)
def glu(input: NestedTensor, dim: int = -1) -> NestedTensor:
    r"""
    The gated linear unit. Computes: .. math :: \text{GLU}(a, b) = a \otimes \sigma(b) where `input` is split in half
    along `dim` to form `a` and `b`, :math:`\sigma` is the sigmoid function and :math:`\otimes` is the element-wise
    product between matrices.
    See also [torch.nn.functional.glu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([[5.0, 6.0, 7.0, 8.0]]))
        >>> torch.allclose(F.glu(nt, dim=-1), F.glu(nt.tensor, dim=-1))
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="glu")
    return _apply(input, F.glu, dim=dim_adj)


@NestedTensorFuncRegistry.implement(F.silu)
def silu(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    See also [torch.nn.functional.silu][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.silu(nt), F.silu(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.silu, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.hardtanh)
def hardtanh(input: NestedTensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> NestedTensor:
    r"""
    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more details.
    See also [torch.nn.functional.hardtanh][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.hardtanh(nt), F.hardtanh(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.hardtanh, min_val=min_val, max_val=max_val, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.hardsigmoid)
def hardsigmoid(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    r"""
    Applies the Hardsigmoid function element-wise.
    See also [torch.nn.functional.hardsigmoid][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.hardsigmoid(nt), F.hardsigmoid(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.hardsigmoid, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.mish)
def mish(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    r"""
    Applies the Mish function, element-wise.
    See also [torch.nn.functional.mish][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.mish(nt), F.mish(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.mish, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.relu6)
def relu6(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    r"""
    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.
    See also [torch.nn.functional.relu6][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.relu6(nt), F.relu6(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.relu6, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.hardswish)
def hardswish(input: NestedTensor, inplace: bool = False) -> NestedTensor:
    r"""
    Applies hardswish function, element-wise.
    See also [torch.nn.functional.hardswish][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.hardswish(nt), F.hardswish(nt.tensor))
        True
    """
    return _apply_elementwise(input, F.hardswish, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.logsigmoid)
def logsigmoid(input: NestedTensor) -> NestedTensor:
    r"""
    logsigmoid(input) -> Tensor Applies element-wise :math:`\text{LogSigmoid}(x_i) = \log \left(\frac{1}{1 +
    \exp(-x_i)}\right)` See :class:`~torch.nn.LogSigmoid` for more details.
    See also [torch.nn.functional.logsigmoid][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.logsigmoid(nt), F.logsigmoid(nt.tensor))
        True
    """
    return _apply_concat(input, F.logsigmoid)


@NestedTensorFuncRegistry.implement(F.sigmoid)
def sigmoid(input: NestedTensor) -> NestedTensor:
    r"""
    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}` See :class:`~torch.nn.Sigmoid`
    for more details.
    See also [torch.nn.functional.sigmoid][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.sigmoid(nt), F.sigmoid(nt.tensor))
        True
    """
    return _apply_concat(input, F.sigmoid)


@NestedTensorFuncRegistry.implement(F.softplus)
def softplus(input: NestedTensor, beta: float = 1.0, threshold: float = 20.0) -> NestedTensor:
    r"""
    Applies element-wise, the function :math:`\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))`.
    See also [torch.nn.functional.softplus][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.softplus(nt), F.softplus(nt.tensor))
        True
    """
    return _apply_concat(input, F.softplus, beta=beta, threshold=threshold)


@NestedTensorFuncRegistry.implement(F.hardshrink)
def hardshrink(input: NestedTensor, lambd: float = 0.5) -> NestedTensor:
    r"""
    Applies the hard shrinkage function element-wise See :class:`~torch.nn.Hardshrink` for more details.
    See also [torch.nn.functional.hardshrink][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.hardshrink(nt), F.hardshrink(nt.tensor))
        True
    """
    return _apply_concat(input, F.hardshrink, lambd=lambd)


@NestedTensorFuncRegistry.implement(F.softsign)
def softsign(input: NestedTensor) -> NestedTensor:
    r"""
    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}` See :class:`~torch.nn.Softsign`
    for more details.
    See also [torch.nn.functional.softsign][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.softsign(nt), F.softsign(nt.tensor))
        True
    """
    return _apply_concat(input, F.softsign)


@NestedTensorFuncRegistry.implement(F.tanh)
def tanh(input: NestedTensor) -> NestedTensor:
    r"""
    Applies element-wise, :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}` See
    :class:`~torch.nn.Tanh` for more details.
    See also [torch.nn.functional.tanh][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.tanh(nt), F.tanh(nt.tensor))
        True
    """
    return _apply_concat(input, F.tanh)


@NestedTensorFuncRegistry.implement(F.tanhshrink)
def tanhshrink(input: NestedTensor) -> NestedTensor:
    r"""
    Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)` See :class:`~torch.nn.Tanhshrink` for more
    details.
    See also [torch.nn.functional.tanhshrink][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.tanhshrink(nt), F.tanhshrink(nt.tensor))
        True
    """
    return _apply_concat(input, F.tanhshrink)


@NestedTensorFuncRegistry.implement(F.softmax)
def softmax(input: NestedTensor, dim: int, _stacklevel: int = 3, dtype=None):
    r"""
    Applies a softmax function over the specified dimension. See also [torch.nn.functional.softmax][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.softmax(nt, dim=-1)
        >>> ref = F.softmax(nt.tensor, dim=-1)
        >>> torch.allclose(out, ref)
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="softmax")
    concat_dim = _concat_dim_for_tensor_dim(input, dim_adj)
    if concat_dim is None:
        return _apply(input, F.softmax, dim_adj, _stacklevel=_stacklevel, dtype=dtype)
    return _apply_concat(input, F.softmax, concat_dim, _stacklevel=_stacklevel, dtype=dtype)


@NestedTensorFuncRegistry.implement(F.log_softmax)
def log_softmax(input: NestedTensor, dim: int, _stacklevel: int = 3, dtype=None):
    r"""
    Applies log-softmax over the specified dimension. See also [torch.nn.functional.log_softmax][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.log_softmax(nt, dim=-1)
        >>> ref = F.log_softmax(nt.tensor, dim=-1)
        >>> torch.allclose(out, ref)
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="log_softmax")
    concat_dim = _concat_dim_for_tensor_dim(input, dim_adj)
    if concat_dim is None:
        return _apply(input, F.log_softmax, dim_adj, _stacklevel=_stacklevel, dtype=dtype)
    return _apply_concat(input, F.log_softmax, concat_dim, _stacklevel=_stacklevel, dtype=dtype)


@NestedTensorFuncRegistry.implement(F.softmin)
def softmin(input: NestedTensor, dim: int, _stacklevel: int = 3, dtype=None):
    r"""
    Applies a softmin function.
    See also [torch.nn.functional.softmin][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> torch.allclose(F.softmin(nt, dim=-1), F.softmin(nt.tensor, dim=-1))
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="softmin")
    concat_dim = _concat_dim_for_tensor_dim(input, dim_adj)
    if concat_dim is None:
        return _apply(input, F.softmin, dim_adj, _stacklevel=_stacklevel, dtype=dtype)
    return _apply_concat(input, F.softmin, concat_dim, _stacklevel=_stacklevel, dtype=dtype)


@NestedTensorFuncRegistry.implement(F.gumbel_softmax)
def gumbel_softmax(input: NestedTensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-10, dim: int = -1):
    r"""
    Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretize.
    See also [torch.nn.functional.gumbel_softmax][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5]))
        >>> out = F.gumbel_softmax(nt, tau=1.0, hard=False, dim=-1)
        >>> torch.allclose(out[0].sum(), torch.tensor(1.0)) and torch.allclose(out[1].sum(), torch.tensor(1.0))
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="gumbel_softmax")
    return _apply(input, F.gumbel_softmax, tau=tau, hard=hard, eps=eps, dim=dim_adj)


@NestedTensorFuncRegistry.implement(F.softshrink)
def softshrink(input: NestedTensor, lambd: float = 0.5) -> NestedTensor:
    r"""
    Applies the soft shrinkage function elementwise See :class:`~torch.nn.Softshrink` for more details.
    See also [torch.nn.functional.softshrink][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.softshrink(nt), F.softshrink(nt.tensor))
        True
    """
    return _apply_concat(input, F.softshrink, lambd)


@NestedTensorFuncRegistry.implement(F.threshold)
def threshold(input: NestedTensor, threshold: float, value: float, inplace: bool = False) -> NestedTensor:
    r"""
    Applies a threshold to each element of the input Tensor.
    See also [torch.nn.functional.threshold][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([-3.0, 4.0]))
        >>> torch.allclose(F.threshold(nt, 0.0, 0.0), F.threshold(nt.tensor, 0.0, 0.0))
        True
    """
    return _apply_elementwise(input, F.threshold, threshold, value, inplace=inplace)


# Geometric


@NestedTensorFuncRegistry.implement(F.grid_sample)
def grid_sample(
    input: NestedTensor,
    grid: NestedTensor | Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool | None = None,
) -> NestedTensor:
    r"""
    Compute grid sample.
    See also [torch.nn.functional.grid_sample][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> a = torch.arange(4.0).view(1, 1, 2, 2)
        >>> b = torch.arange(8.0).view(2, 1, 2, 2)
        >>> grid_a = F.affine_grid(torch.eye(2, 3).unsqueeze(0), a.size(), align_corners=False)
        >>> theta_b = torch.eye(2, 3).unsqueeze(0).repeat(2, 1, 1)
        >>> grid_b = F.affine_grid(theta_b, b.size(), align_corners=False)
        >>> nt = NestedTensor(a, b)
        >>> nt_grid = NestedTensor(grid_a, grid_b)
        >>> out = F.grid_sample(nt, nt_grid, align_corners=False)
        >>> ref = NestedTensor(
        ...     F.grid_sample(a, grid_a, align_corners=False),
        ...     F.grid_sample(b, grid_b, align_corners=False),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _apply_pair(
        input,
        grid,
        F.grid_sample,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


@NestedTensorFuncRegistry.implement(F.affine_grid)
def affine_grid(theta: NestedTensor, size: Tensor, align_corners: bool | None = None) -> NestedTensor:
    r"""
    Generate 2D or 3D flow field (sampling grid), given a batch of affine matrices :attr:`theta`.
    See also [torch.nn.functional.affine_grid][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> theta = torch.eye(2, 3).unsqueeze(0)
        >>> nt = NestedTensor(theta, theta)
        >>> out = F.affine_grid(nt, torch.Size((1, 1, 2, 2)), align_corners=False)
        >>> ref = NestedTensor(
        ...     F.affine_grid(theta, torch.Size((1, 1, 2, 2)), align_corners=False),
        ...     F.affine_grid(theta, torch.Size((1, 1, 2, 2)), align_corners=False),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(theta, F.affine_grid, size, align_corners=align_corners)


# Convolutions


@NestedTensorFuncRegistry.implement(F.conv1d)
def conv1d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    r"""
    Applies a 1D convolution over an input signal composed of several input planes.
    See also [torch.nn.functional.conv1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> weight = torch.ones(1, 1, 2)
        >>> out = F.conv1d(nt, weight)
        >>> ref = F.conv1d(nt.tensor, weight)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.conv1d,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@NestedTensorFuncRegistry.implement(F.conv2d)
def conv2d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    r"""
    Applies a 2D convolution over an input image composed of several input planes.
    See also [torch.nn.functional.conv2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(16.0).view(1, 4, 4), torch.arange(20.0).view(1, 5, 4))
        >>> weight = torch.ones(1, 1, 2, 2)
        >>> out = F.conv2d(nt, weight)
        >>> ref = F.conv2d(nt.tensor, weight)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.conv2d,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@NestedTensorFuncRegistry.implement(F.conv3d)
def conv3d(
    input: NestedTensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1
) -> NestedTensor:
    r"""
    Applies a 3D convolution over an input image composed of several input planes.
    See also [torch.nn.functional.conv3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(1, 2, 2, 2), torch.arange(12.0).view(1, 3, 2, 2))
        >>> weight = torch.ones(1, 1, 2, 2, 2)
        >>> out = F.conv3d(nt, weight)
        >>> ref = F.conv3d(nt.tensor, weight)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.conv3d,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
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
    r"""
    Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also
    called "deconvolution".
    See also [torch.nn.functional.conv_transpose1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> weight = torch.ones(1, 1, 2)
        >>> out = F.conv_transpose1d(nt, weight)
        >>> ref = F.conv_transpose1d(nt.tensor, weight)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.conv_transpose1d,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
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
    r"""
    Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also
    called "deconvolution".
    See also [torch.nn.functional.conv_transpose2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(16.0).view(1, 4, 4), torch.arange(20.0).view(1, 5, 4))
        >>> weight = torch.ones(1, 1, 2, 2)
        >>> out = F.conv_transpose2d(nt, weight)
        >>> ref = F.conv_transpose2d(nt.tensor, weight)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.conv_transpose2d,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
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
    r"""
    Applies a 3D transposed convolution operator over an input image composed of several input planes, sometimes also
    called "deconvolution" This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
    See also [torch.nn.functional.conv_transpose3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(1, 2, 2, 2), torch.arange(12.0).view(1, 3, 2, 2))
        >>> weight = torch.ones(1, 1, 2, 2, 2)
        >>> out = F.conv_transpose3d(nt, weight)
        >>> ref = F.conv_transpose3d(nt.tensor, weight)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.conv_transpose3d,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


# Pooling


@NestedTensorFuncRegistry.implement(F.avg_pool1d)
def avg_pool1d(
    input: NestedTensor, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True
) -> NestedTensor:
    r"""
    Applies a 1D average pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.avg_pool1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> out = F.avg_pool1d(nt, kernel_size=2)
        >>> ref = F.avg_pool1d(nt.tensor, kernel_size=2)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.avg_pool1d,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
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
    r"""
    Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size :math:`sH \times sW` steps. The
    number of output features is equal to the number of input planes.
    See also [torch.nn.functional.avg_pool2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(16.0).view(1, 4, 4), torch.arange(20.0).view(1, 5, 4))
        >>> out = F.avg_pool2d(nt, kernel_size=2)
        >>> ref = F.avg_pool2d(nt.tensor, kernel_size=2)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.avg_pool2d,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
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
    r"""
    Applies 3D average-pooling operation in :math:`kT \times kH \times kW` regions by step size :math:`sT \times sH
    \times sW` steps. The number of output features is equal to :math:`\lfloor\frac{\text{input planes}}{sT}\rfloor`.
    See also [torch.nn.functional.avg_pool3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(1, 2, 2, 2), torch.arange(12.0).view(1, 3, 2, 2))
        >>> out = F.avg_pool3d(nt, kernel_size=2)
        >>> ref = F.avg_pool3d(nt.tensor, kernel_size=2)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(
        input,
        F.avg_pool3d,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@NestedTensorFuncRegistry.implement(F.max_pool1d)
def max_pool1d(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False
):
    r"""
    Applies a 1D max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.max_pool1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> out = F.max_pool1d(nt, kernel_size=2)
        >>> ref = F.max_pool1d(nt.tensor, kernel_size=2)
        >>> torch.equal(out, ref)
        True
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor max_pool1d.")
    return _apply(
        input,
        F.max_pool1d,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@NestedTensorFuncRegistry.implement(F.max_pool2d)
def max_pool2d(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False
):
    r"""
    Applies a 2D max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.max_pool2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(16.0).view(1, 4, 4), torch.arange(20.0).view(1, 5, 4))
        >>> out = F.max_pool2d(nt, kernel_size=2)
        >>> ref = F.max_pool2d(nt.tensor, kernel_size=2)
        >>> torch.equal(out, ref)
        True
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor max_pool2d.")
    return _apply(
        input,
        F.max_pool2d,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@NestedTensorFuncRegistry.implement(F.max_pool3d)
def max_pool3d(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False
):
    r"""
    Applies a 3D max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.max_pool3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(1, 2, 2, 2), torch.arange(12.0).view(1, 3, 2, 2))
        >>> out = F.max_pool3d(nt, kernel_size=2)
        >>> ref = F.max_pool3d(nt.tensor, kernel_size=2)
        >>> torch.equal(out, ref)
        True
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor max_pool3d.")
    return _apply(
        input,
        F.max_pool3d,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool1d)
def adaptive_avg_pool1d(input: NestedTensor, output_size) -> NestedTensor:
    r"""
    Applies a 1D adaptive average pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.adaptive_avg_pool1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> out = F.adaptive_avg_pool1d(nt, output_size=2)
        >>> ref = NestedTensor(
        ...     F.adaptive_avg_pool1d(nt[0], output_size=2),
        ...     F.adaptive_avg_pool1d(nt[1], output_size=2),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(input, F.adaptive_avg_pool1d, output_size)


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool2d)
def adaptive_avg_pool2d(input: NestedTensor, output_size) -> NestedTensor:
    r"""
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.adaptive_avg_pool2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(16.0).view(1, 4, 4), torch.arange(20.0).view(1, 5, 4))
        >>> out = F.adaptive_avg_pool2d(nt, output_size=(2, 2))
        >>> ref = NestedTensor(
        ...     F.adaptive_avg_pool2d(nt[0], output_size=(2, 2)),
        ...     F.adaptive_avg_pool2d(nt[1], output_size=(2, 2)),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(input, F.adaptive_avg_pool2d, output_size)


@NestedTensorFuncRegistry.implement(F.adaptive_avg_pool3d)
def adaptive_avg_pool3d(input: NestedTensor, output_size) -> NestedTensor:
    r"""
    Applies a 3D adaptive average pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.adaptive_avg_pool3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(1, 2, 2, 2), torch.arange(12.0).view(1, 3, 2, 2))
        >>> out = F.adaptive_avg_pool3d(nt, output_size=(1, 1, 1))
        >>> ref = NestedTensor(
        ...     F.adaptive_avg_pool3d(nt[0], output_size=(1, 1, 1)),
        ...     F.adaptive_avg_pool3d(nt[1], output_size=(1, 1, 1)),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(input, F.adaptive_avg_pool3d, output_size)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool1d)
def adaptive_max_pool1d(input: NestedTensor, output_size, return_indices: bool = False):
    r"""
    Applies a 1D adaptive max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.adaptive_max_pool1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> out = F.adaptive_max_pool1d(nt, output_size=2)
        >>> ref = NestedTensor(
        ...     F.adaptive_max_pool1d(nt[0], output_size=2),
        ...     F.adaptive_max_pool1d(nt[1], output_size=2),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor adaptive_max_pool1d.")
    return _apply(input, F.adaptive_max_pool1d, output_size)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool2d)
def adaptive_max_pool2d(input: NestedTensor, output_size, return_indices: bool = False):
    r"""
    Applies a 2D adaptive max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.adaptive_max_pool2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(16.0).view(1, 4, 4), torch.arange(20.0).view(1, 5, 4))
        >>> out = F.adaptive_max_pool2d(nt, output_size=(2, 2))
        >>> ref = NestedTensor(
        ...     F.adaptive_max_pool2d(nt[0], output_size=(2, 2)),
        ...     F.adaptive_max_pool2d(nt[1], output_size=(2, 2)),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor adaptive_max_pool2d.")
    return _apply(input, F.adaptive_max_pool2d, output_size)


@NestedTensorFuncRegistry.implement(F.adaptive_max_pool3d)
def adaptive_max_pool3d(input: NestedTensor, output_size, return_indices: bool = False):
    r"""
    Applies a 3D adaptive max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.adaptive_max_pool3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(1, 2, 2, 2), torch.arange(12.0).view(1, 3, 2, 2))
        >>> out = F.adaptive_max_pool3d(nt, output_size=(1, 1, 1))
        >>> ref = F.adaptive_max_pool3d(nt.tensor, output_size=(1, 1, 1))
        >>> torch.equal(out, ref)
        True
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported for NestedTensor adaptive_max_pool3d.")
    return _apply(input, F.adaptive_max_pool3d, output_size)


@NestedTensorFuncRegistry.implement(F.lp_pool1d)
def lp_pool1d(input: NestedTensor, norm_type: float, kernel_size, stride=None, ceil_mode: bool = False) -> NestedTensor:
    r"""
    Applies a 1D power-average pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.lp_pool1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> out = F.lp_pool1d(nt, norm_type=2.0, kernel_size=2)
        >>> ref = F.lp_pool1d(nt.tensor, norm_type=2.0, kernel_size=2)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(input, F.lp_pool1d, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)


@NestedTensorFuncRegistry.implement(F.lp_pool2d)
def lp_pool2d(input: NestedTensor, norm_type: float, kernel_size, stride=None, ceil_mode: bool = False) -> NestedTensor:
    r"""
    Applies a 2D power-average pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.lp_pool2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(16.0).view(1, 4, 4), torch.arange(20.0).view(1, 5, 4))
        >>> out = F.lp_pool2d(nt, norm_type=2.0, kernel_size=2)
        >>> ref = F.lp_pool2d(nt.tensor, norm_type=2.0, kernel_size=2)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(input, F.lp_pool2d, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)


@NestedTensorFuncRegistry.implement(F.lp_pool3d)
def lp_pool3d(input: NestedTensor, norm_type: float, kernel_size, stride=None, ceil_mode: bool = False) -> NestedTensor:
    r"""
    Applies a 3D power-average pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.lp_pool3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(1, 2, 2, 2), torch.arange(12.0).view(1, 3, 2, 2))
        >>> out = F.lp_pool3d(nt, norm_type=2.0, kernel_size=2)
        >>> ref = F.lp_pool3d(nt.tensor, norm_type=2.0, kernel_size=2)
        >>> torch.allclose(out, ref)
        True
    """
    return _apply(input, F.lp_pool3d, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)


@NestedTensorFuncRegistry.implement(F.pixel_shuffle)
def pixel_shuffle(input: NestedTensor, upscale_factor: int) -> NestedTensor:
    r"""
    Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a tensor of shape :math:`(*, C, H \times
    r, W \times r)`, where r is the :attr:`upscale_factor`.
    See also [torch.nn.functional.pixel_shuffle][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(4, 1, 1), torch.arange(8.0).view(8, 1, 1))
        >>> out = F.pixel_shuffle(nt, 2)
        >>> ref = F.pixel_shuffle(nt.tensor, 2)
        >>> torch.equal(out, ref)
        True
    """
    return _apply(input, F.pixel_shuffle, upscale_factor)


@NestedTensorFuncRegistry.implement(F.pixel_unshuffle)
def pixel_unshuffle(input: NestedTensor, downscale_factor: int) -> NestedTensor:
    r"""
    Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a tensor of shape :math:`(*, C, H
    \times r, W \times r)` to a tensor of shape :math:`(*, C \times r^2, H, W)`, where r is the
    :attr:`downscale_factor`.
    See also [torch.nn.functional.pixel_unshuffle][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 2, 2), torch.arange(8.0).view(2, 2, 2))
        >>> out = F.pixel_unshuffle(nt, 2)
        >>> ref = F.pixel_unshuffle(nt.tensor, 2)
        >>> torch.equal(out, ref)
        True
    """
    return _apply(input, F.pixel_unshuffle, downscale_factor)


@NestedTensorFuncRegistry.implement(F.max_unpool1d)
def max_unpool1d(
    input: NestedTensor, indices: NestedTensor | Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    r"""
    Compute a partial inverse of :class:`MaxPool1d`.
    See also [torch.nn.functional.max_unpool1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> pooled, indices = F.max_pool1d_with_indices(nt, kernel_size=2)
        >>> out = F.max_unpool1d(pooled, indices, kernel_size=2)
        >>> ref = NestedTensor(
        ...     F.max_unpool1d(pooled[0], indices[0], kernel_size=2),
        ...     F.max_unpool1d(pooled[1], indices[1], kernel_size=2),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    return _apply_pair(
        input,
        indices,
        F.max_unpool1d,
        kernel_size,
        stride=stride,
        padding=padding,
        output_size=output_size,
    )


@NestedTensorFuncRegistry.implement(F.max_unpool2d)
def max_unpool2d(
    input: NestedTensor, indices: NestedTensor | Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    r"""
    Compute a partial inverse of :class:`MaxPool2d`.
    See also [torch.nn.functional.max_unpool2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(16.0).view(1, 4, 4),
        ...     torch.arange(20.0).view(1, 5, 4),
        ... )
        >>> pooled, indices = F.max_pool2d_with_indices(nt, kernel_size=2)
        >>> out = F.max_unpool2d(pooled, indices, kernel_size=2)
        >>> ref = NestedTensor(
        ...     F.max_unpool2d(pooled[0], indices[0], kernel_size=2),
        ...     F.max_unpool2d(pooled[1], indices[1], kernel_size=2),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    return _apply_pair(
        input,
        indices,
        F.max_unpool2d,
        kernel_size,
        stride=stride,
        padding=padding,
        output_size=output_size,
    )


@NestedTensorFuncRegistry.implement(F.max_unpool3d)
def max_unpool3d(
    input: NestedTensor, indices: NestedTensor | Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    r"""
    Compute a partial inverse of :class:`MaxPool3d`.
    See also [torch.nn.functional.max_unpool3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(8.0).view(1, 2, 2, 2),
        ...     torch.arange(12.0).view(1, 3, 2, 2),
        ... )
        >>> pooled, indices = F.max_pool3d_with_indices(nt, kernel_size=2)
        >>> out = F.max_unpool3d(pooled, indices, kernel_size=2)
        >>> ref = NestedTensor(
        ...     F.max_unpool3d(pooled[0], indices[0], kernel_size=2),
        ...     F.max_unpool3d(pooled[1], indices[1], kernel_size=2),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    return _apply_pair(
        input,
        indices,
        F.max_unpool3d,
        kernel_size,
        stride=stride,
        padding=padding,
        output_size=output_size,
    )


@NestedTensorFuncRegistry.implement(F.max_pool1d_with_indices)
def max_pool1d_with_indices(
    input: NestedTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=True
):
    r"""
    max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a
    1D max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.max_pool1d_with_indices][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> out, idx = F.max_pool1d_with_indices(nt, kernel_size=2)
        >>> ref0, idx0 = F.max_pool1d_with_indices(nt[0], kernel_size=2)
        >>> ref1, idx1 = F.max_pool1d_with_indices(nt[1], kernel_size=2)
        >>> ref_out = NestedTensor(ref0, ref1)
        >>> ref_idx = NestedTensor(idx0, idx1)
        >>> torch.equal(out, ref_out) and torch.equal(idx, ref_idx)
        True
    """
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
    r"""
    max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a
    2D max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.max_pool2d_with_indices][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(16.0).view(1, 4, 4),
        ...     torch.arange(20.0).view(1, 5, 4),
        ... )
        >>> out, idx = F.max_pool2d_with_indices(nt, kernel_size=2)
        >>> ref0, idx0 = F.max_pool2d_with_indices(nt[0], kernel_size=2)
        >>> ref1, idx1 = F.max_pool2d_with_indices(nt[1], kernel_size=2)
        >>> ref_out = NestedTensor(ref0, ref1)
        >>> ref_idx = NestedTensor(idx0, idx1)
        >>> torch.equal(out, ref_out) and torch.equal(idx, ref_idx)
        True
    """
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
    r"""
    max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a
    3D max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.max_pool3d_with_indices][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(8.0).view(1, 2, 2, 2),
        ...     torch.arange(12.0).view(1, 3, 2, 2),
        ... )
        >>> out, idx = F.max_pool3d_with_indices(nt, kernel_size=2)
        >>> ref0, idx0 = F.max_pool3d_with_indices(nt[0], kernel_size=2)
        >>> ref1, idx1 = F.max_pool3d_with_indices(nt[1], kernel_size=2)
        >>> ref_out = NestedTensor(ref0, ref1)
        >>> ref_idx = NestedTensor(idx0, idx1)
        >>> torch.equal(out, ref_out) and torch.equal(idx, ref_idx)
        True
    """
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
    r"""
    adaptive_max_pool1d(input, output_size, return_indices=False) Applies a 1D adaptive max pooling over an input signal
    composed of several input planes.
    See also [torch.nn.functional.adaptive_max_pool1d_with_indices][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(6.0).view(1, 6))
        >>> out, idx = F.adaptive_max_pool1d_with_indices(nt, output_size=2)
        >>> ref0, idx0 = F.adaptive_max_pool1d_with_indices(nt[0], output_size=2)
        >>> ref1, idx1 = F.adaptive_max_pool1d_with_indices(nt[1], output_size=2)
        >>> ref_out = NestedTensor(ref0, ref1)
        >>> ref_idx = NestedTensor(idx0, idx1)
        >>> torch.equal(out, ref_out) and torch.equal(idx, ref_idx)
        True
    """
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
    r"""
    adaptive_max_pool2d(input, output_size, return_indices=False) Applies a 2D adaptive max pooling over an input signal
    composed of several input planes.
    See also [torch.nn.functional.adaptive_max_pool2d_with_indices][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(16.0).view(1, 4, 4),
        ...     torch.arange(20.0).view(1, 5, 4),
        ... )
        >>> out, idx = F.adaptive_max_pool2d_with_indices(nt, output_size=(2, 2))
        >>> ref0, idx0 = F.adaptive_max_pool2d_with_indices(nt[0], output_size=(2, 2))
        >>> ref1, idx1 = F.adaptive_max_pool2d_with_indices(nt[1], output_size=(2, 2))
        >>> ref_out = NestedTensor(ref0, ref1)
        >>> ref_idx = NestedTensor(idx0, idx1)
        >>> torch.equal(out, ref_out) and torch.equal(idx, ref_idx)
        True
    """
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
    r"""
    adaptive_max_pool3d(input, output_size, return_indices=False) Applies a 3D adaptive max pooling over an input signal
    composed of several input planes.
    See also [torch.nn.functional.adaptive_max_pool3d_with_indices][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(27.0).view(1, 3, 3, 3),
        ...     torch.arange(36.0).view(1, 4, 3, 3),
        ... )
        >>> out, idx = F.adaptive_max_pool3d_with_indices(nt, output_size=(1, 1, 1))
        >>> ref0, idx0 = F.adaptive_max_pool3d_with_indices(nt[0], output_size=(1, 1, 1))
        >>> ref1, idx1 = F.adaptive_max_pool3d_with_indices(nt[1], output_size=(1, 1, 1))
        >>> ref_out = NestedTensor(ref0, ref1)
        >>> ref_idx = NestedTensor(idx0, idx1)
        >>> torch.equal(out, ref_out) and torch.equal(idx, ref_idx)
        True
    """
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
    r"""
    fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False,
    _random_samples=None) Applies 2D fractional max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.fractional_max_pool2d_with_indices][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(16.0).view(1, 1, 4, 4),
        ...     torch.arange(20.0).view(1, 1, 5, 4),
        ... )
        >>> rs = torch.rand(1, 1, 2)
        >>> out, idx = F.fractional_max_pool2d_with_indices(
        ...     nt, kernel_size=2, output_ratio=(0.5, 0.5), _random_samples=rs
        ... )
        >>> ref0, _ = F.fractional_max_pool2d(
        ...     nt._storage[0],
        ...     kernel_size=2,
        ...     output_ratio=(0.5, 0.5),
        ...     return_indices=True,
        ...     _random_samples=rs,
        ... )
        >>> ref1, _ = F.fractional_max_pool2d(
        ...     nt._storage[1],
        ...     kernel_size=2,
        ...     output_ratio=(0.5, 0.5),
        ...     return_indices=True,
        ...     _random_samples=rs,
        ... )
        >>> ref = NestedTensor(ref0, ref1)
        >>> torch.allclose(out, ref)
        True
    """
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
    r"""
    fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False,
    _random_samples=None) Applies 3D fractional max pooling over an input signal composed of several input planes.
    See also [torch.nn.functional.fractional_max_pool3d_with_indices][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(27.0).view(1, 1, 3, 3, 3),
        ...     torch.arange(36.0).view(1, 1, 4, 3, 3),
        ... )
        >>> rs = torch.rand(1, 1, 3)
        >>> out, idx = F.fractional_max_pool3d_with_indices(
        ...     nt, kernel_size=2, output_ratio=(0.5, 0.5, 0.5), _random_samples=rs
        ... )
        >>> ref0, _ = F.fractional_max_pool3d(
        ...     nt._storage[0],
        ...     kernel_size=2,
        ...     output_ratio=(0.5, 0.5, 0.5),
        ...     return_indices=True,
        ...     _random_samples=rs,
        ... )
        >>> ref1, _ = F.fractional_max_pool3d(
        ...     nt._storage[1],
        ...     kernel_size=2,
        ...     output_ratio=(0.5, 0.5, 0.5),
        ...     return_indices=True,
        ...     _random_samples=rs,
        ... )
        >>> ref = NestedTensor(ref0, ref1)
        >>> torch.allclose(out, ref)
        True
    """
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
    r"""
    Extract sliding local blocks from a batched input tensor.
    See also [torch.nn.functional.unfold][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> a = torch.arange(4.0).view(1, 1, 2, 2)
        >>> b = torch.arange(12.0).view(1, 2, 3, 2)
        >>> nt = NestedTensor(a, b)
        >>> out = F.unfold(nt, kernel_size=2)
        >>> ref = NestedTensor(F.unfold(a, kernel_size=2), F.unfold(b, kernel_size=2))
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.unfold(t, kernel_size, dilation=dilation, padding=padding, stride=stride) for t in input._storage
    )


@NestedTensorFuncRegistry.implement(F.fold)
def fold(input: NestedTensor, output_size, kernel_size, dilation=1, padding=0, stride=1) -> NestedTensor:
    r"""
    Combine an array of sliding local blocks into a large containing tensor.
    See also [torch.nn.functional.fold][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> a = torch.arange(9.0).view(1, 1, 3, 3)
        >>> b = torch.arange(18.0).view(1, 2, 3, 3)
        >>> unfolded_a = F.unfold(a, kernel_size=2)
        >>> unfolded_b = F.unfold(b, kernel_size=2)
        >>> nt = NestedTensor(unfolded_a, unfolded_b)
        >>> out = F.fold(nt, output_size=(3, 3), kernel_size=2)
        >>> ref = NestedTensor(
        ...     F.fold(unfolded_a, output_size=(3, 3), kernel_size=2),
        ...     F.fold(unfolded_b, output_size=(3, 3), kernel_size=2),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    return NestedTensor(
        F.fold(t, output_size, kernel_size, dilation=dilation, padding=padding, stride=stride) for t in input._storage
    )


if hasattr(F, "scaled_dot_product_attention"):

    @NestedTensorFuncRegistry.implement(F.scaled_dot_product_attention)
    def scaled_dot_product_attention(
        query: NestedTensor,
        key: NestedTensor | Tensor,
        value: NestedTensor | Tensor,
        attn_mask: NestedTensor | Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> NestedTensor:
        r"""
        Compute scaled dot-product attention for variable-length batches.
        See also [torch.nn.functional.scaled_dot_product_attention][].

        Examples:
            >>> import torch
            >>> from torch.nn import functional as F
            >>> from danling.tensors import NestedTensor
            >>> q = NestedTensor(torch.randn(2, 3, 4), torch.randn(2, 2, 4))
            >>> out = F.scaled_dot_product_attention(q, q, q, dropout_p=0.0)
            >>> ref = NestedTensor(
            ...     F.scaled_dot_product_attention(q[0], q[0], q[0], dropout_p=0.0),
            ...     F.scaled_dot_product_attention(q[1], q[1], q[1], dropout_p=0.0),
            ... )
            >>> torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
            True
        """
        from .nested_tensor import NestedTensor

        if not isinstance(query, NestedTensor):
            raise TypeError("query must be a NestedTensor")
        if isinstance(key, Tensor) and key.shape == query.shape:
            key = query.nested_like(key, strict=False)
        if isinstance(value, Tensor) and value.shape == query.shape:
            value = query.nested_like(value, strict=False)

        if isinstance(key, NestedTensor) and len(query) != len(key):
            raise ValueError(f"NestedTensor batch length mismatch: {len(query)} vs {len(key)}")
        if isinstance(value, NestedTensor) and len(query) != len(value):
            raise ValueError(f"NestedTensor batch length mismatch: {len(query)} vs {len(value)}")
        if isinstance(attn_mask, NestedTensor) and len(query) != len(attn_mask):
            raise ValueError(f"NestedTensor batch length mismatch: {len(query)} vs {len(attn_mask)}")
        if not query._storage:
            return NestedTensor([], **query._state)

        mask_is_batched_tensor = (
            isinstance(attn_mask, Tensor)
            and attn_mask.dim() > 0
            and attn_mask.size(0) == len(query)
            and not isinstance(attn_mask, NestedTensor)
        )
        storage = []
        for i, q in enumerate(query._storage):
            k = key._storage[i] if isinstance(key, NestedTensor) else key
            v = value._storage[i] if isinstance(value, NestedTensor) else value
            if isinstance(attn_mask, NestedTensor):
                m = attn_mask._storage[i]
            elif mask_is_batched_tensor:
                m = attn_mask[i]  # type: ignore[index]
            else:
                m = attn_mask

            if isinstance(m, Tensor) and m.device != q.device:
                m = m.to(device=q.device)
            if isinstance(k, Tensor) and k.device != q.device:
                k = k.to(device=q.device)
            if isinstance(v, Tensor) and v.device != q.device:
                v = v.to(device=q.device)

            storage.append(
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=m,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            )
        return NestedTensor(storage, **query._state)


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
    r"""
    Forward method for MultiHeadAttention.
    See also [torch.nn.functional.multi_head_attention_forward][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> q1 = torch.randn(2, 4)
        >>> q2 = torch.randn(3, 4)
        >>> query = NestedTensor(q1, q2)
        >>> embed_dim = 4
        >>> num_heads = 2
        >>> in_w = torch.randn(3 * embed_dim, embed_dim)
        >>> in_b = torch.randn(3 * embed_dim)
        >>> out_w = torch.randn(embed_dim, embed_dim)
        >>> out_b = torch.randn(embed_dim)
        >>> out, attn = F.multi_head_attention_forward(
        ...     query, query, query, embed_dim, num_heads, in_w, in_b, None, None, False, 0.0, out_w, out_b,
        ...     training=False, need_weights=True,
        ... )
        >>> q_t = query.tensor.transpose(0, 1)
        >>> key_padding_mask = ~query.mask
        >>> ref_out, ref_attn = F.multi_head_attention_forward(
        ...     q_t, q_t, q_t, embed_dim, num_heads, in_w, in_b, None, None, False, 0.0, out_w, out_b,
        ...     training=False, need_weights=True, key_padding_mask=key_padding_mask,
        ... )
        >>> ref_out = ref_out.transpose(0, 1)
        >>> torch.allclose(out, ref_out)
        True
    """
    from .nested_tensor import NestedTensor

    if not isinstance(query, NestedTensor):
        raise TypeError("query must be a NestedTensor")
    if isinstance(key, Tensor) and key.shape == query.shape:
        key = query.nested_like(key, strict=False)
    if isinstance(value, Tensor) and value.shape == query.shape:
        value = query.nested_like(value, strict=False)
    if isinstance(key, NestedTensor) and len(query) != len(key):
        raise ValueError(f"NestedTensor batch length mismatch: {len(query)} vs {len(key)}")
    if isinstance(value, NestedTensor) and len(query) != len(value):
        raise ValueError(f"NestedTensor batch length mismatch: {len(query)} vs {len(value)}")
    if not query._storage:
        empty = NestedTensor([], **query._state)
        if need_weights:
            return empty, torch.empty(0, dtype=query.dtype, device=query.device)
        return empty

    output_storage = []
    attn_weight_storage = []
    for i, q in enumerate(query._storage):
        if isinstance(key, NestedTensor):
            k = key._storage[i]
        elif isinstance(key, Tensor) and key.dim() == q.dim() + 1:
            if query.batch_first and key.size(0) == len(query):
                k = key[i]
            elif not query.batch_first and key.size(1) == len(query):
                k = key[:, i]
            else:
                k = key
        else:
            k = key

        if isinstance(value, NestedTensor):
            v = value._storage[i]
        elif isinstance(value, Tensor) and value.dim() == q.dim() + 1:
            if query.batch_first and value.size(0) == len(query):
                v = value[i]
            elif not query.batch_first and value.size(1) == len(query):
                v = value[:, i]
            else:
                v = value
        else:
            v = value

        if key_padding_mask is None:
            sample_key_padding_mask = None
        elif key_padding_mask.dim() >= 2 and key_padding_mask.size(0) == len(query):
            sample_key_padding_mask = key_padding_mask[i]
        elif key_padding_mask.dim() >= 2 and key_padding_mask.size(0) == 1:
            sample_key_padding_mask = key_padding_mask[0]
        else:
            sample_key_padding_mask = key_padding_mask

        if isinstance(attn_mask, Tensor) and attn_mask.dim() >= 3 and attn_mask.size(0) == len(query):
            sample_attn_mask = attn_mask[i]
        elif isinstance(attn_mask, Tensor) and attn_mask.dim() >= 3 and attn_mask.size(0) == len(query) * num_heads:
            start = i * num_heads
            sample_attn_mask = attn_mask[start : start + num_heads]  # noqa: E203
        else:
            sample_attn_mask = attn_mask

        if isinstance(k, Tensor) and k.device != q.device:
            k = k.to(device=q.device)
        if isinstance(v, Tensor) and v.device != q.device:
            v = v.to(device=q.device)
        if isinstance(sample_key_padding_mask, Tensor) and sample_key_padding_mask.device != q.device:
            sample_key_padding_mask = sample_key_padding_mask.to(device=q.device)
        if isinstance(sample_attn_mask, Tensor) and sample_attn_mask.device != q.device:
            sample_attn_mask = sample_attn_mask.to(device=q.device)

        sample_output, sample_weights = F.multi_head_attention_forward(  # type: ignore[call-arg]
            q,
            k,
            v,
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
            key_padding_mask=sample_key_padding_mask,
            need_weights=need_weights,
            attn_mask=sample_attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        output_storage.append(sample_output)
        if need_weights:
            attn_weight_storage.append(sample_weights)

    nt_output = NestedTensor(output_storage, **query._state)
    if need_weights:
        weights = NestedTensor(attn_weight_storage, batch_first=True, padding_value=0.0, mask_value=False).tensor
        return nt_output, weights
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
    r"""
    Compute the Connectionist Temporal Classification loss.
    See also [torch.nn.functional.ctc_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    def _to_time_first(nested: NestedTensor) -> Tensor:
        logits = nested.tensor
        return logits.transpose(0, 1) if nested.batch_first else logits

    logits = _to_time_first(input) if isinstance(input, NestedTensor) else input
    targets = target.concat if isinstance(target, NestedTensor) else target
    return F.ctc_loss(
        logits,
        targets,
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
    r"""
    Compute the cosine embedding loss.
    See also [torch.nn.functional.cosine_embedding_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    target_concat, input1_concat, input2_concat = concat_tensors(target, input1, input2)
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
    r"""
    Compute the hinge embedding loss.
    See also [torch.nn.functional.hinge_embedding_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the cross entropy loss between input logits and target.
    See also [torch.nn.functional.cross_entropy][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute Binary Cross Entropy between the target and input probabilities.
    See also [torch.nn.functional.binary_cross_entropy][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute Binary Cross Entropy between target and input logits.
    See also [torch.nn.functional.binary_cross_entropy_with_logits][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the negative log likelihood loss.
    See also [torch.nn.functional.nll_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the Gaussian negative log likelihood loss.
    See also [torch.nn.functional.gaussian_nll_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the Poisson negative log likelihood loss.
    See also [torch.nn.functional.poisson_nll_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the element-wise mean squared error, with optional weighting.
    See also [torch.nn.functional.mse_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the L1 loss, with optional weighting.
    See also [torch.nn.functional.l1_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the Smooth L1 loss.
    See also [torch.nn.functional.smooth_l1_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the soft margin loss.
    See also [torch.nn.functional.soft_margin_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the multi margin loss, with optional weighting.
    See also [torch.nn.functional.multi_margin_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the multilabel margin loss.
    See also [torch.nn.functional.multilabel_margin_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the multilabel soft margin loss.
    See also [torch.nn.functional.multilabel_soft_margin_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the triplet loss between given input tensors and a margin greater than 0.
    See also [torch.nn.functional.triplet_margin_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    anchor_is_nt = isinstance(anchor, NestedTensor)
    positive_is_nt = isinstance(positive, NestedTensor)
    negative_is_nt = isinstance(negative, NestedTensor)

    mask = None
    if anchor_is_nt:
        mask = anchor.mask if not anchor.mask_value else ~anchor.mask
    elif positive_is_nt:
        mask = positive.mask if not positive.mask_value else ~positive.mask
    elif negative_is_nt:
        mask = negative.mask if not negative.mask_value else ~negative.mask

    anchor_concat = anchor.concat if anchor_is_nt else (anchor[mask] if mask is not None else anchor)
    positive_concat = positive.concat if positive_is_nt else (positive[mask] if mask is not None else positive)
    negative_concat = negative.concat if negative_is_nt else (negative[mask] if mask is not None else negative)
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
    r"""
    Compute the triplet margin loss for input tensors using a custom distance function.
    See also [torch.nn.functional.triplet_margin_with_distance_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    anchor_is_nt = isinstance(anchor, NestedTensor)
    positive_is_nt = isinstance(positive, NestedTensor)
    negative_is_nt = isinstance(negative, NestedTensor)

    mask = None
    if anchor_is_nt:
        mask = anchor.mask if not anchor.mask_value else ~anchor.mask
    elif positive_is_nt:
        mask = positive.mask if not positive.mask_value else ~positive.mask
    elif negative_is_nt:
        mask = negative.mask if not negative.mask_value else ~negative.mask

    anchor_concat = anchor.concat if anchor_is_nt else (anchor[mask] if mask is not None else anchor)
    positive_concat = positive.concat if positive_is_nt else (positive[mask] if mask is not None else positive)
    negative_concat = negative.concat if negative_is_nt else (negative[mask] if mask is not None else negative)
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
    r"""
    Compute the margin ranking loss.
    See also [torch.nn.functional.margin_ranking_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    input1_is_nt = isinstance(input1, NestedTensor)
    input2_is_nt = isinstance(input2, NestedTensor)
    target_is_nt = isinstance(target, NestedTensor)

    mask = None
    if target_is_nt:
        mask = target.mask if not target.mask_value else ~target.mask
    elif input1_is_nt:
        mask = input1.mask if not input1.mask_value else ~input1.mask
    elif input2_is_nt:
        mask = input2.mask if not input2.mask_value else ~input2.mask

    input1_concat = input1.concat if input1_is_nt else (input1[mask] if mask is not None else input1)
    input2_concat = input2.concat if input2_is_nt else (input2[mask] if mask is not None else input2)
    target_concat = target.concat if target_is_nt else (target[mask] if mask is not None else target)
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
    r"""
    Compute the Huber loss, with optional weighting.
    See also [torch.nn.functional.huber_loss][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    Compute the KL Divergence loss.
    See also [torch.nn.functional.kl_div][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functional import concat_tensors
        >>> input = NestedTensor(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        >>> target = NestedTensor(torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]))
        >>> out = F.mse_loss(input, target)
        >>> input_concat, target_concat = concat_tensors(input, target)
        >>> ref = F.mse_loss(input_concat, target_concat)
        >>> torch.allclose(out, ref)
        True
    """
    input_concat, target_concat = concat_tensors(input, target)
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
    r"""
    During training, randomly zeroes some elements of the input tensor. See also [torch.nn.functional.dropout][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0]))
        >>> out = F.dropout(nt, training=False)
        >>> ref = F.dropout(nt.tensor, training=False)
        >>> torch.equal(out, ref)
        True
    """
    return _apply_elementwise(input, F.dropout, p, training, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.dropout1d)
def dropout1d(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    r"""
    Randomly zero out entire channels (a channel is a 1D feature map).
    See also [torch.nn.functional.dropout1d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(6.0).view(1, 2, 3),
        ...     torch.arange(8.0).view(1, 2, 4),
        ... )
        >>> out = F.dropout1d(nt, p=0.0, training=False)
        >>> ref = NestedTensor(
        ...     F.dropout1d(nt._storage[0], p=0.0, training=False),
        ...     F.dropout1d(nt._storage[1], p=0.0, training=False),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    return _apply(input, F.dropout1d, p, training, inplace)


@NestedTensorFuncRegistry.implement(F.dropout2d)
def dropout2d(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    r"""
    Randomly zero out entire channels (a channel is a 2D feature map).
    See also [torch.nn.functional.dropout2d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(4.0).view(1, 1, 2, 2),
        ...     torch.arange(6.0).view(1, 1, 3, 2),
        ... )
        >>> out = F.dropout2d(nt, p=0.0, training=False)
        >>> ref = NestedTensor(
        ...     F.dropout2d(nt._storage[0], p=0.0, training=False),
        ...     F.dropout2d(nt._storage[1], p=0.0, training=False),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    return _apply(input, F.dropout2d, p, training, inplace)


@NestedTensorFuncRegistry.implement(F.dropout3d)
def dropout3d(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    r"""
    Randomly zero out entire channels (a channel is a 3D feature map).
    See also [torch.nn.functional.dropout3d][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(8.0).view(1, 1, 2, 2, 2),
        ...     torch.arange(12.0).view(1, 1, 3, 2, 2),
        ... )
        >>> out = F.dropout3d(nt, p=0.0, training=False)
        >>> ref = NestedTensor(
        ...     F.dropout3d(nt._storage[0], p=0.0, training=False),
        ...     F.dropout3d(nt._storage[1], p=0.0, training=False),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    return _apply(input, F.dropout3d, p, training, inplace)


@NestedTensorFuncRegistry.implement(F.alpha_dropout)
def alpha_dropout(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> NestedTensor:
    r"""
    Applies alpha dropout to the input.
    See also [torch.nn.functional.alpha_dropout][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = F.alpha_dropout(nt, p=0.0, training=False)
        >>> ref = F.alpha_dropout(nt.tensor, p=0.0, training=False)
        >>> torch.equal(out, ref)
        True
    """
    return _apply_elementwise(input, F.alpha_dropout, p, training, inplace=inplace)


@NestedTensorFuncRegistry.implement(F.feature_alpha_dropout)
def feature_alpha_dropout(
    input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> NestedTensor:
    r"""
    Randomly masks out entire channels (a channel is a feature map).
    See also [torch.nn.functional.feature_alpha_dropout][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = F.feature_alpha_dropout(nt, p=0.0, training=False)
        >>> ref = F.feature_alpha_dropout(nt.tensor, p=0.0, training=False)
        >>> torch.equal(out, ref)
        True
    """
    return _apply(input, F.feature_alpha_dropout, p, training, inplace)


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
    r"""
    Applies Batch Normalization for each channel across a batch of data.
    See also [torch.nn.functional.batch_norm][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(4.0).view(1, 1, 4),
        ...     torch.arange(6.0).view(1, 1, 6),
        ... )
        >>> running_mean = torch.zeros(1)
        >>> running_var = torch.ones(1)
        >>> out = F.batch_norm(nt, running_mean, running_var, training=False)
        >>> ref = F.batch_norm(nt.tensor, running_mean, running_var, training=False)
        >>> torch.allclose(out, ref)
        True
    """
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
    r"""
    Applies Instance Normalization independently for each channel in every data sample within a batch.
    See also [torch.nn.functional.instance_norm][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(5.0).view(1, 5))
        >>> out = F.instance_norm(nt, use_input_stats=True)
        >>> ref = NestedTensor(
        ...     F.instance_norm(nt[0].unsqueeze(0), use_input_stats=True).squeeze(0),
        ...     F.instance_norm(nt[1].unsqueeze(0), use_input_stats=True).squeeze(0),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _map_storage(
        input,
        lambda t: F.instance_norm(
            t.unsqueeze(0),
            running_mean,
            running_var,
            weight,
            bias,
            use_input_stats,
            momentum,
            eps,
        ).squeeze(0),
    )


@NestedTensorFuncRegistry.implement(F.group_norm)
def group_norm(
    input: NestedTensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> NestedTensor:
    r"""
    Applies Group Normalization for last certain number of dimensions.
    See also [torch.nn.functional.group_norm][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(4, 2), torch.arange(12.0).view(6, 2))
        >>> out = F.group_norm(nt, num_groups=2)
        >>> ref = NestedTensor(
        ...     F.group_norm(nt[0].unsqueeze(0), num_groups=2).squeeze(0),
        ...     F.group_norm(nt[1].unsqueeze(0), num_groups=2).squeeze(0),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _map_storage(input, lambda t: F.group_norm(t.unsqueeze(0), num_groups, weight, bias, eps).squeeze(0))


@NestedTensorFuncRegistry.implement(F.layer_norm)
def layer_norm(
    input: NestedTensor,
    normalized_shape: Tuple,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> NestedTensor:
    r"""
    Applies Layer Normalization over the last certain number of dimensions.
    See also [torch.nn.functional.layer_norm][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.layer_norm(nt, (2,))
        >>> ref = F.layer_norm(nt.tensor, (2,))
        >>> torch.allclose(out, ref)
        True
    """
    normalized = tuple(normalized_shape)
    if _can_concat_normalize(input, normalized):
        return _apply_concat(input, F.layer_norm, normalized, weight, bias, eps)
    return _apply(input, F.layer_norm, normalized, weight, bias, eps)


@NestedTensorFuncRegistry.implement(F.rms_norm)
def rms_norm(
    input: NestedTensor, normalized_shape: Tuple, weight: Tensor | None = None, eps: float = 1e-5
) -> NestedTensor:
    r"""
    Applies Root Mean Square Layer Normalization. See also [torch.nn.functional.rms_norm][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.rms_norm(nt, (2,))
        >>> ref = F.rms_norm(nt.tensor, (2,))
        >>> torch.allclose(out, ref)
        True
    """
    normalized = tuple(normalized_shape)
    if _can_concat_normalize(input, normalized):
        return _apply_concat(input, F.rms_norm, normalized, weight, eps)
    return _apply(input, F.rms_norm, normalized, weight, eps)


@NestedTensorFuncRegistry.implement(F.local_response_norm)
def local_response_norm(
    input: NestedTensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
) -> NestedTensor:
    r"""
    Applies local response normalization over an input signal.
    See also [torch.nn.functional.local_response_norm][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(5.0).view(1, 5))
        >>> out = F.local_response_norm(nt, size=2)
        >>> ref = F.local_response_norm(nt.tensor, size=2)
        >>> torch.allclose(out, ref)
        True
    """
    return _map_storage(input, lambda t: F.local_response_norm(t.unsqueeze(0), size, alpha, beta, k).squeeze(0))


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
    r"""
    Down/up samples the input.
    See also [torch.nn.functional.interpolate][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 2, 2), torch.arange(6.0).view(1, 2, 3))
        >>> out = F.interpolate(nt, scale_factor=2, mode="nearest")
        >>> ref = NestedTensor(
        ...     F.interpolate(nt._storage[0], scale_factor=2, mode="nearest"),
        ...     F.interpolate(nt._storage[1], scale_factor=2, mode="nearest"),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    return _apply(
        input,
        F.interpolate,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
        antialias=antialias,
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
        r"""
        Applies grouped matrix multiplication.
        See also [torch.nn.functional.grouped_mm][].

        Examples:
            >>> import torch
            >>> from torch.nn import functional as F
            >>> from danling.tensors import NestedTensor
            >>> a = NestedTensor(torch.eye(2), torch.eye(2))
            >>> out = F.grouped_mm(a, torch.eye(2))
            >>> ref = NestedTensor(F.grouped_mm(torch.eye(2), torch.eye(2)), F.grouped_mm(torch.eye(2), torch.eye(2)))
            >>> torch.allclose(out, ref)
            True
        """
        from .nested_tensor import NestedTensor

        if isinstance(mat_b, NestedTensor):
            if len(mat_a) != len(mat_b):
                raise ValueError(f"NestedTensor batch length mismatch: {len(mat_a)} vs {len(mat_b)}")
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
        r"""
        Applies scaled matrix multiplication.
        See also [torch.nn.functional.scaled_mm][].

        Examples:
            >>> import torch
            >>> from torch.nn import functional as F
            >>> from danling.tensors import NestedTensor
            >>> a = NestedTensor(torch.eye(2), torch.eye(2))
            >>> out = F.scaled_mm(a, torch.eye(2))
            >>> ref = NestedTensor(F.scaled_mm(torch.eye(2), torch.eye(2)), F.scaled_mm(torch.eye(2), torch.eye(2)))
            >>> torch.allclose(out, ref)
            True
        """
        from .nested_tensor import NestedTensor

        if isinstance(mat_b, NestedTensor):
            if len(mat_a) != len(mat_b):
                raise ValueError(f"NestedTensor batch length mismatch: {len(mat_a)} vs {len(mat_b)}")
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
        r"""
        Applies scaled grouped matrix multiplication.
        See also [torch.nn.functional.scaled_grouped_mm][].

        Examples:
            >>> import torch
            >>> from torch.nn import functional as F
            >>> from danling.tensors import NestedTensor
            >>> a = NestedTensor(torch.eye(2), torch.eye(2))
            >>> out = F.scaled_grouped_mm(a, torch.eye(2))
            >>> ref0 = F.scaled_grouped_mm(torch.eye(2), torch.eye(2))
            >>> ref1 = F.scaled_grouped_mm(torch.eye(2), torch.eye(2))
            >>> ref = NestedTensor(ref0, ref1)
            >>> torch.allclose(out, ref)
            True
        """
        from .nested_tensor import NestedTensor

        if isinstance(mat_b, NestedTensor):
            if len(mat_a) != len(mat_b):
                raise ValueError(f"NestedTensor batch length mismatch: {len(mat_a)} vs {len(mat_b)}")
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
    r"""
    Pads tensor.
    See also [torch.nn.functional.pad][].

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = F.pad(nt, (1, 1))
        >>> ref = F.pad(nt.tensor, (1, 1))
        >>> torch.equal(out, ref)
        True
    """
    return _apply(input, F.pad, pad, mode, value)
