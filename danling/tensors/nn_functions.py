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
r"""
``torch.nn.functional.*`` overrides for NestedTensor via ``__torch_function__``.

This module is the **Level 3** dispatch layer, registering handlers for
``F.linear``, ``F.conv*``, ``F.max_pool*``, ``F.embedding``,
``F.layer_norm``, ``F.scaled_dot_product_attention``, and other
``torch.nn.functional`` ops.

Most ops here use **per-element dispatch** ([_apply_per_element][]) because
they require spatial-dimension awareness that the packed ``_values`` tensor
cannot provide.  A few normalisation ops (``F.layer_norm``, ``F.rms_norm``)
have a fast **packed path** ([_apply_packed][]) when the normalised shape
covers the full non-batch extent.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from ._streams import _run_on_streams
from .ops import (
    NestedTensorFuncRegistry,
    _concat_apply,
    _concat_apply_same_shape,
    _concat_dim_for_tensor_dim,
    _ensure_nested_input,
    _map_storage,
    _map_storage_pair,
    _translate_non_batch_dim,
)

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor

_HAS_JAGGED = hasattr(torch, "jagged")


# Helpers


def _extract_concat_from_padded(ref, v: Tensor) -> Tensor:
    r"""
    Extract valid elements from padded tensor *v*, matching ``ref.concat`` layout.

    Uses direct slicing via ``_shape_tensor`` (CPU) instead of boolean mask
    gather, avoiding the O(B * max_len * ...) mask materialisation.
    """
    from .nested_tensor import _cat_ragged_parts

    shapes = ref._shape_tensor.tolist()
    batch_first = ref.batch_first
    parts: list[Tensor] = []
    for idx, shape in enumerate(shapes):
        while shape and shape[-1] == 0:
            shape.pop()
        if not shape:
            parts.append(v.select(0 if batch_first else 1, idx).unsqueeze(0))
            continue
        if batch_first:
            slices = (idx, *(slice(0, s) for s in shape))
        else:
            slices = (slice(0, shape[0]), idx, *(slice(0, s) for s in shape[1:]))
        parts.append(v[slices])

    return _cat_ragged_parts(parts, ref.size(), batch_first)


def _concat_tensors(*values: NestedTensor | Tensor) -> tuple[Tensor, ...]:
    r"""
    Return tensors for each input, expanding NestedTensor values to their concatenated representation.

    Args:
        *values: NestedTensor or Tensor inputs to expand.

    Returns:
        tuple[Tensor, ...]: A tuple of tensors with NestedTensors expanded to concatenated form.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _concat_tensors
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> a, b = _concat_tensors(nt, nt)
        >>> torch.equal(a, nt.concat) and torch.equal(b, nt.concat)
        True
    """
    from .nested_tensor import NestedTensor

    ref = None
    for v in values:
        if isinstance(v, NestedTensor):
            ref = v
            break

    if ref is None:
        return tuple(values)

    result = []
    for v in values:
        if isinstance(v, NestedTensor):
            result.append(v.concat)  # type: ignore[union-attr]
        else:
            if v.shape != ref.shape:  # type: ignore[union-attr]
                raise ValueError(
                    f"Cannot apply NestedTensor mask (shape {tuple(ref.shape)}) "  # type: ignore[union-attr]
                    f"to tensor of shape {tuple(v.shape)}"
                )
            result.append(_extract_concat_from_padded(ref, v))
    return tuple(result)


@torch._dynamo.disable
def _apply_per_element(input: NestedTensor, op: Callable, *args, **kwargs) -> NestedTensor:
    r"""
    Applies an operator to each tensor in a NestedTensor.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _apply_per_element
        >>> nt = NestedTensor(torch.tensor([1.0, -2.0]), torch.tensor([3.0, -4.0, 5.0]))
        >>> out = _apply_per_element(nt, torch.abs)
        >>> ref = torch.abs(nt.tensor)
        >>> torch.equal(out, ref)
        True
    """
    cls = type(input)
    elements = input._storage
    if not elements:
        return cls([], **input._meta(include_dtype=True))
    if not elements[0].is_cuda:
        return cls((op(t, *args, **kwargs) for t in elements), **input._meta())
    return cls(_run_on_streams(elements, lambda t: op(t, *args, **kwargs)), **input._meta())


def _apply_with_indices(input: NestedTensor, pool_fn: Callable, *args, **kwargs):
    r"""Applies a pooling op that returns (output, indices) to each element of a NestedTensor."""
    cls = type(input)
    if len(input) == 0:
        return (
            cls([], **input._meta(include_dtype=True)),
            cls([], dtype=torch.long, **input._meta(include_dtype=False)),
        )
    outputs, indices = [], []
    for t in input._storage:
        out, idx = pool_fn(t, *args, return_indices=True, **kwargs)
        outputs.append(out)
        indices.append(idx)
    return cls(outputs, **input._meta()), cls(indices, **input._meta())


@torch._dynamo.disable
def _apply_pair(input: NestedTensor, other: NestedTensor | Tensor, op: Callable, *args, **kwargs):
    r"""
    Applies a binary operator to a NestedTensor and another operand.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _apply_pair
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> other = torch.tensor([1.0, 1.0])
        >>> out = _apply_pair(nt, other, torch.add)
        >>> ref = torch.add(nt.tensor, other)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    cls = type(input) if isinstance(input, NestedTensor) else type(other)
    input = _ensure_nested_input(input, other, cls)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    if isinstance(other, NestedTensor):
        if len(input) != len(other):
            raise ValueError(
                "NestedTensor batch length mismatch between input and other: " f"input={len(input)}, other={len(other)}"
            )
        elements = input._storage
        if elements and elements[0].is_cuda:
            return cls(
                _run_on_streams(
                    list(zip(elements, other._storage)),
                    lambda pair: op(pair[0], pair[1], *args, **kwargs),
                ),
                **input._meta(),
            )
        return cls((op(x, y, *args, **kwargs) for x, y in zip(elements, other._storage)), **input._meta())
    return cls((op(x, other, *args, **kwargs) for x in input._storage), **input._meta())


def _apply_packed(input: NestedTensor, op: Callable, *args, **kwargs) -> NestedTensor:
    r"""
    Applies an operator to a NestedTensor and return a NestedTensor result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _apply_packed
        >>> nt = NestedTensor(torch.tensor([1.0, -2.0]), torch.tensor([3.0, -4.0, 5.0]))
        >>> out = _apply_packed(nt, torch.abs)
        >>> ref = torch.abs(nt.tensor)
        >>> torch.equal(out, ref)
        True
    """
    return _concat_apply_same_shape(input, lambda t: op(t, *args, **kwargs))


def _from_batch_preserving_values(input: NestedTensor, values: Tensor) -> NestedTensor:
    r"""
    Rebuild NestedTensor metadata for ops that preserve the per-element leading (ragged) dimension.

    The packed ``_values`` layout concatenates per-element dim-0 along global dim-0.
    For ops where dim-0 is batch-like (conv/pool/interpolate), each element keeps its
    original dim-0 length and only trailing dims change uniformly.
    """
    cls = type(input)
    out_shape_tensor = torch.empty((input._shape_tensor.size(0), values.dim()), dtype=input._shape_tensor.dtype)
    out_shape_tensor[:, 0] = input._shape_tensor[:, 0]
    if values.dim() > 1:
        out_shape_tensor[:, 1:] = torch.tensor(list(values.shape[1:]), dtype=out_shape_tensor.dtype)
    max_leading = input._logical_shape[1 if input.batch_first else 0] if len(input._logical_shape) > 1 else 0
    if input.batch_first:
        outer_size = torch.Size((len(input), max_leading, *values.shape[1:]))
    else:
        outer_size = torch.Size((max_leading, len(input), *values.shape[1:]))
    return cls._from_packed(
        values,
        input._offsets,
        out_shape_tensor,
        batch_first=input.batch_first,
        padding_value=input.padding_value,
        mask_value=input.mask_value,
        pin_memory=input._pin_memory,
        outer_size=outer_size,
    )


def _apply_batch_preserving_packed(input: NestedTensor, op: Callable, *args, **kwargs):
    r"""Apply *op* to packed ``_values`` for batch-preserving ops, supporting Tensor or (Tensor, Tensor) returns."""
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    result = op(input._values, *args, **kwargs)
    if isinstance(result, Tensor):
        return _from_batch_preserving_values(input, result)
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], Tensor)
        and isinstance(result[1], Tensor)
    ):
        return _from_batch_preserving_values(input, result[0]), _from_batch_preserving_values(input, result[1])
    raise TypeError(f"Unsupported return type from packed op {op}: {type(result)}")


def _can_concat_normalize(input: NestedTensor, normalized_shape: tuple[int, ...]) -> bool:
    r"""
    Return whether a normalized_shape is compatible with all tensors in a NestedTensor.

    Examples:
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _can_concat_normalize
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


# Linear & Embeddings


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

    Args:
        input: The input NestedTensor of indices.
        weight: The embedding matrix with number of rows equal to the maximum index + 1.
        padding_idx: If specified, entries at this index do not contribute to the gradient.
        max_norm: If given, renormalize embeddings to have norm at most this value.
        norm_type: The p of the p-norm to compute for max_norm.
        scale_grad_by_freq: If True, scale gradients by inverse frequency of the words.
        sparse: If True, gradient w.r.t. weight will be a sparse tensor.

    Returns:
        NestedTensor: The embedded output.

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

    Args:
        input: The input NestedTensor of indices.
        weight: The embedding matrix.
        offsets: Offsets into input for each bag. Required for 1D input.
        max_norm: If given, renormalize embeddings to have norm at most this value.
        norm_type: The p of the p-norm to compute for max_norm.
        scale_grad_by_freq: If True, scale gradients by inverse frequency of the words.
        mode: Aggregation mode: 'mean', 'sum', or 'max'.
        sparse: If True, gradient w.r.t. weight will be a sparse tensor.
        per_sample_weights: Per-sample weights for weighted sum mode.
        include_last_offset: If True, treat the last offset as the size of input.
        padding_idx: If specified, entries at this index do not contribute to the gradient.

    Returns:
        NestedTensor: The embedded bag output.

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

    Args:
        input: The input NestedTensor.
        weight: The weight matrix.
        bias: Optional bias vector.

    Returns:
        NestedTensor: The linearly transformed output.

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
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    # Fast path: single-ragged packing (2D+ _values) → F.linear directly on packed data
    if input._values.dim() >= 2:
        new_values = F.linear(input._values, weight, bias)
        new_shape = input._shape_tensor.clone()
        new_shape[:, -1] = weight.shape[0]
        new_outer_size = list(input._logical_shape)
        if new_outer_size:
            new_outer_size[-1] = int(weight.shape[0])
        return cls._from_packed(
            new_values,
            input._offsets,
            new_shape,
            batch_first=input.batch_first,
            padding_value=input.padding_value,
            mask_value=input.mask_value,
            pin_memory=input._pin_memory,
            outer_size=tuple(new_outer_size),
        )
    # Multi-ragged (flattened 1D _values) or scalar elements: per-element
    return _apply_per_element(input, F.linear, weight, bias)


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

        Args:
            mat_a: The first input NestedTensor.
            mat_b: The second input Tensor or NestedTensor.
            offs: Optional offsets tensor for grouping.
            bias: Optional bias tensor.
            out_dtype: Optional output dtype.

        Returns:
            NestedTensor: The result of grouped matrix multiplication.

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

        cls = type(mat_a)
        if isinstance(mat_b, NestedTensor):
            if len(mat_a) != len(mat_b):
                raise ValueError(
                    "NestedTensor batch length mismatch between mat_a and mat_b: "
                    f"mat_a={len(mat_a)}, mat_b={len(mat_b)}"
                )
            if len(mat_a) == 0:
                return cls([], **mat_a._meta(include_dtype=True))
            outputs = [
                F.grouped_mm(a, b, offs=offs, bias=bias, out_dtype=out_dtype)
                for a, b in zip(mat_a._storage, mat_b._storage)
            ]
        else:
            if len(mat_a) == 0:
                return cls([], **mat_a._meta(include_dtype=True))
            outputs = [F.grouped_mm(a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype) for a in mat_a._storage]
        return cls(outputs, **mat_a._meta())


def _scaled_mm(mat_a: NestedTensor, mat_b, fn, **kwargs) -> NestedTensor:
    r"""Shared implementation for scaled_mm and scaled_grouped_mm."""
    from .nested_tensor import NestedTensor

    cls = type(mat_a)
    if len(mat_a) == 0:
        return cls([], **mat_a._meta(include_dtype=True))
    if isinstance(mat_b, NestedTensor):
        if len(mat_a) != len(mat_b):
            raise ValueError(
                "NestedTensor batch length mismatch between mat_a and mat_b: " f"mat_a={len(mat_a)}, mat_b={len(mat_b)}"
            )
        outputs = [fn(a, b, **kwargs) for a, b in zip(mat_a._storage, mat_b._storage)]
    else:
        outputs = [fn(a, mat_b, **kwargs) for a in mat_a._storage]
    return cls(outputs, **mat_a._meta())


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

        Args:
            mat_a: The first input NestedTensor.
            mat_b: The second input Tensor or NestedTensor.
            scale_a: Scale factor for mat_a.
            scale_recipe_a: Scale recipe for mat_a.
            scale_b: Scale factor for mat_b.
            scale_recipe_b: Scale recipe for mat_b.
            swizzle_a: Optional swizzle pattern for mat_a.
            swizzle_b: Optional swizzle pattern for mat_b.
            bias: Optional bias tensor.
            output_dtype: Optional output dtype.
            contraction_dim: Contraction dimensions.
            use_fast_accum: Whether to use fast accumulation.

        Returns:
            NestedTensor: The result of scaled matrix multiplication.

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
        return _scaled_mm(
            mat_a,
            mat_b,
            F.scaled_mm,
            scale_a=scale_a,
            scale_recipe_a=scale_recipe_a,
            scale_b=scale_b,
            scale_recipe_b=scale_recipe_b,
            swizzle_a=swizzle_a,
            swizzle_b=swizzle_b,
            bias=bias,
            output_dtype=output_dtype,
            contraction_dim=contraction_dim,
            use_fast_accum=use_fast_accum,
        )


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

        Args:
            mat_a: The first input NestedTensor.
            mat_b: The second input Tensor or NestedTensor.
            scale_a: Scale factor for mat_a.
            scale_recipe_a: Scale recipe for mat_a.
            scale_b: Scale factor for mat_b.
            scale_recipe_b: Scale recipe for mat_b.
            swizzle_a: Optional swizzle pattern for mat_a.
            swizzle_b: Optional swizzle pattern for mat_b.
            bias: Optional bias tensor.
            offs: Optional offsets tensor for grouping.
            output_dtype: Optional output dtype.
            contraction_dim: Contraction dimensions.
            use_fast_accum: Whether to use fast accumulation.

        Returns:
            NestedTensor: The result of scaled grouped matrix multiplication.

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
        return _scaled_mm(
            mat_a,
            mat_b,
            F.scaled_grouped_mm,
            scale_a=scale_a,
            scale_recipe_a=scale_recipe_a,
            scale_b=scale_b,
            scale_recipe_b=scale_recipe_b,
            swizzle_a=swizzle_a,
            swizzle_b=swizzle_b,
            bias=bias,
            offs=offs,
            output_dtype=output_dtype,
            contraction_dim=contraction_dim,
            use_fast_accum=use_fast_accum,
        )


# Activations
# Most activations (F.relu, F.gelu, F.silu, etc.) are NOT registered here.
# They fall through to torch.* registrations in torch_functions.py or aten dispatch
# in aten_functions.py via PyTorch's __torch_function__ chain.


# Attention


def _build_sdpa_mask(
    key: NestedTensor | Tensor,
    attn_mask: NestedTensor | Tensor | None,
    is_causal: bool,
    query_tensor: Tensor,
    key_tensor: Tensor,
) -> tuple:
    r"""
    Build combined attention mask for batched SDPA.

    Constructs a mask that prevents attention to padding key positions,
    combines with any user-provided attn_mask, and handles is_causal conflicts.

    Returns:
        (combined_mask, is_causal): The combined mask and updated is_causal flag.
    """
    from .nested_tensor import NestedTensor

    # 1. Build key padding mask from NestedTensor structure
    key_attn_mask = None
    if isinstance(key, NestedTensor):
        key_mask = key.mask  # True=real when mask_value=False
        if not key.batch_first and key_mask.dim() >= 2:
            key_mask = key_mask.transpose(0, 1)
        if key.mask_value:
            key_mask = ~key_mask  # normalize to True=real (SDPA bool: True=attend)

        # Reduce any channel-like trailing dimensions; SDPA key padding mask should
        # describe valid key positions, not feature channels.
        while key_mask.dim() > query_tensor.dim() - 1:
            key_mask = key_mask[..., 0]

        # Expand to match SDPA score dimensions (batch/head..., T_q, T_k).
        # Key mask is (..., T_k) -> (..., 1, T_k).
        target_ndim = query_tensor.dim()
        while key_mask.dim() < target_ndim - 1:
            key_mask = key_mask.unsqueeze(1)
        key_attn_mask = key_mask.unsqueeze(-2)
        while key_attn_mask.dim() < target_ndim:
            key_attn_mask = key_attn_mask.unsqueeze(1)

    # 2. Process user-provided attn_mask
    user_mask = None
    if attn_mask is not None:
        if isinstance(attn_mask, NestedTensor):
            user_mask = attn_mask.tensor  # materialize padded
        else:
            user_mask = attn_mask

    # 3. Handle is_causal + mask conflict (can't pass both to SDPA)
    causal_mask = None
    if is_causal and (key_attn_mask is not None or user_mask is not None):
        q_len = query_tensor.size(-2)
        k_len = key_tensor.size(-2)
        causal_mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query_tensor.device).tril()
        is_causal = False  # handled explicitly via the mask

    # 4. Combine all masks (all in True=attend convention)
    combined = None
    for m in [key_attn_mask, user_mask, causal_mask]:
        if m is None:
            continue
        if combined is None:
            combined = m
        elif combined.dtype == torch.bool and m.dtype == torch.bool:
            combined = combined & m
        else:
            # Mixed bool/float: convert bool to float (-inf for masked)
            if combined.dtype == torch.bool:
                combined = torch.where(
                    combined,
                    torch.zeros((), device=combined.device),
                    torch.tensor(float("-inf"), device=combined.device),
                )
            if m.dtype == torch.bool:
                m = torch.where(m, torch.zeros((), device=m.device), torch.tensor(float("-inf"), device=m.device))
            combined = combined + m

    return combined, is_causal


def _to_jagged(nt: NestedTensor) -> Tensor:
    r"""
    Convert DanLing NestedTensor to torch.nested jagged for SDPA.

    Elements are (heads, seq_i, head_dim): transpose to (seq_i, heads, head_dim)
    for jagged packing along the seq dim, then transpose back to (B, heads, seq, head_dim).
    """
    elements = nt._storage
    elem_ndim = elements[0].dim()
    if elem_ndim == 3:
        # (heads, seq_i, head_dim) -> (seq_i, heads, head_dim) for jagged
        flat = [t.transpose(0, 1).contiguous() for t in elements]
        jagged = torch.nested.nested_tensor(flat, layout=torch.jagged)
        return jagged.transpose(1, 2)  # (B, heads, seq, head_dim)
    # 2D: (seq_i, dim) -> directly packable as (B, seq, dim) jagged
    return torch.nested.nested_tensor(list(elements), layout=torch.jagged)


def _from_jagged(jagged_out: Tensor, query: NestedTensor) -> NestedTensor:
    r"""Convert torch.nested jagged SDPA output back to DanLing NestedTensor."""
    cls = type(query)
    elem_ndim = query._storage[0].dim()
    if elem_ndim == 3:
        # (B, heads, seq, head_dim) -> (B, seq, heads, head_dim) -> unbind
        jagged_out = jagged_out.transpose(1, 2)
        return cls(
            [t.transpose(0, 1).contiguous() for t in jagged_out.unbind()],
            **query._meta(),
        )
    return cls(list(jagged_out.unbind()), **query._meta())


def _sdpa_via_jagged(
    query: NestedTensor,
    key: NestedTensor,
    value: NestedTensor,
    **kwargs,
) -> NestedTensor:
    r"""Run SDPA via torch.nested jagged layout (Flash Attention variable-length path)."""
    q_jagged = _to_jagged(query)
    k_jagged = _to_jagged(key)
    v_jagged = _to_jagged(value)
    out_jagged = F.scaled_dot_product_attention(q_jagged, k_jagged, v_jagged, **kwargs)
    return _from_jagged(out_jagged, query)


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

    Args:
        query: The query NestedTensor.
        key: The key NestedTensor or Tensor.
        value: The value NestedTensor or Tensor.
        embed_dim_to_check: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        in_proj_weight: Input projection weight.
        in_proj_bias: Input projection bias.
        bias_k: Bias of the key projection.
        bias_v: Bias of the value projection.
        add_zero_attn: Whether to add a batch of zeros to key and value.
        dropout_p: Dropout probability.
        out_proj_weight: Output projection weight.
        out_proj_bias: Output projection bias.
        training: Whether in training mode.
        key_padding_mask: Mask for padded key positions.
        need_weights: Whether to return attention weights.
        attn_mask: Attention mask tensor.
        use_separate_proj_weight: Whether to use separate projection weights for q, k, v.
        q_proj_weight: Query projection weight when using separate weights.
        k_proj_weight: Key projection weight when using separate weights.
        v_proj_weight: Value projection weight when using separate weights.
        static_k: Static key tensor.
        static_v: Static value tensor.
        average_attn_weights: Whether to average attention weights over heads.
        is_causal: Whether to apply causal masking.

    Returns:
        tuple[NestedTensor, Tensor]: The attention output and attention weights.

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

    cls = type(query)
    if not isinstance(query, NestedTensor):
        raise TypeError("query must be a NestedTensor")
    if isinstance(key, Tensor) and key.shape == query.shape:
        key = query.nested_like(key, strict=False)
    if isinstance(value, Tensor) and value.shape == query.shape:
        value = query.nested_like(value, strict=False)
    if isinstance(key, NestedTensor):
        if len(query) != len(key):
            raise ValueError(
                "NestedTensor batch length mismatch between query and key: " f"query={len(query)}, key={len(key)}"
            )
        if query.batch_first != key.batch_first:
            raise ValueError(
                "NestedTensor batch_first mismatch between query and key: "
                f"query.batch_first={query.batch_first}, key.batch_first={key.batch_first}. "
                "Use the same batch_first setting for query, key, and value."
            )
    if isinstance(value, NestedTensor):
        if len(query) != len(value):
            raise ValueError(
                "NestedTensor batch length mismatch between query and value: " f"query={len(query)}, value={len(value)}"
            )
        if query.batch_first != value.batch_first:
            raise ValueError(
                "NestedTensor batch_first mismatch between query and value: "
                f"query.batch_first={query.batch_first}, value.batch_first={value.batch_first}. "
                "Use the same batch_first setting for query, key, and value."
            )
    if not query._storage:
        empty = cls([], **query._meta(include_dtype=True))
        return empty, (torch.empty(0, dtype=query.dtype, device=query.device) if need_weights else None)

    # Materialize padded tensors
    q_padded = query.tensor
    k_padded = key.tensor if isinstance(key, NestedTensor) else key
    v_padded = value.tensor if isinstance(value, NestedTensor) else value

    # Transpose to (T, B, E) — F.multi_head_attention_forward always expects this layout
    if query.batch_first:
        q_padded = q_padded.transpose(0, 1)
        if isinstance(k_padded, Tensor) and k_padded.dim() == 3:
            k_padded = k_padded.transpose(0, 1)
        if isinstance(v_padded, Tensor) and v_padded.dim() == 3:
            v_padded = v_padded.transpose(0, 1)

    # Build key_padding_mask from NestedTensor structure if not provided
    # PyTorch convention: True=ignore (padding)
    if key_padding_mask is None and isinstance(key, NestedTensor):
        kpm = key.mask  # True=real when mask_value=False
        if not key.batch_first:
            kpm = kpm.transpose(0, 1)  # → (B, S)
        # Reduce extra dims from squeeze_channel to get (B, S)
        while kpm.dim() > 2:
            kpm = kpm[..., 0]
        if key.mask_value:
            key_padding_mask = kpm  # True=padding matches PyTorch convention
        else:
            key_padding_mask = ~kpm  # True=real → flip to True=ignore

    # Single batched MHA call (replaces per-sample Python loop)
    output, weights = F.multi_head_attention_forward(  # type: ignore[call-arg]
        q_padded,
        k_padded,
        v_padded,
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
        key_padding_mask=key_padding_mask,
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

    # Transpose back to match query's batch_first convention
    if query.batch_first:
        output = output.transpose(0, 1)  # (T, B, E) → (B, T, E)

    # Reconstruct NestedTensor (strips padding via query's structure)
    nt_output = query.nested_like(output, strict=False)

    return nt_output, weights


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

    Args:
        query: The query NestedTensor.
        key: The key NestedTensor or Tensor.
        value: The value NestedTensor or Tensor.
        attn_mask: Optional attention mask.
        dropout_p: Dropout probability.
        is_causal: Whether to apply causal masking.
        scale: Optional scaling factor for attention scores.
        enable_gqa: Whether to enable grouped query attention.

    Returns:
        NestedTensor: The attention output.

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
        raise ValueError(
            "NestedTensor batch length mismatch between query and key: " f"query={len(query)}, key={len(key)}"
        )
    if isinstance(key, NestedTensor) and query.batch_first != key.batch_first:
        raise ValueError(
            "NestedTensor batch_first mismatch between query and key: "
            f"query.batch_first={query.batch_first}, key.batch_first={key.batch_first}. "
            "Use the same batch_first setting for query, key, and value."
        )
    if isinstance(value, NestedTensor) and len(query) != len(value):
        raise ValueError(
            "NestedTensor batch length mismatch between query and value: " f"query={len(query)}, value={len(value)}"
        )
    if isinstance(value, NestedTensor) and query.batch_first != value.batch_first:
        raise ValueError(
            "NestedTensor batch_first mismatch between query and value: "
            f"query.batch_first={query.batch_first}, value.batch_first={value.batch_first}. "
            "Use the same batch_first setting for query, key, and value."
        )
    if isinstance(attn_mask, NestedTensor) and len(query) != len(attn_mask):
        raise ValueError(
            "NestedTensor batch length mismatch between query and attn_mask: "
            f"query={len(query)}, attn_mask={len(attn_mask)}"
        )
    if isinstance(attn_mask, NestedTensor) and query.batch_first != attn_mask.batch_first:
        raise ValueError(
            "NestedTensor batch_first mismatch between query and attn_mask: "
            f"query.batch_first={query.batch_first}, attn_mask.batch_first={attn_mask.batch_first}. "
            "Use the same batch_first setting for query, key, value, and attn_mask."
        )
    if not query._storage:
        return NestedTensor([], **query._meta(include_dtype=True))

    # Fast path: convert to torch.nested jagged for variable-length Flash Attention.
    # Only on CUDA where Flash Attention kernels handle jagged efficiently.
    # On CPU, jagged SDPA decomposes to a slow per-element loop.
    if (
        isinstance(key, NestedTensor)
        and isinstance(value, NestedTensor)
        and attn_mask is None
        and _HAS_JAGGED
        and query._storage[0].is_cuda
    ):
        with suppress(RuntimeError, TypeError, ValueError):
            return _sdpa_via_jagged(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

    # Fallback: pad-based SDPA
    q_padded = query.tensor
    k_padded = key.tensor if isinstance(key, NestedTensor) else key
    v_padded = value.tensor if isinstance(value, NestedTensor) else value
    user_attn_mask: NestedTensor | Tensor | None = (
        attn_mask.tensor if isinstance(attn_mask, NestedTensor) else attn_mask
    )

    # Normalize to batch-leading layout for SDPA fallback and mask broadcast.
    if not query.batch_first:
        if q_padded.dim() >= 2:
            q_padded = q_padded.transpose(0, 1)
        if isinstance(k_padded, Tensor) and k_padded.dim() >= 2:
            k_padded = k_padded.transpose(0, 1)
        if isinstance(v_padded, Tensor) and v_padded.dim() >= 2:
            v_padded = v_padded.transpose(0, 1)
        if isinstance(user_attn_mask, Tensor) and user_attn_mask.dim() >= 3:
            user_attn_mask = user_attn_mask.transpose(0, 1)

    combined_mask, is_causal = _build_sdpa_mask(key, user_attn_mask, is_causal, q_padded, k_padded)

    out_padded = F.scaled_dot_product_attention(
        q_padded,
        k_padded,
        v_padded,
        attn_mask=combined_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    if not query.batch_first and out_padded.dim() >= 2:
        out_padded = out_padded.transpose(0, 1)

    return query.nested_like(out_padded, strict=False)


# Normalization


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

    Args:
        input: The input NestedTensor.
        running_mean: The running mean tensor.
        running_var: The running variance tensor.
        weight: Learnable per-channel affine weight (gamma).
        bias: Learnable per-channel affine bias (beta).
        training: Whether in training mode.
        momentum: Value used for running mean and variance computation.
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

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
    if not training:
        # Eval mode is a per-channel affine transform — padding doesn't affect valid positions
        output = F.batch_norm(input.tensor, running_mean, running_var, weight, bias, False, momentum, eps)
        return input.nested_like(output, strict=False)

    # Training mode: concatenate to compute batch statistics without padding contamination
    from .nested_tensor import NestedTensor

    concat, shapes = input.concatenate()
    normalized = F.batch_norm(concat, running_mean, running_var, weight, bias, training, momentum, eps)
    return NestedTensor.from_concatenated(normalized, shapes, **input._meta())


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

    Args:
        input: The input NestedTensor.
        num_groups: Number of groups to separate the channels into.
        weight: Learnable per-channel affine weight (gamma).
        bias: Learnable per-channel affine bias (beta).
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

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

    Args:
        input: The input NestedTensor.
        running_mean: The running mean tensor.
        running_var: The running variance tensor.
        weight: Learnable per-channel affine weight (gamma).
        bias: Learnable per-channel affine bias (beta).
        use_input_stats: If True, use input statistics; otherwise use running stats.
        momentum: Value used for running mean and variance computation.
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

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


@NestedTensorFuncRegistry.implement(F.layer_norm)
def layer_norm(
    input: NestedTensor,
    normalized_shape: tuple,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> NestedTensor:
    r"""
    Applies Layer Normalization over the last certain number of dimensions.
    See also [torch.nn.functional.layer_norm][].

    Args:
        input: The input NestedTensor.
        normalized_shape: Input shape from an expected input of size.
        weight: Learnable affine weight (gamma).
        bias: Learnable affine bias (beta).
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

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
    try:
        output, _, _ = torch.ops.aten.native_layer_norm.default(input, normalized, weight, bias, eps)
        return output
    except (RuntimeError, TypeError, ValueError):
        return _apply_per_element(input, F.layer_norm, normalized, weight, bias, eps)


@NestedTensorFuncRegistry.implement(F.local_response_norm)
def local_response_norm(
    input: NestedTensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
) -> NestedTensor:
    r"""
    Applies local response normalization over an input signal.
    See also [torch.nn.functional.local_response_norm][].

    Args:
        input: The input NestedTensor.
        size: Amount of neighbouring channels used for normalization.
        alpha: Multiplicative factor.
        beta: Exponent.
        k: Additive factor.

    Returns:
        NestedTensor: The normalized output.

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


@NestedTensorFuncRegistry.implement(F.rms_norm)
def rms_norm(
    input: NestedTensor, normalized_shape: tuple, weight: Tensor | None = None, eps: float = 1e-5
) -> NestedTensor:
    r"""
    Applies Root Mean Square Layer Normalization. See also [torch.nn.functional.rms_norm][].

    Args:
        input: The input NestedTensor.
        normalized_shape: Input shape from an expected input of size.
        weight: Learnable affine weight.
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

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
        from .aten_functions import _packed_like

        try:
            out_values = torch.ops.aten.rms_norm.default(input._values, normalized, weight, eps)
            return _packed_like(input, out_values)
        except (RuntimeError, TypeError, ValueError):
            pass
    return _apply_per_element(input, F.rms_norm, normalized, weight, eps)


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

    Args:
        input: The input NestedTensor of log probabilities.
        target: The target tensor or NestedTensor.
        input_lengths: Lengths of the inputs.
        target_lengths: Lengths of the targets.
        blank: Blank label index.
        reduction: Specifies the reduction: 'none', 'mean', or 'sum'.
        zero_infinity: If True, zero infinite losses and associated gradients.

    Returns:
        Tensor: The computed loss.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> input = NestedTensor(torch.tensor([[-1.0, -2.0]]), torch.tensor([[-2.0, -1.0]]))
        >>> target = NestedTensor(torch.tensor([0]), torch.tensor([1]))
        >>> input_lengths = torch.tensor([1, 1])
        >>> target_lengths = torch.tensor([1, 1])
        >>> out = F.ctc_loss(input, target, input_lengths, target_lengths)
        >>> logits = input.tensor.transpose(0, 1)
        >>> ref = F.ctc_loss(logits, target.concat, input_lengths, target_lengths)
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


# Table-driven registrations — loss functions
# All loss functions concat NestedTensor positional args, then call the original function.

_LOSS_OPS_2 = [
    F.binary_cross_entropy,
    F.binary_cross_entropy_with_logits,
    F.cross_entropy,
    F.gaussian_nll_loss,
    F.hinge_embedding_loss,
    F.huber_loss,
    F.kl_div,
    F.l1_loss,
    F.mse_loss,
    F.multi_margin_loss,
    F.multilabel_margin_loss,
    F.multilabel_soft_margin_loss,
    F.nll_loss,
    F.poisson_nll_loss,
    F.smooth_l1_loss,
    F.soft_margin_loss,
]
_LOSS_OPS_3 = [
    F.cosine_embedding_loss,
    F.margin_ranking_loss,
    F.triplet_margin_loss,
    F.triplet_margin_with_distance_loss,
]

for _op, _n in [*((op, 2) for op in _LOSS_OPS_2), *((op, 3) for op in _LOSS_OPS_3)]:

    @NestedTensorFuncRegistry.implement(_op)
    def _loss_impl(*args, _fn=_op, _n=_n, **kwargs):
        tensor_args = _concat_tensors(*args[:_n])
        return _fn(*tensor_args, *args[_n:], **kwargs)


# Table-driven registrations — per-element apply ops
# These need per-element application because they inspect input.dim() or similar.

_APPLY_OPS = [
    F.affine_grid,
    # Shape-changing / dimensionality-inspecting
    F.pdist,
]

for _op in _APPLY_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _apply_impl(input, *args, _fn=_op, **kwargs):
        return _apply_per_element(input, _fn, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.pad)
def _pad_impl(input, pad, mode="constant", value=None):
    pad_tuple = tuple(pad)
    if input._values.dim() > 1 and len(pad_tuple) % 2 == 0:
        padded_dims = len(pad_tuple) // 2
        # Fast path when pad only touches trailing (static) dims, not ragged dim-0.
        if padded_dims < input._values.dim():
            return _apply_batch_preserving_packed(input, F.pad, pad_tuple, mode=mode, value=value)
    return _apply_per_element(input, F.pad, pad_tuple, mode=mode, value=value)


# Batch-preserving packed fast path:
# for ops where per-element dim-0 is interpreted as a batch-like axis, so applying once
# to packed _values is equivalent to per-element application.
_BATCH_PRESERVING_PACKED_OPS = [
    # Conv
    (F.conv1d, (3,)),
    (F.conv2d, (4,)),
    (F.conv3d, (5,)),
    (F.conv_transpose1d, (3,)),
    (F.conv_transpose2d, (4,)),
    (F.conv_transpose3d, (5,)),
    # Pooling
    (F.avg_pool1d, (3,)),
    (F.avg_pool2d, (4,)),
    (F.avg_pool3d, (5,)),
    (F.max_pool1d, (3,)),
    (F.max_pool2d, (4,)),
    (F.max_pool3d, (5,)),
    (F.adaptive_avg_pool1d, (3,)),
    (F.adaptive_avg_pool2d, (4,)),
    (F.adaptive_avg_pool3d, (5,)),
    (F.adaptive_max_pool1d, (3,)),
    (F.adaptive_max_pool2d, (4,)),
    (F.adaptive_max_pool3d, (5,)),
    (F.lp_pool1d, (3,)),
    (F.lp_pool2d, (4,)),
    (F.lp_pool3d, (5,)),
    # Grid / spatial
    (F.fold, (3,)),
    (F.interpolate, (3, 4, 5)),
    (F.unfold, (4,)),
    # Pixel shuffle / channel
    (F.channel_shuffle, (3, 4, 5)),
    (F.pixel_shuffle, (4, 5)),
    (F.pixel_unshuffle, (4, 5)),
]

for _op, _packed_dims in _BATCH_PRESERVING_PACKED_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _apply_batch_preserving_impl(input, *args, _fn=_op, _dims=_packed_dims, **kwargs):
        if input._values.dim() in _dims:
            return _apply_batch_preserving_packed(input, _fn, *args, **kwargs)
        return _apply_per_element(input, _fn, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.one_hot)
def _one_hot_impl(input, num_classes: int = -1):
    if input._values.dim() > 1 or input._shape_tensor.size(1) == 1:
        out_values = F.one_hot(input._values, num_classes=num_classes)
        out_shape_tensor = torch.empty(
            (input._shape_tensor.size(0), input._shape_tensor.size(1) + 1),
            dtype=input._shape_tensor.dtype,
        )
        if out_shape_tensor.numel() > 0:
            out_shape_tensor[:, :-1] = input._shape_tensor
            out_shape_tensor[:, -1] = out_values.shape[-1]
        return type(input)._from_packed(
            out_values,
            input._offsets,
            out_shape_tensor,
            batch_first=input.batch_first,
            padding_value=input.padding_value,
            mask_value=input.mask_value,
            pin_memory=input._pin_memory,
            outer_size=torch.Size((*input._logical_shape, out_values.shape[-1])),
        )
    return _apply_per_element(input, F.one_hot, num_classes=num_classes)


def _validate_dropout_probability(p: float) -> None:
    # Keep F.* API parity: torch.nn.functional dropout variants raise ValueError
    # for out-of-range probabilities.
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")


@NestedTensorFuncRegistry.implement(F.dropout)
def _dropout_impl(input, p=0.5, training=True, inplace=False):
    from .aten_functions import _packed_like

    _validate_dropout_probability(float(p))
    if (not training) or p == 0:
        return input
    if inplace:
        torch.ops.aten.dropout_.default(input._values, p, training)
        input._cached_storage = None
        return input
    return _packed_like(input, torch.ops.aten.dropout.default(input._values, p, training))


@NestedTensorFuncRegistry.implement(F.alpha_dropout)
def _alpha_dropout_impl(input, p=0.5, training=False, inplace=False):
    from .aten_functions import _packed_like

    _validate_dropout_probability(float(p))
    if (not training) or p == 0:
        return input
    if inplace:
        torch.ops.aten.alpha_dropout_.default(input._values, p, training)
        input._cached_storage = None
        return input
    return _packed_like(input, torch.ops.aten.alpha_dropout.default(input._values, p, training))


# Channel-wise dropout: eval no-op fast path, packed training path when layout matches.
_DROPOUT_CHANNELWISE_PACKED_OPS = [
    (F.dropout1d, (2, 3)),
    (F.dropout2d, (3, 4)),
    (F.dropout3d, (4, 5)),
    (F.feature_alpha_dropout, (2, 3, 4, 5)),
]

for _op, _packed_dims in _DROPOUT_CHANNELWISE_PACKED_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _dropout_channelwise_impl(input, *args, _fn=_op, _dims=_packed_dims, **kwargs):
        training = args[1] if len(args) > 1 else kwargs.get("training", True)
        p = args[0] if len(args) > 0 else kwargs.get("p", 0.5)
        _validate_dropout_probability(float(p))
        if (not training) or p == 0:
            return input
        if input._values.dim() in _dims:
            return _apply_batch_preserving_packed(input, _fn, *args, **kwargs)
        return _apply_per_element(input, _fn, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.bilinear)
def _bilinear_impl(input1, input2, weight, bias=None):
    from .aten_functions import _offsets_match, _packed_new_last_dim
    from .nested_tensor import NestedTensor

    cls = type(input1) if isinstance(input1, NestedTensor) else type(input2)
    input1 = _ensure_nested_input(input1, input2, cls)
    if isinstance(input2, NestedTensor):
        if len(input1) != len(input2):
            raise ValueError(
                "NestedTensor batch length mismatch between input1 and input2: "
                f"input1={len(input1)}, input2={len(input2)}"
            )
        offsets_match = True
        is_fake = False
        with suppress(ImportError):
            from torch._subclasses.fake_tensor import is_fake as _is_fake

            is_fake = _is_fake(input1._values) or _is_fake(input2._values)
        if not is_fake:
            offsets_match = _offsets_match(input1._offsets, input2._offsets)
        if input1._values.dim() >= 2 and input2._values.dim() >= 2 and offsets_match:
            out_values = F.bilinear(input1._values, input2._values, weight, bias)
            return _packed_new_last_dim(input1, out_values, int(weight.size(0)))
    return _apply_pair(input1, input2, F.bilinear, weight, bias)


@NestedTensorFuncRegistry.implement(F.grid_sample)
def _grid_sample_impl(input, grid, *args, **kwargs):
    from .aten_functions import _offsets_match
    from .nested_tensor import NestedTensor

    cls = type(input) if isinstance(input, NestedTensor) else type(grid)
    input = _ensure_nested_input(input, grid, cls)
    if isinstance(grid, NestedTensor):
        if len(input) != len(grid):
            raise ValueError(
                "NestedTensor batch length mismatch between input and grid: " f"input={len(input)}, grid={len(grid)}"
            )
        offsets_match = True
        is_fake = False
        with suppress(ImportError):
            from torch._subclasses.fake_tensor import is_fake as _is_fake

            is_fake = _is_fake(input._values) or _is_fake(grid._values)
        if not is_fake:
            offsets_match = _offsets_match(input._offsets, grid._offsets)
        if input._values.dim() in (4, 5) and grid._values.dim() in (4, 5) and offsets_match:
            return _from_batch_preserving_values(input, F.grid_sample(input._values, grid._values, *args, **kwargs))
    elif isinstance(grid, Tensor):
        if input._values.dim() in (4, 5) and grid.dim() in (4, 5) and grid.size(0) == input._values.size(0):
            return _from_batch_preserving_values(input, F.grid_sample(input._values, grid, *args, **kwargs))
    return _apply_pair(input, grid, F.grid_sample, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.pairwise_distance)
def _pairwise_distance_impl(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    from .aten_functions import _offsets_match
    from .nested_tensor import NestedTensor

    cls = type(x1) if isinstance(x1, NestedTensor) else type(x2)
    x1 = _ensure_nested_input(x1, x2, cls)
    if isinstance(x2, NestedTensor):
        if len(x1) != len(x2):
            raise ValueError("NestedTensor batch length mismatch between x1 and x2: " f"x1={len(x1)}, x2={len(x2)}")
        offsets_match = True
        is_fake = False
        with suppress(ImportError):
            from torch._subclasses.fake_tensor import is_fake as _is_fake

            is_fake = _is_fake(x1._values) or _is_fake(x2._values)
        if not is_fake:
            offsets_match = _offsets_match(x1._offsets, x2._offsets)
        if x1._values.dim() == 2 and x2._values.dim() == 2 and offsets_match:
            out_values = F.pairwise_distance(x1._values, x2._values, p=p, eps=eps, keepdim=keepdim)
            if keepdim:
                out_shape_tensor = x1._shape_tensor.clone()
                if out_shape_tensor.numel() > 0 and out_shape_tensor.size(1) > 0:
                    out_shape_tensor[:, -1] = 1
                out_logical = list(x1._logical_shape)
                if out_logical:
                    out_logical[-1] = 1
            else:
                out_shape_tensor = (
                    x1._shape_tensor[:, :-1].clone() if x1._shape_tensor.size(1) > 0 else x1._shape_tensor
                )
                out_logical = list(x1._logical_shape[:-1]) if len(x1._logical_shape) > 0 else list(x1._logical_shape)
            return cls._from_packed(
                out_values,
                x1._offsets,
                out_shape_tensor,
                batch_first=x1.batch_first,
                padding_value=x1.padding_value,
                mask_value=x1.mask_value,
                pin_memory=x1._pin_memory,
                outer_size=torch.Size(out_logical),
            )
    return _apply_pair(x1, x2, F.pairwise_distance, p=p, eps=eps, keepdim=keepdim)


# Table-driven registrations — pool ops that return (output, indices)

_POOL_WITH_INDICES_OPS = [
    F.max_pool1d_with_indices,
    F.max_pool2d_with_indices,
    F.max_pool3d_with_indices,
    F.adaptive_max_pool1d_with_indices,
    F.adaptive_max_pool2d_with_indices,
    F.adaptive_max_pool3d_with_indices,
    F.fractional_max_pool2d_with_indices,
    F.fractional_max_pool3d_with_indices,
]

for _op in _POOL_WITH_INDICES_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _pool_indices_impl(input, *args, _fn=_op, **kwargs):
        return _map_storage_pair(input, _fn, *args, **kwargs)


# Table-driven registrations — max unpool ops (binary per-element: input + indices)

_MAX_UNPOOL_OPS = [F.max_unpool1d, F.max_unpool2d, F.max_unpool3d]

for _op in _MAX_UNPOOL_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _unpool_impl(input, indices, *args, _fn=_op, **kwargs):
        return _apply_pair(input, indices, _fn, *args, **kwargs)


# Softmax family


@NestedTensorFuncRegistry.implement(F.softmax)
def _f_softmax_impl(input, dim=-1, _stacklevel=3, dtype=None):
    del _stacklevel  # keep F.softmax-compatible signature
    return torch.softmax(input, dim=dim, dtype=dtype)


@NestedTensorFuncRegistry.implement(F.log_softmax)
def _f_log_softmax_impl(input, dim=-1, _stacklevel=3, dtype=None):
    del _stacklevel  # keep F.log_softmax-compatible signature
    return torch.log_softmax(input, dim=dim, dtype=dtype)


@NestedTensorFuncRegistry.implement(F.softmin)
def _f_softmin_impl(input, dim=-1, _stacklevel=3, dtype=None):
    del _stacklevel  # keep F.softmin-compatible signature
    source = input if dtype is None else input.to(dtype=dtype)
    return torch.softmax(torch.neg(source), dim=dim, dtype=None)


@NestedTensorFuncRegistry.implement(F.gumbel_softmax)
def _gumbel_softmax_impl(logits, *args, dim=-1, **kwargs):
    from .aten_functions import _packed_like, _packed_to_padded

    dim_adj = _translate_non_batch_dim(logits, dim)
    if dim_adj == 0:
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(logits, fill_value=float("-inf"))
        out_padded = F.gumbel_softmax(padded, *args, dim=1, **kwargs)
        return _packed_like(logits, out_padded[batch_idx, local_idx])
    concat_dim = _concat_dim_for_tensor_dim(logits, dim_adj)
    if concat_dim is not None:
        return _apply_packed(logits, F.gumbel_softmax, *args, dim=concat_dim, **kwargs)
    return _apply_per_element(logits, F.gumbel_softmax, *args, dim=dim_adj, **kwargs)


# F.normalize — needs dim translation


@NestedTensorFuncRegistry.implement(F.normalize)
def _normalize_impl(input, p=2.0, dim=1, eps=1e-12, out=None):
    dim_adj = _translate_non_batch_dim(input, dim)
    if dim_adj == 0:
        if out is not None:
            raise NotImplementedError("F.normalize(..., out=...) is not supported on ragged dimensions.")
        if isinstance(p, (int, float)) and p > 0:
            from .aten_functions import _packed_to_padded

            padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(input, fill_value=0.0)
            denom = torch.linalg.vector_norm(padded, ord=float(p), dim=1, keepdim=True)
            denom = torch.clamp(denom, min=eps)
            return type(input)._from_packed(
                (padded / denom)[batch_idx, local_idx],
                input._offsets,
                input._shape_tensor,
                batch_first=input.batch_first,
                padding_value=input.padding_value,
                mask_value=input.mask_value,
                pin_memory=input._pin_memory,
                outer_size=input._logical_shape,
            )
        return _apply_per_element(input, F.normalize, p=p, dim=dim_adj, eps=eps, out=None)
    concat_dim = _concat_dim_for_tensor_dim(input, dim_adj)
    if concat_dim is None:
        if out is not None:
            raise NotImplementedError("F.normalize(..., out=...) is not supported on ragged dimensions.")
        return _apply_per_element(input, F.normalize, p=p, dim=dim_adj, eps=eps, out=None)
    return _apply_packed(input, F.normalize, p=p, dim=concat_dim, eps=eps, out=out)


# Fractional max pool — boolean_dispatch wrappers that support return_indices


@NestedTensorFuncRegistry.implement(F.fractional_max_pool2d)
def _fractional_max_pool2d_impl(input, *args, return_indices=False, **kwargs):
    if return_indices:
        return _apply_with_indices(input, F.fractional_max_pool2d, *args, **kwargs)
    random_samples = kwargs.get("_random_samples")
    random_samples_ok = random_samples is None or (
        isinstance(random_samples, Tensor) and random_samples.size(0) == input._values.size(0)
    )
    if input._values.dim() == 4 and random_samples_ok:
        return _apply_batch_preserving_packed(input, F.fractional_max_pool2d, *args, **kwargs)
    return _apply_per_element(input, F.fractional_max_pool2d, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.fractional_max_pool3d)
def _fractional_max_pool3d_impl(input, *args, return_indices=False, **kwargs):
    if return_indices:
        return _apply_with_indices(input, F.fractional_max_pool3d, *args, **kwargs)
    random_samples = kwargs.get("_random_samples")
    random_samples_ok = random_samples is None or (
        isinstance(random_samples, Tensor) and random_samples.size(0) == input._values.size(0)
    )
    if input._values.dim() == 5 and random_samples_ok:
        return _apply_batch_preserving_packed(input, F.fractional_max_pool3d, *args, **kwargs)
    return _apply_per_element(input, F.fractional_max_pool3d, *args, **kwargs)
