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
Core [NestedTensor][] class: packed variable-length tensor with
``__torch_function__`` / ``__torch_dispatch__`` integration.

This module defines the storage contract (``_values``, ``_offsets``,
``_physical_shape``, ``_permutation``), construction, metadata management,
materialization (``.tensor``, ``.mask``, ``.concat``), serialization,
and dispatch entry points.
"""

# pylint: disable=protected-access
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Iterable, SupportsFloat, cast

import torch
from torch import Tensor

from .aten_functions import _is_fake_tensor, per_element_fallback
from .ops import (
    NestedTensorAtenRegistry,
    _batch_leading_valid_mask_from_sizes,
    _check_execution_guard,
    _ExecutionGuardKind,
)

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from torch import nested


class NestedTensor(torch.Tensor):
    r"""
    A container for variable-length tensors that enables efficient batch operations.

    `NestedTensor` solves a fundamental problem in deep learning: handling sequences of different lengths
    in batch operations. Instead of excessive padding or complex bucketing, `NestedTensor` provides an
    elegant solution that maintains both efficiency and usability.

    The class provides three main views of the data:
    - `.tensor`: A padded tensor with zeros (or other value) in place of missing elements
    - `.mask`: A boolean mask indicating which elements are real vs padding
    - `.concat`: The packed tensor containing all elements concatenated without padding

    When indexing a `NestedTensor`, the behavior depends on the index type:
    1. Integer index (`nt[0]`): Returns a single tensor without padding
    2. Slice index (`nt[:]`): Returns a new `NestedTensor` containing the selected batch elements
    3. Tuple index (`nt[:, 1:]`): Returns a new `NestedTensor` with the specified sliced shape

    Attributes:
        _values: Packed tensor data
        _offsets: Top-level cumulative element counts, shape (B+1,)
        _permutation: Canonical logical-to-packed dimension permutation
        _physical_shape: Per-element physical shapes, shape (B, max_ndim)
        batch_first: Whether the first dimension is the batch dimension (B, N, *)
            If `False`, the first dimension is the sequence dimension (N, B, *)
        padding_value: Value used for padding in the padded tensor
        mask_value: Boolean fill value for padding positions in generated masks.
            - ``mask_value=False`` (default): valid positions are ``True`` and padding is ``False``.
            - ``mask_value=True``: padding positions are ``True`` and valid positions are ``False``.

    Args:
        *tensors: Variable-length tensors or sequences to store
        batch_first: Whether to use batch-first representation.
        padding_value: Value to use for padding.
        mask_value: Boolean fill value used for padding positions in masks.

    Raises:
        ValueError: If `tensors` is not an iterable

    Examples:
        Basic usage:
        >>> nested_tensor = NestedTensor(torch.tensor([1, 2, 3]), torch.tensor([4, 5]))
        >>> nested_tensor.shape
        torch.Size([2, 3])
        >>> nested_tensor.tensor  # Padded representation
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> nested_tensor.mask  # Mask showing real vs padding values
        tensor([[ True,  True,  True],
                [ True,  True, False]])
        >>> nested_tensor.concat  # Concatenated version (no padding)
        tensor([1, 2, 3, 4, 5])

        Indexing:
        >>> nested_tensor[0]  # First tensor (no padding)
        tensor([1, 2, 3])
        >>> nested_tensor[:2]  # Returns a NestedTensor slice
        NestedTensor([
            [1, 2, 3],
            [4, 5]
        ])
        >>> nested_tensor[:, 1:]  # Slice operations return a new NestedTensor
        NestedTensor([
            [2, 3],
            [5]
        ])

        Type conversion:
        >>> nested_tensor.to(torch.float).tensor
        tensor([[1., 2., 3.],
                [4., 5., 0.]])
        >>> nested_tensor.half().tensor
        tensor([[1., 2., 3.],
                [4., 5., 0.]], dtype=torch.float16)

        Conversion to Python types:
        >>> nested_tensor.tolist()
        [[1, 2, 3], [4, 5]]

        Creating from Python lists:
        >>> NestedTensor(*[[1, 2, 3], [4, 5]])
        NestedTensor([
            [1, 2, 3],
            [4, 5]
        ])
    """

    _values: Tensor
    _offsets: Tensor
    _permutation: tuple[int, ...]
    _physical_shape: Tensor
    _flatten_sentinel: Tensor = torch.empty(0)
    _logical_shape: torch.Size
    _batch_first: bool
    _padding_value: float
    _mask_value: bool
    _pin_memory: bool
    _packed_sizes: tuple[int, ...] | None
    _element_shapes: tuple[tuple[int, ...], ...] | None
    _cached_storage: tuple[Tensor, ...] | None
    _cached_hierarchical_offsets: tuple[Tensor, ...] | None
    _cached_tensor_view: tuple[bool, float, tuple[int, int, int], Tensor] | None
    _cached_mask_view: tuple[bool, bool, tuple[int, int], Tensor] | None
    _SERIALIZATION_VERSION = 1

    # Construction & Initialization

    @staticmethod
    def __new__(
        cls,
        *tensors: Iterable[Tensor],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool | None = None,
        pin_memory: bool = False,
        batch_first: bool = True,
        padding_value: SupportsFloat = 0.0,
        mask_value: bool = False,
    ):
        if len(tensors) == 1 and not isinstance(tensors[0], Tensor):
            if isinstance(tensors[0], Iterable):
                tensors = tuple(tensors[0])  # type: ignore
            else:
                raise ValueError(f"tensors must be an Iterable, but got {type(tensors[0])}.")

        # Validate and convert tensors
        validated = cls._coerce_tensors(
            tensors, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory
        )

        # Determine dtype/device from validated tensors or fallbacks
        out_dtype = validated[0].dtype if validated else (dtype or torch.get_default_dtype())
        out_device = validated[0].device if validated else (device or torch.device("cpu"))

        # Pack into values, offsets, tensor-shape metadata, and Python metadata.
        values, offsets, shape_tensor, packed_sizes, element_shapes = cls._pack(
            validated,
            dtype=out_dtype,
            device=out_device,
        )
        values = cls._maybe_pin_values(values, pin_memory)
        permutation = cls._permutation_from_element_shapes(element_shapes)

        # Compute logical shape
        logical_shape = cls._compute_logical_shape(validated, batch_first)
        if requires_grad is not None and values.requires_grad != requires_grad:
            values.requires_grad_(requires_grad)
        out_requires_grad = values.requires_grad

        result = torch.Tensor._make_wrapper_subclass(
            cls,
            logical_shape,
            dtype=out_dtype,
            device=out_device,
            requires_grad=out_requires_grad,
        )
        result._values = values
        result._offsets = offsets
        result._permutation = permutation
        result._physical_shape = shape_tensor
        result._logical_shape = logical_shape
        result._set_runtime_config(
            batch_first=batch_first,
            padding_value=padding_value,
            mask_value=mask_value,
        )
        result._pin_memory = bool(pin_memory and values.device.type == "cpu" and values.is_pinned())
        result._packed_sizes = packed_sizes
        result._element_shapes = element_shapes
        result._invalidate_transient_caches()
        cls._validate_packed_metadata(
            result._values,
            result._offsets,
            result._physical_shape,
            permutation=result._permutation,
            logical_shape=result._logical_shape,
            batch_first=result.batch_first,
            packed_sizes=result._packed_sizes,
            element_shapes=result._element_shapes,
        )
        return result

    def __init__(self, *args, **kwargs):
        pass  # All init in __new__

    # ------------------------------------------------------------------
    # Packed representation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_tensors(
        tensors: tuple,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool | None = None,
        pin_memory: bool = False,
    ) -> tuple[Tensor, ...]:
        if not isinstance(tensors, Iterable):
            raise ValueError(f"tensors must be an Iterable, but got {type(tensors)}.")
        if isinstance(tensors, Tensor) and hasattr(tensors, "unbind"):
            tensors = tensors.unbind()

        result: list[Tensor] = []
        common_device: torch.device | None = None
        common_ndim: int | None = None
        # Only track dtype promotion when the caller did not specify an explicit dtype.
        # When dtype is given, t.to(device, dtype=dtype) already handles casting in
        # the first pass, so the promotion loop and second pass are both unnecessary.
        needs_promotion = dtype is None
        common_dtype: torch.dtype | None = None

        for t in tensors:
            if not isinstance(t, Tensor):
                t = torch.tensor(t, dtype=dtype, device=device, pin_memory=pin_memory)
            else:
                t = t.to(device, dtype=dtype)
            if requires_grad is not None:
                t.requires_grad_(requires_grad)

            if common_device is None:
                common_device = t.device
            elif t.device != common_device:
                raise ValueError(
                    f"All tensors in NestedTensor must be on the same device, but got {common_device} and {t.device}"
                )

            if needs_promotion:
                if common_dtype is None:
                    common_dtype = t.dtype
                else:
                    common_dtype = torch.promote_types(common_dtype, t.dtype)

            if common_ndim is None:
                common_ndim = t.ndim
            elif t.ndim != common_ndim:
                raise ValueError(
                    f"All tensors must have the same number of dimensions, got ndim {common_ndim} and {t.ndim}. "
                    "If using a DataLoader with drop_last=False, squeeze the last batch before constructing "
                    "NestedTensor."
                )

            result.append(t)

        if not result:
            return ()

        # Second pass only when dtype=None AND promotion actually changed the dtype.
        if needs_promotion and common_dtype is not None and any(t.dtype != common_dtype for t in result):
            return tuple(t.to(dtype=common_dtype) for t in result)
        return tuple(result)

    @staticmethod
    def _pack(
        tensors: tuple[Tensor, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        permutation: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...], tuple[tuple[int, ...], ...]]:
        r"""Pack a sequence of tensors into values, offsets, tensor metadata, and Python metadata."""
        if not tensors:
            return (
                torch.empty(0, dtype=dtype or torch.get_default_dtype(), device=device),
                torch.zeros(1, dtype=torch.long),
                torch.empty(0, 0, dtype=torch.long),
                (),
                (),
            )

        max_ndim = max(t.ndim for t in tensors)
        element_shapes = tuple(tuple(int(dim) for dim in t.shape) for t in tensors)

        # Offsets and shape_tensor are metadata - always on CPU to avoid CUDA syncs.
        shape_tensor = torch.tensor([list(t.shape) + [0] * (max_ndim - t.ndim) for t in tensors], dtype=torch.long)
        if max_ndim == 0:
            values = torch.stack(tensors)
            sizes = torch.ones(len(tensors), dtype=torch.long)
            packed_sizes = tuple(1 for _ in tensors)
        else:
            if permutation is None:
                varying_dims, static_dims = NestedTensor._pack_layout_from_element_shapes(element_shapes)
                permutation = varying_dims + static_dims
            else:
                permutation = tuple(int(dim) for dim in permutation)
                if len(permutation) != max_ndim or tuple(sorted(permutation)) != tuple(range(max_ndim)):
                    raise ValueError(f"Invalid permutation dims {permutation} for tensors with rank {max_ndim}")
                ragged_rank = len(NestedTensor._hierarchical_level_sizes_from_element_shapes(element_shapes))
                varying_dims = permutation[:ragged_rank]
                static_dims = permutation[ragged_rank:]
            packed = []
            packed_sizes_list = []
            identity_permutation = tuple(range(max_ndim))
            for tensor, shape in zip(tensors, element_shapes):
                packed_size = NestedTensor._packed_size_from_shape(shape, varying_dims)
                packed_sizes_list.append(packed_size)
                packed_tensor = tensor if permutation == identity_permutation else tensor.permute(permutation)
                suffix_shape = tuple(shape[dim] for dim in static_dims)
                packed.append(packed_tensor.reshape((packed_size, *suffix_shape) if suffix_shape else (packed_size,)))
            values = torch.cat(packed, dim=0)
            sizes = torch.tensor(packed_sizes_list, dtype=torch.long)
            packed_sizes = tuple(packed_sizes_list)
        offsets = torch.zeros(len(tensors) + 1, dtype=torch.long)
        torch.cumsum(sizes, dim=0, out=offsets[1:])

        return values, offsets, shape_tensor, packed_sizes, element_shapes

    @staticmethod
    def _maybe_pin_values(values: Tensor, pin_memory: bool) -> Tensor:
        r"""Pin packed storage when requested and the values live on CPU."""
        if pin_memory and values.device.type == "cpu" and not values.is_pinned():
            return values.pin_memory()
        return values

    @staticmethod
    def _trim_shape(shape: Sequence[int]) -> tuple[int, ...]:
        end = len(shape)
        while end > 0 and shape[end - 1] == 0:
            end -= 1
        return tuple(int(shape[i]) for i in range(end))

    @staticmethod
    def _shape_numel(shape: tuple[int, ...]) -> int:
        size = 1
        for dim in shape:
            size *= int(dim)
        return size

    @classmethod
    def _permutation_from_element_shapes(cls, element_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        varying_dims, static_dims = cls._pack_layout_from_element_shapes(element_shapes)
        return varying_dims + static_dims

    @classmethod
    def _permutation_from_physical_shape(
        cls,
        physical_shape: Tensor,
        element_shapes: tuple[tuple[int, ...], ...] | None,
    ) -> tuple[int, ...]:
        varying_dims, static_dims = cls._pack_layout_meta(physical_shape, element_shapes)
        return varying_dims + static_dims

    @staticmethod
    def _offsets_from_sizes(sizes: Sequence[int], *, dtype: torch.dtype = torch.long) -> Tensor:
        offsets = torch.empty((len(sizes) + 1,), dtype=dtype)
        offsets[0] = 0
        if sizes:
            offsets[1:] = torch.cumsum(torch.tensor(sizes, dtype=dtype), dim=0)
        return offsets

    @staticmethod
    def _meta_tensor_equal(lhs: Tensor, rhs: Tensor) -> bool:
        if _is_fake_tensor(lhs) or _is_fake_tensor(rhs):
            return lhs is rhs
        if lhs is rhs:
            return True
        if lhs.shape != rhs.shape:
            return False
        return bool(torch.equal(lhs, rhs))

    @classmethod
    def _hierarchical_level_sizes_from_element_shapes(
        cls,
        element_shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        if not element_shapes:
            return ()
        varying_dims, _ = cls._pack_layout_from_element_shapes(element_shapes)
        if not varying_dims:
            return ()

        level_sizes: list[tuple[int, ...]] = []
        prefix_products = [1] * len(element_shapes)
        for dim in varying_dims:
            sizes: list[int] = []
            next_prefix_products: list[int] = []
            for shape, prefix in zip(element_shapes, prefix_products):
                dim_size = int(shape[dim])
                sizes.extend([dim_size] * prefix)
                next_prefix_products.append(prefix * dim_size)
            level_sizes.append(tuple(sizes))
            prefix_products = next_prefix_products
        return tuple(level_sizes)

    @classmethod
    def _hierarchical_level_sizes_from_physical_shape(
        cls,
        physical_shape: Tensor,
        element_shapes: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        if physical_shape.numel() == 0:
            return ()
        if element_shapes is not None:
            return cls._hierarchical_level_sizes_from_element_shapes(element_shapes)
        if _is_fake_tensor(physical_shape):
            return ()

        varying_dims, _ = cls._pack_layout_meta(physical_shape, None)
        if not varying_dims:
            return ()

        shape_rows = tuple(cls._trim_shape(row) for row in physical_shape.tolist())
        level_sizes: list[tuple[int, ...]] = []
        prefix_products = [1] * len(shape_rows)
        for dim in varying_dims:
            sizes: list[int] = []
            next_prefix_products: list[int] = []
            for shape, prefix in zip(shape_rows, prefix_products):
                dim_size = int(shape[dim]) if dim < len(shape) else 0
                sizes.extend([dim_size] * prefix)
                next_prefix_products.append(prefix * dim_size)
            level_sizes.append(tuple(sizes))
            prefix_products = next_prefix_products
        return tuple(level_sizes)

    @staticmethod
    def _inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
        inverse = [0] * len(permutation)
        for index, dim in enumerate(permutation):
            inverse[dim] = index
        return tuple(inverse)

    @classmethod
    def _pack_layout_from_element_shapes(
        cls,
        element_shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not element_shapes:
            return (), ()
        ndim = len(element_shapes[0])
        if ndim == 0:
            return (), ()
        reference = element_shapes[0]
        static_dims = [
            dim
            for dim in range(ndim)
            if all(len(shape) == ndim and shape[dim] == reference[dim] for shape in element_shapes[1:])
        ]
        if len(static_dims) == ndim:
            static_dims = list(range(1, ndim))
        static_dims_tuple = tuple(static_dims)
        varying_dims = tuple(dim for dim in range(ndim) if dim not in static_dims_tuple)
        return varying_dims, static_dims_tuple

    @classmethod
    def _pack_layout_meta(
        cls,
        physical_shape: Tensor,
        element_shapes: tuple[tuple[int, ...], ...] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if element_shapes is not None and (element_shapes or int(physical_shape.size(1)) == 0):
            return cls._pack_layout_from_element_shapes(element_shapes)
        ndim = int(physical_shape.size(1))
        if ndim == 0:
            return (), ()
        if physical_shape.size(0) == 0:
            return (0,), tuple(range(1, ndim))
        static_dims = tuple(
            dim
            for dim in range(ndim)
            if bool(torch.equal(physical_shape[:, dim], physical_shape[:1, dim].expand(physical_shape.size(0))))
        )
        if len(static_dims) == ndim:
            static_dims = tuple(range(1, ndim))
        varying_dims = tuple(dim for dim in range(ndim) if dim not in static_dims)
        return varying_dims, static_dims

    @staticmethod
    def _packed_size_from_shape(shape: tuple[int, ...], varying_dims: tuple[int, ...]) -> int:
        if not shape or not varying_dims:
            return 1
        size = 1
        for dim in varying_dims:
            size *= int(shape[dim])
        return size

    @classmethod
    def _python_meta_from_packed(
        cls,
        values: Tensor,
        offsets: Tensor,
        shape_tensor: Tensor,
        *,
        packed_sizes: tuple[int, ...] | None = None,
        element_shapes: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        if packed_sizes is None:
            packed_sizes = tuple(int(size) for size in (offsets[1:] - offsets[:-1]).tolist())
        if element_shapes is None:
            element_shapes = tuple(cls._trim_shape(shape) for shape in shape_tensor.tolist())
        return packed_sizes, element_shapes

    @classmethod
    @torch._dynamo.disable
    def _infer_python_meta_from_packed(
        cls,
        values: Tensor,
        offsets: Tensor,
        shape_tensor: Tensor,
        *,
        packed_sizes: tuple[int, ...] | None = None,
        element_shapes: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        return cls._python_meta_from_packed(
            values,
            offsets,
            shape_tensor,
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
        )

    @staticmethod
    def _compute_logical_shape(tensors: tuple[Tensor, ...], batch_first: bool) -> torch.Size:
        r"""Compute the logical shape [B, max_d0, max_d1, ...] from individual tensors."""
        if not tensors:
            return torch.Size((0,))
        if max(t.dim() for t in tensors) == 0:
            return torch.Size((len(tensors),))
        ndim = max(t.dim() for t in tensors)
        size = [max(t.shape[i] if i < len(t.shape) else 0 for t in tensors) for i in range(ndim)]
        size.insert(0 if batch_first else 1, len(tensors))
        return torch.Size(size)

    @staticmethod
    def _logical_shape_from_physical_shape(physical_shape: Tensor, offsets: Tensor, batch_first: bool) -> torch.Size:
        r"""Compute logical shape from packed metadata without unpacking elements."""
        batch_size = len(offsets) - 1
        if batch_size == 0:
            return torch.Size((0,))
        if physical_shape.numel() == 0:
            return torch.Size((batch_size,))
        size = [int(physical_shape[:, d].max().item()) for d in range(physical_shape.size(1))]
        while size and size[-1] == 0:
            size.pop()
        size.insert(0 if batch_first else 1, batch_size)
        return torch.Size(size)

    @staticmethod
    def _batch_dim_from_logical_shape(logical_shape: torch.Size, batch_first: bool) -> int:
        r"""Return the batch dimension index for a logical NestedTensor shape."""
        return 0 if len(logical_shape) <= 1 or batch_first else 1

    @classmethod
    def _validate_packed_metadata(
        cls,
        values: Tensor,
        offsets: Tensor,
        shape_tensor: Tensor,
        *,
        permutation: tuple[int, ...],
        logical_shape: torch.Size,
        batch_first: bool,
        packed_sizes: tuple[int, ...] | None,
        element_shapes: tuple[tuple[int, ...], ...] | None,
    ) -> None:
        r"""Validate that packed storage and metadata describe a coherent NestedTensor layout."""
        if offsets.device.type != "cpu":
            raise ValueError(f"offsets must be on CPU, got {offsets.device}")
        if shape_tensor.device.type != "cpu":
            raise ValueError(f"shape_tensor must be on CPU, got {shape_tensor.device}")
        if offsets.dim() != 1:
            raise ValueError(f"offsets must be 1-D, got shape {tuple(offsets.shape)}")
        if shape_tensor.dim() != 2:
            raise ValueError(f"shape_tensor must be 2-D, got shape {tuple(shape_tensor.shape)}")
        if offsets.dtype.is_floating_point or offsets.dtype.is_complex or offsets.dtype == torch.bool:
            raise ValueError(f"offsets must use an integer dtype, got {offsets.dtype}")
        if shape_tensor.dtype.is_floating_point or shape_tensor.dtype.is_complex or shape_tensor.dtype == torch.bool:
            raise ValueError(f"shape_tensor must use an integer dtype, got {shape_tensor.dtype}")

        batch_size = int(shape_tensor.size(0))
        if offsets.numel() != batch_size + 1:
            raise ValueError(
                "offsets length must equal batch size + 1, got "
                f"offsets.numel()={offsets.numel()}, batch_size={batch_size}"
            )

        physical_rank = int(shape_tensor.size(1))
        if len(logical_shape) != physical_rank + 1:
            raise ValueError(
                "logical shape rank must equal physical rank + 1, got "
                f"logical rank={len(logical_shape)}, physical rank={physical_rank}"
            )
        batch_dim = cls._batch_dim_from_logical_shape(logical_shape, batch_first)
        logical_batch = logical_shape[batch_dim]
        if logical_batch != batch_size:
            raise ValueError(f"logical batch size {logical_batch} does not match metadata batch size {batch_size}")

        if len(permutation) != physical_rank or tuple(sorted(int(dim) for dim in permutation)) != tuple(
            range(physical_rank)
        ):
            raise ValueError(f"Invalid permutation dims {permutation} for shape with {physical_rank} dims")

        if packed_sizes is not None:
            if len(packed_sizes) != batch_size:
                raise ValueError(
                    f"packed_sizes must have one entry per element, got {len(packed_sizes)} for batch size {batch_size}"
                )
            if any(int(size) < 0 for size in packed_sizes):
                raise ValueError("packed_sizes must be non-negative")
            if sum(int(size) for size in packed_sizes) != int(values.shape[0]):
                raise ValueError("packed_sizes must sum to the packed values length")

        if element_shapes is not None:
            if len(element_shapes) != batch_size:
                raise ValueError(
                    "element_shapes must have one entry per element, got "
                    f"{len(element_shapes)} for batch size {batch_size}"
                )
            normalized_shapes = tuple(tuple(int(dim) for dim in shape) for shape in element_shapes)
            if any(len(shape) != physical_rank for shape in normalized_shapes):
                raise ValueError(
                    f"element_shapes rank must match physical rank {physical_rank}, got {normalized_shapes}"
                )
            if any(any(dim < 0 for dim in shape) for shape in normalized_shapes):
                raise ValueError("element_shapes must be non-negative")
            if not _is_fake_tensor(shape_tensor):
                shape_rows = tuple(tuple(int(size) for size in row) for row in shape_tensor.tolist())
                if normalized_shapes != shape_rows:
                    raise ValueError("element_shapes must match shape_tensor exactly")

        if _is_fake_tensor(offsets) or _is_fake_tensor(shape_tensor):
            return

        if bool((shape_tensor < 0).any()):
            raise ValueError("shape_tensor must be non-negative")
        if int(offsets[0].item()) != 0:
            raise ValueError("offsets must start at 0")
        deltas = offsets[1:] - offsets[:-1]
        if bool((deltas < 0).any()):
            raise ValueError("offsets must be monotonically non-decreasing")
        if packed_sizes is None and int(offsets[-1].item()) != int(values.shape[0]):
            raise ValueError(
                f"offsets[-1] must equal packed values length, got offsets[-1]={int(offsets[-1].item())} "
                f"and values.shape[0]={int(values.shape[0])}"
            )

    def _validate_metadata(self) -> None:
        r"""Validate the current packed storage and metadata."""
        type(self)._validate_packed_metadata(
            self._values,
            self._offsets,
            self._physical_shape,
            permutation=self._permutation,
            logical_shape=self._logical_shape,
            batch_first=self.batch_first,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
        )

    @staticmethod
    def _coerce_batch_first(value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"batch_first must be a bool, got {type(value).__name__}")
        return value

    @staticmethod
    def _coerce_mask_value(value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"mask_value must be a bool, got {type(value).__name__}")
        return value

    @staticmethod
    def _coerce_padding_value(value: SupportsFloat) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"padding_value must be float-convertible, got {type(value).__name__}") from exc

    def _set_runtime_config(
        self,
        *,
        batch_first: bool,
        padding_value: SupportsFloat,
        mask_value: bool,
    ) -> None:
        self._batch_first = type(self)._coerce_batch_first(batch_first)
        self._padding_value = type(self)._coerce_padding_value(padding_value)
        self._mask_value = type(self)._coerce_mask_value(mask_value)

    def _invalidate_transient_caches(self) -> None:
        r"""Drop all lazily materialized views derived from packed storage."""
        self._cached_storage = None
        self._cached_hierarchical_offsets = None
        self._cached_tensor_view = None
        self._cached_mask_view = None

    def _values_cache_token(self) -> tuple[int, int, int]:
        r"""Return a cache token for views that depend on packed values and layout metadata.

        Under ``torch.inference_mode`` tensors do not track version counters and
        in-place mutation is forbidden, so the cache is always valid.
        """
        if torch.is_inference_mode_enabled():
            return (0, 0, 0)
        return (int(self._values._version), int(self._offsets._version), int(self._physical_shape._version))

    def _shape_cache_token(self) -> tuple[int, int]:
        r"""Return a cache token for views that depend only on shape metadata."""
        if torch.is_inference_mode_enabled():
            return (0, 0)
        return (int(self._offsets._version), int(self._physical_shape._version))

    @classmethod
    def _validate_serialized_state(cls, state: Mapping) -> None:
        required = (
            "_state_version",
            "_values",
            "_offsets",
            "_permutation",
            "_physical_shape",
            "_logical_shape",
            "batch_first",
            "padding_value",
            "mask_value",
            "_pin_memory",
            "_packed_sizes",
            "_element_shapes",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"Serialized NestedTensor state is missing required keys: {', '.join(missing)}")
        version = state["_state_version"]
        if version != cls._SERIALIZATION_VERSION:
            raise ValueError(f"Unsupported NestedTensor state version {version}; expected {cls._SERIALIZATION_VERSION}")

    @classmethod
    @torch._dynamo.disable
    def _from_packed(
        cls,
        values: Tensor,
        offsets: Tensor,
        shape_tensor: Tensor,
        *,
        permutation: tuple[int, ...] | None = None,
        batch_first: bool = True,
        padding_value: float = 0.0,
        mask_value: bool = False,
        pin_memory: bool = False,
        outer_size: torch.Size | tuple | None = None,
        packed_sizes: tuple[int, ...] | None = None,
        element_shapes: tuple[tuple[int, ...], ...] | None = None,
    ) -> Self:
        r"""Construct a NestedTensor directly from packed representation."""
        # offsets and shape_tensor MUST live on CPU to avoid implicit CUDA syncs
        # when handlers call .item() / .tolist() on them.
        if offsets.device.type != "cpu":
            raise ValueError(f"offsets must be on CPU, got {offsets.device}")
        if shape_tensor.device.type != "cpu":
            raise ValueError(f"shape_tensor must be on CPU, got {shape_tensor.device}")

        if outer_size is not None:
            logical_shape = torch.Size(outer_size)
        else:
            logical_shape = cls._logical_shape_from_physical_shape(shape_tensor, offsets, batch_first)
        if packed_sizes is None and not _is_fake_tensor(offsets):
            packed_sizes = tuple(int(size) for size in (offsets[1:] - offsets[:-1]).tolist())
        if element_shapes is None and not _is_fake_tensor(shape_tensor):
            element_shapes = tuple(cls._trim_shape(shape) for shape in shape_tensor.tolist())

        if _is_fake_tensor(values) and not (_is_fake_tensor(offsets) and _is_fake_tensor(shape_tensor)):
            from torch._subclasses.fake_tensor import maybe_get_fake_mode

            fake_mode = maybe_get_fake_mode(values)
            if fake_mode is not None:
                if not _is_fake_tensor(offsets):
                    offsets = fake_mode.from_tensor(offsets, static_shapes=True, trace=False)
                if not _is_fake_tensor(shape_tensor):
                    shape_tensor = fake_mode.from_tensor(shape_tensor, static_shapes=True, trace=False)

        values = cls._maybe_pin_values(values, pin_memory)
        result = torch.Tensor._make_wrapper_subclass(
            cls,
            logical_shape,
            dtype=values.dtype,
            device=values.device,
            requires_grad=values.requires_grad,
        )
        result._values = values
        result._offsets = offsets
        result._permutation = (
            tuple(int(dim) for dim in permutation)
            if permutation is not None
            else cls._permutation_from_physical_shape(shape_tensor, element_shapes)
        )
        result._physical_shape = shape_tensor
        result._logical_shape = logical_shape
        result._set_runtime_config(
            batch_first=batch_first,
            padding_value=padding_value,
            mask_value=mask_value,
        )
        result._pin_memory = bool(pin_memory and values.device.type == "cpu" and values.is_pinned())
        result._packed_sizes = packed_sizes
        result._element_shapes = element_shapes
        result._invalidate_transient_caches()
        cls._validate_packed_metadata(
            result._values,
            result._offsets,
            result._physical_shape,
            permutation=result._permutation,
            logical_shape=result._logical_shape,
            batch_first=result.batch_first,
            packed_sizes=result._packed_sizes,
            element_shapes=result._element_shapes,
        )
        return result

    # ------------------------------------------------------------------
    # torch.compile support
    # ------------------------------------------------------------------

    def __tensor_flatten__(self):
        # During tracing, wrapper instances can be inspected while being built.
        # Only expose tensor attrs that already exist so Dynamo/FakeTensor can
        # inspect partially constructed wrapper subclasses safely.
        instance_attrs = vars(self)
        inner_tensors = [name for name in ("_values", "_offsets", "_physical_shape") if name in instance_attrs]
        if not inner_tensors:
            inner_tensors = ["_flatten_sentinel"]
        return inner_tensors, {
            "batch_first": getattr(self, "batch_first", True),
            "padding_value": getattr(self, "padding_value", 0.0),
            "mask_value": getattr(self, "mask_value", False),
            "pin_memory": getattr(self, "_pin_memory", False),
            "packed_sizes": getattr(self, "_packed_sizes", ()),
            "element_shapes": getattr(self, "_element_shapes", ()),
            "permutation": getattr(self, "_permutation", ()),
        }

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, ctx, outer_size, outer_stride):
        values = inner_tensors.get("_values", inner_tensors.get("_flatten_sentinel"))
        if values is None:
            raise RuntimeError("NestedTensor requires _values during tensor unflatten.")

        offsets = inner_tensors.get("_offsets")
        shape_tensor = inner_tensors.get("_physical_shape")
        if offsets is not None and shape_tensor is not None:
            # During backward, outer_size may reflect a transposed view
            # (e.g., (seq, batch, hidden) from MHA's batch-dim transpose).
            # Detect and correct so _from_packed validation passes.
            batch_size = len(offsets) - 1
            outer = tuple(outer_size)
            batch_first = ctx.get("batch_first", True)
            if len(outer) >= 2 and (
                (batch_first and outer[0] != batch_size and outer[1] == batch_size)
                or (not batch_first and outer[1] != batch_size and outer[0] == batch_size)
            ):
                outer = (outer[1], outer[0], *outer[2:])
            return cls._from_packed(
                values,
                offsets,
                shape_tensor,
                outer_size=outer,
                **ctx,
            )

        result = torch.Tensor._make_wrapper_subclass(
            cls,
            torch.Size(outer_size),
            dtype=values.dtype,
            device=values.device,
            requires_grad=values.requires_grad,
        )
        result._values = values
        if offsets is not None:
            result._offsets = offsets
        if shape_tensor is not None:
            result._physical_shape = shape_tensor
        result._logical_shape = torch.Size(outer_size)
        result._set_runtime_config(
            batch_first=ctx["batch_first"],
            padding_value=ctx["padding_value"],
            mask_value=ctx["mask_value"],
        )
        result._pin_memory = ctx["pin_memory"]
        result._packed_sizes = ctx["packed_sizes"]
        result._element_shapes = ctx["element_shapes"]
        result._permutation = tuple(int(dim) for dim in ctx["permutation"])
        result._invalidate_transient_caches()
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None) -> Any:
        if kwargs is None:
            kwargs = {}

        # Handle size() specially to avoid infinite recursion
        if func is torch.Tensor.size:
            self = args[0]
            dim = args[1] if len(args) > 1 else kwargs.get("dim")
            return self.size(dim)

        from .ops import NestedTensorFuncRegistry, _compile_unsupported, _is_compiling

        handler = NestedTensorFuncRegistry.get(func)
        if handler is not None:
            if _is_compiling() and not NestedTensorFuncRegistry.is_compile_safe(func, args, kwargs):
                name = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
                _compile_unsupported(name, "handler is marked eager-only")
            return handler(*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None) -> Any:
        if kwargs is None:
            kwargs = {}

        from .ops import _compile_unsupported, _is_compiling

        if func in NestedTensorAtenRegistry:
            if _is_compiling() and not NestedTensorAtenRegistry.is_compile_safe(func, args, kwargs):
                name = getattr(func, "name", None)
                if callable(name):
                    name = name()
                _compile_unsupported(name or repr(func), "aten handler is marked eager-only")
            return NestedTensorAtenRegistry[func](func, args, kwargs)

        if _is_compiling():
            name = getattr(func, "name", None)
            if callable(name):
                name = name()
            _compile_unsupported(name or repr(func), "would fall back to per-element eager execution")
        return per_element_fallback(func, args, kwargs)

    # ------------------------------------------------------------------
    # Layout & Metadata Helpers
    # ------------------------------------------------------------------

    def _unpack(self) -> tuple[Tensor, ...]:
        r"""Reconstruct individual tensors from packed representation."""
        batch_size = len(self._offsets) - 1
        if batch_size == 0:
            return ()

        packed_sizes = self._packed_sizes
        if packed_sizes is None:
            if _is_fake_tensor(self._offsets):
                raise RuntimeError("NestedTensor packed sizes are unavailable for this instance.")
            packed_sizes = tuple(int(size) for size in (self._offsets[1:] - self._offsets[:-1]).tolist())

        element_shapes = self._element_shapes
        if element_shapes is None:
            element_shapes = tuple(tuple(int(dim) for dim in shape) for shape in self._original_shapes())

        splits = self._values.split(packed_sizes, dim=0)
        permutation = self._permutation
        if permutation:
            varying_dims = self._varying_dims
            static_dims = self._static_dims
        else:
            varying_dims, static_dims = type(self)._pack_layout_meta(self._physical_shape, element_shapes)
            permutation = varying_dims + static_dims
        inverse_permutation = type(self)._inverse_permutation(permutation)

        result = []
        for chunk, shape in zip(splits, element_shapes):
            if not shape:
                result.append(chunk[0])
            else:
                packed_shape = tuple(shape[dim] for dim in varying_dims) + tuple(shape[dim] for dim in static_dims)
                unpacked = chunk.reshape(packed_shape)
                if permutation != tuple(range(len(shape))):
                    unpacked = unpacked.permute(inverse_permutation)
                result.append(unpacked)
        return tuple(result)

    def _repack(self, tensors: Sequence) -> None:
        r"""
        Re-pack from already-validated tensors. Skips coercion — callers must ensure
        tensors share device, dtype, and ndim (which is always true for internal paths
        since tensors originate from _unpack or __setitem__ validation)."""
        self._invalidate_transient_caches()
        tensors = tuple(tensors) if not isinstance(tensors, tuple) else tensors
        if tensors and len(self._permutation) != tensors[0].ndim:
            raise RuntimeError(
                "NestedTensor._repack received tensors with rank "
                f"{tensors[0].ndim} but current permutation has rank {len(self._permutation)}"
            )
        values, offsets, shape_tensor, packed_sizes, element_shapes = self._pack(
            tensors,
            permutation=self._permutation if tensors else None,
        )
        values = type(self)._maybe_pin_values(values, self._pin_memory)
        self._values = values
        self._offsets = offsets
        self._physical_shape = shape_tensor
        self._logical_shape = self._compute_logical_shape(tensors, self.batch_first)
        self._packed_sizes = packed_sizes
        self._element_shapes = element_shapes
        self._validate_metadata()

    @property
    def _hierarchical_offsets(self) -> tuple[Tensor, ...]:
        if self._cached_hierarchical_offsets is None:
            level_sizes = type(self)._hierarchical_level_sizes_from_physical_shape(
                self._physical_shape,
                self._element_shapes,
            )
            if not level_sizes:
                if self._element_shapes is None and self._packed_sizes is not None:
                    self._cached_hierarchical_offsets = (
                        type(self)._offsets_from_sizes(self._packed_sizes, dtype=self._offsets.dtype),
                    )
                elif self._element_shapes is None and _is_fake_tensor(self._physical_shape):
                    self._cached_hierarchical_offsets = (self._offsets,)
                else:
                    self._cached_hierarchical_offsets = ()
            else:
                self._cached_hierarchical_offsets = tuple(
                    type(self)._offsets_from_sizes(level_sizes[level], dtype=self._offsets.dtype)
                    for level in range(len(level_sizes))
                )
        return self._cached_hierarchical_offsets

    @property
    def _ragged_rank(self) -> int:
        return len(self._hierarchical_offsets)

    def _ragged_level_offsets(self, level: int = -1) -> Tensor:
        offsets = self._hierarchical_offsets
        if not offsets:
            return self._offsets
        return offsets[level]

    def _ragged_level_sizes(self, level: int = -1) -> Tensor:
        offsets = self._ragged_level_offsets(level)
        return offsets[1:] - offsets[:-1]

    @property
    def _varying_dims(self) -> tuple[int, ...]:
        ragged_rank = self._ragged_rank
        if ragged_rank <= 0:
            return ()
        if self._permutation:
            return tuple(int(dim) for dim in self._permutation[:ragged_rank])
        varying_dims, _ = type(self)._pack_layout_meta(self._physical_shape, self._element_shapes)
        return varying_dims

    @property
    def _static_dims(self) -> tuple[int, ...]:
        ragged_rank = self._ragged_rank
        if self._permutation:
            return tuple(int(dim) for dim in self._permutation[ragged_rank:])
        _, static_dims = type(self)._pack_layout_meta(self._physical_shape, self._element_shapes)
        return static_dims

    def _has_same_structure(self, other: Self) -> bool:
        if self.batch_first != other.batch_first or self._permutation != other._permutation:
            return False
        if self._element_shapes is not None and other._element_shapes is not None:
            lhs_levels = type(self)._hierarchical_level_sizes_from_element_shapes(self._element_shapes)
            rhs_levels = type(self)._hierarchical_level_sizes_from_element_shapes(other._element_shapes)
            if lhs_levels or rhs_levels:
                return lhs_levels == rhs_levels
            return len(self) == len(other)
        lhs_offsets = self._hierarchical_offsets
        rhs_offsets = other._hierarchical_offsets
        if len(lhs_offsets) != len(rhs_offsets):
            return False
        if lhs_offsets:
            return all(type(self)._meta_tensor_equal(lhs, rhs) for lhs, rhs in zip(lhs_offsets, rhs_offsets))
        return type(self)._meta_tensor_equal(self._offsets, other._offsets)

    def _has_same_layout(self, other: Self) -> bool:
        if not self._has_same_structure(other):
            return False
        if self._element_shapes is not None and other._element_shapes is not None:
            if self._element_shapes != other._element_shapes:
                return False
            if self._packed_sizes is not None and other._packed_sizes is not None:
                return self._packed_sizes == other._packed_sizes
            return True
        if (
            self._packed_sizes is not None
            and other._packed_sizes is not None
            and self._packed_sizes != other._packed_sizes
        ):
            return False
        if not type(self)._meta_tensor_equal(self._physical_shape, other._physical_shape):
            return False
        return type(self)._meta_tensor_equal(self._offsets, other._offsets)

    def _packed_flat_index(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.long,
    ) -> Tensor:
        target_device = self.device if device is None else device
        leading = self._values.size(0) if self._values.dim() > 0 else self._values.numel()
        return torch.arange(leading, device=target_device, dtype=dtype)

    def _packed_batch_local_indices(
        self,
        flat_idx: Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.long,
    ) -> tuple[Tensor, Tensor]:
        target_device = self.device if device is None else device
        if flat_idx is None:
            flat_idx = self._packed_flat_index(device=target_device, dtype=dtype)
        offsets = self._offsets.to(device=target_device, dtype=dtype)
        batch_idx = torch.searchsorted(offsets[1:], flat_idx, right=True)
        local_idx = flat_idx - offsets[batch_idx]
        return batch_idx, local_idx

    def _packed_varying_coords(
        self,
        batch_idx: Tensor,
        local_idx: Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.long,
    ) -> tuple[Tensor, ...]:
        target_device = self.device if device is None else device
        varying_dims = self._varying_dims
        if not varying_dims:
            return ()

        varying_sizes = self._physical_shape[:, list(varying_dims)].to(device=target_device, dtype=dtype)[batch_idx]
        strides = torch.ones_like(varying_sizes)
        running = torch.ones(varying_sizes.size(0), dtype=dtype, device=target_device)
        for dim in range(varying_sizes.size(1) - 1, -1, -1):
            strides[:, dim] = running
            running = running * varying_sizes[:, dim]

        coords: list[Tensor] = []
        remainder = local_idx
        for dim in range(varying_sizes.size(1)):
            coord = remainder // strides[:, dim]
            coords.append(coord)
            remainder = remainder - coord * strides[:, dim]
        return tuple(coords)

    def _packed_dense_index(
        self,
        flat_idx: Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.long,
    ) -> tuple[Tensor | slice, ...]:
        target_device = self.device if device is None else device
        batch_idx, local_idx = self._packed_batch_local_indices(flat_idx, device=target_device, dtype=dtype)
        varying_dims = self._varying_dims
        coords = self._packed_varying_coords(batch_idx, local_idx, device=target_device, dtype=dtype)

        dense_index: list[Tensor | slice] = [batch_idx]
        coord_iter = iter(coords)
        for dim in range(self._physical_shape.size(1)):
            dense_index.append(next(coord_iter) if dim in varying_dims else slice(None))
        return tuple(dense_index)

    def _physical_shape_like_batch_dense(self, batch_dense_shape: Sequence[int]) -> Tensor:
        r"""Return per-element shapes for a batch-leading dense tensor with this NestedTensor's ragged structure."""
        expected_ndim = self._physical_shape.size(1) + 1
        if len(batch_dense_shape) != expected_ndim:
            raise ValueError(
                "Batch-leading dense tensor rank does not match NestedTensor layout, "
                f"expected {expected_ndim}, got {len(batch_dense_shape)}"
            )
        shape, _, _ = self._shape_meta_from_components(
            replace_dims={int(dim): int(batch_dense_shape[dim + 1]) for dim in self._static_dims}
        )
        return shape

    def _element_shapes_like_batch_dense(
        self,
        batch_dense_shape: Sequence[int],
    ) -> tuple[tuple[int, ...], ...] | None:
        r"""Return Python element-shape metadata for a batch-leading dense tensor with this NestedTensor's layout."""
        expected_ndim = self._physical_shape.size(1) + 1
        if len(batch_dense_shape) != expected_ndim:
            raise ValueError(
                "Batch-leading dense tensor rank does not match NestedTensor layout, "
                f"expected {expected_ndim}, got {len(batch_dense_shape)}"
            )
        _, _, element_shapes = self._shape_meta_from_components(
            replace_dims={int(dim): int(batch_dense_shape[dim + 1]) for dim in self._static_dims}
        )
        return element_shapes

    def _shape_meta_from_components(
        self,
        *,
        prefix: Sequence[int] = (),
        keep_dims: Sequence[int] | None = None,
        suffix: Sequence[int] = (),
        replace_dims: Mapping[int, int] | None = None,
    ) -> tuple[Tensor, tuple[int, ...] | None, tuple[tuple[int, ...], ...] | None]:
        r"""Build packed shape metadata by keeping selected dims and applying constant prefix/suffix updates."""
        if keep_dims is None:
            keep_dims = tuple(range(self._physical_shape.size(1)))
        keep_dims = tuple(int(dim) for dim in keep_dims)
        prefix = tuple(int(size) for size in prefix)
        suffix = tuple(int(size) for size in suffix)
        updates = {int(dim): int(size) for dim, size in (replace_dims or {}).items()}

        if self._element_shapes:
            element_shapes_list: list[tuple[int, ...]] = []
            for element_shape in self._element_shapes:
                projected = [*prefix, *(int(element_shape[dim]) for dim in keep_dims), *suffix]
                for dim, size in updates.items():
                    projected[dim] = size
                element_shapes_list.append(tuple(projected))
            element_shapes = tuple(element_shapes_list)
            max_ndim = max(len(shape) for shape in element_shapes)
            shape = torch.tensor(
                [list(shape) + [0] * (max_ndim - len(shape)) for shape in element_shapes],
                dtype=torch.long,
            )
            return shape, self._packed_sizes_like(element_shapes), element_shapes

        parts: list[Tensor] = []
        batch_size = len(self)
        if prefix:
            parts.append(self._physical_shape.new_tensor(prefix).reshape(1, -1).expand(batch_size, -1))
        if keep_dims:
            parts.append(self._physical_shape[:, list(keep_dims)].clone())
        if suffix:
            parts.append(self._physical_shape.new_tensor(suffix).reshape(1, -1).expand(batch_size, -1))
        if parts:
            shape = torch.cat(parts, dim=1)
        else:
            shape = self._physical_shape.new_empty((batch_size, 0))
        for dim, size in updates.items():
            shape[:, dim] = size
        return shape, None, None

    def _max_physical_dims(self) -> tuple[int, ...]:
        r"""Return the maximum per-element size for each physical dimension (excluding batch)."""
        batch_dim = 0 if self.batch_first else 1
        return tuple(int(size) for index, size in enumerate(self._logical_shape) if index != batch_dim)

    def _logical_shape_from_physical_dims(self, physical_dims: Sequence[int]) -> torch.Size:
        r"""Build a logical outer shape from non-batch physical-dimension sizes."""
        physical_dims = tuple(int(size) for size in physical_dims)
        batch_size = len(self)
        if self.batch_first:
            return torch.Size((batch_size, *physical_dims))
        if not physical_dims:
            return torch.Size((batch_size,))
        return torch.Size((physical_dims[0], batch_size, *physical_dims[1:]))

    def _logical_shape_from_components(
        self,
        *,
        prefix: Sequence[int] = (),
        keep_dims: Sequence[int] | None = None,
        suffix: Sequence[int] = (),
        replace_dims: Mapping[int, int] | None = None,
    ) -> torch.Size:
        r"""Build a logical outer shape by projecting the current physical-dimension extents."""
        physical_dims = list(self._max_physical_dims())
        if keep_dims is None:
            keep_dims = tuple(range(len(physical_dims)))
        projected = [*(int(prefix_dim) for prefix_dim in prefix), *(physical_dims[int(dim)] for dim in keep_dims)]
        projected.extend(int(suffix_dim) for suffix_dim in suffix)
        for dim, size in (replace_dims or {}).items():
            projected[int(dim)] = int(size)
        return self._logical_shape_from_physical_dims(projected)

    def _leading_dim_preserving_meta(
        self,
        suffix: Sequence[int],
    ) -> tuple[Tensor, torch.Size, tuple[int, ...] | None, tuple[tuple[int, ...], ...] | None]:
        r"""Build metadata for ops that preserve the first per-element dim and replace all trailing dims uniformly."""
        keep_dims = (0,) if self._physical_shape.size(1) > 0 else ()
        shape, packed_sizes, element_shapes = self._shape_meta_from_components(keep_dims=keep_dims, suffix=suffix)
        return shape, self._leading_dim_preserving_outer_size(suffix), packed_sizes, element_shapes

    def _leading_dim_preserving_outer_size(self, suffix: Sequence[int]) -> torch.Size:
        r"""Return logical outer size for ops that preserve per-element dim-0 and replace trailing dims uniformly."""
        suffix = tuple(int(size) for size in suffix)
        batch_size = len(self)
        batch_dim = 0 if self.batch_first else 1
        logical = list(self._logical_shape)
        non_batch = [int(logical[index]) for index in range(len(logical)) if index != batch_dim]

        new_non_batch: list[int] = []
        if self._physical_shape.size(1) > 0 and non_batch:
            new_non_batch.append(non_batch[0])
        new_non_batch.extend(suffix)

        if self.batch_first:
            return torch.Size((batch_size, *new_non_batch))
        if not new_non_batch:
            return torch.Size((batch_size,))
        return torch.Size((new_non_batch[0], batch_size, *new_non_batch[1:]))

    def _drop_trailing_physical_dims_meta(
        self,
        count: int,
        *,
        suffix: Sequence[int] = (),
    ) -> tuple[Tensor, tuple[int, ...] | None, tuple[tuple[int, ...], ...] | None]:
        r"""Build metadata after dropping trailing per-element dims and optionally appending a dense suffix."""
        keep_dims = tuple(range(max(self._physical_shape.size(1) - int(count), 0)))
        return self._shape_meta_from_components(keep_dims=keep_dims, suffix=suffix)

    def _replace_trailing_physical_dims_meta(
        self,
        trailing_sizes: Sequence[int],
    ) -> tuple[Tensor, tuple[int, ...] | None, tuple[tuple[int, ...], ...] | None]:
        r"""Build metadata after replacing the last physical dims with uniform sizes."""
        trailing_sizes = tuple(int(size) for size in trailing_sizes)
        if not trailing_sizes:
            return self._shape_meta_from_components()
        ndim = self._physical_shape.size(1)
        if len(trailing_sizes) > ndim:
            raise ValueError(f"Cannot replace {len(trailing_sizes)} trailing dims for per-element rank {ndim}")
        start = ndim - len(trailing_sizes)
        return self._shape_meta_from_components(
            replace_dims={start + index: size for index, size in enumerate(trailing_sizes)}
        )

    def _permutation_after_dropping_trailing_dims(self, count: int) -> tuple[int, ...]:
        r"""Return the canonical permutation after dropping trailing physical dims."""
        count = int(count)
        new_rank = max(self._physical_shape.size(1) - count, 0)
        if not self._permutation:
            return tuple(range(new_rank))
        return tuple(int(dim) for dim in self._permutation if dim < new_rank)

    def _permutation_after_replacing_trailing_dims(self, removed_count: int, added_count: int) -> tuple[int, ...]:
        r"""Return the canonical permutation after replacing trailing physical dims with a new suffix."""
        removed_count = int(removed_count)
        added_count = int(added_count)
        retained_rank = max(self._physical_shape.size(1) - removed_count, 0)
        retained = self._permutation_after_dropping_trailing_dims(removed_count)
        appended = tuple(range(retained_rank, retained_rank + added_count))
        return retained + appended

    def _scalar_result_meta(
        self,
    ) -> tuple[Tensor, Tensor, torch.Size, tuple[int, ...] | None, tuple[tuple[int, ...], ...] | None]:
        r"""Build metadata for one-scalar-per-element outputs."""
        shape, packed_sizes, element_shapes = self._shape_meta_from_components(keep_dims=())
        offsets = torch.arange(len(self) + 1, dtype=self._offsets.dtype, device=self._offsets.device)
        logical_shape = type(self)._logical_shape_from_physical_shape(shape, self._offsets, self.batch_first)
        return offsets, shape, logical_shape, packed_sizes, element_shapes

    def _from_scalar_result_values(self, values: Tensor) -> Self:
        r"""Wrap one scalar per element using the canonical scalar-result metadata."""
        cls = type(self)
        offsets, shape, outer_size, packed_sizes, element_shapes = self._scalar_result_meta()
        return cls._from_packed(
            values,
            offsets,
            shape,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=outer_size,
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
        )

    @classmethod
    def _cat_batch_packed(cls, tensors: Sequence[Self]) -> Self | None:
        r"""Merge batch-dim concatenation directly from packed storage when layouts are compatible."""
        if not tensors:
            raise ValueError("Expected at least one NestedTensor to concatenate.")

        ref = tensors[0]
        packed_rank = ref._values.dim()
        packed_tail = ref._values.shape[1:]
        reference_permutation = ref._permutation
        for tensor in tensors[1:]:
            if tensor._values.dim() != packed_rank:
                return None
            if tensor._permutation != reference_permutation:
                return None
            if packed_rank > 1 and tensor._values.shape[1:] != packed_tail:
                return None

        new_values = torch.cat([tensor._values for tensor in tensors], dim=0)

        offset_parts = []
        cumulative = 0
        for index, tensor in enumerate(tensors):
            offsets = tensor._offsets if index == 0 else tensor._offsets[1:] + cumulative
            offset_parts.append(offsets)
            cumulative += int(tensor._offsets[-1].item())
        new_offsets = torch.cat(offset_parts, dim=0)

        max_cols = max(tensor._physical_shape.size(1) for tensor in tensors)
        if max_cols > 0:
            padded_shapes = []
            for tensor in tensors:
                physical_shape = tensor._physical_shape
                if physical_shape.size(1) < max_cols:
                    physical_shape = torch.nn.functional.pad(physical_shape, (0, max_cols - physical_shape.size(1)))
                padded_shapes.append(physical_shape)
            new_physical_shape = torch.cat(padded_shapes, dim=0)
        else:
            new_physical_shape = torch.empty(len(new_offsets) - 1, 0, dtype=torch.long)

        batch_dim = 0 if ref.batch_first else 1
        out_logical = list(ref._logical_shape)
        if len(out_logical) <= batch_dim:
            out_logical.extend(0 for _ in range(batch_dim + 1 - len(out_logical)))
        out_logical[batch_dim] = sum(len(tensor) for tensor in tensors)
        for logical_dim in range(len(out_logical)):
            if logical_dim == batch_dim:
                continue
            out_logical[logical_dim] = max(
                int(tensor._logical_shape[logical_dim]) if logical_dim < len(tensor._logical_shape) else 0
                for tensor in tensors
            )

        packed_sizes = None
        if all(tensor._packed_sizes is not None for tensor in tensors):
            packed_sizes = tuple(size for tensor in tensors for size in cast(tuple[int, ...], tensor._packed_sizes))
        element_shapes = None
        if all(tensor._element_shapes is not None for tensor in tensors):
            element_shapes = tuple(
                shape for tensor in tensors for shape in cast(tuple[tuple[int, ...], ...], tensor._element_shapes)
            )

        return cls._from_packed(
            new_values,
            new_offsets,
            new_physical_shape,
            permutation=reference_permutation,
            batch_first=ref.batch_first,
            padding_value=ref.padding_value,
            mask_value=ref.mask_value,
            pin_memory=ref._pin_memory,
            outer_size=tuple(out_logical),
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
        )

    @property
    def _storage(self) -> tuple[Tensor, ...]:
        if self._cached_storage is None:
            self._cached_storage = self._unpack()
        return self._cached_storage

    @_storage.setter
    def _storage(self, tensors: Sequence) -> None:
        self._repack(tensors)

    # ------------------------------------------------------------------
    # Cached materialized views
    # ------------------------------------------------------------------

    def _tensor_cached_view(self) -> Tensor:
        cached = self._cached_tensor_view
        token = self._values_cache_token()
        if (
            cached is not None
            and cached[0] is self.batch_first
            and cached[1] == self.padding_value
            and cached[2] == token
        ):
            return cached[3]
        batch_leading = self._materialize_batch_leading(self.padding_value)
        tensor = batch_leading if self.batch_first else batch_leading.movedim(0, 1)
        self._cached_tensor_view = (self.batch_first, self.padding_value, token, tensor)
        return tensor

    def _mask_cached_view(self) -> Tensor:
        cached = self._cached_mask_view
        token = self._shape_cache_token()
        if cached is not None and cached[0] is self.batch_first and cached[1] is self.mask_value and cached[2] == token:
            return cached[3]
        mask = self._materialize_mask()
        self._cached_mask_view = (self.batch_first, self.mask_value, token, mask)
        return mask

    @property
    def tensor_mask(self) -> tuple[Tensor, Tensor]:
        r"""
        Return a tuple of padded tensor and mask tensor.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tensor_mask
            (tensor([[1, 2, 3],
                    [4, 5, 0]]), tensor([[ True,  True,  True],
                    [ True,  True, False]]))
        """
        return self._tensor_cached_view(), self._mask_cached_view()

    @property
    def tensor(self) -> Tensor:
        r"""
        Return a single tensor by padding all the tensors.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tensor
            tensor([[1, 2, 3],
                    [4, 5, 0]])
        """
        return self._tensor_cached_view()

    @property
    def mask(self) -> Tensor:
        r"""
        Padding mask of `tensor`.

        `mask_value` controls which boolean value denotes padding in this mask.
        With the default `mask_value=False`, `True` means valid data.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.mask
            tensor([[ True,  True,  True],
                    [ True,  True, False]])
        """
        return self._mask_cached_view()

    def _mask_squeezes_channel(self) -> bool:
        return self._physical_shape.size(1) > 1 and (self._physical_shape.size(1) - 1) in self._static_dims

    def _materialize_mask(self) -> Tensor:
        batch_size = len(self)
        logical_shape = self._logical_shape
        squeeze_channel = self._mask_squeezes_channel()

        if batch_size == 0:
            mask_shape = logical_shape[:-1] if squeeze_channel else logical_shape
            return torch.empty(mask_shape, dtype=torch.bool, device=self.device)

        if self._physical_shape.size(1) == 0:
            return torch.full((batch_size,), not self.mask_value, dtype=torch.bool, device=self.device)

        effective_shape = logical_shape[:-1] if squeeze_channel else logical_shape
        batch_dim = 0 if self.batch_first else 1
        non_batch_sizes = [effective_shape[i] for i in range(len(effective_shape)) if i != batch_dim]

        sizes = self._physical_shape[:, :-1] if squeeze_channel else self._physical_shape
        sizes = sizes.to(device=self.device, dtype=torch.long)

        valid = _batch_leading_valid_mask_from_sizes(
            sizes,
            non_batch_sizes,
            device=self.device,
        )

        if not self.batch_first:
            valid = valid.movedim(0, 1)
        return valid if not self.mask_value else ~valid

    def _materialize_batch_leading(self, fill_value) -> Tensor:
        r"""Materialize a padded dense tensor with the batch dimension in front."""
        _check_execution_guard(_ExecutionGuardKind.PADDED_MATERIALIZATION, "NestedTensor._materialize_batch_leading")
        logical_shape = self._logical_shape
        batch_size = len(self)
        if batch_size == 0:
            if self.batch_first:
                return torch.empty(logical_shape, dtype=self._values.dtype, device=self.device)
            if len(logical_shape) <= 1:
                return torch.empty((0,), dtype=self._values.dtype, device=self.device)
            non_batch = list(logical_shape)
            non_batch.pop(1)
            return torch.empty((0, *non_batch), dtype=self._values.dtype, device=self.device)

        if self._physical_shape.size(1) == 0:
            return self._values.reshape((batch_size,))

        tensor_shape = list(logical_shape)
        tensor_shape.pop(0 if self.batch_first else 1)
        batch_leading = self._values.new_full((batch_size, *tensor_shape), fill_value)
        if self._values.size(0) > 0:
            batch_leading[self._packed_dense_index(device=batch_leading.device)] = self._values
        return batch_leading

    def _original_shapes(self) -> tuple[torch.Size, ...]:
        if self._element_shapes is not None:
            return tuple(torch.Size(shape) for shape in self._element_shapes)
        if not _is_fake_tensor(self._physical_shape):
            return tuple(torch.Size(type(self)._trim_shape(row)) for row in self._physical_shape.tolist())
        raise RuntimeError("NestedTensor shape metadata is unavailable for this instance.")

    @property
    def concat(self) -> Tensor:
        r"""
        Flatten elements and concatenate along the ragged dimension (no padding).

        This is particularly useful when calculating loss or passing `Linear` to avoid unnecessary computation.

        Examples:
            >>> nested_tensor = NestedTensor([torch.randn(9, 8), torch.randn(11, 8)])
            >>> nested_tensor.concat.shape
            torch.Size([20, 8])
            >>> nested_tensor = NestedTensor([torch.randn(9, 9, 8), torch.randn(11, 11, 8)])
            >>> nested_tensor.concat.shape
            torch.Size([202, 8])
            >>> nested_tensor = NestedTensor([torch.randn(9, 9, 8, 6), torch.randn(11, 11, 8, 6)])
            >>> nested_tensor.concat.shape
            torch.Size([202, 8, 6])
            >>> nested_tensor = NestedTensor([torch.randn(9, 9, 8, 7), torch.randn(11, 11, 8, 6)])
            >>> nested_tensor.concat.shape
            torch.Size([1293, 8])
            >>> nested_tensor = NestedTensor([torch.randn(1, 9, 9, 5), torch.randn(1, 11, 11, 5)])
            >>> nested_tensor.concat.shape
            torch.Size([202, 1, 5])
        """
        if len(self._offsets) <= 1:
            return torch.empty(0, dtype=self._values.dtype, device=self.device)
        return self._values

    def concatenate(self) -> tuple[Tensor, tuple[torch.Size, ...]]:
        r"""
        Concatenate tensors in padding dimension and return structural information for reconstruction.

        Returns:
            A tuple containing:
            - concat_tensor: The concatenated tensor (same as .concat property)
            - shapes: Tuple of original tensor shapes for reconstruction

        Examples:
            >>> nested_tensor = NestedTensor([torch.randn(9, 8), torch.randn(11, 8)])
            >>> concat_tensor, shapes = nested_tensor.concatenate()
            >>> concat_tensor.shape
            torch.Size([20, 8])
            >>> shapes
            (torch.Size([9, 8]), torch.Size([11, 8]))
            >>> reconstructed = NestedTensor.from_concatenated(concat_tensor, shapes)
            >>> torch.equal(nested_tensor.tensor, reconstructed.tensor)
            True
        """
        batch_size = len(self._offsets) - 1
        if batch_size == 0:
            return torch.empty(0, dtype=self._values.dtype, device=self.device), ()
        return self._values, self._original_shapes()

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        r"""Return the number of tensors in the batch."""
        if not hasattr(self, "_offsets"):
            with torch._C.DisableTorchFunctionSubclass():
                full_size = torch.Tensor.size(self)
            if len(full_size) == 0:
                return 0
            batch_dim = 0 if getattr(self, "batch_first", True) else (1 if len(full_size) > 1 else 0)
            return int(full_size[batch_dim])
        return len(self._offsets) - 1

    def __repr__(self):
        r"""Return a human-readable string representation of the NestedTensor."""
        if torch._dynamo.is_compiling():
            try:
                shape = tuple(self.size())
            except Exception:
                shape = "?"
            return (
                f"{self.__class__.__name__}(shape={shape}, dtype={self.dtype}, "
                f"device={self.device}, batch_first={getattr(self, 'batch_first', True)})"
            )

        try:
            from torch._subclasses.fake_tensor import is_fake

            for name in ("_values", "_offsets", "_physical_shape"):
                value = self.__dict__.get(name)
                if isinstance(value, Tensor) and is_fake(value):
                    shape = tuple(self.size())
                    return (
                        f"{self.__class__.__name__}(shape={shape}, dtype={self.dtype}, "
                        f"device={self.device}, batch_first={getattr(self, 'batch_first', True)})"
                    )
        except Exception:
            pass

        if not all(name in self.__dict__ for name in ("_values", "_offsets", "_physical_shape")):
            try:
                shape = tuple(self.size())
            except Exception:
                shape = "?"
            return (
                f"{self.__class__.__name__}(shape={shape}, dtype={self.dtype}, "
                f"device={self.device}, batch_first={getattr(self, 'batch_first', True)})"
            )

        if len(self) == 0:
            return self.__class__.__name__ + "()"

        storage = self._storage
        truncated = len(storage) > 10
        if truncated:
            storage = storage[:5]

        indent = "    "

        # Strip "tensor(" wrapper from each element's repr,
        # keeping PyTorch's internal number formatting (precision, alignment).
        data_parts = []
        for t in storage:
            s = repr(t)
            paren_idx = s.index("(")
            data = s[paren_idx + 1 : -1]  # noqa: E203
            # Re-indent continuation lines for multi-line element reprs (e.g. 2D tensors)
            if "\n" in data:
                lines = data.split("\n")
                data = lines[0] + "\n" + "\n".join(indent + " " + line.lstrip() for line in lines[1:])
            data_parts.append(data)

        result_lines = [self.__class__.__name__ + "(["]
        for i, part in enumerate(data_parts):
            suffix = "," if i < len(data_parts) - 1 or truncated else ""
            result_lines.append(indent + part + suffix)
        if truncated:
            result_lines.append(indent + f"... ({len(self)} tensors)")
        result_lines.append("])")
        return "\n".join(result_lines)

    def __bool__(self) -> bool:
        r"""NestedTensor follows tensor-style truthiness and never acts like a Python container."""
        raise RuntimeError(
            "Boolean value of NestedTensor is ambiguous. Use .numel(), .any(), .all(), or an explicit reduction."
        )

    def __iter__(self):
        r"""Iterate over the tensors in the batch."""
        _check_execution_guard(_ExecutionGuardKind.ITERATION, "NestedTensor.__iter__")
        return iter(self._storage)

    def __eq__(self, other):  # type: ignore[override]
        r"""Element-wise equality comparison."""
        try:
            return torch.eq(self, other)
        except TypeError:
            return NotImplemented

    def __ne__(self, other):  # type: ignore[override]
        r"""Element-wise inequality comparison."""
        try:
            return torch.ne(self, other)
        except TypeError:
            return NotImplemented

    # Python sets __hash__ = None when __eq__ is overridden in a subclass.
    # Preserve Tensor's identity hash so AOT/torch.compile memoization works.
    __hash__ = Tensor.__hash__

    # Arithmetic, comparison, and in-place operators are handled by the base
    # Tensor class, which routes through C++ → aten → __torch_dispatch__ →
    # aten_functions.py. No Python-level overrides needed.

    # ------------------------------------------------------------------
    # Conversion & Factory Methods
    # ------------------------------------------------------------------

    @classmethod
    def from_concatenated(cls, concat_tensor: Tensor, shapes: tuple[torch.Size, ...], **kwargs) -> Self:
        r"""
        Reconstruct a NestedTensor from a concatenated tensor and shape information.

        Args:
            concat_tensor: The concatenated tensor returned by concatenate()
            shapes: Tuple of original tensor shapes returned by concatenate()
            **kwargs: Additional arguments to pass to NestedTensor constructor

        Returns:
            Reconstructed NestedTensor

        Examples:
            >>> nested_tensor = NestedTensor([torch.randn(9, 9, 8), torch.randn(11, 11, 8)])
            >>> concat_tensor, shapes = nested_tensor.concatenate()
            >>> reconstructed = NestedTensor.from_concatenated(concat_tensor, shapes)
            >>> concat_tensor.shape
            torch.Size([202, 8])
            >>> reconstructed.shape
            torch.Size([2, 11, 11, 8])
            >>> torch.equal(nested_tensor.tensor, reconstructed.tensor)
            True
        """
        if not shapes:
            if "dtype" not in kwargs:
                kwargs["dtype"] = concat_tensor.dtype
            if "device" not in kwargs:
                kwargs["device"] = concat_tensor.device
            return cls([], **kwargs)

        num_elements = [shape.numel() for shape in shapes]
        element_shapes = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
        varying_dims, static_dims = cls._pack_layout_from_element_shapes(element_shapes)
        permutation = varying_dims + static_dims
        identity_permutation = tuple(range(len(element_shapes[0]))) if element_shapes and element_shapes[0] else ()

        if len(set(shapes)) == 1 and permutation == identity_permutation:
            shape = shapes[0]
            total_elements = sum(num_elements)
            if concat_tensor.numel() == total_elements:
                try:
                    reshaped = concat_tensor.reshape(len(shapes), *shape)
                except (RuntimeError, ValueError):
                    # The reshape fast path is opportunistic; a normal unpack fallback
                    # is expected for non-view-compatible inputs.
                    pass
                else:
                    tensors = [t.reshape(shape) for t in reshaped.unbind(0)]
                    return cls(tensors, **kwargs)

        packed_sizes = tuple(cls._packed_size_from_shape(shape, varying_dims) for shape in element_shapes)
        total_expected = sum(num_elements)
        num_provided = concat_tensor.numel()
        if num_provided != total_expected:
            raise ValueError(
                f"Concatenated tensor has {num_provided} elements "
                f"but expected {total_expected} based on shapes {shapes}"
            )

        tensors = []
        start = 0
        inverse_permutation = cls._inverse_permutation(permutation)
        for shape, packed_size in zip(element_shapes, packed_sizes):
            end = start + packed_size
            chunk = concat_tensor.narrow(0, start, packed_size)
            packed_shape = tuple(shape[dim] for dim in varying_dims) + tuple(shape[dim] for dim in static_dims)
            tensor_data = chunk.reshape(packed_shape)
            if permutation != tuple(range(len(shape))):
                tensor_data = tensor_data.permute(inverse_permutation)
            tensors.append(tensor_data)
            start = end

        return cls(tensors, **kwargs)

    @classmethod
    def from_tensor_mask(cls, tensor: Tensor, mask: Tensor, *, batched: bool = False, **kwargs):
        r"""
        Build a `NestedTensor` object from a padded `Tensor` and corresponding mask `Tensor`.

        Args:
            tensor: Padded Tensor.
            mask: Tensor Mask.
                The mask uses the same convention as ``mask_value``:
                padding positions equal ``mask_value`` and valid positions equal ``not mask_value``.
            batched: When ``True`` and ``mask.ndim == 1``, treat ``mask`` as a per-batch-element
                selector (each ``True`` entry selects a row from ``tensor``) rather than a
                contiguous-prefix length indicator.

        Examples:
            >>> padded_tensor = torch.tensor([[1, 2, 3, 0, 0],
            ...                                [4, 5, 0, 0, 0],
            ...                                [6, 7, 8, 9, 0]])
            >>> mask_tensor = torch.tensor([[1, 1, 1, 0, 0],
            ...                             [1, 1, 0, 0, 0],
            ...                             [1, 1, 1, 1, 0]])
            >>> nested_tensor = NestedTensor.from_tensor_mask(padded_tensor, mask_tensor)
            >>> nested_tensor
            NestedTensor([
                [1, 2, 3],
                [4, 5],
                [6, 7, 8, 9]
            ])
        """
        mask = mask.to(dtype=torch.bool)
        mask_value = kwargs.get("mask_value", False)
        effective_mask = ~mask if mask_value else mask

        if mask.ndim == 1:
            if batched:
                indices = effective_mask.nonzero(as_tuple=False).flatten()
                return cls([tensor[int(i)] for i in indices], dtype=tensor.dtype, **kwargs)
            return cls(tensor[effective_mask], dtype=tensor.dtype, **kwargs)
        # ndim >= 2: batch setup is shared, per-element trim differs by rank
        batch_first = kwargs.get("batch_first", True)
        tensor_iter = tensor if batch_first else tensor.transpose(0, 1)
        mask_iter = effective_mask if batch_first else effective_mask.transpose(0, 1)
        if tensor_iter.size(0) != mask_iter.size(0):
            raise ValueError("Tensor/mask batch dimension mismatch: " f"{tensor_iter.size(0)} vs {mask_iter.size(0)}")
        trimmed = []

        def _is_prefix_mask(mask_1d: Tensor) -> bool:
            count = int(mask_1d.sum().item())
            prefix = torch.arange(mask_1d.size(0), device=mask_1d.device, dtype=torch.long) < count
            return bool(torch.equal(mask_1d, prefix))

        def _is_hierarchical_prefix_mask(mask_nd: Tensor) -> bool:
            if mask_nd.dim() == 1:
                return _is_prefix_mask(mask_nd)
            leading_valid = mask_nd.reshape(mask_nd.size(0), -1).any(dim=1)
            valid_count = int(leading_valid.sum().item())
            prefix = torch.arange(mask_nd.size(0), device=mask_nd.device, dtype=torch.long) < valid_count
            if not torch.equal(leading_valid, prefix):
                return False
            return all(_is_hierarchical_prefix_mask(mask_nd[index]) for index in range(valid_count))

        if mask.ndim == 2:
            # 1-D per-element mask: only contiguous-prefix masks can be reconstructed
            # via slicing without changing dense semantics.
            counts = mask_iter.sum(dim=1, dtype=torch.long)
            prefix = torch.arange(mask_iter.size(1), device=mask_iter.device, dtype=torch.long).unsqueeze(0)
            prefix = prefix < counts.unsqueeze(1)
            if not torch.equal(mask_iter, prefix):
                raise ValueError(
                    "from_tensor_mask() with 2-D masks requires each row to be a valid prefix mask; "
                    "interior False gaps are not supported."
                )
            for t, count in zip(tensor_iter, counts.tolist()):
                trimmed.append(t[:count])
        else:
            # N-D per-element mask: only hierarchical ragged-prefix masks are representable as NestedTensor.
            extents = torch.zeros((mask_iter.size(0), mask_iter.dim() - 1), dtype=torch.long, device=mask_iter.device)
            nonzero = mask_iter.nonzero(as_tuple=False)
            if nonzero.numel() > 0:
                batch_index = nonzero[:, :1].expand(-1, extents.size(1))
                extents.scatter_reduce_(0, batch_index, nonzero[:, 1:] + 1, reduce="amax", include_self=False)
            extent_rows = extents.cpu().tolist()
            for t, em, sizes in zip(tensor_iter, mask_iter, extent_rows):
                if not _is_hierarchical_prefix_mask(em):
                    raise ValueError(
                        "from_tensor_mask() with N-D masks requires each element mask to be a valid hierarchical "
                        "ragged prefix; "
                        "interior False gaps are not supported."
                    )
                slices = tuple(slice(0, size) for size in sizes)
                t_slice = t[slices]
                m_slice = em[slices]
                valid_mask = m_slice
                if t_slice.dim() > m_slice.dim():
                    valid_mask = m_slice.view(m_slice.shape + (1,) * (t_slice.dim() - m_slice.dim()))
                trimmed.append(t_slice.masked_fill(~valid_mask, kwargs.get("padding_value", 0.0)))
        return cls(trimmed, dtype=tensor.dtype, **kwargs)

    def _dense_to_packed_values(self, tensor: Tensor) -> Tensor | None:
        r"""
        Convert a batch-aligned dense tensor to ``self``'s packed ``_values`` layout.

        Returns ``None`` when the dense tensor does not cover the current logical
        padded extents and we must fall back to per-element slicing/repacking.
        """
        batch_leading = tensor.to(device=self.device)
        if self.dim() > 1 and not self.batch_first:
            batch_leading = batch_leading.movedim(1, 0)

        logical_shape = list(self.shape)
        if logical_shape:
            batch_dim = 0 if self.dim() <= 1 or self.batch_first else 1
            logical_shape.pop(batch_dim)
        if batch_leading.dim() != len(logical_shape) + 1:
            return None

        dense_sizes = tuple(int(batch_leading.size(dim + 1)) for dim in range(batch_leading.dim() - 1))
        if any(dense_sizes[dim] < int(size) for dim, size in enumerate(logical_shape)):
            return None

        if logical_shape:
            batch_leading = batch_leading[(slice(None), *[slice(0, int(size)) for size in logical_shape])]

        if batch_leading.dim() <= 1:
            return batch_leading.contiguous()

        return batch_leading[self._packed_dense_index(device=batch_leading.device)].contiguous()

    def _packed_sizes_like(self, element_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        varying_dims, _ = type(self)._pack_layout_from_element_shapes(element_shapes)
        return tuple(type(self)._packed_size_from_shape(shape, varying_dims) for shape in element_shapes)

    def nested_like(self, tensor: Tensor, strict: bool = True) -> Self:
        r"""
        Create a new `NestedTensor` from a `Tensor`.
        The newly created `NestedTensor` will have the same shape as current `NestedTensor`.

        Args:
            tensor: The tensor to be converted to `NestedTensor`.
            strict: Check if the shape of `tensor` is the same as the current `NestedTensor`.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> (nested_tensor == nested_tensor.nested_like(nested_tensor)).all()
            tensor(True)
            >>> tensor = nested_tensor.tensor
            >>> (nested_tensor == nested_tensor.nested_like(tensor)).all()
            tensor(True)
            >>> f = nested_tensor.nested_like(torch.randn(2, 2))
            Traceback (most recent call last):
            ...
            ValueError: The shape of NestedTensor and input tensor does not match, ...
            >>> p = nested_tensor.nested_like(torch.randn(2, 2), False)
            >>> p = nested_tensor.nested_like(torch.randn(3, 3), False)
            Traceback (most recent call last):
            ...
            ValueError: The batch size of NestedTensor and input tensor does not match, 2 != 3
        """

        if isinstance(tensor, NestedTensor):
            return tensor.clone()

        if strict and self.shape != tensor.shape:
            raise ValueError(
                f"The shape of NestedTensor and input tensor does not match, {self.shape} != {tensor.shape}"
            )
        batch_dim = 0 if self.dim() <= 1 or self.batch_first else 1
        if len(self) != tensor.size(batch_dim):
            raise ValueError(
                "The batch size of NestedTensor and input tensor does not match, "
                f"{len(self)} != {tensor.size(batch_dim)}"
            )
        values = self._dense_to_packed_values(tensor)
        if values is not None:
            element_shapes = self._element_shapes
            return self.__class__._from_packed(
                values,
                self._offsets,
                self._physical_shape,
                batch_first=self.batch_first,
                padding_value=self.padding_value,
                mask_value=self.mask_value,
                pin_memory=self._pin_memory,
                outer_size=self._logical_shape,
                packed_sizes=self._packed_sizes,
                element_shapes=element_shapes,
            )
        dense_tensor = tensor.to(device=self.device)
        element_shapes = self._original_shapes()
        new_storage = []
        for idx, shape in enumerate(element_shapes):
            if self.batch_first:
                slices = (idx, *[slice(0, int(dim)) for dim in shape])
            else:
                if len(shape) == 0:
                    slices = (idx,)
                else:
                    slices = (slice(0, int(shape[0])), idx, *[slice(0, int(dim)) for dim in shape[1:]])
            # .contiguous() ensures storage elements don't inherit non-trivial
            # strides from the padded tensor (e.g. after transpose).
            new_storage.append(dense_tensor[slices].contiguous())
        return self.__class__(new_storage, dtype=tensor.dtype, **self._meta(include_dtype=False))

    @property
    def occupancy(self) -> float:
        r"""
        Occupancy of the NestedTensor.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6])])
            >>> nested_tensor.occupancy
            0.75
        """
        if len(self) == 0:
            return 0.0
        denom = self.shape.numel()  # type: ignore[union-attr]
        if denom == 0:
            return 0.0
        return self.numel() / denom  # type: ignore[union-attr]

    def to_torch_nested(self) -> Tensor:
        r"""
        Create a `torch.nested.nested_tensor` object from `self`.

        Examples:
            >>> nested_tensor = NestedTensor([[2, 3, 5], [7, 8]])
            >>> nt = nested_tensor.to_torch_nested()
            >>> nt.layout == torch.jagged
            True
            >>> nt.values()
            tensor([2, 3, 5, 7, 8])
        """
        storage = list(self._storage)
        if not storage or all(t.dim() > 0 for t in storage):
            return nested.nested_tensor(storage, layout=torch.jagged)
        return nested.nested_tensor(storage)

    def unbind(self, dim: int = 0) -> tuple[Tensor, ...]:
        r"""
        Unbind the NestedTensor.
        """
        return torch.unbind(self, dim=dim)

    def _maybe_exact_shape_nested_like(self, tensor: object) -> Self | None:
        r"""
        Convert an exact-shape dense tensor to this NestedTensor's layout.

        This is the shared policy boundary for dense-to-nested alignment used by
        operator helpers: only non-scalar dense tensors with logical shape exactly
        matching ``self.shape`` are converted, and the conversion always uses
        ``nested_like(..., strict=False)``.
        """
        if not isinstance(tensor, Tensor) or isinstance(tensor, type(self)):
            return None
        if tensor.dim() == 0 or tensor.shape != self.shape:
            return None
        return self.nested_like(tensor, strict=False)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, index: int | slice | list | tuple | Tensor | NestedTensor) -> Tensor | NestedTensor:
        r"""Retrieve element(s) by index, slice, list, tuple, or tensor mask."""
        if isinstance(index, int):
            return self._storage[index]
        if isinstance(index, (slice, list)):
            if isinstance(index, list) and index and all(isinstance(i, bool) for i in index):
                if len(index) != len(self):
                    raise IndexError(f"Boolean index has length {len(index)} but batch size is {len(self)}")
                index = [i for i, flag in enumerate(index) if flag]
            storage = tuple(self._storage[index] if isinstance(index, slice) else [self._storage[i] for i in index])
            return self.__class__(storage, **self._meta(include_dtype=True))
        if isinstance(index, tuple):
            if len(index) == 0:
                return self

            # Expand Ellipsis: ``nt[..., :2]`` on a 4-D NestedTensor becomes
            # ``nt[:, :, :, :2]``.  The batch dim is consumed first, so Ellipsis
            # fills the gap between the number of explicit indices and the total
            # number of logical dimensions.
            if Ellipsis in index:
                eidx = index.index(Ellipsis)
                n_explicit = len(index) - 1  # exclude Ellipsis itself
                n_expand = self.dim() - n_explicit
                index = index[:eidx] + (slice(None),) * n_expand + index[eidx + 1 :]

            batch_index, *rest = index

            if isinstance(batch_index, (Tensor, NestedTensor)):
                return self.tensor[index]

            if isinstance(batch_index, list) and batch_index and all(isinstance(i, bool) for i in batch_index):
                if len(batch_index) != len(self):
                    raise IndexError(f"Boolean index has length {len(batch_index)} but batch size is {len(self)}")
                batch_index = [i for i, flag in enumerate(batch_index) if flag]

            if isinstance(batch_index, int):
                tensor = self._storage[batch_index]
                if rest:
                    return tensor[tuple(rest)]
                return tensor
            elif isinstance(batch_index, (slice, list)):
                if isinstance(batch_index, slice):
                    selected = self._storage[batch_index]
                else:
                    selected = tuple(self._storage[i] for i in batch_index)
                if rest:
                    rest_tuple = tuple(rest)
                    selected = tuple(t[rest_tuple] for t in selected)
                return self.__class__(selected, **self._meta(include_dtype=True))
            raise ValueError(f"Unsupported batch index type {type(batch_index)}")
        if isinstance(index, NestedTensor):
            if len(self) != len(index):
                raise ValueError(
                    "NestedTensor batch length mismatch between self and index: "
                    f"self={len(self)}, index={len(index)}"
                )
            return self.__class__(
                [t[i] for t, i in zip(self._storage, index._storage)], **self._meta(include_dtype=True)
            )
        if isinstance(index, Tensor):
            if index.dim() == 0 and index.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                return self._storage[int(index.item())]
            if index.dim() == 1:
                if index.dtype in (torch.bool, torch.uint8):
                    if index.numel() != len(self):
                        raise IndexError(f"Boolean index has length {index.numel()} but batch size is {len(self)}")
                    selected = tuple(self._storage[i] for i, flag in enumerate(index.tolist()) if bool(flag))
                    return self.__class__(selected, **self._meta(include_dtype=True))
                if index.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    return self.__class__(
                        [self._storage[int(i)] for i in index.tolist()],
                        **self._meta(include_dtype=True),
                    )
            index = self.nested_like(index, strict=False)
            return self.__class__(
                [t[i] for t, i in zip(self._storage, index._storage)], **self._meta(include_dtype=True)
            )
        raise ValueError(f"Unsupported index type {type(index)}")

    def __setitem__(self, index: int | slice | list | tuple, value: Tensor | NestedTensor) -> None:
        r"""
        Set values in the NestedTensor at the specified index.

        Args:
            index: The index to modify. Can be an integer, slice, list, or tuple.
            value: The new value to set. Can be a Tensor or NestedTensor.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor[0] = torch.tensor([6, 7, 8])
            >>> nested_tensor[0]
            tensor([6, 7, 8])
            >>> nested_tensor[1] = torch.tensor([9, 10, 11, 12])
            >>> nested_tensor.shape
            torch.Size([2, 4])
        """
        if isinstance(index, int):
            self._invalidate_transient_caches()
            if isinstance(value, NestedTensor):
                if len(value._storage) != 1:
                    raise ValueError(
                        f"When setting with an integer index, value must have a single tensor, but got {len(value)}"
                    )
                value = value._storage[0]
            if not isinstance(value, Tensor):
                value = torch.tensor(value, device=self.device, dtype=self.dtype)
            else:
                value = value.to(device=self.device, dtype=self.dtype)
            if self.requires_grad:
                value.requires_grad_(True)

            # Normalize negative index
            idx = index + len(self) if index < 0 else index
            if idx < 0 or idx >= len(self):
                raise IndexError(f"index {index} is out of range for NestedTensor with {len(self)} elements")
            expected_ndim = self._physical_shape.size(1)
            if value.dim() != expected_ndim:
                raise ValueError(
                    f"Assigned tensor ndim must match existing ndim {expected_ndim}, but got {value.dim()}"
                )

            old_start = int(self._offsets[idx].item())
            old_end = int(self._offsets[idx + 1].item())
            old_size = old_end - old_start
            new_shape_row = torch.tensor(list(value.shape), dtype=self._physical_shape.dtype)

            permutation = self._permutation
            identity_permutation = tuple(range(expected_ndim))
            varying_dims = self._varying_dims
            static_dims = self._static_dims
            packed_size = type(self)._packed_size_from_shape(tuple(int(dim) for dim in value.shape), varying_dims)
            packed_value = value if permutation == identity_permutation else value.permute(permutation)
            suffix_shape = tuple(int(value.shape[dim]) for dim in static_dims)
            new_payload = packed_value.reshape((packed_size, *suffix_shape) if suffix_shape else (packed_size,))
            new_size = packed_size

            if self._values.dim() > 1 and new_payload.shape[1:] != self._values.shape[1:]:
                storage_list = list(self._storage)
                storage_list[idx] = value
                self._repack(storage_list)
                return

            if new_size == old_size:
                # Same packed span size: direct overwrite keeps _values allocation.
                self._values[old_start:old_end] = new_payload
                self._physical_shape[idx] = new_shape_row
            else:
                # Different packed span size: splice _values and shift subsequent offsets.
                self._values = torch.cat([self._values[:old_start], new_payload, self._values[old_end:]], dim=0)
                delta = new_size - old_size
                self._offsets = self._offsets.clone()
                self._offsets[idx + 1 :] += delta  # noqa: E203
                self._physical_shape = self._physical_shape.clone()
                self._physical_shape[idx] = new_shape_row
            self._logical_shape = self._logical_shape_from_physical_shape(
                self._physical_shape, self._offsets, self.batch_first
            )
            if self._element_shapes is not None and self._packed_sizes is not None:
                element_shapes = list(self._element_shapes)
                element_shapes[idx] = tuple(int(dim) for dim in value.shape)
                self._element_shapes = tuple(element_shapes)
                packed_sizes = list(self._packed_sizes)
                packed_sizes[idx] = self._packed_sizes_like((self._element_shapes[idx],))[0]
                self._packed_sizes = tuple(packed_sizes)
            self._validate_metadata()
        elif isinstance(index, (slice, list)):
            if isinstance(index, list) and index and all(isinstance(i, bool) for i in index):
                if len(index) != len(self):
                    raise IndexError(f"Boolean index has length {len(index)} but batch size is {len(self)}")
                index = [i for i, flag in enumerate(index) if flag]

            if isinstance(value, Tensor) and not isinstance(value, NestedTensor):
                if value.dim() > 1 and value.size(0) > 1:
                    value = self.__class__(value.unbind(0), **self._meta())
                else:
                    value = self.__class__([value], **self._meta())

            if isinstance(index, slice):
                start, stop, step = index.indices(len(self))
                indices = range(start, stop, step)
            else:
                indices = index  # type: ignore[assignment]

            if len(indices) != len(value._storage):
                raise ValueError(
                    f"Size mismatch: tried to assign {len(value._storage)} values to {len(indices)} indices"
                )

            storage_list = list(self._storage)
            for i, idx in enumerate(indices):
                storage_list[idx] = value._storage[i]
            self._storage = tuple(storage_list)
        elif isinstance(index, tuple):
            if len(index) == 0:
                return
            if len(index) == 1:
                self[index[0]] = value
                return

            first_idx, rest_idx = index[0], index[1:]
            batch_indices: list[int]
            if isinstance(first_idx, int):
                batch_indices = [first_idx]
            elif isinstance(first_idx, (slice, list)):
                if isinstance(first_idx, list) and first_idx and all(isinstance(i, bool) for i in first_idx):
                    if len(first_idx) != len(self):
                        raise IndexError(f"Boolean index has length {len(first_idx)} but batch size is {len(self)}")
                    batch_indices = [i for i, flag in enumerate(first_idx) if flag]
                elif isinstance(first_idx, slice):
                    start, stop, step = first_idx.indices(len(self))
                    batch_indices = list(range(start, stop, step))
                else:
                    batch_indices = list(first_idx)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unsupported first index type {type(first_idx)}")

            if isinstance(value, NestedTensor):
                if len(batch_indices) != len(value._storage):
                    raise ValueError(
                        f"Size mismatch: tried to assign {len(value._storage)} values to {len(batch_indices)} indices"
                    )
                assigned_values = list(value._storage)
            else:
                assigned_values = [value] * len(batch_indices)

            elems = list(self._storage)
            for position, idx in enumerate(batch_indices):
                elem = elems[idx].clone()
                elem[rest_idx] = assigned_values[position]
                elems[idx] = elem
            self._storage = tuple(elems)
        else:
            raise ValueError(f"Unsupported index type {type(index)}")

    # ------------------------------------------------------------------
    # Properties: runtime config, dtype, device, requires_grad
    # ------------------------------------------------------------------

    @property
    def batch_first(self) -> bool:
        r"""Whether the logical outer shape uses ``(B, ...)`` instead of ``(..., B, ...)``."""
        return self._batch_first

    @batch_first.setter
    def batch_first(self, value: bool):
        new_value = type(self)._coerce_batch_first(value)
        old_value = getattr(self, "_batch_first", None)
        self._batch_first = new_value
        if old_value is None or old_value == new_value:
            return
        if hasattr(self, "_physical_shape") and hasattr(self, "_offsets") and hasattr(self, "_logical_shape"):
            self._logical_shape = type(self)._logical_shape_from_physical_shape(
                self._physical_shape,
                self._offsets,
                new_value,
            )
        if hasattr(self, "_cached_tensor_view"):
            self._invalidate_transient_caches()

    @property
    def padding_value(self) -> float:
        r"""Padding fill value used when materializing dense views."""
        return self._padding_value

    @padding_value.setter
    def padding_value(self, value: SupportsFloat):
        new_value = type(self)._coerce_padding_value(value)
        old_value = getattr(self, "_padding_value", None)
        self._padding_value = new_value
        if old_value is None or old_value == new_value:
            return
        if hasattr(self, "_cached_tensor_view"):
            self._cached_tensor_view = None

    @property
    def mask_value(self) -> bool:
        r"""Boolean value used to denote padding positions in generated masks."""
        return self._mask_value

    @mask_value.setter
    def mask_value(self, value: bool):
        new_value = type(self)._coerce_mask_value(value)
        old_value = getattr(self, "_mask_value", None)
        self._mask_value = new_value
        if old_value is None or old_value == new_value:
            return
        if hasattr(self, "_cached_mask_view"):
            self._cached_mask_view = None

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        r"""Data type of the underlying tensor elements."""
        return self._values.dtype

    @dtype.setter
    def dtype(self, value: torch.dtype | None):
        r"""`dtype` is read-only; use `.to(dtype=...)` to convert."""
        raise AttributeError("NestedTensor.dtype is read-only; use .to(dtype=...) to create a converted tensor.")

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        r"""Device on which the underlying tensor data resides."""
        return self._values.device

    @device.setter
    def device(self, value: torch.device | None):
        r"""`device` is read-only; use `.to(device=...)` to move tensors."""
        raise AttributeError("NestedTensor.device is read-only; use .to(device=...) to create a moved tensor.")

    @property
    def requires_grad(self) -> bool:  # type: ignore[override]
        r"""Whether gradient computation is enabled for this tensor."""
        return self._values.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        r"""Enable or disable gradient computation for this tensor."""
        self._values.requires_grad_(value)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _meta(self, *, include_dtype: bool | None = None) -> Mapping:
        r"""Metadata used for structure-preserving reconstruction."""
        if include_dtype is None:
            # Empty reconstructions cannot infer dtype from storage; include it by default.
            include_dtype = self._values.numel() == 0
        if include_dtype:
            return {
                "batch_first": self.batch_first,
                "padding_value": self.padding_value,
                "mask_value": self.mask_value,
                "pin_memory": self._pin_memory,
                "device": self._values.device,
                "dtype": self.dtype,
            }
        return {
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "pin_memory": self._pin_memory,
            "device": self._values.device,
        }

    def __getstate__(self) -> dict:
        return {
            "_state_version": self._SERIALIZATION_VERSION,
            "_values": self._values,
            "_offsets": self._offsets,
            "_permutation": self._permutation,
            "_physical_shape": self._physical_shape,
            "_logical_shape": self._logical_shape,
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "_pin_memory": self._pin_memory,
            "_packed_sizes": self._packed_sizes,
            "_element_shapes": self._element_shapes,
        }

    def __setstate__(self, state: Mapping) -> None:
        type(self)._validate_serialized_state(state)
        self._values = state["_values"]
        self._offsets = state["_offsets"].cpu()
        self._permutation = tuple(int(dim) for dim in state["_permutation"])
        self._physical_shape = state["_physical_shape"].cpu()
        self._logical_shape = state["_logical_shape"]
        self._set_runtime_config(
            batch_first=state["batch_first"],
            padding_value=state["padding_value"],
            mask_value=state["mask_value"],
        )
        self._pin_memory = bool(state["_pin_memory"] and self._values.device.type == "cpu" and self._values.is_pinned())
        self._packed_sizes = state["_packed_sizes"]
        self._element_shapes = state["_element_shapes"]
        # Serialized state intentionally excludes transient caches.
        self._invalidate_transient_caches()
        self._validate_metadata()

    def __reduce__(self):
        return (self.__class__._from_state, (self.__getstate__(),))

    @classmethod
    def _from_state(cls, state: dict) -> Self:
        cls._validate_serialized_state(state)
        return cls._from_packed(
            state["_values"],
            state["_offsets"].cpu(),
            state["_physical_shape"].cpu(),
            permutation=tuple(int(dim) for dim in state["_permutation"]),
            batch_first=state["batch_first"],
            padding_value=state["padding_value"],
            mask_value=state["mask_value"],
            pin_memory=state["_pin_memory"],
            outer_size=state["_logical_shape"],
            packed_sizes=state["_packed_sizes"],
            element_shapes=state["_element_shapes"],
        )

    def __copy__(self):
        r"""Shallow copy: new NestedTensor sharing underlying tensor data."""
        return self.__class__._from_packed(
            self._values,
            self._offsets,
            self._physical_shape,
            permutation=self._permutation,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
        )

    def __deepcopy__(self, memo):
        r"""Deep copy: clones all tensor data."""
        result = self.__class__._from_packed(
            self._values.clone(),
            self._offsets.clone(),
            self._physical_shape.clone(),
            permutation=self._permutation,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
        )
        memo[id(self)] = result
        return result

    # ------------------------------------------------------------------
    # Tensor-like methods
    # ------------------------------------------------------------------

    def all(self, dim: int | None = None, keepdim: bool = False) -> bool | Tensor | NestedTensor:
        r"""
        Tests if all elements in NestedTensor evaluate to True.

        Examples:
            >>> nested_tensor = NestedTensor([torch.ones(2, 4, dtype=torch.bool), torch.ones(3, 5, dtype=torch.bool)])
            >>> nested_tensor.all()
            tensor(True)
            >>> nested_tensor.all(dim=0)
            tensor([True, True])
            >>> nested_tensor.all(dim=0, keepdim=True)
            tensor([[True, True]])
            >>> nested_tensor.all(dim=1)
            NestedTensor([
                [True, True, True, True],
                [True, True, True, True, True]
            ])
            >>> nested_tensor.all(dim=1, keepdim=True)
            NestedTensor([
                [[True, True, True, True]],
                [[True, True, True, True, True]]
            ])
            >>> nested_tensor.batch_first = False
            >>> nested_tensor.all(dim=1)
            tensor([True, True])
            >>> nested_tensor.all(dim=0)
            NestedTensor([
                [True, True, True, True],
                [True, True, True, True, True]
            ])
            >>> nested_tensor.all(dim=-2)
            tensor([True, True])
        """
        return torch.all(self, dim=dim, keepdim=keepdim)

    def any(self, dim: int | None = None, keepdim: bool = False) -> bool | Tensor | NestedTensor:
        r"""
        Tests if any elements in NestedTensor evaluate to True.

        Examples:
            >>> nested_tensor = NestedTensor([torch.zeros(2, dtype=torch.bool), torch.ones(3, dtype=torch.bool)])
            >>> nested_tensor.any()
            tensor(True)
            >>> nested_tensor.any(dim=0)
            tensor([False,  True])
        """
        return torch.any(self, dim=dim, keepdim=keepdim)

    def dim(self) -> int:
        r"""
        Number of dimension of the NestedTensor.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.dim()
            2
        """
        if not hasattr(self, "_logical_shape"):
            with torch._C.DisableTorchFunctionSubclass():
                return len(torch.Tensor.size(self))
        return len(self._logical_shape)

    def max(self, dim: int | None = None, keepdim: bool = False) -> Tensor | NestedTensor:
        r"""Return the maximum value, optionally along a given dimension."""
        if dim is None:
            return torch.max(self)
        return torch.max(self, dim=dim, keepdim=keepdim)

    def mean(
        self,
        dim: int | None = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,  # type: ignore[name-defined]
    ) -> Tensor | NestedTensor:
        r"""Return the mean value, optionally along a given dimension."""
        return torch.mean(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def min(self, dim: int | None = None, keepdim: bool = False) -> Tensor | NestedTensor:
        r"""Return the minimum value, optionally along a given dimension."""
        if dim is None:
            return torch.min(self)
        return torch.min(self, dim=dim, keepdim=keepdim)

    @property
    def mT(self) -> Self:  # type: ignore[override]
        r"""Matrix transpose over the last two per-element dimensions."""
        ndims = self.dim()
        batch_dim = 0 if self.batch_first else 1
        elem_dims = [d for d in range(ndims) if d != batch_dim]
        if len(elem_dims) < 2:
            raise RuntimeError(
                f"tensor.mT is only supported on matrices or batches of matrices. Got {len(elem_dims)}-D tensor."
            )
        return torch.transpose(self, elem_dims[-2], elem_dims[-1])

    @property
    def ndim(self) -> int:
        r"""
        Alias for `dim()`.
        """
        return self.dim()

    def numel(self) -> int:
        r"""
        Number of elements in the NestedTensor.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.numel()
            5
        """
        return self._values.numel()

    def permute(self, *dims) -> Self:
        r"""
        Apply permutation to each tensor in the NestedTensor.

        Args:
            *dims: The desired ordering of dimensions for the NestedTensor (including batch dimension).

        Returns:
            NestedTensor: A new NestedTensor with each tensor permuted.

        Examples:
            >>> nested_tensor = NestedTensor([torch.randn(3, 4, 5), torch.randn(2, 4, 5)])
            >>> permuted = nested_tensor.permute(0, 3, 1, 2)
            >>> permuted.shape
            torch.Size([2, 5, 3, 4])
        """
        return torch.permute(self, dims)

    def moveaxis(self, source, destination) -> Self:
        r"""Move per-element dimensions to new positions."""
        return torch.moveaxis(self, source, destination)

    def movedim(self, source, destination) -> Self:
        r"""Alias for `moveaxis()`."""
        return torch.movedim(self, source, destination)

    # to(), clone(), detach(), contiguous(), half(), float(), double(), etc.
    # are all handled by aten dispatch in aten_functions.py (aten._to_copy, aten.clone,
    # aten.detach). No custom Python methods needed.

    def pin_memory(self) -> Self:
        r"""Pin the underlying tensor memory for faster host-to-device transfer."""
        return type(self)._from_packed(
            self._values.pin_memory(),
            self._offsets,
            self._physical_shape,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=True,
            outer_size=self._logical_shape,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
        )

    def prod(
        self,
        dim: int | None = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,  # type: ignore[name-defined]
    ) -> Tensor | NestedTensor:
        r"""Return the product of elements, optionally along a given dimension."""
        return torch.prod(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def requires_grad_(self, requires_grad: bool = True):
        r"""Enable or disable gradient computation in-place."""
        self.requires_grad = requires_grad
        return self

    def reshape(self, *shape) -> Self:
        r"""
        Reshape each tensor in the NestedTensor.

        Args:
            *shape: The desired size of each dimension for the underlying tensors.

        Returns:
            NestedTensor: A new NestedTensor with each tensor reshaped.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])])
            >>> reshaped = nested_tensor.reshape(4)
            >>> reshaped.shape
            torch.Size([2, 4])
        """
        if not shape:
            raise TypeError("reshape() missing shape")
        target_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else shape
        return torch.reshape(self, target_shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        r"""Flatten each tensor in the NestedTensor."""
        return torch.flatten(self, start_dim=start_dim, end_dim=end_dim)

    def flip(self, dims) -> Self:
        r"""Flip each tensor in the NestedTensor along the given dimensions."""
        return torch.flip(self, dims)

    @property
    def shape(self) -> torch.Size:  # type: ignore[override, name-defined]
        r"""
        Alias for `size()`.
        """
        return self.size()

    def size(self, dim: int | None = None) -> torch.Size | int:  # type: ignore[override, name-defined]
        r"""
        Returns the size of the self `NestedTensor`.

        Args:
            dim: If not specified, the returned value is a `torch.Size`, a subclass of `tuple`.
                If specified, returns an `int` holding the size of that dimension.
                Defaults to `None`.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.size()
            torch.Size([2, 3])
            >>> nested_tensor.size(0)
            2
            >>> nested_tensor[1] = torch.tensor([4, 5, 6, 7])
            >>> nested_tensor.shape
            torch.Size([2, 4])
            >>> nested_tensor.size(1)
            4
        """
        if hasattr(self, "_logical_shape"):
            full_size = self._logical_shape
        else:
            with torch._C.DisableTorchFunctionSubclass():
                full_size = torch.Tensor.size(self)
        if dim is not None:
            dim = dim + len(full_size) if dim < 0 else dim
            return full_size[dim]
        return full_size

    def sum(
        self,
        dim: int | Sequence[int] | None = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,  # type: ignore[name-defined]
    ) -> Tensor | NestedTensor:
        r"""
        Returns the sum of each tensor over the given dimension(s).

        Args:
            dim: The dimension or dimensions to reduce. If None, sum over all dimensions.
                Supports int, Sequence[int], or None. Negative dimensions are supported.
            keepdim: Whether to retain reduced dimensions with size 1.
            dtype: The desired data type of returned tensor.

        Returns:
            Tensor or NestedTensor depending on the dimensions being reduced.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.sum()
            tensor(15)
            >>> nested_tensor.sum(dim=0)  # when dim=0, sum across batch dimension
            tensor([6, 9])
            >>> nested_tensor.sum(dim=1)
            tensor([6, 9])
            >>> nested_tensor.sum(dim=[0, 1])
            tensor(15)
            >>> nested_tensor.sum(dim=0, keepdim=True)
            tensor([[6, 9]])
            >>> nested_tensor.sum(dtype=torch.float32)
            tensor(15.)
        """
        return torch.sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    @property
    def T(self) -> Self:  # type: ignore[override]
        r"""Transpose: reverse per-element dims while keeping batch dim fixed."""
        ndims = self.dim()
        if ndims <= 1:
            return self
        batch_dim = 0 if self.batch_first else 1
        elem_dims = [d for d in range(ndims) if d != batch_dim]
        order = list(reversed(elem_dims))
        order.insert(batch_dim, batch_dim)
        return torch.permute(self, tuple(order))

    def tolist(self) -> list:
        r"""
        Convert a NestedTensor to a list of lists of values.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tolist()
            [[1, 2, 3], [4, 5]]
        """
        return [t.tolist() for t in self._storage]

    def transpose(self, dim0: int, dim1: int) -> Self:  # type: ignore[valid-type]
        r"""
        Transpose dimensions dim0 and dim1 for each tensor in the NestedTensor.

        Args:
            dim0: First dimension to transpose (in NestedTensor coordinate system).
            dim1: Second dimension to transpose (in NestedTensor coordinate system).

        Returns:
            NestedTensor: A new NestedTensor with each tensor transposed.

        Examples:
            >>> nested_tensor = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
            >>> # NestedTensor shape is [2, 3, 4], underlying tensors are [3, 4] and [2, 4]
            >>> transposed = nested_tensor.transpose(1, 2)  # transpose dims 1 and 2
            >>> transposed.shape  # batch dimension is still first
            torch.Size([2, 4, 3])
        """
        return torch.transpose(self, dim0, dim1)

    def swapaxes(self, axis0: int, axis1: int) -> Self:
        r"""Alias for `transpose()`."""
        return torch.swapaxes(self, axis0, axis1)

    def swapdims(self, dim0: int, dim1: int) -> Self:
        r"""Alias for `swapaxes()`."""
        return torch.swapdims(self, dim0, dim1)

    def squeeze(self, dim: int | None = None) -> Self:  # type: ignore[valid-type]
        r"""Squeeze singleton dimensions from each tensor in the NestedTensor."""
        return torch.squeeze(self, dim=dim)

    def unsqueeze(self, dim: int) -> Self:  # type: ignore[valid-type]
        r"""
        Unsqueeze each tensor in the NestedTensor by adding a singleton dimension at the specified position.

        Args:
            dim: The dimension at which to add the singleton dimension. This is in the NestedTensor's
                coordinate system (where dim 0 is the batch dimension).

        Returns:
            NestedTensor: A new NestedTensor with each tensor unsqueezed at the specified dimension.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> # Original shape: [2, 3] (batch_size=2, max_seq_len=3)
            >>> unsqueezed = nested_tensor.unsqueeze(1)
            >>> unsqueezed.shape
            torch.Size([2, 1, 3])
            >>> # Now each underlying tensor has shape [1, seq_len] instead of [seq_len]

            >>> nested_tensor_2d = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
            >>> # Original shape: [2, 3, 4] (batch_size=2, max_len1=3, max_len2=4)
            >>> unsqueezed_2d = nested_tensor_2d.unsqueeze(2)
            >>> unsqueezed_2d.shape
            torch.Size([2, 3, 1, 4])
            >>> # Now each underlying tensor has shape [len1, 1, len2] instead of [len1, len2]
        """
        return torch.unsqueeze(self, dim)

    def unflatten(self, dim: int, sizes) -> Self:  # type: ignore[valid-type]
        r"""Unflatten one dimension of each tensor in the NestedTensor."""
        return torch.unflatten(self, dim, sizes)

    def roll(self, shifts, dims=None) -> Self:
        r"""Roll each tensor in the NestedTensor along the given dimensions."""
        return torch.roll(self, shifts, dims=dims)

    def rot90(self, k: int = 1, dims: Sequence[int] = (0, 1)) -> Self:
        r"""Rotate each tensor in the NestedTensor by 90 degrees in the given plane."""
        return torch.rot90(self, k, dims)

    def view(self, *shape) -> Self:
        r"""
        View each tensor in the NestedTensor with a different shape.

        Args:
            *shape: The desired size of each dimension for the underlying tensors.

        Returns:
            NestedTensor: A new NestedTensor with each tensor viewed with the new shape.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])])
            >>> viewed = nested_tensor.view(4)  # View each 2x2 tensor as 4
            >>> viewed.shape
            torch.Size([2, 4])
            >>> type(viewed).__name__
            'NestedTensor'
        """
        if not shape:
            raise TypeError("view() missing shape")
        target_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else shape
        return NestedTensorAtenRegistry[torch.ops.aten.view.default](
            torch.ops.aten.view.default, (self, list(target_shape)), {}
        )

    def _view_shapes(self, shape) -> list[tuple[int, ...]]:  # type: ignore[valid-type]
        r"""
        Compute per-element view shapes, adjusting ragged dimensions.

        Batch-dim detection rules:
        1. If ``shape[batch_dim]`` does not match the batch size, batch dim is NOT included.
        2. If ``len(shape) != self.dim()``, batch dim IS included (unambiguous).
        3. If ``len(shape) == self.dim()`` (ambiguous), batch dim is included only when
           at least one other dimension matches max_sizes or is -1.

        For ragged dimensions, each target dimension that matches the corresponding
        max size is substituted with the element's actual size. When a target dimension
        matches a max size at a different position (e.g. after inserting a dim), a
        single-candidate search resolves the mapping.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])

        batch_dim = 0 if self.batch_first else 1
        batch_size = len(self)

        # Step 1: Determine if batch dim is in the target shape
        include_batch = False
        if len(shape) > batch_dim:
            if shape[batch_dim] == batch_size and len(shape) != self.dim():
                include_batch = True
            elif shape[batch_dim] in (-1, batch_size) and len(shape) == self.dim():
                # Ambiguous: same dim count → confirm via dimension matching
                max_sizes = list(self.size())  # type: ignore[arg-type]
                if max_sizes:
                    max_sizes.pop(batch_dim)
                non_batch = [i for i in range(len(shape)) if i != batch_dim]
                include_batch = any(
                    j < len(max_sizes) and (shape[d] == -1 or shape[d] == max_sizes[j]) for j, d in enumerate(non_batch)
                )

        # Step 2: Strip batch dim from target shape
        target = list(shape)
        if include_batch:
            if target[batch_dim] == -1:
                target[batch_dim] = batch_size
            if target[batch_dim] != batch_size:
                raise ValueError(f"Batch dimension mismatch: expected {batch_size} but got {target[batch_dim]}")
            target.pop(batch_dim)

        # Step 3: Per-element shape adjustment (ragged dim substitution)
        max_sizes = list(self.size())  # type: ignore[arg-type]
        if max_sizes:
            max_sizes.pop(batch_dim)

        element_shapes = self._element_shapes
        if element_shapes is None:
            element_shapes = tuple(tuple(shape) for shape in self._original_shapes())

        view_shapes = []
        for element_shape in element_shapes:
            adjusted = list(target)
            available = list(range(len(max_sizes)))
            for i in range(min(len(adjusted), len(max_sizes))):
                if adjusted[i] == -1:
                    continue
                # Direct match: same position in max_sizes
                if adjusted[i] == max_sizes[i]:
                    adjusted[i] = element_shape[i]
                    if i in available:
                        available.remove(i)
                    continue
                # Indirect match: search remaining positions for unique candidate
                candidates = [j for j in available if max_sizes[j] == adjusted[i]]
                if len(candidates) == 1:
                    j = candidates[0]
                    adjusted[i] = element_shape[j]
                    available.remove(j)
            if adjusted.count(-1) == 1:
                missing = adjusted.index(-1)
                known = 1
                for dim in adjusted:
                    if dim != -1:
                        known *= dim
                element_numel = type(self)._shape_numel(element_shape)
                if known != 0 and element_numel % known == 0:
                    adjusted[missing] = element_numel // known
            view_shapes.append(tuple(adjusted))
        return view_shapes

    def where(self, condition: Tensor | NestedTensor, other: Tensor | NestedTensor | SupportsFloat) -> Self:
        r"""
        Return a NestedTensor of elements selected from either self or other, depending on condition.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.where(nested_tensor > 2, torch.tensor([[6, 5, 4], [3, 2, 1]]))
            NestedTensor([
                [6, 5, 3],
                [4, 5]
            ])
            >>> nested_tensor.where(nested_tensor > 2, NestedTensor([[6, 5, 4], [3, 2]]))
            NestedTensor([
                [6, 5, 3],
                [4, 5]
            ])
            >>> nested_tensor.where(torch.tensor(True), NestedTensor([[6, 5, 4], [3, 2]]))
            NestedTensor([
                [1, 2, 3],
                [4, 5]
            ])
        """
        return torch.where(condition, self, other)
