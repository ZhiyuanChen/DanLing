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

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Iterable, SupportsFloat, cast

import torch
from torch import Tensor

from .aten_functions import _is_fake_tensor, per_element_fallback
from .ops import (
    NestedTensorAtenRegistry,
    _batch_leading_valid_mask_from_sizes,
    _check_execution_guard,
    _compile_unsupported,
    _ExecutionGuardKind,
    _is_compiling,
    _physical_to_values_dim,
)

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from torch import nested

_INT64_MAX = torch.iinfo(torch.int64).max
_INT64_SQUARE_ROOT_MAX = math.isqrt(_INT64_MAX)


def _validate_concrete_square_lengths(lengths: Tensor) -> None:
    r"""Reject square metadata that cannot be represented by signed int64."""
    concrete_lengths = lengths.tolist()
    if any(length < 0 for length in concrete_lengths):
        raise ValueError("lengths must be non-negative")
    if any(length > _INT64_SQUARE_ROOT_MAX for length in concrete_lengths):
        raise ValueError("each squared length must fit in torch.int64")
    if sum(length * length for length in concrete_lengths) > _INT64_MAX:
        raise ValueError("the cumulative sum of squared lengths must fit in torch.int64")


@torch.library.custom_op("danling::_metadata_tensors_equal", mutates_args=())
def _metadata_tensors_equal(lhs: Tensor, rhs: Tensor) -> Tensor:
    r"""Compare metadata tensors opaquely, including shape and values."""
    return lhs.new_tensor(torch.equal(lhs, rhs), dtype=torch.bool)


@_metadata_tensors_equal.register_fake
def _metadata_tensors_equal_fake(lhs: Tensor, rhs: Tensor) -> Tensor:
    del rhs
    return lhs.new_empty((), dtype=torch.bool)


@torch.library.custom_op("danling::_square_row_splits", mutates_args=())
def _square_row_splits(lengths: Tensor) -> Tensor:
    r"""Build the innermost CSR row splits for square ragged elements."""
    # Keep the allocation boundary independently guarded. Under torch.compile this
    # custom op executes with concrete tensors, while its fake implementation below
    # only describes the output. The duplicate check therefore prevents backend
    # scheduling from reaching repeat_interleave before the graph assertions run.
    _validate_concrete_square_lengths(lengths)
    row_widths = torch.repeat_interleave(lengths, lengths)
    return torch.nn.functional.pad(row_widths.cumsum(0), (1, 0))


@_square_row_splits.register_fake
def _square_row_splits_fake(lengths: Tensor) -> Tensor:
    ctx = torch.library.get_ctx()
    row_splits_size = ctx.new_dynamic_size(min=1)
    return lengths.new_empty((row_splits_size,))


@torch.library.custom_op("danling::_rectangular_row_splits", mutates_args=())
def _rectangular_row_splits(row_lengths: Tensor, column_lengths: Tensor) -> Tensor:
    r"""Build innermost CSR row splits for rectangular ragged elements."""
    row_widths = torch.repeat_interleave(column_lengths, row_lengths)
    return torch.nn.functional.pad(row_widths.cumsum(0), (1, 0))


@_rectangular_row_splits.register_fake
def _rectangular_row_splits_fake(row_lengths: Tensor, column_lengths: Tensor) -> Tensor:
    del column_lengths
    ctx = torch.library.get_ctx()
    row_splits_size = ctx.new_dynamic_size(min=1)
    return row_lengths.new_empty((row_splits_size,))


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
        ragged_dims: Optional ordered element dimensions that remain ragged even
            when their sizes happen to be equal within a batch. ``None`` keeps
            the existing shape-inference behavior.
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
    _compile_max_length_binding: Tensor
    _logical_shape: torch.Size
    _ragged_dims: tuple[int, ...]
    _ragged_dims_explicit: bool
    _batch_first: bool
    _padding_value: float
    _mask_value: bool
    _pin_memory: bool
    _packed_sizes: tuple[int, ...] | None
    _element_shapes: tuple[tuple[int, ...], ...] | None
    _cached_storage: tuple[Tensor, ...] | None
    _cached_hierarchical_offsets: tuple[Tensor, ...] | None
    _cached_tensor_view: tuple[bool, float, tuple[int, ...], Tensor] | None
    _cached_mask_view: tuple[bool, bool, tuple[int, ...], Tensor] | None
    _cached_packed_batch_indices: dict[tuple[str, torch.dtype, tuple[int, ...]], Tensor] | None
    _cached_packed_local_indices: dict[tuple[int, str, torch.dtype, tuple[int, ...]], Tensor] | None
    _cached_packed_offsets: dict[tuple[str, torch.dtype, tuple[int, ...]], Tensor] | None
    _cached_ragged_level_offsets: dict[tuple[int, str, torch.dtype, tuple[int, ...]], Tensor] | None
    _aot_concat_projection: Tensor | None
    _allow_aot_concat_update: bool
    _RAGGED_OFFSETS_PREFIX = "_ragged_offsets_"
    _SERIALIZATION_VERSION = 3
    _AOT_CACHE_HASH_VERSION = 3

    # Construction & Initialization

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._compiled_packed_constructor = staticmethod(_make_nested_tensor_from_packed_constructor(cls))

    @staticmethod
    def __new__(
        cls,
        *tensors: Iterable[Tensor],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool | None = None,
        pin_memory: bool = False,
        ragged_dims: tuple[int, ...] | None = None,
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
            ragged_dims=ragged_dims,
        )
        values = cls._maybe_pin_values(values, pin_memory)
        if ragged_dims is None:
            resolved_ragged_dims, static_dims = cls._pack_layout_from_element_shapes(element_shapes)
        else:
            resolved_ragged_dims, static_dims = cls._pack_layout_from_declared_ragged_dims(element_shapes, ragged_dims)
        permutation = resolved_ragged_dims + static_dims

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
        result._ragged_dims = resolved_ragged_dims
        result._ragged_dims_explicit = ragged_dims is not None
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
        ragged_offsets = cls._resolve_persistent_ragged_offsets(
            offsets,
            shape_tensor,
            permutation=permutation,
            ragged_dims=resolved_ragged_dims if ragged_dims is not None else None,
            element_shapes=element_shapes,
        )
        cls._install_persistent_ragged_offsets(result, ragged_offsets)
        result._invalidate_transient_caches()
        result._mark_tensor_backed_dynamic_dims()
        cls._validate_packed_metadata(
            result._values,
            result._offsets,
            result._physical_shape,
            permutation=result._permutation,
            ragged_dims=result._ragged_dims,
            logical_shape=result._logical_shape,
            batch_first=result.batch_first,
            packed_sizes=result._packed_sizes,
            element_shapes=result._element_shapes,
            ragged_offsets=ragged_offsets,
        )
        if torch.is_grad_enabled() and values.requires_grad:
            return _PackedLikeAutograd.apply(values, _PackedStructureReference(result))
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
        ragged_dims: tuple[int, ...] | None = None,
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
        declared_layout = None
        if ragged_dims is not None:
            declared_layout = NestedTensor._pack_layout_from_declared_ragged_dims(element_shapes, ragged_dims)

        # Offsets and shape_tensor are metadata - always on CPU to avoid CUDA syncs.
        shape_tensor = torch.tensor([list(t.shape) + [0] * (max_ndim - t.ndim) for t in tensors], dtype=torch.long)
        if max_ndim == 0:
            values = torch.stack(tensors)
            sizes = torch.ones(len(tensors), dtype=torch.long)
            packed_sizes = tuple(1 for _ in tensors)
        else:
            if permutation is None:
                if declared_layout is None:
                    varying_dims, static_dims = NestedTensor._pack_layout_from_element_shapes(element_shapes)
                else:
                    varying_dims, static_dims = declared_layout
                permutation = varying_dims + static_dims
            else:
                permutation = tuple(int(dim) for dim in permutation)
                if len(permutation) != max_ndim or tuple(sorted(permutation)) != tuple(range(max_ndim)):
                    raise ValueError(f"Invalid permutation dims {permutation} for tensors with rank {max_ndim}")
                if declared_layout is None:
                    ragged_rank = len(NestedTensor._hierarchical_level_sizes_from_element_shapes(element_shapes))
                    varying_dims = permutation[:ragged_rank]
                    static_dims = permutation[ragged_rank:]
                else:
                    varying_dims, _ = declared_layout
                    if permutation[: len(varying_dims)] != varying_dims:
                        raise ValueError(
                            "permutation must begin with ragged_dims in the declared order, "
                            f"got permutation={permutation} and ragged_dims={varying_dims}"
                        )
                    static_dims = permutation[len(varying_dims) :]
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
    def _normalize_ragged_dims(ragged_dims: tuple[int, ...], physical_rank: int) -> tuple[int, ...]:
        if not isinstance(ragged_dims, tuple):
            raise TypeError(f"ragged_dims must be a tuple of ints or None, got {type(ragged_dims).__name__}")
        normalized: list[int] = []
        for dim in ragged_dims:
            if isinstance(dim, torch.SymInt):
                # AOTAutograd can round-trip literal layout dimensions through
                # wrapper-subclass metadata as constant SymInts.  Layout axes are
                # structural integers, never data-dependent sizes, so normalize
                # those constants before applying the public type/range contract.
                dim = int(dim)
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise TypeError(f"ragged_dims must contain only ints, got {type(dim).__name__}")
            normalized_dim = dim + physical_rank if dim < 0 else dim
            if normalized_dim < 0 or normalized_dim >= physical_rank:
                raise ValueError(f"ragged_dims contains dim {dim} outside element rank {physical_rank}")
            if normalized_dim in normalized:
                raise ValueError(f"ragged_dims must not contain duplicate dimensions, got {ragged_dims}")
            normalized.append(normalized_dim)
        return tuple(normalized)

    @staticmethod
    def _is_tensor_backed_layout(
        permutation: tuple[int, ...] | None,
        ragged_dims: tuple[int, ...] | None,
    ) -> bool:
        r"""Return whether tensor metadata fully encodes an explicit packed-prefix layout."""
        if not ragged_dims or permutation is None:
            return False
        ragged_rank = len(ragged_dims)
        return tuple(permutation[:ragged_rank]) == tuple(ragged_dims)

    @classmethod
    def _ragged_offset_names(cls, ragged_rank: int) -> tuple[str, ...]:
        r"""Return stable wrapper-child names for persistent multi-level row splits."""
        if ragged_rank <= 1:
            return ()
        return tuple(f"{cls._RAGGED_OFFSETS_PREFIX}{level}" for level in range(ragged_rank))

    @classmethod
    def _build_explicit_ragged_offsets(
        cls,
        shape_tensor: Tensor,
        ragged_dims: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, ...]:
        r"""Build CSR row splits for an explicit packed-prefix ragged hierarchy."""
        if _is_fake_tensor(shape_tensor):
            raise RuntimeError("Cannot derive explicit multi-ragged offsets from data-less FakeTensor shape metadata.")
        batch_size = int(shape_tensor.size(0))
        parent_counts = torch.ones(batch_size, dtype=torch.long, device=shape_tensor.device)
        offsets: list[Tensor] = []
        for dim in ragged_dims:
            widths = torch.repeat_interleave(shape_tensor[:, dim].to(torch.long), parent_counts)
            offsets.append(cls._offsets_from_sizes(widths, dtype=dtype).contiguous())
            parent_counts = parent_counts * shape_tensor[:, dim].to(torch.long)
        return tuple(offsets)

    @classmethod
    def _resolve_persistent_ragged_offsets(
        cls,
        offsets: Tensor,
        shape_tensor: Tensor,
        *,
        permutation: tuple[int, ...] | None,
        ragged_dims: tuple[int, ...] | None,
        ragged_offsets: tuple[Tensor, ...] | None = None,
        element_shapes: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[Tensor, ...] | None:
        r"""Resolve persistent row splits for an explicit packed-prefix ragged layout."""
        if not cls._is_tensor_backed_layout(permutation, ragged_dims):
            if ragged_offsets is not None:
                raise ValueError("ragged_offsets require an explicit layout whose packed order begins with ragged_dims")
            return None
        assert ragged_dims is not None
        ragged_rank = len(ragged_dims)
        if ragged_rank == 1:
            if ragged_offsets is not None:
                if len(ragged_offsets) != 1:
                    raise ValueError(f"Expected one ragged offset tensor, got {len(ragged_offsets)}")
                supplied = ragged_offsets[0]
                if (
                    supplied is not offsets
                    and not (_is_fake_tensor(supplied) or _is_fake_tensor(offsets))
                    and not torch.equal(supplied, offsets)
                ):
                    raise ValueError("The supplied single-level ragged offsets must match offsets")
            return (offsets,)
        if ragged_offsets is not None:
            if len(ragged_offsets) != ragged_rank:
                raise ValueError(f"Expected {ragged_rank} ragged offset tensors, got {len(ragged_offsets)}")
            return tuple(ragged_offsets)
        if _is_fake_tensor(shape_tensor):
            if element_shapes is None:
                _compile_unsupported(
                    "NestedTensor._from_packed",
                    "explicit multi-ragged FakeTensor rebuilds require concrete element shapes or persistent offsets",
                )
            assert element_shapes is not None
            level_sizes = cls._hierarchical_level_sizes_from_element_shapes(element_shapes, ragged_dims)
            return tuple(offsets.new_empty((len(sizes) + 1,), dtype=offsets.dtype) for sizes in level_sizes)
        if _is_compiling():
            _compile_unsupported(
                "NestedTensor._from_packed",
                "explicit multi-ragged rebuilds require persistent ragged offset tensors",
            )
        return cls._build_explicit_ragged_offsets(shape_tensor, ragged_dims, dtype=offsets.dtype)

    @classmethod
    def _install_persistent_ragged_offsets(
        cls,
        result: Self,
        ragged_offsets: tuple[Tensor, ...] | None,
    ) -> None:
        r"""Install persistent multi-level row splits as traceable wrapper children."""
        for name in tuple(vars(result)):
            if name.startswith(cls._RAGGED_OFFSETS_PREFIX):
                delattr(result, name)
        if ragged_offsets is None or len(ragged_offsets) <= 1:
            return
        for name, level_offsets in zip(cls._ragged_offset_names(len(ragged_offsets)), ragged_offsets):
            setattr(result, name, level_offsets)

    def _persistent_ragged_offsets(self) -> tuple[Tensor, ...] | None:
        r"""Return tensor-backed row splits when this instance owns a complete topology."""
        instance_attrs = vars(self)
        required = ("_offsets", "_permutation", "_ragged_dims", "_ragged_dims_explicit")
        if any(name not in instance_attrs for name in required):
            return None
        declared_ragged_dims = self._ragged_dims if self._ragged_dims_explicit else None
        if not type(self)._is_tensor_backed_layout(self._permutation, declared_ragged_dims):
            return None
        if self._ragged_rank == 1:
            # Every explicit single-ragged layout is already packed with that ragged
            # dimension first. Its row splits and physical-shape tensor completely
            # describe the topology, including the ordinary leading-ragged case. Do not
            # retain legacy per-element Python shape caches in the compile contract.
            return (self._offsets,)
        names = type(self)._ragged_offset_names(self._ragged_rank)
        if any(name not in instance_attrs for name in names):
            return None
        return tuple(instance_attrs[name] for name in names)

    @classmethod
    def _pack_layout_from_declared_ragged_dims(
        cls,
        element_shapes: tuple[tuple[int, ...], ...],
        ragged_dims: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        physical_rank = len(element_shapes[0]) if element_shapes else 0
        varying_dims = cls._normalize_ragged_dims(ragged_dims, physical_rank)
        static_dims = tuple(dim for dim in range(physical_rank) if dim not in varying_dims)
        if element_shapes:
            reference = element_shapes[0]
            for dim in static_dims:
                if any(len(shape) != physical_rank or shape[dim] != reference[dim] for shape in element_shapes[1:]):
                    raise ValueError(
                        "Dimensions not listed in ragged_dims must have identical sizes across elements, "
                        f"but dim {dim} varies for shapes {element_shapes}"
                    )
        return varying_dims, static_dims

    def _project_declared_ragged_dims(
        self,
        *,
        prefix: Sequence[int] = (),
        keep_dims: Sequence[int] | None = None,
    ) -> tuple[int, ...] | None:
        r"""Remap declared ragged dimensions through a basic physical-dimension projection."""
        if not self._ragged_dims_explicit:
            return None
        if keep_dims is None:
            keep_dims = tuple(range(self._physical_shape.size(1)))
        old_to_new = {int(dim): len(prefix) + index for index, dim in enumerate(keep_dims)}
        return tuple(old_to_new[dim] for dim in self._ragged_dims if dim in old_to_new)

    def _project_permutation(
        self,
        *,
        prefix: Sequence[int] = (),
        keep_dims: Sequence[int] | None = None,
        suffix: Sequence[int] = (),
    ) -> tuple[int, ...]:
        r"""Remap packed dimension order through a projection with dense prefix/suffix dimensions."""
        if keep_dims is None:
            keep_dims = tuple(range(self._physical_shape.size(1)))
        keep_dims = tuple(int(dim) for dim in keep_dims)
        prefix_rank = len(prefix)
        old_to_new = {dim: prefix_rank + index for index, dim in enumerate(keep_dims)}
        retained = tuple(old_to_new[dim] for dim in self._permutation if dim in old_to_new)
        prefix_dims = tuple(range(prefix_rank))
        suffix_start = prefix_rank + len(keep_dims)
        suffix_dims = tuple(range(suffix_start, suffix_start + len(suffix)))
        retained_set = set(retained)
        added_static = tuple(dim for dim in (*prefix_dims, *suffix_dims) if dim not in retained_set)
        return retained + added_static

    @classmethod
    def _ragged_dims_from_packed_layout(
        cls,
        values: Tensor,
        physical_rank: int,
        permutation: tuple[int, ...],
        ragged_dims: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if ragged_dims is not None:
            resolved = cls._normalize_ragged_dims(ragged_dims, physical_rank)
            if permutation[: len(resolved)] != resolved:
                raise ValueError(
                    "permutation must begin with ragged_dims in the declared order, "
                    f"got permutation={permutation} and ragged_dims={resolved}"
                )
            return resolved
        ragged_rank = physical_rank - max(values.dim() - 1, 0)
        if ragged_rank < 0 or ragged_rank > physical_rank:
            raise ValueError(
                "Packed values rank is inconsistent with the element rank, "
                f"got values rank {values.dim()} and element rank {physical_rank}"
            )
        return permutation[:ragged_rank]

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
        sizes_tensor = sizes if isinstance(sizes, Tensor) else torch.tensor(sizes, dtype=dtype)
        if sizes_tensor.dtype != dtype:
            sizes_tensor = sizes_tensor.to(dtype)
        if sizes_tensor.numel() == 0:
            return torch.zeros((1,), dtype=dtype, device=sizes_tensor.device)
        return torch.cat([sizes_tensor.new_zeros((1,)), torch.cumsum(sizes_tensor, dim=0)])

    @staticmethod
    def _meta_tensor_equal(
        lhs: Tensor,
        rhs: Tensor,
        message: str = "NestedTensor metadata must match",
        *,
        runtime_assert: bool = False,
    ) -> bool:
        if lhs is rhs:
            return True
        if lhs.dim() != rhs.dim():
            return False
        if _is_compiling() or _is_fake_tensor(lhs) or _is_fake_tensor(rhs):
            from torch.fx.experimental.symbolic_shapes import statically_known_true

            if any(statically_known_true(lhs_size != rhs_size) for lhs_size, rhs_size in zip(lhs.shape, rhs.shape)):
                return False
            if (
                not runtime_assert
                and not _is_compiling()
                and all(statically_known_true(lhs_size == rhs_size) for lhs_size, rhs_size in zip(lhs.shape, rhs.shape))
            ):
                # Standalone FakeTensor execution has nowhere to retain a runtime
                # value assertion. Preserve the conservative eager-Fake contract
                # when the shapes are already known; an unbacked shape still takes
                # the opaque assertion path used by compiled reconstruction.
                return False
            torch._assert_async(_metadata_tensors_equal(lhs, rhs), message)
            return True
        if lhs.shape != rhs.shape:
            return False
        if runtime_assert:
            torch._assert_async(torch.all(lhs == rhs), message)
            return True
        return bool(torch.equal(lhs, rhs))

    @classmethod
    def _hierarchical_level_sizes_from_element_shapes(
        cls,
        element_shapes: tuple[tuple[int, ...], ...],
        ragged_dims: tuple[int, ...] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        if not element_shapes:
            return ()
        if ragged_dims is None:
            varying_dims, _ = cls._pack_layout_from_element_shapes(element_shapes)
        else:
            varying_dims, _ = cls._pack_layout_from_declared_ragged_dims(element_shapes, ragged_dims)
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
        ragged_dims: tuple[int, ...] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        if physical_shape.numel() == 0:
            return ()
        if element_shapes is not None:
            return cls._hierarchical_level_sizes_from_element_shapes(element_shapes, ragged_dims)
        if _is_fake_tensor(physical_shape):
            return ()

        if ragged_dims is None:
            varying_dims, _ = cls._pack_layout_meta(physical_shape, None)
        else:
            varying_dims = cls._normalize_ragged_dims(ragged_dims, int(physical_shape.size(1)))
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
        ragged_dims: tuple[int, ...],
        logical_shape: torch.Size,
        batch_first: bool,
        packed_sizes: tuple[int, ...] | None,
        element_shapes: tuple[tuple[int, ...], ...] | None,
        ragged_offsets: tuple[Tensor, ...] | None,
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
        normalized_ragged_dims = cls._normalize_ragged_dims(ragged_dims, physical_rank)
        if permutation[: len(normalized_ragged_dims)] != normalized_ragged_dims:
            raise ValueError(
                "permutation must begin with ragged_dims in the declared order, "
                f"got permutation={permutation} and ragged_dims={normalized_ragged_dims}"
            )
        static_dims = tuple(int(dim) for dim in permutation[len(normalized_ragged_dims) :])
        expected_values_rank = 1 + len(static_dims)
        if values.dim() != expected_values_rank:
            raise ValueError(
                "Packed values rank is inconsistent with ragged_dims, "
                f"got values rank {values.dim()}, expected {expected_values_rank} for "
                f"physical rank {physical_rank} and ragged_dims={normalized_ragged_dims}"
            )

        tensor_backed_layout = cls._is_tensor_backed_layout(permutation, normalized_ragged_dims)
        if ragged_offsets is not None:
            if not tensor_backed_layout:
                raise ValueError("ragged_offsets require an explicit layout whose packed order begins with ragged_dims")
            if len(ragged_offsets) != len(normalized_ragged_dims):
                raise ValueError(
                    f"Expected {len(normalized_ragged_dims)} ragged offset tensors, got {len(ragged_offsets)}"
                )
            for level, level_offsets in enumerate(ragged_offsets):
                if level_offsets.device.type != "cpu":
                    raise ValueError(f"ragged_offsets[{level}] must be on CPU, got {level_offsets.device}")
                if level_offsets.dim() != 1:
                    raise ValueError(
                        f"ragged_offsets[{level}] must be one-dimensional, got shape {tuple(level_offsets.shape)}"
                    )
                if (
                    level_offsets.dtype.is_floating_point
                    or level_offsets.dtype.is_complex
                    or level_offsets.dtype == torch.bool
                ):
                    raise ValueError(f"ragged_offsets[{level}] must use an integer dtype, got {level_offsets.dtype}")

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
            # The packed leading dimension is the product of declared ragged
            # dimensions.  Every remaining dimension is represented directly
            # in the packed value tail and therefore must be uniform.
            if normalized_shapes:
                cls._pack_layout_from_declared_ragged_dims(normalized_shapes, normalized_ragged_dims)
            expected_packed_sizes = tuple(
                cls._packed_size_from_shape(shape, normalized_ragged_dims) for shape in normalized_shapes
            )
            if packed_sizes is not None and tuple(int(size) for size in packed_sizes) != expected_packed_sizes:
                raise ValueError(
                    "packed_sizes must equal the product of ragged dimensions for every element, "
                    f"got {packed_sizes} and expected {expected_packed_sizes}"
                )
            if normalized_shapes:
                expected_tail = tuple(normalized_shapes[0][dim] for dim in static_dims)
                if tuple(values.shape[1:]) != expected_tail:
                    raise ValueError(
                        "Packed values tail must match static dimensions in permutation order, "
                        f"got {tuple(values.shape[1:])} and expected {expected_tail}"
                    )

        if _is_fake_tensor(offsets) or _is_fake_tensor(shape_tensor):
            return

        if bool((shape_tensor < 0).any()):
            raise ValueError("shape_tensor must be non-negative")
        if int(offsets[0].item()) != 0:
            raise ValueError("offsets must start at 0")
        deltas = offsets[1:] - offsets[:-1]
        if bool((deltas < 0).any()):
            raise ValueError("offsets must be monotonically non-decreasing")
        if packed_sizes is not None:
            delta_sizes = tuple(int(size) for size in deltas.tolist())
            normalized_packed_sizes = tuple(int(size) for size in packed_sizes)
            if delta_sizes != normalized_packed_sizes:
                raise ValueError(
                    "offset deltas must match packed_sizes exactly, " f"got {delta_sizes} and {normalized_packed_sizes}"
                )
        if packed_sizes is None and int(offsets[-1].item()) != int(values.shape[0]):
            raise ValueError(
                f"offsets[-1] must equal packed values length, got offsets[-1]={int(offsets[-1].item())} "
                f"and values.shape[0]={int(values.shape[0])}"
            )
        if ragged_offsets is not None:
            expected_ragged_offsets = cls._build_explicit_ragged_offsets(
                shape_tensor,
                normalized_ragged_dims,
                dtype=offsets.dtype,
            )
            for level, (actual, expected) in enumerate(zip(ragged_offsets, expected_ragged_offsets)):
                if not torch.equal(actual, expected):
                    raise ValueError(f"ragged_offsets[{level}] does not match physical_shape")
            sample_leaf_offsets = ragged_offsets[0]
            for level_offsets in ragged_offsets[1:]:
                sample_leaf_offsets = level_offsets.index_select(0, sample_leaf_offsets.to(torch.long))
            if not torch.equal(sample_leaf_offsets.to(offsets.dtype), offsets):
                raise ValueError("ragged_offsets do not reproduce the packed sample offsets")
            if int(ragged_offsets[-1][-1].item()) != int(values.shape[0]):
                raise ValueError("Final ragged offsets must cover the packed values leading dimension")

    def _validate_metadata(self) -> None:
        r"""Validate the current packed storage and metadata."""
        type(self)._validate_packed_metadata(
            self._values,
            self._offsets,
            self._physical_shape,
            permutation=self._permutation,
            ragged_dims=self._ragged_dims,
            logical_shape=self._logical_shape,
            batch_first=self.batch_first,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
            ragged_offsets=self._persistent_ragged_offsets(),
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
        self._cached_packed_batch_indices = None
        self._cached_packed_local_indices = None
        self._cached_packed_offsets = None
        self._cached_ragged_level_offsets = None
        self._aot_concat_projection = None
        self._allow_aot_concat_update = False

    def _mark_tensor_backed_dynamic_dims(self, *, mark_values: bool = True) -> None:
        r"""Mark packed and logical ragged extents dynamic without guarded user state."""
        ragged_offsets = self._persistent_ragged_offsets()
        if ragged_offsets is None:
            return

        # Go through the public marker rather than writing the private attribute behind it, so
        # the mark keeps working if PyTorch renames its bookkeeping. This is the same call
        # torch's own jagged NestedTensor makes for its ragged dim: a hint that a dimension may
        # vary, which lets a NestedTensor cross a graph boundary without specializing its ragged
        # or packed extent. The stricter `mark_dynamic` is wrong here because it asserts the
        # dimension really is dynamic, and layouts whose ragged extent is fixed would fail it.
        from torch._dynamo import maybe_mark_dynamic

        for physical_dim in self._ragged_dims:
            logical_dim = physical_dim + 1 if self.batch_first or physical_dim > 0 else physical_dim
            maybe_mark_dynamic(self, logical_dim)
        if mark_values:
            maybe_mark_dynamic(self._values, 0)
        for level_offsets in ragged_offsets:
            maybe_mark_dynamic(level_offsets, 0)

    def _values_cache_token(self) -> tuple[int, ...]:
        r"""Return a cache token for views that depend on packed values and layout metadata.

        Tensors created under ``torch.inference_mode`` do not track version
        counters, even after leaving the context. Fall back to object identity
        for those immutable tensors so cached views remain usable.
        """
        return (self._cache_version(self._values), *self._shape_cache_token())

    def _shape_cache_token(self) -> tuple[int, ...]:
        r"""Return a cache token for views that depend only on shape metadata."""
        ragged_offsets = self._persistent_ragged_offsets() or ()
        return (
            self._cache_version(self._offsets),
            self._cache_version(self._physical_shape),
            *(self._cache_version(level_offsets) for level_offsets in ragged_offsets),
        )

    @staticmethod
    def _cache_version(tensor: Tensor) -> int:
        try:
            return int(tensor._version)
        except RuntimeError as exc:
            if "Inference tensors do not track version counter" not in str(exc):
                raise
            return id(tensor)

    @staticmethod
    def _offset_conversion_device_key(device: torch.device) -> str | None:
        r"""Return an unambiguous per-device cache key for derived offsets.

        An index-less non-CPU device follows PyTorch's current-device semantics.
        Its concrete target can therefore change between calls, so it must not
        participate in the conversion cache.  CPU has no per-process current
        index and remains safely cacheable.
        """
        if device.type != "cpu" and device.index is None:
            return None
        return str(device)

    @classmethod
    def _validate_serialized_state(cls, state: Mapping) -> None:
        required = (
            "_state_version",
            "_values",
            "_offsets",
            "_permutation",
            "_ragged_dims",
            "_physical_shape",
            "_logical_shape",
            "batch_first",
            "padding_value",
            "mask_value",
            "_pin_memory",
            "_packed_sizes",
            "_element_shapes",
            "_ragged_offsets",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"Serialized NestedTensor state is missing required keys: {', '.join(missing)}")
        version = state["_state_version"]
        if version != cls._SERIALIZATION_VERSION:
            raise ValueError(f"Unsupported NestedTensor state version {version}; expected {cls._SERIALIZATION_VERSION}")

    @classmethod
    def _from_packed(
        cls,
        values: Tensor,
        offsets: Tensor,
        shape_tensor: Tensor,
        *,
        permutation: tuple[int, ...] | None = None,
        ragged_dims: tuple[int, ...] | None = None,
        batch_first: bool = True,
        padding_value: float = 0.0,
        mask_value: bool = False,
        pin_memory: bool = False,
        outer_size: torch.Size | tuple | None = None,
        packed_sizes: tuple[int, ...] | None = None,
        element_shapes: tuple[tuple[int, ...], ...] | None = None,
        ragged_offsets: tuple[Tensor, ...] | None = None,
        validate: bool = True,
        materialize_python_metadata: bool = True,
        mark_values_dynamic: bool = True,
    ) -> Self:
        r"""Construct a NestedTensor directly from packed representation."""
        # offsets and shape_tensor MUST live on CPU to avoid implicit CUDA syncs
        # when handlers call .item() / .tolist() on them.
        if offsets.device.type != "cpu":
            raise ValueError(f"offsets must be on CPU, got {offsets.device}")
        if shape_tensor.device.type != "cpu":
            raise ValueError(f"shape_tensor must be on CPU, got {shape_tensor.device}")

        compiling = _is_compiling()
        if validate and compiling:
            _compile_unsupported("NestedTensor._from_packed", "metadata validation is eager-only")
        if outer_size is not None:
            logical_shape = torch.Size(outer_size)
        elif compiling:
            _compile_unsupported("NestedTensor._from_packed", "outer_size must be provided for compile-safe rebuilds")
        else:
            logical_shape = cls._logical_shape_from_physical_shape(shape_tensor, offsets, batch_first)
        tensor_backed_layout = cls._is_tensor_backed_layout(permutation, ragged_dims)
        if packed_sizes is None and compiling and not tensor_backed_layout:
            _compile_unsupported(
                "NestedTensor._from_packed",
                "packed_sizes may be omitted only for an explicit tensor-backed layout",
            )
        if element_shapes is None and compiling and not tensor_backed_layout:
            _compile_unsupported(
                "NestedTensor._from_packed",
                "element_shapes may be omitted only for an explicit tensor-backed layout",
            )
        if ragged_offsets is not None and packed_sizes is None and element_shapes is None:
            materialize_python_metadata = False
        if packed_sizes is None and materialize_python_metadata and not _is_fake_tensor(offsets):
            packed_sizes = tuple(int(size) for size in (offsets[1:] - offsets[:-1]).tolist())
        if element_shapes is None and materialize_python_metadata and not _is_fake_tensor(shape_tensor):
            element_shapes = tuple(cls._trim_shape(shape) for shape in shape_tensor.tolist())

        physical_rank = int(shape_tensor.size(1))
        if permutation is None:
            if ragged_dims is None:
                resolved_permutation = cls._permutation_from_physical_shape(shape_tensor, element_shapes)
            else:
                resolved_ragged_dims = cls._normalize_ragged_dims(ragged_dims, physical_rank)
                resolved_permutation = resolved_ragged_dims + tuple(
                    dim for dim in range(physical_rank) if dim not in resolved_ragged_dims
                )
        else:
            resolved_permutation = tuple(int(dim) for dim in permutation)
        resolved_ragged_dims = cls._ragged_dims_from_packed_layout(
            values,
            physical_rank,
            resolved_permutation,
            ragged_dims,
        )
        ragged_dims_explicit = ragged_dims is not None
        resolved_ragged_offsets = cls._resolve_persistent_ragged_offsets(
            offsets,
            shape_tensor,
            permutation=resolved_permutation,
            ragged_dims=resolved_ragged_dims if ragged_dims_explicit else None,
            ragged_offsets=ragged_offsets,
            element_shapes=element_shapes,
        )

        if (
            not compiling
            and _is_fake_tensor(values)
            and not (_is_fake_tensor(offsets) and _is_fake_tensor(shape_tensor))
        ):
            from torch._subclasses.fake_tensor import maybe_get_fake_mode

            fake_mode = maybe_get_fake_mode(values)
            if fake_mode is not None:
                if not _is_fake_tensor(offsets):
                    offsets = fake_mode.from_tensor(offsets, static_shapes=True, trace=False)
                if not _is_fake_tensor(shape_tensor):
                    shape_tensor = fake_mode.from_tensor(shape_tensor, static_shapes=True, trace=False)
                if resolved_ragged_offsets is not None:
                    if len(resolved_ragged_offsets) == 1:
                        resolved_ragged_offsets = (offsets,)
                    else:
                        resolved_ragged_offsets = tuple(
                            (
                                level_offsets
                                if _is_fake_tensor(level_offsets)
                                else fake_mode.from_tensor(level_offsets, static_shapes=True, trace=False)
                            )
                            for level_offsets in resolved_ragged_offsets
                        )

        values = cls._maybe_pin_values(values, pin_memory)
        if compiling:
            constructor = cls._compiled_packed_constructor
            result = constructor(
                values,
                offsets,
                shape_tensor,
                logical_shape,
                resolved_permutation,
                resolved_ragged_dims,
                ragged_dims_explicit,
                bool(batch_first),
                float(padding_value),
                bool(mask_value),
                bool(pin_memory and values.device.type == "cpu" and values.is_pinned()),
                packed_sizes,
                element_shapes,
                resolved_ragged_offsets,
            )
        else:
            result = torch.Tensor._make_wrapper_subclass(
                cls,
                logical_shape,
                dtype=values.dtype,
                device=values.device,
                requires_grad=values.requires_grad,
            )
            result._values = values
            result._offsets = offsets
            result._permutation = resolved_permutation
            result._ragged_dims = resolved_ragged_dims
            result._ragged_dims_explicit = ragged_dims_explicit
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
            cls._install_persistent_ragged_offsets(result, resolved_ragged_offsets)
            result._invalidate_transient_caches()
        result._mark_tensor_backed_dynamic_dims(mark_values=mark_values_dynamic)
        if validate:
            cls._validate_packed_metadata(
                result._values,
                result._offsets,
                result._physical_shape,
                permutation=result._permutation,
                ragged_dims=result._ragged_dims,
                logical_shape=result._logical_shape,
                batch_first=result.batch_first,
                packed_sizes=result._packed_sizes,
                element_shapes=result._element_shapes,
                ragged_offsets=resolved_ragged_offsets,
            )
        return result

    # ------------------------------------------------------------------
    # torch.compile support
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_metadata_for_caching(tensor: Tensor) -> tuple[Any, ...]:
        r"""Return stable tensor metadata without inspecting storage contents.

        Shapes and strides may contain ``SymInt`` expressions.  Their ``repr`` is
        intentionally retained by :meth:`_stable_hash_for_caching`, matching the
        representation-based stable hash used by PyTorch's ``DTensor`` cache
        extension.  Storage addresses and tensor values are excluded;
        ``storage_offset`` itself remains ordinary tensor metadata.
        """
        return (
            str(tensor.dtype),
            tensor.device.type,
            tensor.device.index,
            str(tensor.layout),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            bool(tensor.requires_grad),
            bool(tensor.is_conj()),
            bool(tensor.is_neg()),
            bool(tensor.is_inference()),
        )

    def _stable_hash_for_caching(self) -> str:
        r"""Return a deterministic metadata hash for PyTorch's AOTAutograd cache.

        Tensor-backed offsets, physical shapes, and hierarchical row splits are
        flattened children.  Only their tensor metadata participates in this hash;
        their data does not.  Consequently, two dynamic calls with the same static
        structure can reuse one cache entry even when their ragged lengths differ.
        Legacy layouts whose topology still lives in ``packed_sizes`` or
        ``element_shapes`` retain those tuples in the static flatten context and
        therefore remain safely layout-specific.
        """
        inner_tensor_names, context = self.__tensor_flatten__()
        tensor_metadata = type(self)._tensor_metadata_for_caching
        payload = (
            "danling.NestedTensor.aot_autograd",
            type(self)._AOT_CACHE_HASH_VERSION,
            type(self).__module__,
            type(self).__qualname__,
            tensor_metadata(self),
            tuple((name, tensor_metadata(getattr(self, name))) for name in inner_tensor_names),
            tuple(context.items()),
        )
        return hashlib.blake2b(repr(payload).encode("utf-8"), digest_size=16).hexdigest()

    @property
    def _max_length_binding(self) -> Tensor:
        r"""Bind the data-derived logical maximum with one tensor dimension."""
        binding = vars(self).get("_compile_max_length_binding")
        if binding is not None:
            return binding
        max_length = self._physical_shape[:, 0].max().item() if self._physical_shape.size(0) else 0
        return self._offsets.new_empty(()).expand(max_length)

    @_max_length_binding.setter
    def _max_length_binding(self, binding: Tensor) -> None:
        self._compile_max_length_binding = binding

    def __tensor_flatten__(self):
        # During tracing, wrapper instances can be inspected while being built.
        # Only expose tensor attrs that already exist so Dynamo/FakeTensor can
        # inspect partially constructed wrapper subclasses safely.
        instance_attrs = vars(self)
        inner_tensors = [name for name in ("_values", "_offsets", "_physical_shape") if name in instance_attrs]
        if "_compile_max_length_binding" in instance_attrs:
            inner_tensors.append("_max_length_binding")
        # Dynamo cannot source-track a Tensor returned by a wrapper property on
        # its own, so the public packed projection remains an explicit child
        # beside its raw storage. AOT may rewrite this alias when coercing a
        # runtime tangent's memory format; the one-shot setter below handles
        # that update without replacing ``_values``.
        if "_values" in instance_attrs:
            inner_tensors.append("concat")
        if not inner_tensors:
            inner_tensors = ["_flatten_sentinel"]
        permutation = getattr(self, "_permutation", ())
        ragged_dims = getattr(self, "_ragged_dims", ()) if getattr(self, "_ragged_dims_explicit", False) else None
        ragged_offsets = self._persistent_ragged_offsets() if "_offsets" in instance_attrs else None
        if ragged_offsets is not None and len(ragged_offsets) > 1:
            inner_tensors.extend(type(self)._ragged_offset_names(len(ragged_offsets)))
        tensor_backed_layout = ragged_offsets is not None
        return inner_tensors, {
            "requires_grad": self.requires_grad,
            "allow_aot_concat_update": vars(self).get("_allow_aot_concat_update", False),
            "batch_first": getattr(self, "batch_first", True),
            "padding_value": getattr(self, "padding_value", 0.0),
            "mask_value": getattr(self, "mask_value", False),
            "pin_memory": getattr(self, "_pin_memory", False),
            "packed_sizes": None if tensor_backed_layout else getattr(self, "_packed_sizes", ()),
            "element_shapes": None if tensor_backed_layout else getattr(self, "_element_shapes", ()),
            "permutation": permutation,
            "ragged_dims": ragged_dims,
        }

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, ctx, outer_size, outer_stride):
        values = inner_tensors.get("_values", inner_tensors.get("_flatten_sentinel"))
        if values is None:
            raise RuntimeError("NestedTensor requires _values during tensor unflatten.")
        ctx = dict(ctx)
        wrapper_requires_grad = bool(ctx.pop("requires_grad", values.requires_grad))
        allow_aot_concat_update = bool(ctx.pop("allow_aot_concat_update", False))

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
            max_length_binding = inner_tensors.get("_max_length_binding")
            preserve_tensor_metadata = (
                cls._is_tensor_backed_layout(ctx.get("permutation"), ctx.get("ragged_dims"))
                and ctx.get("packed_sizes") is None
                and ctx.get("element_shapes") is None
            )
            ragged_rank = len(ctx.get("ragged_dims") or ())
            if preserve_tensor_metadata and ragged_rank > 1:
                names = cls._ragged_offset_names(ragged_rank)
                missing = tuple(name for name in names if name not in inner_tensors)
                if missing:
                    raise RuntimeError(
                        "NestedTensor tensor-backed multi-ragged unflatten is missing row-split children: "
                        + ", ".join(missing)
                    )
                ragged_offsets = tuple(inner_tensors[name] for name in names)
            elif preserve_tensor_metadata and ragged_rank == 1:
                ragged_offsets = (offsets,)
            else:
                ragged_offsets = None
            result = cls._from_packed(
                values,
                offsets,
                shape_tensor,
                outer_size=outer,
                validate=False,
                materialize_python_metadata=not preserve_tensor_metadata,
                ragged_offsets=ragged_offsets,
                **ctx,
            )
            if torch.Tensor.requires_grad.__get__(result) != wrapper_requires_grad:
                torch.Tensor.requires_grad.__set__(result, wrapper_requires_grad)
            if max_length_binding is not None:
                result._max_length_binding = max_length_binding
            result._allow_aot_concat_update = allow_aot_concat_update
            return result

        result = torch.Tensor._make_wrapper_subclass(
            cls,
            torch.Size(outer_size),
            dtype=values.dtype,
            device=values.device,
            requires_grad=wrapper_requires_grad,
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
        declared_ragged_dims = ctx.get("ragged_dims")
        result._ragged_dims = cls._ragged_dims_from_packed_layout(
            values,
            len(result._permutation),
            result._permutation,
            declared_ragged_dims,
        )
        result._ragged_dims_explicit = declared_ragged_dims is not None
        ragged_rank = len(result._ragged_dims)
        names = cls._ragged_offset_names(ragged_rank)
        ragged_offsets = tuple(inner_tensors[name] for name in names if name in inner_tensors)
        if ragged_rank == 1 and ctx.get("packed_sizes") is None and ctx.get("element_shapes") is None:
            ragged_offsets = (offsets,) if offsets is not None else ()
        cls._install_persistent_ragged_offsets(result, ragged_offsets or None)
        max_length_binding = inner_tensors.get("_max_length_binding")
        if max_length_binding is not None:
            result._max_length_binding = max_length_binding
        result._invalidate_transient_caches()
        result._allow_aot_concat_update = allow_aot_concat_update
        result._mark_tensor_backed_dynamic_dims()
        return result

    def __coerce_tangent_metadata__(self):
        r"""Allow one AOT-only update of the flattened packed alias."""
        self._allow_aot_concat_update = True
        return self

    def __coerce_same_metadata_as_tangent__(self, expected_meta, expected_type=None):
        r"""Apply the one-shot alias marker expected by AOT's runtime tangent."""
        if expected_type is not None and expected_type is not type(self):
            return None
        self._allow_aot_concat_update = bool(expected_meta.get("allow_aot_concat_update", False))
        return self

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
        _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, "NestedTensor._unpack")
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
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
        )
        values = type(self)._maybe_pin_values(values, self._pin_memory)
        self._values = values
        self._offsets = offsets
        self._physical_shape = shape_tensor
        self._logical_shape = self._compute_logical_shape(tensors, self.batch_first)
        self._packed_sizes = packed_sizes
        self._element_shapes = element_shapes
        if not self._ragged_dims_explicit:
            self._ragged_dims = type(self)._ragged_dims_from_packed_layout(
                values,
                int(shape_tensor.size(1)),
                self._permutation,
                None,
            )
        ragged_offsets = type(self)._resolve_persistent_ragged_offsets(
            offsets,
            shape_tensor,
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
        )
        type(self)._install_persistent_ragged_offsets(self, ragged_offsets)
        self._mark_tensor_backed_dynamic_dims()
        self._validate_metadata()

    @property
    def _hierarchical_offsets(self) -> tuple[Tensor, ...]:
        persistent = self._persistent_ragged_offsets()
        if persistent is not None:
            return persistent
        if self._cached_hierarchical_offsets is None:
            level_sizes = type(self)._hierarchical_level_sizes_from_physical_shape(
                self._physical_shape,
                self._element_shapes,
                self._ragged_dims,
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
            elif len(level_sizes) == 1:
                self._cached_hierarchical_offsets = (self._offsets,)
            else:
                self._cached_hierarchical_offsets = tuple(
                    type(self)._offsets_from_sizes(level_sizes[level], dtype=self._offsets.dtype)
                    for level in range(len(level_sizes))
                )
        return self._cached_hierarchical_offsets

    @property
    def _ragged_rank(self) -> int:
        return len(self._ragged_dims)

    def _ragged_level_offsets(self, level: int = -1) -> Tensor:
        offsets = self._hierarchical_offsets
        if not offsets:
            return self._offsets
        return offsets[level]

    def packed_offsets(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        r"""Return logical-batch boundaries in the packed leading dimension.

        The returned offsets delimit each logical batch element in ``concat``.
        This differs from :meth:`ragged_level_offsets` for multi-ragged layouts:
        ``packed_offsets`` always addresses complete per-sample packed chunks,
        while ragged-level offsets address rows within the ragged hierarchy.
        Device and dtype conversions are cached per ``NestedTensor`` instance.
        """
        offsets = self._offsets
        target_device = offsets.device if device is None else torch.device(device)
        target_dtype = offsets.dtype if dtype is None else dtype
        if offsets.device == target_device and offsets.dtype == target_dtype:
            return offsets
        device_key = type(self)._offset_conversion_device_key(target_device)
        key = None if device_key is None else (device_key, target_dtype, self._shape_cache_token())
        if key is not None and self._cached_packed_offsets is not None:
            cached = self._cached_packed_offsets.get(key)
            if cached is not None:
                return cached
        elif key is not None and not _is_fake_tensor(offsets):
            self._cached_packed_offsets = {}
        converted = offsets.to(device=target_device, dtype=target_dtype)
        if key is not None and self._cached_packed_offsets is not None:
            self._cached_packed_offsets[key] = converted
        return converted

    def element_sizes(self) -> Tensor:
        r"""Return every logical element shape as a CPU integer tensor.

        Rows follow logical batch order and columns follow logical element
        dimension order, independent of ``batch_first`` and physical packed
        storage order.  The returned ``(batch_size, element_rank)`` tensor is
        the canonical tensor-backed shape metadata and is therefore returned
        without a copy.
        """
        return self._physical_shape

    def ragged_level_offsets(
        self,
        level: int = -1,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        r"""Return ragged-level offsets, caching device and dtype conversions."""
        offsets = self._ragged_level_offsets(level)
        target_device = offsets.device if device is None else torch.device(device)
        target_dtype = offsets.dtype if dtype is None else dtype
        if offsets.device == target_device and offsets.dtype == target_dtype:
            return offsets
        device_key = type(self)._offset_conversion_device_key(target_device)
        key = None if device_key is None else (int(level), device_key, target_dtype, self._shape_cache_token())
        if key is not None and self._cached_ragged_level_offsets is not None:
            cached = self._cached_ragged_level_offsets.get(key)
            if cached is not None:
                return cached
        elif key is not None and not _is_fake_tensor(offsets):
            self._cached_ragged_level_offsets = {}
        converted = offsets.to(device=target_device, dtype=target_dtype)
        if key is not None and self._cached_ragged_level_offsets is not None:
            self._cached_ragged_level_offsets[key] = converted
        return converted

    def _ragged_level_sizes(self, level: int = -1) -> Tensor:
        offsets = self._ragged_level_offsets(level)
        return offsets[1:] - offsets[:-1]

    def packed_local_indices(
        self,
        level: int = 0,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.long,
    ) -> Tensor:
        r"""Return local coordinates within the selected packed ragged level."""
        target_device = self.device if device is None else torch.device(device)
        level = int(level)
        key = (level, str(target_device), dtype, self._shape_cache_token())
        if self._cached_packed_local_indices is not None:
            cached = self._cached_packed_local_indices.get(key)
            if cached is not None:
                return cached
        elif not _is_fake_tensor(self._offsets) and not _is_compiling():
            self._cached_packed_local_indices = {}

        packed_sizes = self._packed_sizes
        if (
            level == 0
            and self._ragged_rank == 1
            and packed_sizes is not None
            and (_is_compiling() or _is_fake_tensor(self._offsets))
        ):
            lengths_tuple = tuple(int(size) for size in packed_sizes)
            starts_tuple: list[int] = []
            running = 0
            for length in lengths_tuple:
                starts_tuple.append(running)
                running += length
            lengths = torch.tensor(lengths_tuple, dtype=torch.long, device=target_device)
            starts_source = torch.tensor(starts_tuple, dtype=dtype, device=target_device)
            total = running
        else:
            hierarchical_offsets = self._hierarchical_offsets
            if hierarchical_offsets:
                normalized_level = level if level >= 0 else len(hierarchical_offsets) + level
                if normalized_level < 0 or normalized_level >= len(hierarchical_offsets):
                    raise IndexError(f"ragged level {level} is out of range for rank {len(hierarchical_offsets)}")
                offsets = hierarchical_offsets[normalized_level]
                is_last_level = normalized_level == len(hierarchical_offsets) - 1
            else:
                normalized_level = 0
                offsets = self._offsets
                is_last_level = True
            lengths = offsets[1:] - offsets[:-1]
            total = self._values.shape[0] if is_last_level else hierarchical_offsets[normalized_level + 1].numel() - 1
            starts_source = offsets[:-1].to(device=target_device, dtype=dtype)
        positions = torch.arange(total, dtype=dtype, device=target_device)
        starts = torch.repeat_interleave(starts_source, lengths.to(target_device), output_size=total)
        local_indices = positions - starts
        if self._cached_packed_local_indices is not None:
            self._cached_packed_local_indices[key] = local_indices
        return local_indices

    def packed_batch_indices(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.long,
    ) -> Tensor:
        r"""Return batch coordinates for packed values."""
        target_device = self.device if device is None else torch.device(device)
        key = (str(target_device), dtype, self._shape_cache_token())
        if self._cached_packed_batch_indices is not None:
            cached = self._cached_packed_batch_indices.get(key)
            if cached is not None:
                return cached
        elif not _is_fake_tensor(self._offsets) and not _is_compiling():
            self._cached_packed_batch_indices = {}

        packed_sizes = self._packed_sizes
        if packed_sizes is not None and (_is_compiling() or _is_fake_tensor(self._offsets)):
            lengths_tuple = tuple(int(size) for size in packed_sizes)
            lengths = torch.tensor(lengths_tuple, dtype=torch.long, device=target_device)
            total = sum(lengths_tuple)
            batch_source = torch.arange(len(lengths_tuple), dtype=dtype, device=target_device)
        else:
            offsets = self._offsets
            lengths = offsets[1:] - offsets[:-1]
            total = self._values.shape[0]
            batch_source = torch.arange(offsets.numel() - 1, dtype=dtype, device=target_device)
        batch_indices = torch.repeat_interleave(batch_source, lengths.to(target_device), output_size=total)
        if self._cached_packed_batch_indices is not None:
            self._cached_packed_batch_indices[key] = batch_indices
        return batch_indices

    @property
    def _varying_dims(self) -> tuple[int, ...]:
        return self._ragged_dims

    @property
    def _static_dims(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._permutation[len(self._ragged_dims) :])

    def _has_same_structure(self, other: Self) -> bool:
        if (
            self.batch_first != other.batch_first
            or self._permutation != other._permutation
            or self._ragged_dims != other._ragged_dims
        ):
            return False
        if self._element_shapes is not None and other._element_shapes is not None:
            lhs_levels = type(self)._hierarchical_level_sizes_from_element_shapes(
                self._element_shapes,
                self._ragged_dims,
            )
            rhs_levels = type(self)._hierarchical_level_sizes_from_element_shapes(
                other._element_shapes,
                other._ragged_dims,
            )
            if lhs_levels or rhs_levels:
                return lhs_levels == rhs_levels
            return len(self) == len(other)
        lhs_offsets = self._hierarchical_offsets
        rhs_offsets = other._hierarchical_offsets
        if len(lhs_offsets) != len(rhs_offsets):
            return False
        runtime_assert = self._element_shapes is None or other._element_shapes is None
        if lhs_offsets:
            return all(
                type(self)._meta_tensor_equal(
                    lhs,
                    rhs,
                    "NestedTensor ragged offsets must match",
                    runtime_assert=runtime_assert,
                )
                for lhs, rhs in zip(lhs_offsets, rhs_offsets)
            )
        return type(self)._meta_tensor_equal(
            self._offsets,
            other._offsets,
            "NestedTensor ragged offsets must match",
            runtime_assert=runtime_assert,
        )

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
        runtime_assert = self._element_shapes is None or other._element_shapes is None
        if not type(self)._meta_tensor_equal(
            self._physical_shape,
            other._physical_shape,
            "NestedTensor physical shapes must match",
            runtime_assert=runtime_assert,
        ):
            return False
        return type(self)._meta_tensor_equal(
            self._offsets,
            other._offsets,
            "NestedTensor ragged offsets must match",
            runtime_assert=runtime_assert,
        )

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
            batch_idx = self.packed_batch_indices(device=target_device, dtype=dtype)
            offsets = self._offsets.to(device=target_device, dtype=dtype)
            lookup_idx = batch_idx if batch_idx.dtype == torch.long else batch_idx.to(dtype=torch.long)
            return batch_idx, flat_idx - offsets[lookup_idx]
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
        coord_by_dim = dict(zip(varying_dims, coords))

        dense_index: list[Tensor | slice] = [batch_idx]
        for dim in range(self._physical_shape.size(1)):
            dense_index.append(coord_by_dim[dim] if dim in coord_by_dim else slice(None))
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
            output_ragged_dims = None
            if self._ragged_dims_explicit:
                output_ragged_dims = tuple(
                    len(prefix) + keep_dims.index(dim) for dim in self._ragged_dims if dim in keep_dims
                )
            return shape, self._packed_sizes_like(element_shapes, output_ragged_dims), element_shapes

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
        batch_dim = type(self)._batch_dim_from_logical_shape(self._logical_shape, self.batch_first)
        return tuple(size for index, size in enumerate(self._logical_shape) if index != batch_dim)

    def _logical_shape_from_physical_dims(self, physical_dims: Sequence[int]) -> torch.Size:
        r"""Build a logical outer shape from non-batch physical-dimension sizes."""
        physical_dims = tuple(physical_dims)
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
        projected = [*prefix, *(physical_dims[int(dim)] for dim in keep_dims)]
        projected.extend(suffix)
        for dim, size in (replace_dims or {}).items():
            projected[int(dim)] = size
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
            validate=False,
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

        # Rebasing each operand's offsets on the running row total is tensor arithmetic; reading
        # that total back with ``.item()`` makes the rebase depend on a value, which is what
        # stopped batch concatenation from tracing while the slower non-batch path traced fine.
        offset_parts = [tensors[0]._offsets]
        for tensor in tensors[1:]:
            offset_parts.append(tensor._offsets[1:] + offset_parts[-1][-1])
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
            ragged_dims=ref._ragged_dims if ref._ragged_dims_explicit else None,
            batch_first=ref.batch_first,
            padding_value=ref.padding_value,
            mask_value=ref.mask_value,
            pin_memory=ref._pin_memory,
            outer_size=tuple(out_logical),
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
            validate=False,
        )

    @property
    def _storage(self) -> tuple[Tensor, ...]:
        cached = self._cached_storage
        if cached is None or self._storage_cache_dropped_grad(cached):
            cached = self._unpack()
            self._cached_storage = cached
        return cached

    @_storage.setter
    def _storage(self, tensors: Sequence) -> None:
        self._repack(tensors)

    def _storage_cache_dropped_grad(self, cached: tuple[Tensor, ...]) -> bool:
        r"""
        Report whether a cached unpack predates the autograd graph now on ``_values``.

        ``_cached_storage`` keeps whatever the first access produced, so a cache filled
        under ``no_grad`` (or below the autograd layer inside ``__torch_dispatch__``) holds
        detached views. Serving those later would silently cut the graph for every consumer
        that reads elements instead of ``_values``.
        """
        return bool(cached) and self._values.requires_grad and cached[0].grad_fn is None

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
            if self._persistent_ragged_offsets() is not None:
                # Tensor-backed explicit layouts have a fixed, exact physical
                # rank.  Their trailing zeros are real dimensions rather than
                # padding columns (for example an empty square is ``(0, 0)``).
                return tuple(torch.Size(row) for row in self._physical_shape.tolist())
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
        aot_projection = vars(self).get("_aot_concat_projection")
        if aot_projection is not None:
            return aot_projection
        # Eager results already carry their history on ``_values`` and retain
        # the historical identity contract.  A wrapper returned across an AOT
        # boundary instead owns the autograd edge while its flattened child is
        # detached.  Bridge that edge back to packed values on demand.
        outer_grad_fn = self.grad_fn
        if outer_grad_fn is not None and not self._values.requires_grad:
            return _project_packed_values(self)
        return self._values

    @concat.setter
    def concat(self, values: Tensor) -> None:
        # AOTAutograd may coerce a tangent alias to a contiguous memory
        # format while reconstructing flattened subclass children.  Keep that
        # projection separate: the raw packed tangent remains the value that
        # the wrapper's backward consumes.
        if not vars(self).get("_allow_aot_concat_update", False):
            raise AttributeError("NestedTensor.concat is read-only")
        self._allow_aot_concat_update = False
        if (
            not isinstance(values, Tensor)
            or isinstance(values, NestedTensor)
            or values.shape != self._values.shape
            or values.dtype != self._values.dtype
            or values.device != self._values.device
            or values.requires_grad
            or values.grad_fn is not None
        ):
            raise AttributeError("Invalid AOT tangent projection for NestedTensor.concat")
        self._aot_concat_projection = values

    @property
    def packed_dim_order(self) -> tuple[int, ...]:
        r"""Logical element dimensions in physical packed-storage order.

        The tuple is a read-only structural descriptor.  Identity order means
        packed storage follows the element's logical dimension order; operations
        that permute logical dimensions may retain the same packed values while
        changing this mapping.
        """
        return self._permutation

    @property
    def ragged_dims(self) -> tuple[int, ...]:
        r"""Logical element dimensions represented by packed ragged levels.

        The order is stable when ``ragged_dims`` was declared at construction,
        even when all elements in a particular batch happen to have equal sizes.
        """
        return self._ragged_dims

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

    @staticmethod
    def _operator_result(op, *args):
        try:
            return op(*args)
        except TypeError:
            return NotImplemented

    def __add__(self, other):
        return self._operator_result(torch.add, self, other)

    def __radd__(self, other):
        return self._operator_result(torch.add, other, self)

    def __sub__(self, other):
        return self._operator_result(torch.sub, self, other)

    def __rsub__(self, other):
        return self._operator_result(torch.sub, other, self)

    def __mul__(self, other):
        return self._operator_result(torch.mul, self, other)

    def __rmul__(self, other):
        return self._operator_result(torch.mul, other, self)

    def __truediv__(self, other):
        return self._operator_result(torch.true_divide, self, other)

    def __rtruediv__(self, other):
        return self._operator_result(torch.true_divide, other, self)

    def __floordiv__(self, other):
        return self._operator_result(torch.floor_divide, self, other)

    def __rfloordiv__(self, other):
        return self._operator_result(torch.floor_divide, other, self)

    def __mod__(self, other):
        return self._operator_result(torch.remainder, self, other)

    def __rmod__(self, other):
        return self._operator_result(torch.remainder, other, self)

    def __pow__(self, other):
        return self._operator_result(torch.pow, self, other)

    def __rpow__(self, other):
        return self._operator_result(torch.pow, other, self)

    def __matmul__(self, other):
        return self._operator_result(torch.matmul, self, other)

    def __rmatmul__(self, other):
        return self._operator_result(torch.matmul, other, self)

    def __neg__(self):
        return self._operator_result(torch.neg, self)

    def __abs__(self):
        return self._operator_result(torch.abs, self)

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
        declared_ragged_dims = kwargs.get("ragged_dims")
        if declared_ragged_dims is None:
            varying_dims, static_dims = cls._pack_layout_from_element_shapes(element_shapes)
        else:
            varying_dims, static_dims = cls._pack_layout_from_declared_ragged_dims(element_shapes, declared_ragged_dims)
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

    def _packed_sizes_like(
        self,
        element_shapes: tuple[tuple[int, ...], ...],
        ragged_dims: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        if ragged_dims is None:
            varying_dims, _ = type(self)._pack_layout_from_element_shapes(element_shapes)
        else:
            varying_dims, _ = type(self)._pack_layout_from_declared_ragged_dims(element_shapes, ragged_dims)
        return tuple(type(self)._packed_size_from_shape(shape, varying_dims) for shape in element_shapes)

    def _packed_like_unchecked_raw(self, packed_values: Tensor) -> Self:
        r"""Rebuild directly when the caller already proved shape compatibility."""
        result = type(self)._from_packed(
            packed_values,
            self._offsets,
            self._physical_shape,
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
            ragged_offsets=self._persistent_ragged_offsets(),
            validate=False,
        )
        if (
            self._cached_hierarchical_offsets is not None
            and result._offsets is self._offsets
            and result._physical_shape is self._physical_shape
        ):
            result._cached_hierarchical_offsets = self._cached_hierarchical_offsets
        return result

    def _packed_like_unchecked(self, packed_values: Tensor) -> Self:
        r"""Rebuild from compatible packed values, preserving compiled autograd edges."""
        if _is_compiling() or (torch.is_grad_enabled() and packed_values.requires_grad):
            return _PackedLikeAutograd.apply(packed_values, _PackedStructureReference(self))
        return self._packed_like_unchecked_raw(packed_values)

    def packed_like(self, packed_values: Tensor) -> Self:
        r"""Wrap packed values with this ``NestedTensor``'s structure.

        ``packed_values`` must have exactly the same shape as :attr:`concat`.
        The returned ``NestedTensor`` shares ``packed_values`` directly, so its
        dtype, device, strides, pinning, and autograd history all come from the
        supplied tensor.  Ragged offsets, element shapes, permutation, logical
        shape, and runtime configuration are inherited from ``self``.

        Args:
            packed_values: Dense packed storage for the returned
                ``NestedTensor``.

        Returns:
            A ``NestedTensor`` with ``self``'s structure and
            ``packed_values`` as its packed storage.

        Raises:
            TypeError: If ``packed_values`` is not a dense ``Tensor``.
            ValueError: If its shape differs from ``self.concat.shape``.

        Examples:
            >>> reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
            >>> values = torch.ones_like(reference.concat)
            >>> output = reference.packed_like(values)
            >>> output.concat is values
            True
            >>> output.shape == reference.shape
            True
        """
        if (
            not isinstance(packed_values, Tensor)
            or isinstance(packed_values, NestedTensor)
            or packed_values.is_nested
            or packed_values.layout != torch.strided
        ):
            raise TypeError(
                "packed_values must be a dense Tensor with torch.strided layout, "
                f"got {type(packed_values).__name__} with layout "
                f"{getattr(packed_values, 'layout', None)}"
            )
        if packed_values.shape != self._values.shape:
            raise ValueError(
                "packed_values must have exactly the same shape as the reference packed storage, "
                f"got {packed_values.shape} and expected {self._values.shape}"
            )
        pin_memory = bool(packed_values.device.type == "cpu" and packed_values.is_pinned())
        result = type(self)._from_packed(
            packed_values,
            self._offsets,
            self._physical_shape,
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=pin_memory,
            outer_size=self._logical_shape,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
            ragged_offsets=self._persistent_ragged_offsets(),
            validate=False,
        )
        if (
            self._cached_hierarchical_offsets is not None
            and result._offsets is self._offsets
            and result._physical_shape is self._physical_shape
        ):
            result._cached_hierarchical_offsets = self._cached_hierarchical_offsets
        if torch.is_grad_enabled() and packed_values.requires_grad:
            return _PackedLikeAutograd.apply(packed_values, _PackedStructureReference(result))
        return result

    def packed_with_static_tail(self, packed_values: Tensor) -> Self:
        r"""Wrap packed values after replacing this tensor's static tail.

        This operation preserves every declared ragged level and replaces all
        static element dimensions with ``packed_values.shape[1:]``.  A
        canonical reference may change the number of static dimensions.  An
        explicit tensor-backed reference with one non-leading logical ragged
        dimension may instead replace its existing static dimensions in packed
        order, but must keep the same static rank.  This preserves where those
        dimensions appear in the logical element layout.  The returned tensor
        shares ``packed_values`` directly.

        Args:
            packed_values: Dense packed storage whose leading dimension equals
                ``self.concat.shape[0]``.  Remaining dimensions become the new
                static tail.

        Returns:
            A ``NestedTensor`` preserving the declared ragged topology and
            supported packed layout, with ``packed_values`` as its packed
            storage.

        Raises:
            TypeError: If ``packed_values`` is not a dense strided tensor.
            ValueError: If the reference has no ragged dimensions, is neither
                canonical nor a supported tensor-backed non-leading layout,
                changes the static rank of such a non-leading layout, or has a
                mismatched packed leading dimension.

        Examples:
            >>> atoms = NestedTensor([torch.zeros(2), torch.zeros(4)])
            >>> values = torch.randn(6, 3)
            >>> output = atoms.packed_with_static_tail(values)
            >>> [tuple(element.shape) for element in output]
            [(2, 3), (4, 3)]
            >>> output.concat is values
            True
        """
        if (
            not isinstance(packed_values, Tensor)
            or isinstance(packed_values, NestedTensor)
            or packed_values.is_nested
            or packed_values.layout != torch.strided
        ):
            raise TypeError(
                "packed_values must be a dense Tensor with torch.strided layout, "
                f"got {type(packed_values).__name__} with layout "
                f"{getattr(packed_values, 'layout', None)}"
            )
        if packed_values.dim() == 0:
            raise ValueError("packed_values must have a leading packed dimension")

        ragged_rank = len(self._ragged_dims)
        canonical_ragged_dims = tuple(range(ragged_rank))
        canonical_order = tuple(range(len(self._permutation)))
        if ragged_rank == 0:
            raise ValueError("packed_with_static_tail requires at least one ragged dimension")
        canonical_layout = self._ragged_dims == canonical_ragged_dims and self._permutation == canonical_order
        nonleading_tensor_backed_layout = (
            self._ragged_dims_explicit
            and ragged_rank == 1
            and self._ragged_dims[0] != 0
            and self._permutation[0] == self._ragged_dims[0]
            and self._persistent_ragged_offsets() is not None
        )
        if not canonical_layout and not nonleading_tensor_backed_layout:
            raise ValueError(
                "packed_with_static_tail requires canonical packed order or an explicit tensor-backed "
                "single non-leading ragged dimension, "
                f"got ragged_dims={self._ragged_dims} and packed_dim_order={self._permutation}"
            )
        if packed_values.shape[0] != self._values.shape[0]:
            raise ValueError(
                "packed_values leading dimension must equal the reference packed length, "
                f"got {packed_values.shape[0]} and expected {self._values.shape[0]}"
            )

        static_tail = tuple(packed_values.shape[1:])
        if canonical_layout:
            physical_shape, packed_sizes, element_shapes = self._shape_meta_from_components(
                keep_dims=self._ragged_dims,
                suffix=static_tail,
            )
            permutation = tuple(range(ragged_rank + len(static_tail)))
            ragged_dims = canonical_ragged_dims
            outer_size = self._logical_shape_from_components(
                keep_dims=self._ragged_dims,
                suffix=static_tail,
            )
        else:
            static_dims = self._static_dims
            if len(static_tail) != len(static_dims):
                raise ValueError(
                    "packed_with_static_tail requires a non-leading ragged layout to keep the same "
                    "number of packed static dimensions, "
                    f"got {len(static_tail)} and expected {len(static_dims)}"
                )
            replacements = dict(zip(static_dims, static_tail))
            # This is the production packed path: update the fixed-rank tensor metadata
            # directly instead of rebuilding one Python shape tuple per sample.  Dropping
            # those legacy caches also ensures that outputs remain layout-dynamic when an
            # eager reference enters a compiled graph.
            physical_shape = self._physical_shape
            if _is_fake_tensor(packed_values) and not _is_fake_tensor(physical_shape):
                from torch._subclasses.fake_tensor import maybe_get_fake_mode

                fake_mode = maybe_get_fake_mode(packed_values)
                if fake_mode is not None:
                    physical_shape = fake_mode.from_tensor(physical_shape, static_shapes=True, trace=False)
            physical_shape = physical_shape.clone()
            for dim, size in replacements.items():
                physical_shape[:, dim] = size
            packed_sizes = None
            element_shapes = None
            permutation = self._permutation
            ragged_dims = self._ragged_dims
            outer_size = self._logical_shape_from_components(replace_dims=replacements)
        if len(self) == 0:
            packed_sizes = ()
            element_shapes = ()
        pin_memory = bool(packed_values.device.type == "cpu" and packed_values.is_pinned())
        return type(self)._from_packed(
            packed_values,
            self._offsets,
            physical_shape,
            permutation=permutation,
            ragged_dims=ragged_dims,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=pin_memory,
            outer_size=outer_size,
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
            ragged_offsets=self._persistent_ragged_offsets(),
            validate=False,
        )

    def packed_with_lengths(self, packed_values: Tensor, lengths: Tensor) -> Self:
        r"""Wrap packed values with new leading ragged lengths.

        ``lengths`` defines one canonical leading ragged dimension and
        ``packed_values.shape[1:]`` defines the static tail.  The reference
        supplies only the batch size, subclass, and runtime configuration; its
        existing ragged lengths are replaced.  Both packed values and their
        autograd history are shared without copying. Compiled reconstruction
        keeps offsets and element shapes as tensor-backed graph data, so fixed
        batch sizes do not create per-element Python output metadata.

        Args:
            packed_values: Dense packed storage with shape
                ``(sum(lengths), *static_tail)``.
            lengths: One-dimensional CPU integer tensor with one non-negative
                length per batch element.

        Returns:
            A canonical one-ragged-dimension ``NestedTensor`` backed directly
            by ``packed_values``.

        Raises:
            TypeError: If either tensor has an unsupported type, dtype, or
                layout.
            ValueError: If ``lengths`` is not valid one-dimensional CPU
                metadata, has the wrong batch size, contains a negative value,
                or does not sum to the packed leading dimension.

        Examples:
            >>> reference = NestedTensor([torch.zeros(4, 2), torch.zeros(6, 2)])
            >>> lengths = torch.tensor([2, 3])
            >>> values = torch.randn(5, 7)
            >>> output = reference.packed_with_lengths(values, lengths)
            >>> [tuple(element.shape) for element in output]
            [(2, 7), (3, 7)]
            >>> output.concat is values
            True
        """
        if (
            not isinstance(packed_values, Tensor)
            or isinstance(packed_values, NestedTensor)
            or packed_values.is_nested
            or packed_values.layout != torch.strided
        ):
            raise TypeError(
                "packed_values must be a dense Tensor with torch.strided layout, "
                f"got {type(packed_values).__name__} with layout "
                f"{getattr(packed_values, 'layout', None)}"
            )
        if packed_values.dim() == 0:
            raise ValueError("packed_values must have a leading packed dimension")
        if (
            not isinstance(lengths, Tensor)
            or isinstance(lengths, NestedTensor)
            or lengths.is_nested
            or lengths.layout != torch.strided
        ):
            raise TypeError(
                "lengths must be a dense Tensor with torch.strided layout, "
                f"got {type(lengths).__name__} with layout {getattr(lengths, 'layout', None)}"
            )
        if lengths.device.type != "cpu":
            raise ValueError(f"lengths must be on CPU, got {lengths.device}")
        if lengths.dtype.is_floating_point or lengths.dtype.is_complex or lengths.dtype == torch.bool:
            raise TypeError(f"lengths must use an integer dtype, got {lengths.dtype}")
        if lengths.dim() != 1:
            raise ValueError(f"lengths must be one-dimensional, got shape {tuple(lengths.shape)}")
        if lengths.numel() != len(self):
            raise ValueError(
                "lengths must contain one value per batch element, "
                f"got {lengths.numel()} values for batch size {len(self)}"
            )
        symbolic_lengths = _is_fake_tensor(lengths)
        if symbolic_lengths:
            if not _is_fake_tensor(packed_values):
                raise ValueError(
                    "FakeTensor lengths require FakeTensor packed_values; "
                    "pass concrete CPU lengths for standalone FakeTensor reconstruction"
                )
            torch._assert_async(torch.all(lengths >= 0), "lengths must be non-negative")
            torch._assert_async(
                lengths.sum() == packed_values.shape[0],
                "lengths must sum to the packed values leading dimension",
            )
        else:
            if bool(torch.any(lengths < 0)):
                raise ValueError("lengths must be non-negative")
            packed_length = int(lengths.sum().item())
            if packed_length != packed_values.shape[0]:
                raise ValueError(
                    "lengths must sum to the packed values leading dimension, "
                    f"got sum {packed_length} and packed length {packed_values.shape[0]}"
                )

        static_tail = tuple(packed_values.shape[1:])
        physical_rank = 1 + len(static_tail)
        if lengths.numel():
            tail_shape = lengths.new_empty((lengths.numel(), len(static_tail)), dtype=torch.long)
            for dim, size in enumerate(static_tail):
                tail_shape[:, dim] = size
            physical_shape = torch.cat((lengths.to(dtype=torch.long).reshape(-1, 1), tail_shape), dim=1)
        else:
            physical_shape = lengths.new_empty((0, physical_rank), dtype=torch.long)
        offsets = torch.nn.functional.pad(lengths.to(dtype=torch.long).cumsum(0), (1, 0))
        max_length = lengths.max().item() if lengths.numel() else 0
        if self.batch_first:
            logical_shape = torch.Size((len(self), max_length, *static_tail))
        else:
            logical_shape = torch.Size((max_length, len(self), *static_tail))
        pin_memory = bool(packed_values.device.type == "cpu" and packed_values.is_pinned())
        result = type(self)._from_packed(
            packed_values,
            offsets,
            physical_shape,
            permutation=tuple(range(physical_rank)),
            ragged_dims=(0,),
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=pin_memory,
            outer_size=logical_shape,
            packed_sizes=None,
            element_shapes=None,
            validate=False,
            materialize_python_metadata=False,
        )
        if symbolic_lengths:
            result._max_length_binding = result._offsets.new_empty(()).expand(max_length)
        return result

    def packed_with_square_lengths(self, packed_values: Tensor, lengths: Tensor) -> Self:
        r"""Wrap packed values as canonical square ragged elements.

        ``lengths`` defines two canonical ragged dimensions with equal sizes,
        so batch element ``i`` has shape
        ``(lengths[i], lengths[i], *packed_values.shape[1:])``.  The reference
        supplies only the batch size, subclass, and runtime configuration; its
        existing topology is not retained.  The returned tensor shares
        ``packed_values`` and its autograd history directly.

        Args:
            packed_values: Dense packed storage with shape
                ``(sum(lengths.square()), *static_tail)``.
            lengths: One-dimensional CPU integer tensor with one non-negative
                length per batch element.

        Returns:
            A canonical two-ragged-dimension ``NestedTensor`` backed directly
            by ``packed_values``.

        Raises:
            TypeError: If either tensor has an unsupported type, dtype, or
                layout.
            ValueError: If ``lengths`` is not valid one-dimensional CPU
                metadata, has the wrong batch size, contains a negative value,
                its squared sizes or their cumulative sum exceed the int64
                metadata range, or they do not sum to the packed leading
                dimension.

        Examples:
            >>> reference = NestedTensor([torch.zeros(1), torch.zeros(1)])
            >>> lengths = torch.tensor([2, 3])
            >>> values = torch.randn(13, 4)
            >>> output = reference.packed_with_square_lengths(values, lengths)
            >>> [tuple(element.shape) for element in output]
            [(2, 2, 4), (3, 3, 4)]
            >>> output.concat is values
            True
        """
        if (
            not isinstance(packed_values, Tensor)
            or isinstance(packed_values, NestedTensor)
            or packed_values.is_nested
            or packed_values.layout != torch.strided
        ):
            raise TypeError(
                "packed_values must be a dense Tensor with torch.strided layout, "
                f"got {type(packed_values).__name__} with layout "
                f"{getattr(packed_values, 'layout', None)}"
            )
        if packed_values.dim() == 0:
            raise ValueError("packed_values must have a leading packed dimension")
        if (
            not isinstance(lengths, Tensor)
            or isinstance(lengths, NestedTensor)
            or lengths.is_nested
            or lengths.layout != torch.strided
        ):
            raise TypeError(
                "lengths must be a dense Tensor with torch.strided layout, "
                f"got {type(lengths).__name__} with layout {getattr(lengths, 'layout', None)}"
            )
        if lengths.device.type != "cpu":
            raise ValueError(f"lengths must be on CPU, got {lengths.device}")
        if lengths.dtype.is_floating_point or lengths.dtype.is_complex or lengths.dtype == torch.bool:
            raise TypeError(f"lengths must use an integer dtype, got {lengths.dtype}")
        if lengths.dim() != 1:
            raise ValueError(f"lengths must be one-dimensional, got shape {tuple(lengths.shape)}")
        if lengths.numel() != len(self):
            raise ValueError(
                "lengths must contain one value per batch element, "
                f"got {lengths.numel()} values for batch size {len(self)}"
            )

        long_lengths = lengths.to(dtype=torch.long)
        symbolic_lengths = _is_fake_tensor(lengths)
        if symbolic_lengths:
            if not _is_fake_tensor(packed_values):
                raise ValueError(
                    "FakeTensor lengths require FakeTensor packed_values; "
                    "pass concrete CPU lengths for standalone FakeTensor reconstruction"
                )
            torch._assert_async(torch.all(long_lengths >= 0), "lengths must be non-negative")
            torch._assert_async(
                torch.all(long_lengths <= _INT64_SQUARE_ROOT_MAX),
                "each squared length must fit in torch.int64",
            )
        else:
            _validate_concrete_square_lengths(long_lengths)

        packed_sizes = long_lengths.square()
        packed_prefix = packed_sizes.cumsum(0)
        if symbolic_lengths:
            torch._assert_async(
                torch.all(packed_prefix >= 0),
                "the cumulative sum of squared lengths must fit in torch.int64",
            )
            torch._assert_async(
                packed_sizes.sum() == packed_values.shape[0],
                "squared lengths must sum to the packed values leading dimension",
            )
        else:
            packed_length = int(packed_sizes.sum().item())
            if packed_length != packed_values.shape[0]:
                raise ValueError(
                    "squared lengths must sum to the packed values leading dimension, "
                    f"got sum {packed_length} and packed length {packed_values.shape[0]}"
                )

        static_tail = tuple(packed_values.shape[1:])
        physical_rank = 2 + len(static_tail)
        if lengths.numel():
            square_shape = long_lengths.reshape(-1, 1).expand(-1, 2)
            tail_shape = long_lengths.new_empty((lengths.numel(), len(static_tail)))
            for dim, size in enumerate(static_tail):
                tail_shape[:, dim] = size
            physical_shape = torch.cat((square_shape, tail_shape), dim=1)
        else:
            physical_shape = long_lengths.new_empty((0, physical_rank))

        outer_offsets = torch.nn.functional.pad(packed_prefix, (1, 0))
        level_zero_offsets = torch.nn.functional.pad(long_lengths.cumsum(0), (1, 0))
        level_one_offsets = _square_row_splits(long_lengths)
        max_length = long_lengths.max().item() if lengths.numel() else 0
        if self.batch_first:
            logical_shape = torch.Size((len(self), max_length, max_length, *static_tail))
        else:
            logical_shape = torch.Size((max_length, len(self), max_length, *static_tail))
        pin_memory = bool(packed_values.device.type == "cpu" and packed_values.is_pinned())
        # Both ragged axes and their offsets already carry the dynamic shape contract. Marking
        # caller-owned packed storage here would mutate a compiled graph input after Dynamo has
        # installed its guards, forcing an otherwise unnecessary second compilation.
        result = type(self)._from_packed(
            packed_values,
            outer_offsets,
            physical_shape,
            permutation=tuple(range(physical_rank)),
            ragged_dims=(0, 1),
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=pin_memory,
            outer_size=logical_shape,
            packed_sizes=None,
            element_shapes=None,
            ragged_offsets=(level_zero_offsets, level_one_offsets),
            validate=False,
            materialize_python_metadata=False,
            mark_values_dynamic=False,
        )
        if symbolic_lengths:
            result._max_length_binding = result._offsets.new_empty(()).expand(max_length)
        return result

    def packed_with_rectangular_lengths(
        self,
        packed_values: Tensor,
        row_lengths: Tensor,
        column_lengths: Tensor,
    ) -> Self:
        r"""Wrap packed values as canonical rectangular ragged elements.

        Batch element ``i`` has shape
        ``(row_lengths[i], column_lengths[i], *packed_values.shape[1:])``.
        Both ragged axes and their hierarchical row splits remain tensor-backed,
        so a fixed outer batch can reuse one dynamic fullgraph across different
        rectangular layouts. The returned tensor shares ``packed_values`` and
        its autograd history directly.

        Args:
            packed_values: Dense storage with leading length
                ``sum(row_lengths * column_lengths)``.
            row_lengths: One-dimensional CPU integer row counts.
            column_lengths: One-dimensional CPU integer column counts.

        Returns:
            A canonical two-ragged-dimension ``NestedTensor``.

        Examples:
            >>> reference = NestedTensor([torch.zeros(1), torch.zeros(1)])
            >>> rows, columns = torch.tensor([2, 3]), torch.tensor([4, 1])
            >>> values = torch.randn(11, 5)
            >>> output = reference.packed_with_rectangular_lengths(values, rows, columns)
            >>> [tuple(element.shape) for element in output]
            [(2, 4, 5), (3, 1, 5)]
            >>> output.concat is values
            True
        """
        if (
            not isinstance(packed_values, Tensor)
            or isinstance(packed_values, NestedTensor)
            or packed_values.is_nested
            or packed_values.layout != torch.strided
        ):
            raise TypeError(
                "packed_values must be a dense Tensor with torch.strided layout, "
                f"got {type(packed_values).__name__} with layout "
                f"{getattr(packed_values, 'layout', None)}"
            )
        if packed_values.dim() == 0:
            raise ValueError("packed_values must have a leading packed dimension")

        batch_size = self._physical_shape.shape[0]
        for name, lengths in (("row_lengths", row_lengths), ("column_lengths", column_lengths)):
            if (
                not isinstance(lengths, Tensor)
                or isinstance(lengths, NestedTensor)
                or lengths.is_nested
                or lengths.layout != torch.strided
            ):
                raise TypeError(
                    f"{name} must be a dense Tensor with torch.strided layout, "
                    f"got {type(lengths).__name__} with layout {getattr(lengths, 'layout', None)}"
                )
            if lengths.device.type != "cpu":
                raise ValueError(f"{name} must be on CPU, got {lengths.device}")
            if lengths.dtype.is_floating_point or lengths.dtype.is_complex or lengths.dtype == torch.bool:
                raise TypeError(f"{name} must use an integer dtype, got {lengths.dtype}")
            if lengths.dim() != 1:
                raise ValueError(f"{name} must be one-dimensional, got shape {tuple(lengths.shape)}")
            if lengths.numel() != batch_size:
                raise ValueError(
                    f"{name} must contain one value per batch element, "
                    f"got {lengths.numel()} values for batch size {batch_size}"
                )

        long_rows = row_lengths.to(dtype=torch.long)
        long_columns = column_lengths.to(dtype=torch.long)
        packed_sizes = long_rows * long_columns
        symbolic_lengths = _is_fake_tensor(row_lengths) or _is_fake_tensor(column_lengths)
        if symbolic_lengths:
            if not (
                _is_fake_tensor(row_lengths) and _is_fake_tensor(column_lengths) and _is_fake_tensor(packed_values)
            ):
                raise ValueError(
                    "FakeTensor rectangular lengths require FakeTensor packed_values and matching FakeTensor metadata"
                )
            torch._assert_async(torch.all(long_rows >= 0), "row_lengths must be non-negative")
            torch._assert_async(torch.all(long_columns >= 0), "column_lengths must be non-negative")
            torch._assert_async(
                packed_sizes.sum() == packed_values.shape[0],
                "rectangular lengths must cover the packed values leading dimension",
            )
        else:
            if bool(torch.any(long_rows < 0)):
                raise ValueError("row_lengths must be non-negative")
            if bool(torch.any(long_columns < 0)):
                raise ValueError("column_lengths must be non-negative")
            packed_length = int(packed_sizes.sum().item())
            if packed_length != packed_values.shape[0]:
                raise ValueError(
                    "row_lengths * column_lengths must sum to the packed values leading dimension, "
                    f"got sum {packed_length} and packed length {packed_values.shape[0]}"
                )

        static_tail = tuple(packed_values.shape[1:])
        physical_rank = 2 + len(static_tail)
        if row_lengths.numel():
            rectangular_shape = torch.stack((long_rows, long_columns), dim=1)
            tail_shape = long_rows.new_empty((row_lengths.numel(), len(static_tail)))
            for dim, size in enumerate(static_tail):
                tail_shape[:, dim] = size
            physical_shape = torch.cat((rectangular_shape, tail_shape), dim=1)
        else:
            physical_shape = long_rows.new_empty((0, physical_rank))

        outer_offsets = torch.nn.functional.pad(packed_sizes.cumsum(0), (1, 0))
        level_zero_offsets = torch.nn.functional.pad(long_rows.cumsum(0), (1, 0))
        level_one_offsets = _rectangular_row_splits(long_rows, long_columns)
        max_rows = long_rows.max().item() if row_lengths.numel() else 0
        max_columns = long_columns.max().item() if column_lengths.numel() else 0
        if self.batch_first:
            logical_shape = torch.Size((batch_size, max_rows, max_columns, *static_tail))
        else:
            logical_shape = torch.Size((max_rows, batch_size, max_columns, *static_tail))
        pin_memory = bool(packed_values.device.type == "cpu" and packed_values.is_pinned())
        result = type(self)._from_packed(
            packed_values,
            outer_offsets,
            physical_shape,
            permutation=tuple(range(physical_rank)),
            ragged_dims=(0, 1),
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=pin_memory,
            outer_size=logical_shape,
            packed_sizes=None,
            element_shapes=None,
            ragged_offsets=(level_zero_offsets, level_one_offsets),
            validate=False,
            materialize_python_metadata=False,
        )
        # The generic fallback binds only physical dim 0. Rectangular outputs
        # carry two independent data-derived maxima, so expose a zero-stride
        # two-dimensional child for both eager and Fake/compiled rebuilds.
        result._max_length_binding = result._offsets.new_empty(()).expand(max_rows, max_columns)
        return result

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
                permutation=self._permutation,
                ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
                batch_first=self.batch_first,
                padding_value=self.padding_value,
                mask_value=self.mask_value,
                pin_memory=self._pin_memory,
                outer_size=self._logical_shape,
                packed_sizes=self._packed_sizes,
                element_shapes=element_shapes,
                validate=False,
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

    def _batch_select_static_position(self, position: int, tail_index: tuple = ()) -> Tensor | None:
        r"""Select one ragged position from every batch element without unpacking storage."""
        if self._varying_dims != (0,):
            return None
        if len(self) == 0:
            extent = int(self._max_physical_dims()[0])
            normalized_position = int(position)
            if normalized_position < 0:
                normalized_position += extent
            if normalized_position < 0 or normalized_position >= extent:
                raise IndexError(f"index {position} is out of bounds for dimension 0 with size {extent}")
            return self._values.narrow(0, 0, 0)[(slice(None), *tail_index)]

        if self._packed_sizes is not None:
            lengths = self._packed_sizes
        elif _is_compiling() or _is_fake_tensor(self._offsets):
            _compile_unsupported(
                "NestedTensor select",
                "tensor-backed per-element index bounds are not implemented",
            )
        else:
            lengths = tuple(int(size) for size in (self._offsets[1:] - self._offsets[:-1]).tolist())

        idx = int(position)
        if idx >= 0:
            if any(idx >= int(length) for length in lengths):
                raise IndexError(f"index {position} is out of bounds for at least one NestedTensor element")
            local: Tensor | int = idx
        else:
            normalized: list[int] = []
            for length in lengths:
                idx = int(position)
                idx += int(length)
                if idx < 0 or idx >= int(length):
                    raise IndexError(f"index {position} is out of bounds for at least one NestedTensor element")
                normalized.append(idx)
            local = torch.as_tensor(normalized, dtype=torch.long, device=self._values.device)

        offsets = self._offsets[:-1].to(device=self._values.device, dtype=torch.long)
        selected = self._values.index_select(0, offsets + local)
        if tail_index:
            selected = selected[(slice(None), *tail_index)]
        return selected

    def _packed_physical_slice(self, rest: tuple) -> Self | None:
        r"""Slice packed physical dims when the batch dim is untouched."""
        if not self.batch_first:
            return None
        physical_rank = int(self._physical_shape.size(1))
        if len(rest) > physical_rank:
            return None
        index = tuple(rest) + (slice(None),) * (physical_rank - len(rest))
        if not all(isinstance(selector, slice) for selector in index):
            return None

        static_dims = self._static_dims
        static_lookup = {dim: axis for axis, dim in enumerate(static_dims)}
        varying_dims = self._varying_dims
        physical_dims = list(self._max_physical_dims())
        value_index: list[slice] = [slice(None)] * self._values.dim()
        replace_dims: dict[int, int] = {}
        ragged_selector: tuple[int, slice] | None = None
        changed = False
        for dim, selector in enumerate(index):
            if selector.step is not None and selector.step <= 0:
                raise ValueError("step must be greater than zero")
            axis = static_lookup.get(dim)
            if axis is None:
                if selector.start is None and selector.stop is None and selector.step is None:
                    continue
                if len(varying_dims) != 1 or dim != varying_dims[0] or ragged_selector is not None:
                    return None
                ragged_selector = (dim, selector)
                size = int(physical_dims[dim])
                start, stop, step = selector.indices(size)
                retained_size = len(range(start, stop, step))
                replace_dims[dim] = retained_size
                physical_dims[dim] = retained_size
                changed = True
            else:
                size = int(physical_dims[dim])
                start, stop, step = selector.indices(size)
                new_size = len(range(start, stop, step))
                value_index[1 + axis] = slice(start, stop, step)
                replace_dims[dim] = new_size
                physical_dims[dim] = new_size
                changed = changed or start != 0 or stop != size or step != 1
        if not changed:
            return self

        packed_sizes = self._packed_sizes
        offsets = self._offsets
        values = self._values[tuple(value_index)]
        if ragged_selector is not None:
            ragged_dim, selector = ragged_selector
            if self._packed_sizes is None and (
                _is_compiling() or _is_fake_tensor(self._offsets) or _is_fake_tensor(self._physical_shape)
            ):
                _compile_unsupported(
                    "NestedTensor slice",
                    "tensor-backed ragged slice metadata is not implemented",
                )
            if _is_fake_tensor(self._offsets):
                return None
            starts = []
            new_sizes = []
            lengths = (self._offsets[1:] - self._offsets[:-1]).tolist()
            step = 1
            for length in lengths:
                start, stop, step = selector.indices(int(length))
                starts.append(start)
                new_sizes.append(len(range(start, stop, step)))
            offsets = type(self)._offsets_from_sizes(new_sizes, dtype=self._offsets.dtype)
            total = int(offsets[-1].item()) if len(new_sizes) > 0 else 0
            if total == 0:
                gather = torch.empty((0,), device=self._values.device, dtype=torch.long)
            else:
                offsets_dev = self._offsets.to(device=self._values.device, dtype=torch.long)
                new_offsets_dev = offsets.to(device=self._values.device, dtype=torch.long)
                starts_dev = torch.as_tensor(starts, device=self._values.device, dtype=torch.long)
                sizes_dev = torch.as_tensor(new_sizes, device=self._values.device, dtype=torch.long)
                batch_idx = torch.arange(
                    len(new_sizes), device=self._values.device, dtype=torch.long
                ).repeat_interleave(sizes_dev, output_size=total)
                local_rank = (
                    torch.arange(total, device=self._values.device, dtype=torch.long) - new_offsets_dev[batch_idx]
                )
                gather = offsets_dev[batch_idx] + starts_dev[batch_idx] + local_rank * step
            values = self._values.index_select(0, gather)[tuple(value_index)]
            if new_sizes or not (len(self) == 0 and self._ragged_dims_explicit):
                replace_dims[ragged_dim] = max(new_sizes, default=0)
                physical_dims[ragged_dim] = max(new_sizes, default=0)
            packed_sizes = tuple(new_sizes)

        element_shapes = None
        if self._element_shapes is not None:
            if any(len(shape) != physical_rank for shape in self._element_shapes):
                return None
            rows = []
            for shape in self._element_shapes:
                row = []
                for dim, size in enumerate(shape):
                    if ragged_selector is not None and dim == ragged_selector[0]:
                        start, stop, step = ragged_selector[1].indices(int(size))
                        row.append(len(range(start, stop, step)))
                    else:
                        row.append(replace_dims.get(dim, int(size)))
                rows.append(tuple(row))
            element_shapes = tuple(rows)

        new_physical_shape = self._physical_shape.clone()
        for dim, size in replace_dims.items():
            if ragged_selector is not None and dim == ragged_selector[0] and packed_sizes is not None:
                new_physical_shape[:, dim] = new_physical_shape.new_tensor(packed_sizes)
            else:
                new_physical_shape[:, dim] = size
        return type(self)._from_packed(
            values,
            offsets,
            new_physical_shape,
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape_from_physical_dims(physical_dims),
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
            ragged_offsets=self._persistent_ragged_offsets() if ragged_selector is None else None,
            validate=False,
        )

    def _packed_static_integer_index(self, rest: tuple) -> Self | None:
        r"""Consume one static element axis directly on packed values."""
        physical_rank = int(self._physical_shape.size(1))
        if len(rest) > physical_rank:
            return None
        selectors = tuple(rest) + (slice(None),) * (physical_rank - len(rest))
        integer_dims = [dim for dim, selector in enumerate(selectors) if type(selector) is int]
        if len(integer_dims) != 1:
            return None
        physical_dim = integer_dims[0]
        if any(
            not (
                type(selector) is int
                or (
                    isinstance(selector, slice)
                    and selector.start is None
                    and selector.stop is None
                    and selector.step is None
                )
            )
            for selector in selectors
        ):
            return None
        values_dim = _physical_to_values_dim(self, physical_dim)
        if values_dim is None:
            return None
        from .aten_functions import _packed_without_dim

        values = self._values.select(values_dim, int(selectors[physical_dim]))
        return cast(Self, _packed_without_dim(self, physical_dim, values))

    def _packed_newaxis_index(self, rest: tuple) -> Self | None:
        r"""Insert basic ``None`` axes without unpacking an untouched batch."""
        if not self.batch_first or not any(selector is None for selector in rest):
            return None
        physical_rank = int(self._physical_shape.size(1))
        consumed = sum(selector is not None for selector in rest)
        if consumed > physical_rank:
            return None
        selectors = tuple(rest) + (slice(None),) * (physical_rank - consumed)
        if any(
            selector is not None
            and not (
                isinstance(selector, slice)
                and selector.start is None
                and selector.stop is None
                and selector.step is None
            )
            for selector in selectors
        ):
            return None

        result = self
        for logical_dim, selector in enumerate(selectors, start=1):
            if selector is None:
                result = result.unsqueeze(logical_dim)
        return result

    def _empty_batch_like(self) -> Self:
        r"""Return a source-derived empty batch without discarding element-rank topology."""
        outer_size = list(self._logical_shape)
        batch_dim = type(self)._batch_dim_from_logical_shape(self._logical_shape, self.batch_first)
        outer_size[batch_dim] = 0
        return type(self)._from_packed(
            self._values.narrow(0, 0, 0),
            self._offsets.new_zeros((1,)),
            self._physical_shape[:0],
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=torch.Size(outer_size),
            packed_sizes=(),
            element_shapes=(),
            validate=False,
        )

    def _empty_batch_basic_index(self, rest: tuple) -> Self | None:
        r"""Project a source-derived empty batch through basic element indexing.

        With no physical rows, ``_physical_shape`` cannot report the output
        extents. ``_empty_batch_like`` deliberately retains those maxima in the
        logical shape, so use them as structural extents and rebuild metadata
        without inventing a representative element.
        """
        if len(self) != 0 or not self._ragged_dims_explicit:
            return None

        physical_rank = int(self._physical_shape.size(1))
        selectors: list[tuple[int | None, object]] = []
        old_dim = 0
        for selector in rest:
            if selector is None:
                selectors.append((None, selector))
                continue
            if old_dim >= physical_rank:
                raise IndexError(f"too many indices for NestedTensor with element rank {physical_rank}")
            selectors.append((old_dim, selector))
            old_dim += 1
        while old_dim < physical_rank:
            selectors.append((old_dim, slice(None)))
            old_dim += 1

        if any(
            type(selector) is not int and not isinstance(selector, slice) and selector is not None
            for _, selector in selectors
        ):
            return None

        physical_extents = self._max_physical_dims()
        static_lookup = {dim: axis for axis, dim in enumerate(self._static_dims)}
        static_selectors: dict[int, int | slice] = {}
        old_to_new: dict[int, int] = {}
        inserted_dims: list[int] = []
        projected_extents: list[int] = []

        for dim, selector in selectors:
            if dim is None:
                inserted_dims.append(len(projected_extents))
                projected_extents.append(1)
                continue

            extent = int(physical_extents[dim])
            if type(selector) is int:
                index = int(selector)
                if index < 0:
                    index += extent
                if index < 0 or index >= extent:
                    raise IndexError(f"index {selector} is out of bounds for dimension {dim} with size {extent}")
                if dim in static_lookup:
                    static_selectors[dim] = index
                continue

            assert isinstance(selector, slice)
            if selector.step is not None and selector.step <= 0:
                raise ValueError("step must be greater than zero")
            start, stop, step = selector.indices(extent)
            old_to_new[dim] = len(projected_extents)
            projected_extents.append(len(range(start, stop, step)))
            if dim in static_lookup:
                static_selectors[dim] = slice(start, stop, step)

        values_index = (slice(None), *(static_selectors.get(dim, slice(None)) for dim in self._static_dims))
        values = self._values[values_index]
        packed_static_dims = [old_to_new[dim] for dim in self._static_dims if dim in old_to_new]
        for dim in inserted_dims:
            values = values.unsqueeze(-1)
            packed_static_dims.append(dim)

        ragged_dims = tuple(old_to_new[dim] for dim in self._ragged_dims if dim in old_to_new)
        static_dims = tuple(dim for dim in range(len(projected_extents)) if dim not in ragged_dims)
        values = values.permute((0, *(1 + packed_static_dims.index(dim) for dim in static_dims)))
        offsets = self._offsets.new_zeros((1,))
        physical_shape = self._physical_shape.new_empty((0, len(projected_extents)))
        ragged_offsets = tuple(offsets for _ in ragged_dims) if ragged_dims else None
        return type(self)._from_packed(
            values,
            offsets,
            physical_shape,
            permutation=ragged_dims + static_dims,
            ragged_dims=ragged_dims,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape_from_physical_dims(projected_extents),
            packed_sizes=(),
            element_shapes=(),
            ragged_offsets=ragged_offsets,
            validate=False,
        )

    def _meta_after_basic_index(self, rest: tuple, *, include_dtype: bool = True) -> Mapping:
        r"""Return reconstruction metadata after basic indexing of physical element dimensions."""
        meta = dict(self._meta(include_dtype=include_dtype))
        if not self._ragged_dims_explicit:
            return meta

        old_to_new: dict[int, int] = {}
        old_dim = 0
        new_dim = 0
        for selector in rest:
            if selector is None:
                new_dim += 1
                continue
            if old_dim >= self._physical_shape.size(1):
                meta["ragged_dims"] = None
                return meta
            if isinstance(selector, int):
                old_dim += 1
                continue
            if not isinstance(selector, slice):
                # Advanced indexing can reorder dimensions; without explicit
                # provenance it is safer to drop the declaration than guess.
                meta["ragged_dims"] = None
                return meta
            old_to_new[old_dim] = new_dim
            old_dim += 1
            new_dim += 1
        while old_dim < self._physical_shape.size(1):
            old_to_new[old_dim] = new_dim
            old_dim += 1
            new_dim += 1
        meta["ragged_dims"] = tuple(old_to_new[dim] for dim in self._ragged_dims if dim in old_to_new)
        return meta

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
            if not storage and self._ragged_dims_explicit:
                return self._empty_batch_like()
            return self.__class__(storage, **self._meta(include_dtype=True))
        if isinstance(index, tuple):
            if len(index) == 0:
                return self

            # Expand Ellipsis: ``nt[..., :2]`` on a 4-D NestedTensor becomes
            # ``nt[:, :, :, :2]``.  The batch dim is consumed first, so Ellipsis
            # fills the gap between the number of explicit indices and the total
            # number of logical dimensions.
            if index.count(Ellipsis) > 1:
                raise IndexError("an index can only have a single ellipsis ('...')")
            if Ellipsis in index:
                eidx = index.index(Ellipsis)
                n_explicit = sum(1 for entry in index if entry is not Ellipsis and entry is not None)
                n_expand = self.dim() - n_explicit
                index = index[:eidx] + (slice(None),) * n_expand + index[eidx + 1 :]

            batch_index, *rest = index

            symbolic_tensor_metadata = self._packed_sizes is None and (
                _is_compiling() or _is_fake_tensor(self._offsets) or _is_fake_tensor(self._physical_shape)
            )
            if symbolic_tensor_metadata and batch_index == slice(None) and rest:
                newaxis_output = self._packed_newaxis_index(tuple(rest))
                if newaxis_output is not None:
                    return newaxis_output
                integer_output = self._packed_static_integer_index(tuple(rest))
                if integer_output is not None:
                    return integer_output
                first_selector = rest[0]
                if isinstance(first_selector, int):
                    _compile_unsupported(
                        "NestedTensor select",
                        "tensor-backed per-element index bounds are not implemented",
                    )
                if isinstance(first_selector, slice) and first_selector != slice(None):
                    _compile_unsupported(
                        "NestedTensor slice",
                        "tensor-backed ragged slice metadata is not implemented",
                    )

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
                if (
                    self.batch_first
                    and isinstance(batch_index, slice)
                    and batch_index == slice(None)
                    and rest
                    and isinstance(rest[0], int)
                ):
                    static_position = self._batch_select_static_position(rest[0], tuple(rest[1:]))
                    if static_position is not None:
                        return static_position
                if isinstance(batch_index, slice) and batch_index == slice(None) and rest:
                    newaxis_output = self._packed_newaxis_index(tuple(rest))
                    if newaxis_output is not None:
                        return newaxis_output
                    integer_output = self._packed_static_integer_index(tuple(rest))
                    if integer_output is not None:
                        return integer_output
                    slice_output = self._packed_physical_slice(tuple(rest))
                    if slice_output is not None:
                        return slice_output
                    empty_output = self._empty_batch_basic_index(tuple(rest))
                    if empty_output is not None:
                        return empty_output
                if isinstance(batch_index, slice):
                    selected = self._storage[batch_index]
                else:
                    selected = tuple(self._storage[i] for i in batch_index)
                if rest:
                    rest_tuple = tuple(rest)
                    selected = tuple(t[rest_tuple] for t in selected)
                    meta = self._meta_after_basic_index(rest_tuple, include_dtype=True)
                else:
                    meta = self._meta(include_dtype=True)
                if not selected and self._ragged_dims_explicit:
                    empty = self._empty_batch_like()
                    if not rest:
                        return empty
                    rest_tuple = tuple(rest)
                    projected = empty._empty_batch_basic_index(rest_tuple)
                    if projected is not None:
                        return projected
                return self.__class__(selected, **meta)
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
                    if not selected and self._ragged_dims_explicit:
                        return self._empty_batch_like()
                    return self.__class__(selected, **self._meta(include_dtype=True))
                if index.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    if index.numel() == 0 and self._ragged_dims_explicit:
                        return self._empty_batch_like()
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
                packed_sizes[idx] = self._packed_sizes_like(
                    (self._element_shapes[idx],),
                    self._ragged_dims if self._ragged_dims_explicit else None,
                )[0]
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
            # Expand Ellipsis (e.g. ``nt[..., 0] = 0``) the same way __getitem__ does:
            # the batch dim is consumed first, so Ellipsis fills the gap between the
            # explicit indices and the total number of logical dimensions.
            if index.count(Ellipsis) > 1:
                raise IndexError("an index can only have a single ellipsis ('...')")
            if Ellipsis in index:
                eidx = index.index(Ellipsis)
                n_explicit = sum(1 for entry in index if entry is not Ellipsis and entry is not None)
                n_expand = self.dim() - n_explicit
                index = index[:eidx] + (slice(None),) * n_expand + index[eidx + 1 :]
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
        values = vars(self).get("_values")
        if isinstance(values, Tensor):
            return values.dtype
        return torch.Tensor.dtype.__get__(self)

    @dtype.setter
    def dtype(self, value: torch.dtype | None):
        r"""`dtype` is read-only; use `.to(dtype=...)` to convert."""
        raise AttributeError("NestedTensor.dtype is read-only; use .to(dtype=...) to create a converted tensor.")

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        r"""Device on which the underlying tensor data resides."""
        values = vars(self).get("_values")
        if isinstance(values, Tensor):
            return values.device
        return torch.Tensor.device.__get__(self)

    @device.setter
    def device(self, value: torch.device | None):
        r"""`device` is read-only; use `.to(device=...)` to move tensors."""
        raise AttributeError("NestedTensor.device is read-only; use .to(device=...) to create a moved tensor.")

    @property
    def requires_grad(self) -> bool:  # type: ignore[override]
        r"""Whether gradient computation is enabled for this tensor."""
        return super().requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        r"""Enable or disable gradient computation for this tensor."""
        if super().requires_grad == value:
            return
        values = vars(self).get("_values")
        if not self.is_leaf:
            raise RuntimeError("you can only change requires_grad flags of leaf variables")
        if isinstance(values, Tensor) and values.requires_grad != value and not values.is_leaf:
            raise RuntimeError("you can only change requires_grad flags of leaf variables")
        torch.Tensor.requires_grad.__set__(self, value)  # type: ignore[attr-defined]
        if isinstance(values, Tensor) and values.requires_grad != value:
            values.requires_grad_(value)

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
                "ragged_dims": self._ragged_dims if self._ragged_dims_explicit else None,
                "device": self._values.device,
                "dtype": self.dtype,
            }
        return {
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "pin_memory": self._pin_memory,
            "ragged_dims": self._ragged_dims if self._ragged_dims_explicit else None,
            "device": self._values.device,
        }

    def __getstate__(self) -> dict:
        return {
            "_state_version": self._SERIALIZATION_VERSION,
            "_values": self._values,
            "_offsets": self._offsets,
            "_permutation": self._permutation,
            "_ragged_dims": self._ragged_dims if self._ragged_dims_explicit else None,
            "_physical_shape": self._physical_shape,
            "_logical_shape": self._logical_shape,
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "_pin_memory": self._pin_memory,
            "_packed_sizes": self._packed_sizes,
            "_element_shapes": self._element_shapes,
            "_ragged_offsets": self._persistent_ragged_offsets(),
        }

    def __setstate__(self, state: Mapping) -> None:
        type(self)._validate_serialized_state(state)
        self._values = state["_values"]
        self._offsets = state["_offsets"].cpu()
        self._permutation = tuple(int(dim) for dim in state["_permutation"])
        declared_ragged_dims = state["_ragged_dims"]
        self._ragged_dims = type(self)._ragged_dims_from_packed_layout(
            self._values,
            len(self._permutation),
            self._permutation,
            declared_ragged_dims,
        )
        self._ragged_dims_explicit = declared_ragged_dims is not None
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
        serialized_ragged_offsets = state["_ragged_offsets"]
        if serialized_ragged_offsets is not None:
            serialized_ragged_offsets = tuple(level_offsets.cpu() for level_offsets in serialized_ragged_offsets)
        ragged_offsets = type(self)._resolve_persistent_ragged_offsets(
            self._offsets,
            self._physical_shape,
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
            ragged_offsets=serialized_ragged_offsets,
        )
        type(self)._install_persistent_ragged_offsets(self, ragged_offsets)
        # Serialized state intentionally excludes transient caches.
        self._invalidate_transient_caches()
        self._mark_tensor_backed_dynamic_dims()
        self._validate_metadata()

    def __reduce__(self):
        return (self.__class__._from_state, (self.__getstate__(),))

    @classmethod
    def _from_state(cls, state: dict) -> Self:
        cls._validate_serialized_state(state)
        serialized_ragged_offsets = state["_ragged_offsets"]
        if serialized_ragged_offsets is not None:
            serialized_ragged_offsets = tuple(level_offsets.cpu() for level_offsets in serialized_ragged_offsets)
        return cls._from_packed(
            state["_values"],
            state["_offsets"].cpu(),
            state["_physical_shape"].cpu(),
            permutation=tuple(int(dim) for dim in state["_permutation"]),
            ragged_dims=state["_ragged_dims"],
            batch_first=state["batch_first"],
            padding_value=state["padding_value"],
            mask_value=state["mask_value"],
            pin_memory=state["_pin_memory"],
            outer_size=state["_logical_shape"],
            packed_sizes=state["_packed_sizes"],
            element_shapes=state["_element_shapes"],
            ragged_offsets=serialized_ragged_offsets,
        )

    def __copy__(self):
        r"""Shallow copy: new NestedTensor sharing underlying tensor data."""
        return self.__class__._from_packed(
            self._values,
            self._offsets,
            self._physical_shape,
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
            ragged_offsets=self._persistent_ragged_offsets(),
            validate=False,
        )

    def __deepcopy__(self, memo):
        r"""Deep copy: clones all tensor data."""
        result = self.__class__._from_packed(
            self._values.clone(),
            self._offsets.clone(),
            self._physical_shape.clone(),
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
            ragged_offsets=(
                tuple(level_offsets.clone() for level_offsets in self._persistent_ragged_offsets() or ()) or None
            ),
            validate=False,
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
            permutation=self._permutation,
            ragged_dims=self._ragged_dims if self._ragged_dims_explicit else None,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=True,
            outer_size=self._logical_shape,
            packed_sizes=self._packed_sizes,
            element_shapes=self._element_shapes,
            ragged_offsets=self._persistent_ragged_offsets(),
            validate=False,
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

    def repeat_batch(self, repeats: int, *, output_size: int | None = None) -> Self:
        r"""Repeat complete logical batch elements in interleaved order."""
        from .torch_functions import _repeat_interleave_packed_batch

        return _repeat_interleave_packed_batch(self, repeats, output_size=output_size)

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
            if _is_compiling() or _is_fake_tensor(self._physical_shape):
                _compile_unsupported(
                    "NestedTensor view/reshape",
                    "tensor-backed per-element view shape remapping is not implemented",
                )
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

    def cdist(
        self,
        other: Tensor | NestedTensor,
        p: float = 2.0,
        compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
    ) -> NestedTensor:
        r"""Compute per-sample pairwise distances through the traceable packed path.

        This method shares :func:`torch.cdist`'s ``p`` and ``compute_mode``
        contract. Unlike the built-in spelling, Dynamo can inline this explicit
        NestedTensor handler, so AOT/Inductor preserve gradients to prebuilt
        packed leaves when the result is consumed inside the compiled region.
        Wrapper-only compiled outputs retain their outer autograd edge, and
        :attr:`concat` projects that edge back to packed values without padding.
        """
        from .torch_functions import cdist

        return cdist(self, other, p, compute_mode)

    def cumprod(self, dim: int, *, dtype: torch.dtype | None = None) -> NestedTensor:
        r"""Compute cumulative products through the traceable packed segmented path.

        The explicit method keeps gradients connected to prebuilt packed leaves
        across AOTAutograd/Inductor, including wrapper-only compiled outputs.
        """
        op = torch.ops.aten.cumprod.default
        kwargs = {} if dtype is None else {"dtype": dtype}
        return NestedTensorAtenRegistry[op](op, (self, dim), kwargs)


class _PackedStructureReference:
    r"""Non-Tensor carrier for layout metadata used by the packed autograd bridge."""

    __slots__ = ("value",)

    def __init__(self, value: NestedTensor) -> None:
        self.value = value


class _PackedLikeAutograd(torch.autograd.Function):
    r"""Attach a derived packed tensor to a wrapper produced inside a compiled graph."""

    @staticmethod
    def forward(ctx, packed_values: Tensor, structure: _PackedStructureReference) -> NestedTensor:
        ctx.structure = structure
        ctx.packed_shape = packed_values.shape
        reference = structure.value
        result = reference._packed_like_unchecked_raw(packed_values)
        max_length_binding = vars(reference).get("_compile_max_length_binding")
        if max_length_binding is not None:
            result._max_length_binding = max_length_binding
        return result

    @staticmethod
    def backward(ctx, grad_output: NestedTensor):
        reference = ctx.structure.value
        values_grad = grad_output._values if isinstance(grad_output, NestedTensor) else grad_output
        reference_static = reference._static_dims
        if isinstance(grad_output, NestedTensor):
            grad_static = grad_output._static_dims
            packed_tangent = grad_output._ragged_dims == reference._ragged_dims and set(grad_static) == set(
                reference_static
            )
        else:
            grad_static = ()
            packed_tangent = False
        if packed_tangent:
            packed_order = (0, *(1 + grad_static.index(dim) for dim in reference_static))
            values_grad = values_grad.permute(packed_order)
        elif values_grad.shape != ctx.packed_shape:
            packed_grad = reference._dense_to_packed_values(values_grad)
            if packed_grad is None:
                raise RuntimeError(
                    "NestedTensor could not project the wrapper tangent back to packed values: "
                    f"got tangent shape {tuple(values_grad.shape)} for packed shape {tuple(ctx.packed_shape)}"
                )
            values_grad = packed_grad
        return values_grad, None


class _PackedValuesAutograd(torch.autograd.Function):
    r"""Project a wrapper-owned autograd edge back onto its packed values."""

    @staticmethod
    def forward(ctx, input: NestedTensor) -> Tensor:
        ctx.save_for_backward(input)
        values = input._values
        from torch._subclasses.functional_tensor import FunctionalTensor

        outer_grad_fn = input.grad_fn
        if isinstance(values, FunctionalTensor) or type(outer_grad_fn).__name__ == "CompiledFunctionBackward":
            return values
        projected = values.view_as(values)
        from torch._dynamo import maybe_mark_dynamic

        maybe_mark_dynamic(projected, 0)
        return projected

    @staticmethod
    def backward(ctx, grad_values: Tensor):
        (input,) = ctx.saved_tensors
        return input._packed_like_unchecked_raw(grad_values)


@torch.compiler.allow_in_graph
def _project_packed_values(input: NestedTensor) -> Tensor:
    r"""Give Dynamo an explicit source for the wrapper-to-packed projection."""
    return _PackedValuesAutograd.apply(input)


_cdist_eager = NestedTensor.cdist


@torch.compiler.substitute_in_graph(_cdist_eager)
def _traceable_cdist(
    self: NestedTensor,
    other: Tensor | NestedTensor,
    p: float = 2.0,
    compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
) -> NestedTensor:
    r"""Preserve packed-leaf autograd while tracing the explicit method."""
    from .torch_functions import cdist

    return cdist(self, other, p, compute_mode)


# Wrapper-subclass inputs need the decorated replacement to be the bound
# method itself; registering the substitution without assigning it lets AOT
# build a backward graph but disconnects pre-existing packed leaves.
NestedTensor.cdist = _traceable_cdist  # type: ignore[method-assign]


_cumprod_eager = NestedTensor.cumprod


@torch.compiler.substitute_in_graph(_cumprod_eager)
def _traceable_cumprod(
    self: NestedTensor,
    dim: int,
    *,
    dtype: torch.dtype | None = None,
) -> NestedTensor:
    r"""Preserve packed-leaf autograd while tracing segmented cumprod."""
    op = torch.ops.aten.cumprod.default
    kwargs = {} if dtype is None else {"dtype": dtype}
    return NestedTensorAtenRegistry[op](op, (self, dim), kwargs)


NestedTensor.cumprod = _traceable_cumprod  # type: ignore[method-assign]


_repeat_batch_eager = NestedTensor.repeat_batch


@torch.compiler.substitute_in_graph(_repeat_batch_eager)
def _traceable_repeat_batch(self: NestedTensor, repeats: int, *, output_size: int | None = None) -> NestedTensor:
    r"""Repeat complete logical batch elements in interleaved order."""
    from .torch_functions import _repeat_interleave_packed_batch

    return _repeat_interleave_packed_batch(self, repeats, output_size=output_size)


NestedTensor.repeat_batch = _traceable_repeat_batch  # type: ignore[method-assign]


def _make_nested_tensor_from_packed_impl(
    nested_tensor_cls: type[NestedTensor],
    values: Tensor,
    offsets: Tensor,
    shape_tensor: Tensor,
    logical_shape: torch.Size,
    permutation: tuple[int, ...],
    ragged_dims: tuple[int, ...],
    ragged_dims_explicit: bool,
    batch_first: bool,
    padding_value: float,
    mask_value: bool,
    pin_memory: bool,
    packed_sizes: tuple[int, ...] | None,
    element_shapes: tuple[tuple[int, ...], ...] | None,
    ragged_offsets: tuple[Tensor, ...] | None,
) -> NestedTensor:
    result = torch.Tensor._make_wrapper_subclass(
        nested_tensor_cls,
        logical_shape,
        dtype=values.dtype,
        device=values.device,
        requires_grad=values.requires_grad,
    )
    result._values = values
    result._offsets = offsets
    result._permutation = tuple(int(dim) for dim in permutation)
    result._ragged_dims = tuple(int(dim) for dim in ragged_dims)
    result._ragged_dims_explicit = ragged_dims_explicit
    result._physical_shape = shape_tensor
    result._logical_shape = logical_shape
    result._set_runtime_config(
        batch_first=batch_first,
        padding_value=padding_value,
        mask_value=mask_value,
    )
    result._pin_memory = pin_memory
    result._packed_sizes = packed_sizes
    result._element_shapes = element_shapes
    nested_tensor_cls._install_persistent_ragged_offsets(result, ragged_offsets)
    result._invalidate_transient_caches()
    return result


def _make_nested_tensor_from_packed(
    values: Tensor,
    offsets: Tensor,
    shape_tensor: Tensor,
    logical_shape: torch.Size,
    permutation: tuple[int, ...],
    ragged_dims: tuple[int, ...],
    ragged_dims_explicit: bool,
    batch_first: bool,
    padding_value: float,
    mask_value: bool,
    pin_memory: bool,
    packed_sizes: tuple[int, ...] | None,
    element_shapes: tuple[tuple[int, ...], ...] | None,
    ragged_offsets: tuple[Tensor, ...] | None,
) -> NestedTensor:
    r"""Construct the base wrapper through a stable module-level graph target."""
    return _make_nested_tensor_from_packed_impl(
        NestedTensor,
        values,
        offsets,
        shape_tensor,
        logical_shape,
        permutation,
        ragged_dims,
        ragged_dims_explicit,
        batch_first,
        padding_value,
        mask_value,
        pin_memory,
        packed_sizes,
        element_shapes,
        ragged_offsets,
    )


if hasattr(torch, "compiler") and hasattr(torch.compiler, "allow_in_graph"):
    _make_nested_tensor_from_packed = torch.compiler.allow_in_graph(_make_nested_tensor_from_packed)


def _make_nested_tensor_from_packed_constructor(nested_tensor_cls: type[NestedTensor]):
    r"""Capture a concrete wrapper class without passing its type through FX."""

    def constructor(*args) -> NestedTensor:
        return _make_nested_tensor_from_packed_impl(nested_tensor_cls, *args)

    if hasattr(torch, "compiler") and hasattr(torch.compiler, "allow_in_graph"):
        constructor = torch.compiler.allow_in_graph(constructor)
    return constructor


NestedTensor._compiled_packed_constructor = staticmethod(_make_nested_tensor_from_packed)
