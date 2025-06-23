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

# pylint: disable=protected-access
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Iterable, SupportsFloat

import torch
from torch import Tensor

from .aten_functions import per_element_fallback
from .ops import NestedTensorAtenRegistry
from .utils import mask_tensor, pad_tensor, tensor_mask

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from torch import nested


def _cat_ragged_parts(parts: list[Tensor], logical_size: torch.Size | Sequence[int], batch_first: bool) -> Tensor:
    r"""
    Reshape variable-length parts to collapse ragged dims, then concatenate along dim 0.

    Shared by ``concatenate()`` and ``_extract_concat_from_padded`` to ensure
    they always produce the same layout.
    """
    if parts[0].ndim == 0:
        return torch.stack(parts, dim=0)

    inner = list(logical_size[1:] if batch_first else [logical_size[0]] + list(logical_size[2:]))
    for i in range(len(inner)):
        if len({p.size(i) for p in parts}) > 1:
            inner[i] = -1
    target = [-1] + [s for s in inner if s != -1]
    if len(target) > parts[0].ndim:
        return torch.cat(parts, dim=0)
    return torch.cat([p.reshape(target) for p in parts], dim=0)


def _torch_function_impl(cls, func, types, args, kwargs):
    r"""
    Standalone __torch_function__ handler.

    Individual non-traceable helpers (those that iterate _storage via Python
    loops) are decorated with @torch._dynamo.disable instead, so that
    elementwise ops operating directly on _values can be captured by the
    torch.compile graph.
    """
    if kwargs is None:
        kwargs = {}

    # Handle size() specially to avoid infinite recursion
    if func is torch.Tensor.size:
        self = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        return self.size(dim)

    from .ops import NestedTensorFuncRegistry

    handler = NestedTensorFuncRegistry.get(func)
    if handler is not None:
        return handler(*args, **kwargs)

    # Fall through to aten decomposition → __torch_dispatch__
    with torch._C.DisableTorchFunctionSubclass():
        return func(*args, **kwargs)


class NestedTensor(torch.Tensor):
    r"""
    A container for variable-length tensors that enables efficient batch operations.

    `NestedTensor` solves a fundamental problem in deep learning: handling sequences of different lengths
    in batch operations. Instead of excessive padding or complex bucketing, `NestedTensor` provides an
    elegant solution that maintains both efficiency and usability.

    The class provides three main views of the data:
    - `.tensor`: A padded tensor with zeros (or other value) in place of missing elements
    - `.mask`: A boolean mask indicating which elements are real vs padding
    - `.concat`: A flattened tensor containing all elements concatenated (no padding)

    When indexing a `NestedTensor`, the behavior depends on the index type:
    1. Integer index (`nt[0]`): Returns a single tensor without padding
    2. Slice index (`nt[:]`): Returns a tuple of (padded_tensor, mask)
    3. Tuple index (`nt[:, 1:]`): Returns a new `NestedTensor` with the specified sliced shape

    Attributes:
        _values: Packed tensor data (concatenated along dim 0 or flattened)
        _offsets: Cumulative element counts, shape (B+1,)
        _shape_tensor: Per-element shapes, shape (B, max_ndim)
        batch_first: Whether the first dimension is the batch dimension (B, N, *)
            If `False`, the first dimension is the sequence dimension (N, B, *)
        padding_value: Value used for padding in the padded tensor
        mask_value: Value used in the mask to indicate padding positions (usually False)

    Args:
        *tensors: Variable-length tensors or sequences to store
        batch_first: Whether to use batch-first representation.
        padding_value: Value to use for padding.
        mask_value: Value to use for padding positions in mask.

    Raises:
        ValueError: If `tensors` is not an iterable or is empty

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
        >>> nested_tensor[:2]  # Padded tensor and mask
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
    _shape_tensor: Tensor
    _logical_shape: torch.Size
    batch_first: bool
    padding_value: float
    mask_value: bool
    _pin_memory: bool
    _cached_storage: tuple[Tensor, ...] | None

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

        # Pack into (values, offsets, shape_tensor)
        values, offsets, shape_tensor = cls._pack(validated, dtype=out_dtype)

        # Compute logical shape
        logical_shape = cls._compute_logical_shape(validated, batch_first)
        out_requires_grad = requires_grad if requires_grad is not None else False

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            logical_shape,
            dtype=out_dtype,
            device=out_device,
            requires_grad=out_requires_grad,
        )
        r._values = values
        r._offsets = offsets
        r._shape_tensor = shape_tensor
        r._logical_shape = logical_shape
        r.batch_first = batch_first
        r.padding_value = float(padding_value)
        r.mask_value = mask_value
        r._pin_memory = pin_memory
        r._cached_storage = None
        return r

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

        result = []
        for t in tensors:
            if not isinstance(t, Tensor):
                t = torch.tensor(t, dtype=dtype, device=device, pin_memory=pin_memory)
            else:
                t = t.to(device, dtype=dtype)
            if requires_grad is not None:
                t.requires_grad_(requires_grad)
            result.append(t)

        if not result:
            return ()

        devices = {t.device for t in result}
        if len(devices) != 1:
            raise ValueError(
                "All tensors in NestedTensor must be on the same device, " f"but got {sorted(str(d) for d in devices)}"
            )
        common_dtype = result[0].dtype
        needs_cast = False
        for t in result[1:]:
            promoted = torch.promote_types(common_dtype, t.dtype)
            if promoted != common_dtype:
                common_dtype = promoted
                needs_cast = True
            elif t.dtype != common_dtype:
                needs_cast = True
        if needs_cast:
            result = [t.to(dtype=common_dtype) for t in result]

        ndims = {t.ndim for t in result}
        if len(ndims) > 1:
            raise ValueError(
                f"All tensors must have the same number of dimensions, got ndim in {sorted(ndims)}. "
                "If using a DataLoader with drop_last=False, squeeze the last batch before constructing NestedTensor."
            )

        return tuple(result)

    @staticmethod
    def _pack(tensors: tuple[Tensor, ...], *, dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor, Tensor]:
        r"""Pack a sequence of tensors into (values, offsets, shape_tensor)."""
        if not tensors:
            return (
                torch.empty(0, dtype=dtype or torch.get_default_dtype()),
                torch.zeros(1, dtype=torch.long),
                torch.empty(0, 0, dtype=torch.long),
            )

        max_ndim = max(t.ndim for t in tensors)

        # Offsets and shape_tensor are metadata — always on CPU to avoid CUDA syncs.
        shape_tensor = torch.tensor([list(t.shape) + [0] * (max_ndim - t.ndim) for t in tensors], dtype=torch.long)

        # Determine packing layout from data:
        # - If all trailing dims match: cat along dim 0 → N-D values
        # - Otherwise: flatten each element → 1-D values
        if len(tensors) <= 1 or tensors[0].ndim == 0 or all(t.shape[1:] == tensors[0].shape[1:] for t in tensors[1:]):
            values = torch.cat(tensors, dim=0) if tensors[0].ndim > 0 else torch.stack(tensors)
            sizes = torch.tensor([t.size(0) if t.ndim > 0 else 1 for t in tensors], dtype=torch.long)
        else:
            flat = [t.flatten() for t in tensors]
            values = torch.cat(flat, dim=0)
            sizes = torch.tensor([t.numel() for t in flat], dtype=torch.long)
        offsets = torch.zeros(len(tensors) + 1, dtype=torch.long)
        torch.cumsum(sizes, dim=0, out=offsets[1:])

        return values, offsets, shape_tensor

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
    def _logical_shape_from_shape_tensor(shape_tensor: Tensor, offsets: Tensor, batch_first: bool) -> torch.Size:
        r"""Compute logical shape from packed metadata without unpacking elements."""
        batch_size = len(offsets) - 1
        if batch_size == 0:
            return torch.Size((0,))
        if shape_tensor.numel() == 0:
            return torch.Size((batch_size,))
        size = [int(shape_tensor[:, d].max().item()) for d in range(shape_tensor.size(1))]
        while size and size[-1] == 0:
            size.pop()
        size.insert(0 if batch_first else 1, batch_size)
        return torch.Size(size)

    @classmethod
    def _from_packed(
        cls,
        values: Tensor,
        offsets: Tensor,
        shape_tensor: Tensor,
        *,
        batch_first: bool = True,
        padding_value: float = 0.0,
        mask_value: bool = False,
        pin_memory: bool = False,
        outer_size: torch.Size | tuple | None = None,
    ) -> Self:
        r"""Construct a NestedTensor directly from packed representation (no validation)."""
        # offsets and shape_tensor MUST live on CPU to avoid implicit CUDA syncs
        # when handlers call .item() / .tolist() on them.
        if offsets.device.type != "cpu":
            raise ValueError(f"offsets must be on CPU, got {offsets.device}")
        if shape_tensor.device.type != "cpu":
            raise ValueError(f"shape_tensor must be on CPU, got {shape_tensor.device}")

        if outer_size is not None:
            logical_shape = torch.Size(outer_size)
        else:
            logical_shape = cls._logical_shape_from_shape_tensor(shape_tensor, offsets, batch_first)

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            logical_shape,
            dtype=values.dtype,
            device=values.device,
            requires_grad=values.requires_grad,
        )
        r._values = values
        r._offsets = offsets
        r._shape_tensor = shape_tensor
        r._logical_shape = logical_shape
        r.batch_first = batch_first
        r.padding_value = padding_value
        r.mask_value = mask_value
        r._pin_memory = pin_memory
        r._cached_storage = None
        return r

    # ------------------------------------------------------------------
    # torch.compile support
    # ------------------------------------------------------------------

    def __tensor_flatten__(self):
        return ["_values", "_offsets", "_shape_tensor"], {
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "pin_memory": self._pin_memory,
        }

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, ctx, outer_size, outer_stride):
        return cls._from_packed(
            inner_tensors["_values"],
            inner_tensors["_offsets"],
            inner_tensors["_shape_tensor"],
            outer_size=outer_size,
            **ctx,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None) -> Any:
        return _torch_function_impl(cls, func, types, args, kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None) -> Any:
        if kwargs is None:
            kwargs = {}

        if func in NestedTensorAtenRegistry:
            return NestedTensorAtenRegistry[func](func, args, kwargs)

        # Per-element fallback
        return per_element_fallback(func, args, kwargs)

    def _unpack(self) -> tuple[Tensor, ...]:
        r"""Reconstruct individual tensors from packed representation."""
        batch_size = len(self._offsets) - 1
        if batch_size == 0:
            return ()

        # _offsets and _shape_tensor are on CPU — no CUDA syncs here.
        sizes = (self._offsets[1:] - self._offsets[:-1]).tolist()
        splits = self._values.split(sizes, dim=0)

        # Reshape each chunk to its original shape.
        # For N-D packed values (single-ragged), reshape is a no-op view.
        shapes = self._shape_tensor.tolist()
        result = []
        for chunk, shape in zip(splits, shapes):
            while shape and shape[-1] == 0:
                shape.pop()
            if not shape:
                result.append(chunk[0])
            else:
                result.append(chunk.reshape(shape))
        return tuple(result)

    def _repack(self, tensors: Sequence) -> None:
        r"""
        Re-pack from already-validated tensors. Skips coercion — callers must ensure
        tensors share device, dtype, and ndim (which is always true for internal paths
        since tensors originate from _unpack or __setitem__ validation)."""
        self._cached_storage = None
        tensors = tuple(tensors) if not isinstance(tensors, tuple) else tensors
        values, offsets, shape_tensor = self._pack(tensors)
        self._values = values
        self._offsets = offsets
        self._shape_tensor = shape_tensor
        self._logical_shape = self._compute_logical_shape(tensors, self.batch_first)

    @property
    def _storage(self) -> tuple[Tensor, ...]:
        if self._cached_storage is None:
            self._cached_storage = self._unpack()
        return self._cached_storage

    @_storage.setter
    def _storage(self, tensors: Sequence) -> None:
        self._repack(tensors)

    # ------------------------------------------------------------------
    # Properties: dtype, device, requires_grad
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        r"""Data type of the underlying tensor elements."""
        return self._values.dtype

    @dtype.setter
    def dtype(self, value: torch.dtype | None):
        r"""Set the data type and cast all elements accordingly."""
        if value is not None and self._values.numel() > 0:
            self._cached_storage = None
            self._values = self._values.to(dtype=value)

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        r"""Device on which the underlying tensor data resides."""
        return self._values.device

    @device.setter
    def device(self, value: torch.device | None):
        r"""Set the device and move all elements accordingly."""
        if value is not None and self._values.numel() > 0:
            self._cached_storage = None
            self._values = self._values.to(device=value)

    @property
    def requires_grad(self) -> bool:  # type: ignore[override]
        r"""Whether gradient computation is enabled for this tensor."""
        return self._values.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        r"""Enable or disable gradient computation for this tensor."""
        if self._values.numel() > 0:
            self._values.requires_grad_(value)

    # ------------------------------------------------------------------
    # Cached padded views
    # ------------------------------------------------------------------

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
        return self._tensor_mask(self._storage, self.batch_first, self.padding_value, self.mask_value)

    def _tensor_mask(
        self, storage: tuple, batch_first: bool, padding_value: float, mask_value: bool
    ) -> tuple[Tensor, Tensor]:
        r"""Compute padded tensor and boolean mask from raw storage."""
        if not storage:
            size = self.size()
            t = torch.empty(size, dtype=self._values.dtype, device=self.device)
            m = torch.empty(size, dtype=torch.bool, device=self.device)
            return t, m
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0), torch.full(
                (len(storage),), not mask_value, dtype=torch.bool, device=self.device
            )
        return tensor_mask(
            storage,
            size=self.size(),
            batch_first=batch_first,
            padding_value=padding_value,
            mask_value=mask_value,
        )

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
        return self._tensor(self._storage, self.batch_first, self.padding_value)

    def _tensor(self, storage: tuple, batch_first: bool, padding_value: float) -> Tensor:
        r"""Pad and stack raw storage into a single dense tensor."""
        if not storage:
            return torch.empty(self.size(), dtype=self._values.dtype, device=self.device)
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0)
        return pad_tensor(storage, size=self.size(), batch_first=batch_first, padding_value=padding_value)

    @property
    def mask(self) -> Tensor:
        r"""
        Padding mask of `tensor`.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.mask
            tensor([[ True,  True,  True],
                    [ True,  True, False]])
        """
        return self._mask(self._storage, self.batch_first, self.mask_value)

    def _mask(self, storage: tuple, batch_first: bool, mask_value: bool) -> Tensor:
        r"""Compute the boolean padding mask from raw storage."""
        if not storage:
            return torch.empty(self.size(), dtype=torch.bool, device=self.device)
        if storage[0].dim() == 0:
            return torch.full((len(storage),), not mask_value, dtype=torch.bool, device=self.device)
        return mask_tensor(storage, size=self.size(), batch_first=batch_first, mask_value=mask_value)

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
        """
        return self.concatenate()[0]

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

        storage = self._storage
        original_shapes = tuple(t.shape for t in storage)
        concat_tensor = _cat_ragged_parts(list(storage), self.size(), self.batch_first)
        return concat_tensor, original_shapes

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
            return cls([], **kwargs)

        num_elements = [shape.numel() for shape in shapes]

        if len(set(shapes)) == 1:
            shape = shapes[0]
            total_elements = sum(num_elements)
            if concat_tensor.numel() == total_elements:
                try:
                    reshaped = concat_tensor.reshape(len(shapes), *shape)
                except (RuntimeError, ValueError) as e:
                    warnings.warn(
                        f"from_concatenated reshape fast-path failed ({e}), " "falling back to per-element unpack",
                        stacklevel=2,
                    )
                else:
                    tensors = [t.reshape(shape) for t in reshaped.unbind(0)]
                    return cls(tensors, **kwargs)

        flattened = concat_tensor.flatten()
        total_expected = sum(num_elements)
        num_provided = flattened.numel()
        if num_provided != total_expected:
            raise ValueError(
                f"Concatenated tensor has {num_provided} elements "
                f"but expected {total_expected} based on shapes {shapes}"
            )

        tensors = []
        start = 0
        for shape in shapes:
            end = start + shape.numel()
            tensor_data = flattened[start:end].reshape(shape)
            tensors.append(tensor_data)
            start = end

        return cls(tensors, **kwargs)

    @property
    def torch(self) -> Tensor:
        r"""
        Create a `torch.nested.nested_tensor` object from `self`.

        Examples:
            >>> nested_tensor = NestedTensor([[2, 3, 5], [7, 8]])
            >>> nested_tensor.torch
            nested_tensor([
              tensor([2, 3, 5]),
              tensor([7, 8])
            ])
        """
        return nested.nested_tensor(list(self._storage))

    def unbind(self, dim: int = 0) -> tuple[Tensor, ...]:
        r"""
        Unbind the NestedTensor.
        """
        return torch.unbind(self, dim=dim)

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

    @classmethod
    def from_tensor_mask(cls, tensor: Tensor, mask: Tensor, *, batched: bool = False, **kwargs):
        r"""
        Build a `NestedTensor` object from a padded `Tensor` and corresponding mask `Tensor`.

        Args:
            tensor: Padded Tensor.
            mask: Tensor Mask.
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
                return cls([tensor[int(i)] for i in indices], **kwargs)
            valid = int(effective_mask.sum().item())
            return cls(tensor[:valid], **kwargs)
        # ndim >= 2: batch setup is shared, per-element trim differs by rank
        batch_first = kwargs.get("batch_first", True)
        tensor_iter = tensor if batch_first else tensor.transpose(0, 1)
        mask_iter = effective_mask if batch_first else effective_mask.transpose(0, 1)
        if tensor_iter.size(0) != mask_iter.size(0):
            raise ValueError("Tensor/mask batch dimension mismatch: " f"{tensor_iter.size(0)} vs {mask_iter.size(0)}")
        trimmed = []
        if mask.ndim == 2:
            # 1-D per-element mask: contiguous-prefix fast path avoids nonzero()
            for t, m in zip(tensor_iter, mask_iter):
                count = int(m.sum().item())
                if count == 0 or bool(m[:count].all().item()):
                    trimmed.append(t[:count])
                else:
                    trimmed.append(t[m])
        else:
            # N-D per-element mask: bounding-box slice via nonzero()
            for t, em in zip(tensor_iter, mask_iter):
                nonzero = em.nonzero(as_tuple=False)
                if nonzero.numel() == 0:
                    slices = tuple(slice(0, 0) for _ in range(em.dim()))
                else:
                    max_indices = nonzero.max(dim=0).values + 1
                    slices = tuple(slice(0, int(i.item())) for i in max_indices)
                trimmed.append(t[slices])
        return cls(trimmed, **kwargs)

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
            ValueError: The shape of NestedTensor and input tensor does not match, torch.Size([2, 3]) != torch.Size([2, 2])
            >>> p = nested_tensor.nested_like(torch.randn(2, 2), False)
            >>> p = nested_tensor.nested_like(torch.randn(3, 3), False)
            Traceback (most recent call last):
            ValueError: The batch size of NestedTensor and input tensor does not match, 2 != 3
        """
          # noqa: E501

        if isinstance(tensor, NestedTensor):
            return tensor.clone()

        if strict and self.shape != tensor.shape:
            raise ValueError(
                f"The shape of NestedTensor and input tensor does not match, {self.shape} != {tensor.shape}"
            )
        batch_dim = 0 if self.batch_first else 1
        if len(self) != tensor.size(batch_dim):
            raise ValueError(
                "The batch size of NestedTensor and input tensor does not match, "
                f"{len(self)} != {tensor.size(batch_dim)}"
            )
        new_storage = []
        for idx, t in enumerate(self._storage):
            if self.batch_first:
                slices = (idx, *[slice(0, dim) for dim in t.shape])
            else:
                slices = (slice(0, t.size(0)), idx, *[slice(0, dim) for dim in t.shape[1:]])
            # .contiguous() ensures storage elements don't inherit non-trivial
            # strides from the padded tensor (e.g. after transpose).
            new_storage.append(tensor[slices].contiguous())
        return self.__class__(new_storage, **self._meta)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, index: int | slice | list | tuple | Tensor | NestedTensor) -> Tensor | NestedTensor:
        r"""Retrieve element(s) by index, slice, list, tuple, or tensor mask."""
        if isinstance(index, int):
            return self._storage[index]
        if isinstance(index, (slice, list)):
            storage = tuple(self._storage[index] if isinstance(index, slice) else [self._storage[i] for i in index])
            return self.__class__(storage, **self._meta)
        if isinstance(index, tuple):
            if len(index) == 0:
                return self
            batch_index, *rest = index

            if batch_index is Ellipsis:
                batch_index = slice(None)

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
                return self.__class__(selected, **self._meta)
            raise ValueError(f"Unsupported batch index type {type(batch_index)}")
        if isinstance(index, Tensor):
            index = self.nested_like(index, strict=False)
        if isinstance(index, NestedTensor):
            if len(self) != len(index):
                raise ValueError(f"NestedTensor batch length mismatch: {len(self)} vs {len(index)}")
            return self.__class__([t[i] for t, i in zip(self._storage, index._storage)], **self._meta)
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
            self._cached_storage = None
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

            flat_value = value.reshape(-1)
            old_start = int(self._offsets[idx].item())
            old_end = int(self._offsets[idx + 1].item())
            old_numel = old_end - old_start

            if flat_value.numel() == old_numel:
                # Same flattened size: copy directly into _values (no reallocation)
                self._values[old_start:old_end] = flat_value
                # Update shape_tensor row in case shape changed (e.g. [6] -> [2,3])
                new_shape = list(value.shape) + [0] * (self._shape_tensor.size(1) - value.dim())
                self._shape_tensor[idx] = torch.tensor(new_shape, dtype=self._shape_tensor.dtype)
            else:
                # Different size: splice _values and rebuild offsets/shape_tensor.
                # Clone metadata tensors first to avoid corrupting shallow copies
                # that share the same _offsets/_shape_tensor objects.
                self._values = torch.cat([self._values[:old_start], flat_value, self._values[old_end:]])
                delta = flat_value.numel() - old_numel
                self._offsets = self._offsets.clone()
                self._offsets[idx + 1 :] += delta  # noqa: E203
                self._shape_tensor = self._shape_tensor.clone()
                new_shape = list(value.shape) + [0] * (self._shape_tensor.size(1) - value.dim())
                self._shape_tensor[idx] = torch.tensor(new_shape, dtype=self._shape_tensor.dtype)
            self._logical_shape = self._logical_shape_from_shape_tensor(
                self._shape_tensor, self._offsets, self.batch_first
            )
        elif isinstance(index, (slice, list)):
            if isinstance(value, Tensor) and not isinstance(value, NestedTensor):
                if value.dim() > 1 and value.size(0) > 1:
                    value = self.__class__(value.unbind(0), **self._meta)
                else:
                    value = self.__class__([value], **self._meta)

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
            if len(index) < 2:
                raise ValueError("Tuple index must have at least two elements")

            first_idx, rest_idx = index[0], index[1:]

            # _storage returns views into _values, so in-place writes through
            # the view already modify _values — no repack needed.
            # Recompute _logical_shape defensively for consistency with other
            # branches, even though in-place indexed assignment cannot change
            # element shapes in PyTorch.
            if isinstance(first_idx, int):
                self._storage[first_idx][rest_idx] = value
            elif isinstance(first_idx, (slice, list)):
                if isinstance(first_idx, slice):
                    start, stop, step = first_idx.indices(len(self))
                    indices = range(start, stop, step)
                else:
                    indices = first_idx  # type: ignore[assignment]

                elements = self._storage
                for idx in indices:
                    elements[idx][rest_idx] = value
            else:
                raise ValueError(f"Unsupported first index type {type(first_idx)}")
            self._logical_shape = self._logical_shape_from_shape_tensor(
                self._shape_tensor, self._offsets, self.batch_first
            )
        else:
            raise ValueError(f"Unsupported index type {type(index)}")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @property
    def _meta(self) -> Mapping:
        return {
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "pin_memory": self._pin_memory,
            "device": self._values.device,
        }

    def __getstate__(self) -> dict:
        return {
            "_values": self._values,
            "_offsets": self._offsets,
            "_shape_tensor": self._shape_tensor,
            "_logical_shape": self._logical_shape,
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "_pin_memory": self._pin_memory,
        }

    def __setstate__(self, state: Mapping) -> None:
        self.__dict__.update(state)

    def __reduce__(self):
        return (self.__class__._from_state, (self.__getstate__(),))

    @classmethod
    def _from_state(cls, state: dict) -> Self:
        return cls._from_packed(
            state["_values"],
            state["_offsets"],
            state["_shape_tensor"],
            batch_first=state["batch_first"],
            padding_value=state["padding_value"],
            mask_value=state["mask_value"],
            pin_memory=state["_pin_memory"],
            outer_size=state["_logical_shape"],
        )

    def __copy__(self):
        r"""Shallow copy: new NestedTensor sharing underlying tensor data."""
        return self.__class__._from_packed(
            self._values,
            self._offsets,
            self._shape_tensor,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape,
        )

    def __deepcopy__(self, memo):
        r"""Deep copy: clones all tensor data."""
        result = self.__class__._from_packed(
            self._values.clone(),
            self._offsets.clone(),
            self._shape_tensor.clone(),
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=self._pin_memory,
            outer_size=self._logical_shape,
        )
        memo[id(self)] = result
        return result

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        r"""Return the number of tensors in the batch."""
        return len(self._offsets) - 1

    def __repr__(self):
        r"""Return a human-readable string representation of the NestedTensor."""
        if len(self) == 0:
            return self.__class__.__name__ + "()"

        storage = self._storage
        truncated = len(storage) > 10
        if truncated:
            storage = storage[:5]

        indent = "    "

        # Strip "tensor(" / "PNTensor(" wrapper from each element's repr,
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
        r"""Return True if the NestedTensor contains any elements."""
        return len(self) > 0

    def __iter__(self):
        r"""Iterate over the tensors in the batch."""
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

    # Arithmetic, comparison, and in-place operators are handled by the base
    # Tensor class, which routes through C++ → aten → __torch_dispatch__ →
    # aten_functions.py. No Python-level overrides needed.

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
            tensor([False, True])
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
    def ndim(self) -> int:
        r"""
        Alias for `dim()`.
        """
        return self.dim()

    @property
    def T(self) -> Self:  # type: ignore[override]
        r"""Transpose: applies .T to each element in storage."""
        return self.__class__([t.T for t in self._storage], **self._meta)

    @property
    def mT(self) -> Self:  # type: ignore[override]
        r"""Batch matrix transpose: applies .mT to each element in storage."""
        return self.__class__([t.mT for t in self._storage], **self._meta)

    @property
    def shape(self) -> torch.Size:  # type: ignore[override, name-defined]
        r"""
        Alias for `size()`.
        """
        return self.size()

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
        full_size = self._logical_shape
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

    def prod(
        self,
        dim: int | None = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,  # type: ignore[name-defined]
    ) -> Tensor | NestedTensor:
        r"""Return the product of elements, optionally along a given dimension."""
        return torch.prod(self, dim=dim, keepdim=keepdim, dtype=dtype)

    # to(), clone(), detach(), contiguous(), half(), float(), double(), etc.
    # are all handled by aten dispatch in aten_functions.py (aten._to_copy, aten.clone,
    # aten.detach). No custom Python methods needed.

    def pin_memory(self) -> Self:
        r"""Pin the underlying tensor memory for faster host-to-device transfer."""
        return type(self)._from_packed(
            self._values.pin_memory(),
            self._offsets,
            self._shape_tensor,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
            mask_value=self.mask_value,
            pin_memory=True,
            outer_size=self._logical_shape,
        )

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
        if len(self) == 0:
            return self.__class__([], **self._meta)
        return self.__class__([t.view(s) for t, s in zip(self._storage, self._view_shapes(shape))], **self._meta)

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
        if len(shape) > batch_dim and shape[batch_dim] in (-1, batch_size):
            if len(shape) != self.dim():
                # Unambiguous: different dim count → batch is included
                include_batch = True
            else:
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

        view_shapes = []
        for t in self._storage:
            adjusted = list(target)
            available = list(range(len(max_sizes)))
            for i in range(min(len(adjusted), len(max_sizes))):
                if adjusted[i] == -1:
                    continue
                # Direct match: same position in max_sizes
                if adjusted[i] == max_sizes[i]:
                    adjusted[i] = t.size(i)
                    if i in available:
                        available.remove(i)
                    continue
                # Indirect match: search remaining positions for unique candidate
                candidates = [j for j in available if max_sizes[j] == adjusted[i]]
                if len(candidates) == 1:
                    j = candidates[0]
                    adjusted[i] = t.size(j)
                    available.remove(j)
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
