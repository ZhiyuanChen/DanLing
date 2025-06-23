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

import operator
from collections.abc import Mapping, Sequence
from typing import Any, Iterable, SupportsFloat, Tuple

import torch
from torch import Tensor

from .dispatch import DISPATCH_TABLE, per_element_fallback
from .utils import mask_tensor, pad_tensor, tensor_mask

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

try:
    from torch import nested
except ImportError:
    nested = None


_METHOD_TO_FUNC: dict = {}


def _init_method_to_func():
    """Lazily populate the Tensor-method → torch-function mapping.

    Some torch methods may not exist in all versions, so we guard each entry.
    """
    if _METHOD_TO_FUNC:
        return
    _candidates = [
        (torch.Tensor.split, torch.split),
        (torch.Tensor.chunk, torch.chunk),
    ]
    for method, func in _candidates:
        _METHOD_TO_FUNC[method] = func


@torch._dynamo.disable
def _torch_function_impl(cls, func, types, args, kwargs):
    """Standalone __torch_function__ handler, disabled for dynamo tracing.

    By wrapping with torch._dynamo.disable, dynamo executes this eagerly
    during compilation instead of tracing through it. This avoids graph breaks
    from _make_wrapper_subclass inside __torch_dispatch__ handlers, at the
    cost of an opaque call boundary (the op won't be captured in the graph).
    """
    if kwargs is None:
        kwargs = {}

    # Handle size() specially to avoid infinite recursion
    if func is torch.Tensor.size:
        self = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        return self.size(dim)

    # Map Tensor methods to their torch.* function equivalents
    _init_method_to_func()
    func = _METHOD_TO_FUNC.get(func, func)

    from .functions import NestedTensorFuncRegistry

    if func in NestedTensorFuncRegistry:
        return NestedTensorFuncRegistry[func](*args, **kwargs)

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
        NestedTensor([[1, 2, 3],
                [4, 5, 0]])
        >>> nested_tensor[:, 1:]  # Slice operations return a new NestedTensor
        NestedTensor([[2, 3],
                [5, 0]])

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
        NestedTensor([[1, 2, 3],
                [4, 5, 0]])
    """

    _values: Tensor
    _offsets: Tensor
    _shape_tensor: Tensor
    _logical_shape: torch.Size
    _cache: dict

    batch_first: bool
    padding_value: float
    mask_value: bool
    _pin_memory: bool
    _orig_dtype: torch.dtype | None
    _orig_device: torch.device | None
    _orig_requires_grad: bool | None

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
        validated = cls._validate_tensors(
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
        r._cache = {}
        r.batch_first = batch_first
        r.padding_value = float(padding_value)
        r.mask_value = mask_value
        r._pin_memory = pin_memory
        r._orig_dtype = dtype
        r._orig_device = device
        r._orig_requires_grad = requires_grad
        return r

    def __init__(self, *args, **kwargs):
        pass  # All init in __new__

    # ------------------------------------------------------------------
    # Packed representation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_tensors(
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
        for t in result[1:]:
            common_dtype = torch.promote_types(common_dtype, t.dtype)
        if any(t.dtype != common_dtype for t in result):
            result = [t.to(dtype=common_dtype) for t in result]

        # Silent squeeze: if drop_last=False, the last element may have an extra batch dimension
        ndims = {t.ndim for t in result[:-1]}
        if len(ndims) == 1:
            (ndim,) = ndims
            if result[-1].ndim == ndim + 1 and result[-1].size(0) == 1:
                result[-1] = result[-1].squeeze(0)

        return tuple(result)

    @staticmethod
    def _pack(tensors: tuple[Tensor, ...], *, dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Pack a sequence of tensors into (values, offsets, shape_tensor)."""
        if not tensors:
            return (
                torch.empty(0, dtype=dtype or torch.get_default_dtype()),
                torch.zeros(1, dtype=torch.long),
                torch.empty(0, 0, dtype=torch.long),
            )

        max_ndim = max(t.ndim for t in tensors)
        device = tensors[0].device

        shape_tensor = torch.zeros(len(tensors), max_ndim, dtype=torch.long, device=device)
        for i, t in enumerate(tensors):
            for d in range(t.ndim):
                shape_tensor[i, d] = t.size(d)

        # Determine packing layout from data
        if _all_static_except_dim0(tensors):
            # Single-ragged: cat along dim 0, values is N-D
            values = torch.cat(tensors, dim=0) if tensors[0].ndim > 0 else torch.stack(tensors)
            offsets = torch.zeros(len(tensors) + 1, dtype=torch.long, device=device)
            for i, t in enumerate(tensors):
                offsets[i + 1] = offsets[i] + (t.size(0) if t.ndim > 0 else 1)
        else:
            # Multi-ragged: flatten each, cat, values is 1-D
            flat = [t.flatten() for t in tensors]
            values = torch.cat(flat, dim=0)
            offsets = torch.zeros(len(tensors) + 1, dtype=torch.long, device=device)
            for i, t in enumerate(flat):
                offsets[i + 1] = offsets[i] + t.numel()

        return values, offsets, shape_tensor

    @staticmethod
    def _compute_logical_shape(tensors: tuple[Tensor, ...], batch_first: bool) -> torch.Size:
        """Compute the logical shape [B, max_d0, max_d1, ...] from individual tensors."""
        if not tensors:
            return torch.Size((0,))
        if max(t.dim() for t in tensors) == 0:
            return torch.Size((len(tensors),))
        ndim = max(t.dim() for t in tensors)
        size = [max(t.shape[i] if i < len(t.shape) else 0 for t in tensors) for i in range(ndim)]
        size.insert(0 if batch_first else 1, len(tensors))
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
        outer_size: torch.Size | tuple | None = None,
    ) -> Self:
        """Construct a NestedTensor directly from packed representation (no validation)."""
        if outer_size is not None:
            logical_shape = torch.Size(outer_size)
        else:
            batch_size = len(offsets) - 1
            if batch_size == 0:
                logical_shape = torch.Size((0,))
            elif shape_tensor.numel() == 0:
                logical_shape = torch.Size((batch_size,))
            else:
                max_ndim = shape_tensor.size(1)
                size = [int(shape_tensor[:, d].max().item()) for d in range(max_ndim)]
                size.insert(0 if batch_first else 1, batch_size)
                logical_shape = torch.Size(size)

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
        r._cache = {}
        r.batch_first = batch_first
        r.padding_value = padding_value
        r.mask_value = mask_value
        r._pin_memory = False
        r._orig_dtype = None
        r._orig_device = None
        r._orig_requires_grad = None
        return r

    # ------------------------------------------------------------------
    # torch.compile support
    # ------------------------------------------------------------------

    def __tensor_flatten__(self):
        return ["_values", "_offsets", "_shape_tensor"], {
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
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

        if func in DISPATCH_TABLE:
            return DISPATCH_TABLE[func](func, args, kwargs)

        # Per-element fallback
        return per_element_fallback(func, args, kwargs)

    # ------------------------------------------------------------------
    # _storage: derived property for backward compatibility
    # ------------------------------------------------------------------

    @property
    def _storage(self) -> Tuple[Tensor, ...]:
        if "_storage" in self._cache:
            return self._cache["_storage"]
        result = self._unpack()
        self._cache["_storage"] = result
        return result

    @_storage.setter
    def _storage(self, tensors: Sequence):
        """Re-pack storage. Used by legacy code paths."""
        validated = self._validate_tensors(
            tuple(tensors) if not isinstance(tensors, tuple) else tensors,
            dtype=self._orig_dtype,
            device=self._orig_device,
            requires_grad=self._orig_requires_grad,
            pin_memory=self._pin_memory,
        )
        values, offsets, shape_tensor = self._pack(validated)
        self._values = values
        self._offsets = offsets
        self._shape_tensor = shape_tensor
        self._logical_shape = self._compute_logical_shape(validated, self.batch_first)
        self._cache = {}
        if validated:
            self._orig_dtype = validated[0].dtype
            self._orig_device = validated[0].device
            self._orig_requires_grad = validated[0].requires_grad

    def _unpack(self) -> Tuple[Tensor, ...]:
        """Reconstruct individual tensors from packed representation."""
        batch_size = len(self._offsets) - 1
        if batch_size == 0:
            return ()
        is_single_ragged = _all_static_except_dim0_from_shapes(self._shape_tensor)
        result = []
        for i in range(batch_size):
            start = int(self._offsets[i].item())
            end = int(self._offsets[i + 1].item())
            shape = self._shape_tensor[i].tolist()
            # Remove trailing zeros for tensors with fewer dims
            while shape and shape[-1] == 0:
                shape.pop()
            if not shape:
                # Scalar element
                result.append(self._values[start])
            elif is_single_ragged:
                result.append(self._values[start:end])
            else:
                result.append(self._values[start:end].reshape(shape))
        return tuple(result)

    def storage(self):
        return self._storage

    # ------------------------------------------------------------------
    # Properties: dtype, device, requires_grad
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        return self._values.dtype

    @dtype.setter
    def dtype(self, value: torch.dtype | None):
        self._orig_dtype = value
        if value is not None and self._values.numel() > 0:
            self._storage = tuple(t.to(dtype=value) for t in self._storage)

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return self._values.device

    @device.setter
    def device(self, value: torch.device | None):
        self._orig_device = value
        if value is not None and self._values.numel() > 0:
            self._storage = tuple(t.to(device=value) for t in self._storage)

    @property
    def requires_grad(self) -> bool:  # type: ignore[override]
        return self._values.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self._orig_requires_grad = value
        if self._values.numel() > 0:
            self._values.requires_grad_(value)

    def _default_dtype(self):
        return self._values.dtype

    # ------------------------------------------------------------------
    # Cached padded views
    # ------------------------------------------------------------------

    @property
    def tensor_mask(self) -> Tuple[Tensor, Tensor]:
        r"""
        Return a tuple of padded tensor and mask tensor.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tensor_mask
            (tensor([[1, 2, 3],
                    [4, 5, 0]]), tensor([[ True,  True,  True],
                    [ True,  True, False]]))
        """
        if "tensor_mask" in self._cache:
            return self._cache["tensor_mask"]
        storage = self._storage
        if not storage:
            size = self.size()
            t = torch.empty(size, dtype=self._default_dtype(), device=self.device)
            m = torch.empty(size, dtype=torch.bool, device=self.device)
            result = (t, m)
        else:
            result = self._tensor_mask(storage, self.batch_first, self.padding_value, self.mask_value)
        self._cache["tensor_mask"] = result
        self._cache["tensor"] = result[0]
        self._cache["mask"] = result[1]
        return result

    def _tensor_mask(
        self, storage: tuple, batch_first: bool, padding_value: float, mask_value: bool
    ) -> Tuple[Tensor, Tensor]:
        if not storage:
            size = self.size()
            t = torch.empty(size, dtype=self._default_dtype(), device=self.device)
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
        if "tensor" in self._cache:
            return self._cache["tensor"]
        storage = self._storage
        if not storage:
            result = torch.empty(self.size(), dtype=self._default_dtype(), device=self.device)
        else:
            result = self._tensor(storage, self.batch_first, self.padding_value)
        self._cache["tensor"] = result
        return result

    def _tensor(self, storage: tuple, batch_first: bool, padding_value: float) -> Tensor:
        if not storage:
            return torch.empty(self.size(), dtype=self._default_dtype(), device=self.device)
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
        if "mask" in self._cache:
            return self._cache["mask"]
        storage = self._storage
        if not storage:
            result = torch.empty(self.size(), dtype=torch.bool, device=self.device)
        else:
            result = self._mask(storage, self.batch_first, self.mask_value)
        self._cache["mask"] = result
        return result

    def _mask(self, storage: tuple, batch_first: bool, mask_value: bool) -> Tensor:
        if not storage:
            return torch.empty(self.size(), dtype=torch.bool, device=self.device)
        if storage[0].dim() == 0:
            return torch.full((len(storage),), not mask_value, dtype=torch.bool, device=self.device)
        return mask_tensor(storage, size=self.size(), batch_first=batch_first, mask_value=mask_value)

    @property
    def concat(self) -> Tensor:
        r"""
        Concat `tensor` in padding dim.

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

    def concatenate(self) -> Tuple[Tensor, Tuple[torch.Size, ...]]:
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
        storage = self._storage
        if not storage:
            return torch.empty(0, dtype=self._default_dtype(), device=self.device), ()

        original_shapes = tuple(t.shape for t in storage)
        elem = storage[0]
        if elem.ndim == 0:
            concat_tensor = torch.stack(storage, dim=0)
            return concat_tensor, original_shapes

        shape = list(self.size())  # type: ignore[arg-type]
        shape = shape[1:] if self.batch_first else [shape[0]] + shape[2:]
        if elem.shape == torch.Size(shape) and all(t.shape == elem.shape for t in storage):
            concat_tensor = torch.cat(storage, dim=0)
            return concat_tensor, original_shapes

        static_dims = set(range(len(shape)))
        for i, s in enumerate(shape):
            if not all(t.size(i) == s for t in storage):
                shape[i] = -1
                static_dims.remove(i)
        target_shape = [-1] + [s for s in shape if s != -1]
        reshaped = [t.reshape(target_shape) for t in storage]
        concat_tensor = torch.cat(reshaped, dim=0)
        return concat_tensor, original_shapes

    @classmethod
    def from_concatenated(cls, concat_tensor: Tensor, shapes: Tuple[torch.Size, ...], **kwargs) -> Self:
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
                except (RuntimeError, ValueError):
                    pass
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
        if nested is None:
            raise ImportError("torch.nested is not available, please install torch with nested support.")
        return nested.nested_tensor(list(self._storage))

    def unbind(self, dim: int = 0) -> Tuple[Tensor, ...]:
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
    def from_tensor_mask(cls, tensor: Tensor, mask: Tensor, **kwargs):
        r"""
        Build a `NestedTensor` object from a padded `Tensor` and corresponding mask `Tensor`.

        Args:
            tensor: Padded Tensor.
            mask: Tensor Mask.

        Examples:
            >>> padded_tensor = torch.tensor([[1, 2, 3, 0, 0],
            ...                                [4, 5, 0, 0, 0],
            ...                                [6, 7, 8, 9, 0]])
            >>> mask_tensor = torch.tensor([[1, 1, 1, 0, 0],
            ...                             [1, 1, 0, 0, 0],
            ...                             [1, 1, 1, 1, 0]])
            >>> nested_tensor = NestedTensor.from_tensor_mask(padded_tensor, mask_tensor)
            >>> nested_tensor
            NestedTensor([[1, 2, 3, 0],
                    [4, 5, 0, 0],
                    [6, 7, 8, 9]])
        """
        batched = kwargs.pop("batched", False)
        mask = mask.to(dtype=torch.bool)
        mask_value = kwargs.get("mask_value", False)
        effective_mask = ~mask if mask_value else mask

        if mask.ndim == 1:
            if batched:
                indices = effective_mask.nonzero(as_tuple=False).flatten()
                return cls([tensor[int(i)] for i in indices], **kwargs)
            valid = int(effective_mask.sum().item())
            return cls(tensor[:valid], **kwargs)
        if mask.ndim == 2:
            batch_first = kwargs.get("batch_first", True)
            tensor_iter = tensor if batch_first else tensor.transpose(0, 1)
            mask_iter = effective_mask if batch_first else effective_mask.transpose(0, 1)
            if tensor_iter.size(0) != mask_iter.size(0):
                raise ValueError(
                    "Tensor/mask batch dimension mismatch: " f"{tensor_iter.size(0)} vs {mask_iter.size(0)}"
                )
            trimmed = [t[: int(m.sum().item())] for t, m in zip(tensor_iter, mask_iter)]
            return cls(trimmed, **kwargs)
        batch_first = kwargs.get("batch_first", True)
        tensor_iter = tensor if batch_first else tensor.transpose(0, 1)
        mask_iter = effective_mask if batch_first else effective_mask.transpose(0, 1)
        if tensor_iter.size(0) != mask_iter.size(0):
            raise ValueError("Tensor/mask batch dimension mismatch: " f"{tensor_iter.size(0)} vs {mask_iter.size(0)}")
        trimmed_tensors = []
        for t, em in zip(tensor_iter, mask_iter):
            nonzero = em.nonzero(as_tuple=False)
            if nonzero.numel() == 0:
                slices = tuple(slice(0, 0) for _ in range(em.dim()))
            else:
                max_indices = nonzero.max(dim=0).values + 1
                slices = tuple(slice(0, int(i.item())) for i in max_indices)
            trimmed_tensors.append(t[slices])
        return cls(trimmed_tensors, **kwargs)

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
        """  # noqa: E501

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
            new_storage.append(tensor[slices])
        return self.__class__(new_storage, **self._state)

    # ------------------------------------------------------------------
    # Dispatch fallback (used by __torch_function__ for unregistered ops)
    # ------------------------------------------------------------------

    @classmethod
    def _dispatch_fallback(cls, func, args, kwargs) -> Any:
        source = next((a for a in args if isinstance(a, NestedTensor)), None)
        if source is None:
            source = next((v for v in kwargs.values() if isinstance(v, NestedTensor)), None)
        args = [a.tensor if hasattr(a, "tensor") and isinstance(a, NestedTensor) else a for a in args]
        for k, v in kwargs.items():
            if isinstance(v, NestedTensor):
                kwargs[k] = v.tensor
        output = func(*args, **kwargs)
        if isinstance(output, NestedTensor):
            return output
        if (
            isinstance(output, Tensor)
            and source is not None
            and output.dim() >= source.tensor.dim()
            and output.shape[: source.tensor.dim()] == source.tensor.shape
        ):
            batched = source.tensor.dim() == 1 and source.mask.dim() == 1
            return cls.from_tensor_mask(output, source.mask, batched=batched, **source._state)
        return output

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, index: int | slice | list | tuple | Tensor | NestedTensor) -> Tensor | NestedTensor:
        if isinstance(index, int):
            return self._storage[index]
        if isinstance(index, (slice, list)):
            storage = tuple(self._storage[index] if isinstance(index, slice) else [self._storage[i] for i in index])
            return self.__class__(storage, **self._state)
        if isinstance(index, tuple):
            if len(index) == 0:
                return self
            batch_index, *rest = index

            if batch_index is Ellipsis:
                batch_index = slice(None)

            if isinstance(batch_index, (Tensor, NestedTensor)):
                return self.tensor[index]

            if isinstance(batch_index, list) and batch_index and all(isinstance(i, bool) for i in batch_index):
                if len(batch_index) != len(self._storage):
                    raise IndexError(
                        f"Boolean index has length {len(batch_index)} but batch size is {len(self._storage)}"
                    )
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
                return self.__class__(selected, **self._state)
            raise ValueError(f"Unsupported batch index type {type(batch_index)}")
        if isinstance(index, Tensor):
            index = self.nested_like(index, strict=False)
        if isinstance(index, NestedTensor):
            if len(self) != len(index):
                raise ValueError(f"NestedTensor batch length mismatch: {len(self)} vs {len(index)}")
            return self.__class__([t[i] for t, i in zip(self._storage, index._storage)], **self._state)
        raise ValueError(f"Unsupported index type {type(index)}")

    def __setitem__(self, index: int | slice | list | tuple, value: Tensor | NestedTensor) -> None:
        """
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
            if self._orig_requires_grad is not None:
                value.requires_grad_(self._orig_requires_grad)
            # Repack with the new element
            storage_list = list(self._storage)
            storage_list[index] = value
            self._storage = tuple(storage_list)
        elif isinstance(index, (slice, list)):
            if isinstance(value, Tensor) and not isinstance(value, NestedTensor):
                if value.dim() > 1 and value.size(0) > 1:
                    value = self.__class__(value.unbind(0), **self._state)
                else:
                    value = self.__class__([value], **self._state)

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

            if isinstance(first_idx, int):
                storage_list = list(self._storage)
                tensor = storage_list[first_idx]
                tensor[rest_idx] = value
                storage_list[first_idx] = tensor
                self._storage = tuple(storage_list)
            elif isinstance(first_idx, (slice, list)):
                if isinstance(first_idx, slice):
                    start, stop, step = first_idx.indices(len(self))
                    indices = range(start, stop, step)
                else:
                    indices = first_idx  # type: ignore[assignment]

                storage_list = list(self._storage)
                for idx in indices:
                    tensor = storage_list[idx]
                    tensor[rest_idx] = value
                    storage_list[idx] = tensor
                self._storage = tuple(storage_list)
            else:
                raise ValueError(f"Unsupported first index type {type(first_idx)}")
        else:
            raise ValueError(f"Unsupported index type {type(index)}")

    # ------------------------------------------------------------------
    # Attribute access fallback
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        storage = self._storage
        if not storage:
            raise ValueError(f"Unable to get {name} from an empty {self.__class__.__name__}")
        ret = [getattr(i, name) for i in storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            try:
                return torch.stack(ret)
            except (RuntimeError, ValueError, TypeError):
                return self.__class__(ret, **self._state)
        if callable(elem):

            def wrapper(*args, **kwargs):
                results = [f(*args, **kwargs) for f in ret]
                if not results or results[0] is None:
                    return None
                if isinstance(results[0], Tensor):
                    try:
                        return torch.stack(results)
                    except (RuntimeError, ValueError, TypeError):
                        return self.__class__(results, **self._state)
                return results

            return wrapper
        return ret

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @property
    def _state(self) -> Mapping:
        return self._build_state(return_dtype=False, return_device=True, return_requires_grad=False)

    def _build_state(
        self, return_dtype: bool = True, return_device: bool = True, return_requires_grad: bool = False
    ) -> Mapping:
        state: dict[str, Any] = {
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "pin_memory": self._pin_memory,
        }
        if return_dtype:
            state["dtype"] = self.dtype
        if return_device:
            state["device"] = self.device
        if return_requires_grad:
            state["requires_grad"] = self.requires_grad
        return state

    def __getstate__(self) -> dict:
        return {
            "_values": self._values,
            "_offsets": self._offsets,
            "_shape_tensor": self._shape_tensor,
            "batch_first": self.batch_first,
            "padding_value": self.padding_value,
            "mask_value": self.mask_value,
            "_pin_memory": self._pin_memory,
            "_orig_dtype": self._orig_dtype,
            "_orig_device": self._orig_device,
            "_orig_requires_grad": self._orig_requires_grad,
        }

    def __setstate__(self, state: Mapping) -> None:
        self._cache = {}
        self.__dict__.update(state)
        if "_logical_shape" not in self.__dict__:
            self._logical_shape = self._compute_logical_shape(self._storage, self.batch_first)

    def __reduce__(self):
        return (self.__class__._from_state, (self.__getstate__(),))

    @classmethod
    def _from_state(cls, state: dict) -> Self:
        return cls._from_packed(
            state["_values"],
            state["_offsets"],
            state["_shape_tensor"],
            batch_first=state.get("batch_first", True),
            padding_value=state.get("padding_value", 0.0),
            mask_value=state.get("mask_value", False),
        )

    def _inherit_state_from(self, other: Self) -> Self:
        self.batch_first = other.batch_first
        self.padding_value = other.padding_value
        self.mask_value = other.mask_value
        self._pin_memory = other._pin_memory
        return self

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __repr__(self):
        if len(self) == 0:
            return self.__class__.__name__ + "()"
        if len(self) > 10:
            contents = ",\n  ".join(repr(t) for t in self._storage[:5])
            return f"{self.__class__.__name__}([\n  {contents},\n  ... ({len(self)} tensors)\n])"
        contents = ",\n  ".join(repr(t) for t in self._storage)
        return f"{self.__class__.__name__}([\n  {contents}\n])"

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self):
        return iter(self._storage)

    # ------------------------------------------------------------------
    # In-place binary ops
    # ------------------------------------------------------------------

    def _inplace_binary_op(self, other: Tensor | NestedTensor | SupportsFloat, op, *, cast_other: bool = False):
        if isinstance(other, Tensor) and not isinstance(other, NestedTensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if cast_other and hasattr(other, "to"):
            other = other.to(self.dtype)

        # Fast path: operate directly on packed _values
        if len(self) > 0:
            if isinstance(other, NestedTensor):
                if len(self) != len(other):
                    raise ValueError(f"NestedTensor batch length mismatch: {len(self)} vs {len(other)}")
                if torch.equal(self._offsets, other._offsets):
                    op(self._values, other._values)
                    self._cache = {}
                    return self
            elif not isinstance(other, Tensor) or other.dim() == 0:
                from .ops import _as_tensor_like

                op(self._values, _as_tensor_like(other, self._values))
                self._cache = {}
                return self

        # Fallback: per-element
        storage = list(self._storage)
        if isinstance(other, NestedTensor):
            if len(self) != len(other):
                raise ValueError(f"NestedTensor batch length mismatch: {len(self)} vs {len(other)}")
            for idx, (x, y) in enumerate(zip(storage, other._storage)):
                result = op(x, y)
                if result is not None and result is not x:
                    storage[idx] = result
        else:
            for idx, x in enumerate(storage):
                result = op(x, other)
                if result is not None and result is not x:
                    storage[idx] = result

        self._storage = tuple(storage)
        return self

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def __gt__(self, other):
        return torch.gt(self, other)

    def __ge__(self, other):
        return torch.ge(self, other)

    def __eq__(self, other):  # type: ignore[override]
        try:
            return torch.eq(self, other)
        except TypeError:
            return NotImplemented

    def __ne__(self, other):  # type: ignore[override]
        try:
            return torch.ne(self, other)
        except TypeError:
            return NotImplemented

    def __le__(self, other):
        return torch.le(self, other)

    def __lt__(self, other):
        return torch.lt(self, other)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __abs__(self):
        return torch.abs(self)

    def __add__(self, other):
        return torch.add(self, other)

    def __radd__(self, other):
        return torch.add(other, self)

    def __iadd__(self, other):
        return self._inplace_binary_op(other, operator.iadd, cast_other=True)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __isub__(self, other):
        return self._inplace_binary_op(other, operator.isub, cast_other=True)

    def __pos__(self):
        return torch.positive(self)

    def __neg__(self):
        return torch.neg(self)

    def __invert__(self):
        return torch.bitwise_not(self)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __rmul__(self, other):
        return torch.mul(other, self)

    def __imul__(self, other):
        return self._inplace_binary_op(other, operator.imul, cast_other=True)

    def __pow__(self, other):
        return torch.pow(self, other)

    def __rpow__(self, other):
        return torch.pow(other, self)

    def __ipow__(self, other):
        return self._inplace_binary_op(other, operator.ipow, cast_other=True)

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def __imatmul__(self, other):
        return self._inplace_binary_op(other, operator.imatmul, cast_other=True)

    def __truediv__(self, other):
        return torch.true_divide(self, other)

    def __rtruediv__(self, other):
        return torch.true_divide(other, self)

    def __itruediv__(self, other):
        return self._inplace_binary_op(other, operator.itruediv, cast_other=True)

    def __floordiv__(self, other):
        return torch.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return torch.floor_divide(other, self)

    def __ifloordiv__(self, other):
        return self._inplace_binary_op(other, operator.ifloordiv, cast_other=True)

    def __mod__(self, other):
        return torch.remainder(self, other)

    def __rmod__(self, other):
        return torch.remainder(other, self)

    def __imod__(self, other):
        return self._inplace_binary_op(other, operator.imod, cast_other=True)

    def __and__(self, other):
        return torch.bitwise_and(self, other)

    def __rand__(self, other):
        return torch.bitwise_and(other, self)

    def __iand__(self, other):
        return self._inplace_binary_op(other, operator.iand, cast_other=True)

    def __or__(self, other):
        return torch.bitwise_or(self, other)

    def __ror__(self, other):
        return torch.bitwise_or(other, self)

    def __ior__(self, other):
        return self._inplace_binary_op(other, operator.ior, cast_other=True)

    def __xor__(self, other):
        return torch.bitwise_xor(self, other)

    def __rxor__(self, other):
        return torch.bitwise_xor(other, self)

    def __ixor__(self, other):
        return self._inplace_binary_op(other, operator.ixor, cast_other=True)

    def __lshift__(self, other):
        return torch.bitwise_left_shift(self, other)

    def __rlshift__(self, other):
        return torch.bitwise_left_shift(other, self)

    def __ilshift__(self, other):
        return self._inplace_binary_op(other, operator.ilshift, cast_other=True)

    def __rshift__(self, other):
        return torch.bitwise_right_shift(self, other)

    def __rrshift__(self, other):
        return torch.bitwise_right_shift(other, self)

    def __irshift__(self, other):
        return self._inplace_binary_op(other, operator.irshift, cast_other=True)

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
            NestedTensor([[ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]])
            >>> nested_tensor.all(dim=1, keepdim=True)
            NestedTensor([[[ True,  True,  True,  True, False]],
            <BLANKLINE>
                    [[ True,  True,  True,  True,  True]]])
            >>> nested_tensor.batch_first = False
            >>> nested_tensor.all(dim=1)
            tensor([True, True])
            >>> nested_tensor.all(dim=0)
            NestedTensor([[ True,  True],
                    [ True,  True],
                    [ True,  True],
                    [ True,  True],
                    [False,  True]])
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
        return torch.mean(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def min(self, dim: int | None = None, keepdim: bool = False) -> Tensor | NestedTensor:
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
        """Transpose: applies .T to each element in storage."""
        return self.__class__([t.T for t in self._storage], **self._state)

    @property
    def mT(self) -> Self:  # type: ignore[override]
        """Batch matrix transpose: applies .mT to each element in storage."""
        return self.__class__([t.mT for t in self._storage], **self._state)

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
        return torch.prod(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def to(self, *args, **kwargs):
        if args and isinstance(args[0], NestedTensor):
            other = args[0]
            if "device" not in kwargs:
                kwargs["device"] = other.device
            if "dtype" not in kwargs:
                kwargs["dtype"] = other.dtype
            args = ()
        return self.__class__(
            tuple(t.to(*args, **kwargs) for t in self._storage),
            **self._build_state(return_dtype=False, return_device=False),
        )

    # clone(), detach(), contiguous() are handled by aten dispatch in dispatch.py
    # (aten.clone.default, aten.detach.default). No custom Python methods needed.

    def pin_memory(self) -> Self:
        return self.__class__(tuple(t.pin_memory() for t in self._storage), **self._state)

    def half(self) -> Self:
        return self.to(dtype=torch.float16)

    def float(self) -> Self:
        return self.to(dtype=torch.float32)

    def double(self) -> Self:
        return self.to(dtype=torch.float64)

    def bfloat16(self) -> Self:
        return self.to(dtype=torch.bfloat16)

    def int(self) -> Self:
        return self.to(dtype=torch.int32)

    def long(self) -> Self:
        return self.to(dtype=torch.int64)

    def short(self) -> Self:
        return self.to(dtype=torch.int16)

    def bool(self) -> Self:
        return self.to(dtype=torch.bool)

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
        if not self._storage:
            return self.__class__([], **self._state)
        return self.__class__([t.view(s) for t, s in zip(self._storage, self._view_shapes(shape))], **self._state)

    def _view_shapes(self, shape) -> list[tuple[int, ...]]:  # type: ignore[valid-type]
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])

        batch_dim = 0 if self.batch_first else 1
        include_batch = False
        has_batch = len(shape) > batch_dim and shape[batch_dim] in (-1, len(self._storage))
        if has_batch:
            if len(shape) != self.dim():
                include_batch = True
            else:
                max_sizes = list(self.size())  # type: ignore[arg-type]
                if max_sizes:
                    max_sizes.pop(batch_dim)
                tensor_dims = [i for i in range(len(shape)) if i != batch_dim]
                for i, d in enumerate(tensor_dims):
                    if i < len(max_sizes) and (shape[d] == -1 or shape[d] == max_sizes[i]):
                        include_batch = True
                        break

        target_shape = list(shape)
        if include_batch:
            if target_shape[batch_dim] == -1:
                target_shape[batch_dim] = len(self._storage)
            if target_shape[batch_dim] != len(self._storage):
                raise ValueError(
                    f"Batch dimension mismatch: expected {len(self._storage)} but got {target_shape[batch_dim]}"
                )
            target_shape.pop(batch_dim)

        max_sizes = list(self.size())  # type: ignore[arg-type]
        if max_sizes:
            max_sizes.pop(batch_dim)

        view_shapes = []
        for t in self._storage:
            adjusted = list(target_shape)
            available = list(range(len(max_sizes)))
            for i in range(min(len(adjusted), len(max_sizes))):
                if adjusted[i] == -1:
                    continue
                if adjusted[i] == max_sizes[i]:
                    adjusted[i] = t.size(i)
                    if i in available:
                        available.remove(i)
                    continue
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
            NestedTensor([[6, 5, 3],
                    [4, 5, 0]])
            >>> nested_tensor.where(nested_tensor > 2, NestedTensor([[6, 5, 4], [3, 2]]))
            NestedTensor([[6, 5, 3],
                    [4, 5, 0]])
            >>> nested_tensor.where(torch.tensor(True), NestedTensor([[6, 5, 4], [3, 2]]))
            NestedTensor([[1, 2, 3],
                    [4, 5, 0]])
        """
        return torch.where(condition, self, other)

    def to_device(self, device: torch.device | str) -> Self:  # type: ignore[name-defined]
        return self.to(device=device)


def _all_static_except_dim0(tensors: tuple[Tensor, ...]) -> bool:
    """Check if all tensors have the same shape except for dimension 0."""
    if len(tensors) <= 1:
        return True
    if tensors[0].ndim == 0:
        return True
    ref = tensors[0].shape[1:]
    return all(t.shape[1:] == ref for t in tensors[1:])


def _all_static_except_dim0_from_shapes(shape_tensor: Tensor) -> bool:
    """Check if all shapes are the same except for dimension 0, from shape_tensor."""
    if shape_tensor.size(0) <= 1:
        return True
    if shape_tensor.size(1) <= 1:
        return True
    trailing = shape_tensor[:, 1:]
    return bool((trailing == trailing[0:1]).all().item())
