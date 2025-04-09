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

from collections.abc import Mapping, Sequence
from typing import Any, Iterable, SupportsFloat, Tuple

import torch
from torch import Tensor

from ..utils import method_cache
from .functions import NestedTensorFuncRegistry, NestedTensorFuncWrapper
from .utils import mask_tensor, pad_tensor, tensor_mask

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

try:
    from torch import nested
except ImportError:
    nested = None


class NestedTensor:
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
        _storage: The sequence of original tensors (internal use)
        tensor: Padded tensor representing all sequences with padding
        mask: Boolean mask where True indicates real elements, False indicates padding
        concat: Concatenated tensor of all sequences without padding
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

    __storage: Tuple[Tensor, ...]

    dtype: torch.dtype | None = None
    device: torch.device | None = None
    requires_grad: bool | None = None
    _pin_memory: bool = False

    batch_first: bool = True
    padding_value: SupportsFloat = 0.0
    mask_value: bool = False

    def __init__(
        self,
        *tensors: Iterable[Tensor],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool | None = None,
        pin_memory: bool = False,
        batch_first: bool = True,
        padding_value: SupportsFloat = 0.0,
        mask_value: bool = False,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self._pin_memory = pin_memory
        if len(tensors) == 1 and isinstance(tensors, Sequence):
            tensors = tensors[0]  # type: ignore
        self._storage = tensors
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.mask_value = mask_value

    @property
    def _storage(self):
        return self.__storage

    @_storage.setter
    def _storage(self, tensors: Sequence):
        if not isinstance(tensors, Iterable):
            raise ValueError(f"tensors must be an Iterable, bug got {type(tensors)}.")
        if isinstance(tensors, Tensor) and hasattr(tensors, "unbind"):
            tensors = tensors.unbind()
        tensors_ = []
        for t in tensors:
            if not isinstance(t, Tensor):
                t = torch.tensor(
                    t,
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self._pin_memory,
                )
            else:
                t = t.to(self.device, dtype=self.dtype)
            if self.requires_grad is not None:
                t.requires_grad_(self.requires_grad)
            tensors_.append(t)
        if len(tensors_) == 0:
            self.__storage = ()
            return
        tensors = tuple(tensors_)
        self.dtype = tensors[0].dtype
        self.device = tensors[0].device
        self.requires_grad = tensors[0].requires_grad
        # if drop_last=False, the last element is likely not a NestedTensor and has an extra batch dimension
        ndims = {t.ndim for t in tensors[:-1]}
        if len(ndims) == 1:
            (ndim,) = ndims
            if tensors[-1].ndim == ndim + 1 and tensors[-1].size(0) == 1:
                tensors = tensors[:-1] + (tensors[-1].squeeze(0),)
        self.__storage = tensors

    def storage(self):
        return self._storage

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

        return self._tensor_mask(
            self._storage, self.batch_first, self.padding_value, self.mask_value, self.requires_grad
        )

    @method_cache(maxsize=1)
    def _tensor_mask(
        self, storage: tuple, batch_first: bool, padding_value: SupportsFloat, mask_value: bool, requires_grad: bool
    ) -> Tensor:
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0), torch.full(
                (len(storage),), not mask_value, dtype=torch.bool, device=self.device
            )
        return tensor_mask(
            storage,
            size=self.size(),
            batch_first=batch_first,
            padding_value=float(padding_value),
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

        return self._tensor(self._storage, self.batch_first, self.padding_value, self.requires_grad)

    @method_cache(maxsize=1)
    def _tensor(self, storage: tuple, batch_first: bool, padding_value: SupportsFloat, requires_grad: bool) -> Tensor:
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0)
        return pad_tensor(storage, size=self.size(), batch_first=batch_first, padding_value=float(padding_value))

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

        return self._mask(self._storage, self.batch_first, self.mask_value, self.requires_grad)

    @method_cache(maxsize=1)
    def _mask(self, storage: tuple, batch_first: bool, mask_value: bool, requires_grad: bool) -> Tensor:
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
        shape = list(self.size())  # type: ignore[arg-type]
        shape = shape[1:] if self.batch_first else shape[0] + shape[2:]
        elem = self._storage[0]
        if elem.shape == shape:
            return torch.cat(self._storage, dim=1 if self.batch_first else 0)
        static_dims = set(range(len(shape)))
        for i, s in enumerate(shape):
            if not all(t.size(i) == s for t in self._storage):
                shape[i] = -1
                static_dims.remove(i)
        target_shape = [-1] + [s for s in shape if s != -1]
        storage = [i.reshape(target_shape) for i in self._storage]
        return torch.cat(storage, dim=0 if self.batch_first else 1)

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
        if dim != 0:
            raise ValueError(f"NestedTensor can only be unbound along dimension 0, got dimension {dim} instead.")
        return self._storage

    @property
    def occupancy(self) -> float:
        r"""
        Occupancy of the NestedTensor.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6])])
            >>> nested_tensor.occupancy
            0.75
        """

        return self.numel() / self.shape.numel()  # type: ignore[union-attr]

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

        if mask.ndim == 1:
            return cls(tensor, **kwargs)
        if mask.ndim == 2:
            return cls((t[slice(0, m.sum())] for t, m in zip(tensor, mask)), **kwargs)
        return cls(
            (
                t[[slice(0, (m.sum(dim=dim) > 0).sum().item()) for dim in reversed(range(m.dim()))]]
                for t, m in zip(tensor, mask)
            ),
            **kwargs,
        )

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
        if self.size(0) != tensor.size(0):
            raise ValueError(
                f"The batch size of NestedTensor and input tensor does not match, {self.size(0)} != {tensor.size(0)}"
            )
        return self.__class__([o[tuple(slice(0, dim) for dim in t.shape)] for t, o in zip(self._storage, tensor)])

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None) -> Self:
        if kwargs is None:
            kwargs = {}
        if func in NestedTensorFuncRegistry:
            return NestedTensorFuncRegistry[func](*args, **kwargs)
        args = [a.tensor if hasattr(a, "tensor") else a for a in args]
        for k, v in kwargs.items():
            if hasattr(v, "tensor"):
                kwargs[k] = v.tensor
        output = func(*args, **kwargs)
        if isinstance(output, (Tensor, NestedTensor)):
            return output
        return cls(output)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None) -> Self:
        args = [a.tensor if hasattr(a, "tensor") else a for a in args]
        for k, v in kwargs.items():
            if hasattr(v, "tensor"):
                kwargs[k] = v.tensor
        output = func(*args, **kwargs)
        if isinstance(output, (Tensor, NestedTensor)):
            return output
        return cls(output)

    def __getitem__(self, index: int | slice | list | tuple | Tensor | NestedTensor) -> Tensor | NestedTensor:
        if isinstance(index, int):
            return self._storage[index]
        if isinstance(index, (slice, list)):
            storage = tuple(self._storage[index] if isinstance(index, slice) else [self._storage[i] for i in index])
            return NestedTensor(storage, **self._state)
        if isinstance(index, tuple):
            return NestedTensor([t[index[0]][index[1:]] for t in self._storage])
        if isinstance(index, Tensor):
            index = self.nested_like(index, strict=False)
        if isinstance(index, NestedTensor):
            return NestedTensor([t[i] for t, i in zip(self._storage, index._storage)])
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
                value = torch.tensor(value, device=self.device)
            # Create a new list of tensors to modify
            storage_list = list(self._storage)
            storage_list[index] = value
            self._storage = tuple(storage_list)
        elif isinstance(index, (slice, list)):
            if isinstance(value, Tensor):
                # Convert tensor to NestedTensor if it's a regular tensor
                if value.dim() > 1 and value.size(0) > 1:
                    value = NestedTensor(value.unbind(0))
                else:
                    value = NestedTensor([value])

            if isinstance(index, slice):
                start, stop, step = index.indices(len(self._storage))
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
                # Handle single tensor modification
                storage_list = list(self._storage)
                tensor = storage_list[first_idx]
                tensor[rest_idx] = value
                storage_list[first_idx] = tensor
                self._storage = tuple(storage_list)
            elif isinstance(first_idx, (slice, list)):
                # Handle multiple tensor modification
                if isinstance(first_idx, slice):
                    start, stop, step = first_idx.indices(len(self._storage))
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

    def __getattr__(self, name: str) -> Any:
        if not self._storage:
            raise ValueError(f"Unable to get {name} from an empty {self.__class__.__name__}")
        ret = [getattr(i, name) for i in self._storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret, **self._state)
        if callable(elem):
            return NestedTensorFuncWrapper(ret, state=self._state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret

    def __iter__(self):
        return iter(self._storage)

    @property
    def _state(self) -> Mapping:
        return self.__state__(return_dtype=False, return_device=True, return_requires_grad=False)

    def __state__(
        self, return_dtype: bool = True, return_device: bool = True, return_requires_grad: bool = False
    ) -> Mapping:
        state = {k: v for k, v in self.__dict__.items() if not (k.startswith("_") or k.endswith("_"))}
        if not return_dtype:
            state.pop("dtype", None)
        if not return_device:
            state.pop("device", None)
        if not return_requires_grad:
            state.pop("requires_grad", None)
        return state

    def __setstate__(self, state: Mapping) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self):
        if not self._storage:
            return self.__class__.__name__ + "()"
        return self.__class__.__name__ + repr(self.tensor)[len(self.tensor.__class__.__name__) :]  # noqa: E203

    def __bool__(self) -> int:
        return all(bool(x) for x in self._storage)

    def __gt__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i > j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x > other for x in self._storage], **self._state)
        raise TypeError(f"> not supported between instances of '{type(self)}' and '{type(other)}'")

    def __ge__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i >= j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x >= other for x in self._storage], **self._state)
        raise TypeError(f">= not supported between instances of '{type(self)}' and '{type(other)}'")

    def __eq__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i == j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x == other for x in self._storage], **self._state)
        return False

    def __ne__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i != j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x != other for x in self._storage], **self._state)
        return True

    def __le__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i <= j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x <= other for x in self._storage], **self._state)
        raise TypeError(f"<= not supported between instances of '{type(self)}' and '{type(other)}'")

    def __lt__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i < j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x < other for x in self._storage], **self._state)
        raise TypeError(f"< not supported between instances of '{type(self)}' and '{type(other)}'")

    def __abs__(self):
        return NestedTensor([abs(value) for value in self._storage], **self._state)

    def __add__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x + y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value + other for value in self._storage], **self._state)

    def __radd__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y + x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other + value for value in self._storage], **self._state)

    def __iadd__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x += y
        else:
            for value in self._storage:
                value += other
        return self

    def __sub__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x - y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value - other for value in self._storage], **self._state)

    def __rsub__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y - x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other - value for value in self._storage], **self._state)

    def __isub__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x -= y
        else:
            for value in self._storage:
                value -= other
        return self

    def __pos__(self):
        return NestedTensor([+x for x in self._storage])

    def __neg__(self):
        return NestedTensor([-x for x in self._storage])

    def __invert__(self):
        return NestedTensor([~x for x in self._storage])

    def __mul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x * y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value * other for value in self._storage], **self._state)

    def __rmul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y * x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other * value for value in self._storage], **self._state)

    def __imul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x *= y
        else:
            for value in self._storage:
                value *= other
        return self

    def __pow__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x**y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value**other for value in self._storage], **self._state)

    def __rpow__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y**x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other**value for value in self._storage], **self._state)

    def __ipow__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x **= y
        else:
            for value in self._storage:
                value **= other
        return self

    def __matmul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x @ y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value @ other for value in self._storage], **self._state)

    def __rmatmul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y @ x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other @ value for value in self._storage], **self._state)

    def __imatmul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x @= y
        else:
            for value in self._storage:
                value @= other
        return self

    def __truediv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x / y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value / other for value in self._storage], **self._state)

    def __rtruediv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y / x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other / value for value in self._storage], **self._state)

    def __itruediv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x /= y
        else:
            for value in self._storage:
                value /= other
        return self

    def __floordiv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x // y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value // other for value in self._storage], **self._state)

    def __rfloordiv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y // x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other // value for value in self._storage], **self._state)

    def __ifloordiv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x //= y
        else:
            for value in self._storage:
                value //= other
        return self

    def __mod__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x % y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value % other for value in self._storage], **self._state)

    def __rmod__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y % x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other % value for value in self._storage], **self._state)

    def __imod__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x %= y
        else:
            for value in self._storage:
                value %= other
        return self

    def __and__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x & y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value & other for value in self._storage], **self._state)

    def __rand__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y & x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other & value for value in self._storage], **self._state)

    def __iand__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x &= y
        else:
            for value in self._storage:
                value &= other
        return self

    def __or__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x | y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value | other for value in self._storage], **self._state)

    def __ror__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y | x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other | value for value in self._storage], **self._state)

    def __ior__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x |= y
        else:
            for value in self._storage:
                value |= other
        return self

    def __xor__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x ^ y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value ^ other for value in self._storage], **self._state)

    def __rxor__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y ^ x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other ^ value for value in self._storage], **self._state)

    def __ixor__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x ^= y
        else:
            for value in self._storage:
                value ^= other
        return self

    def __lshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x << y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value << other for value in self._storage], **self._state)

    def __rlshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y << x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other << value for value in self._storage], **self._state)

    def __ilshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x <<= y
        else:
            for value in self._storage:
                value <<= other
        return self

    def __rshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x >> y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value >> other for value in self._storage], **self._state)

    def __rrshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y >> x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other >> value for value in self._storage], **self._state)

    def __irshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x >>= y
        else:
            for value in self._storage:
                value >>= other
        return self

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
            >>> nested_tensor.batch_first = False
            >>> nested_tensor.all(dim=0)
            NestedTensor([[ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]])
            >>> nested_tensor.all(dim=1)
            tensor([True, True])
        """

        if dim is None:
            return torch.tensor(all(i.all() for i in self._storage))
        if (self.batch_first and dim == 0) or (not self.batch_first and dim == 1):
            if keepdim:
                return torch.tensor([i.all() for i in self._storage]).unsqueeze(0 if self.batch_first else 1)
            return torch.tensor([i.all() for i in self._storage])
        if self.batch_first or dim != 0:
            dim -= 1
        return NestedTensor([i.all(dim=dim, keepdim=keepdim) for i in self._storage])

    def dim(self) -> int:
        r"""
        Number of dimension of the NestedTensor.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.dim()
            2
        """

        return self._dim(self._storage)

    @method_cache(maxsize=1)
    def _dim(self, storage: Tuple[Tensor, ...]) -> int:  # type: ignore[name-defined]
        return max(t.dim() for t in storage) + 1

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

        return sum(t.numel() for t in self._storage)

    def requires_grad_(self, requires_grad: bool = True):
        self.requires_grad = requires_grad
        for t in self._storage:
            t.requires_grad = requires_grad
        return self

    def reshape(self, *shape) -> Tensor:
        r"""
        Returns a torch tensor with a different shape.

        Note:
            since NestedTensor is a collection of tensors, the reshape operation is ambiguous.

            Therefore, it is converted to a tensor and then reshaped.

        Args:
            shape: The desired size of each dimension.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.reshape(3, 2)
            tensor([[1, 2],
                    [3, 4],
                    [5, 0]])
            >>> nested_tensor.reshape(2, 3)
            tensor([[1, 2, 3],
                    [4, 5, 0]])
        """

        return self.tensor.reshape(*shape)

    @property
    def shape(self) -> torch.Size | int:  # type: ignore[name-defined]
        r"""
        Alias for `size()`.
        """

        return self.size()

    def size(self, dim: int | None = None) -> torch.Size | int:  # type: ignore[name-defined]
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

        return self._size(self._storage, dim, self.batch_first)

    @method_cache(maxsize=1)
    def _size(
        self,
        storage: Tuple[Tensor, ...],
        dim: int | None = None,
        batch_first: bool = True,
    ) -> torch.Size | int:  # type: ignore[name-defined]
        if dim is not None:
            if dim == 0:
                return len(storage)
            return max(t.size(dim - 1) for t in storage)
        if max(t.dim() for t in storage) == 0:
            return torch.Size((len(storage),))
        ndim = max(t.dim() for t in storage)
        size = [max(t.shape[i] if i < len(t.shape) else 0 for t in storage) for i in range(ndim)]
        size.insert(0 if batch_first else 1, len(storage))
        return torch.Size(size)

    def to(self, *args, **kwargs):
        return NestedTensor(
            tuple(t.to(*args, **kwargs) for t in self._storage),
            **self.__state__(return_dtype=False, return_device=False),
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

    def view(self, *shape) -> Tensor:
        r"""
        Returns a torch tensor with a different shape.

        Note:
            since NestedTensor is a collection of tensors, the view operation is ambiguous.

            Therefore, it is converted to a tensor and then reshaped.

        Args:
            shape: The desired size of each dimension.

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.view(3, 2)
            tensor([[1, 2],
                    [3, 4],
                    [5, 0]])
            >>> nested_tensor.view(2, 3)
            tensor([[1, 2, 3],
                    [4, 5, 0]])
        """

        return self.tensor.view(*shape)

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

        if isinstance(condition, Tensor) and self.shape == condition.shape:
            condition = self.nested_like(condition)
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(condition, NestedTensor) and isinstance(other, NestedTensor):
            return self.__class__(
                [x.where(c, y) for x, c, y in zip(self._storage, condition._storage, other._storage)], **self._state
            )
        if isinstance(condition, NestedTensor):
            return self.__class__([x.where(c, other) for x, c in zip(self._storage, condition._storage)], **self._state)
        if isinstance(other, NestedTensor):
            return self.__class__([x.where(condition, y) for x, y in zip(self._storage, other._storage)], **self._state)
        return self.__class__(x.where(condition, other) for x in self._storage)
