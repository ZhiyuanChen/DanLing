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

from functools import wraps
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence, Tuple

import torch
from chanfig import Registry
from torch import Tensor

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor


class TorchFuncRegistry(Registry):  # pylint: disable=too-few-public-methods
    """
    Registry for extending PyTorch functions to work with custom tensor types like NestedTensor.

    `TorchFuncRegistry` provides a clean interface for implementing PyTorch function
    overrides for custom tensor types such as NestedTensor. It's used internally by
    NestedTensor to register implementations for various torch functions like
    torch.cat, torch.mean, torch.stack, etc.

    This mechanism enables NestedTensor to behave like a regular torch.Tensor
    when used with standard PyTorch functions by providing custom implementations
    that understand the NestedTensor structure.

    Usage:
    ```python
    # Create a registry
    registry = TorchFuncRegistry("my_tensor_registry")

    # Register an implementation for torch.mean
    @registry.implement(torch.mean)
    def mean_implementation(input, dim=None, keepdim=False, **kwargs):
        # Custom implementation for your tensor type
        pass

    # The registry can be used to look up the implementation
    registry[torch.mean]  # Returns mean_implementation
    ```
    """

    def implement(self, torch_function: Callable) -> Callable:
        r"""
        Register a custom implementation for a PyTorch function.

        Use this decorator to provide implementations for PyTorch functions
        that will work with custom tensor types like NestedTensor. This is
        the key mechanism that allows NestedTensor to integrate seamlessly
        with the PyTorch ecosystem.

        Args:
            torch_function: The original PyTorch function to override (e.g., torch.mean, torch.cat)

        Returns:
            Callable: A decorator function that registers the implementation

        Raises:
            ValueError: If the function is already registered and override=False

        Examples:
            >>> import torch
            >>> registry = TorchFuncRegistry("test")
            >>> @registry.implement(torch.mean)
            ... def mean(input):
            ...     return input.mean()
            >>> registry[torch.mean]  # doctest: +ELLIPSIS
            <function mean at ...>

        Note:
            This is primarily used internally by NestedTensor.__torch_function__
            to provide implementations for various PyTorch functions. You can
            use the same mechanism to extend NestedTensor with additional
            function implementations.
        """

        if torch_function in self and not self.override:
            raise ValueError(f"Torch function {torch_function.__name__} already registered.")

        @wraps(self.register)
        def register(function):
            self.set(torch_function, function)
            return function

        return register


class NestedTensorFuncWrapper:  # pylint: disable=R0903
    r"""
    Function Wrapper to handle NestedTensor as input.
    """

    __storage: Sequence[Callable] = []
    state: Mapping = {}

    def __init__(self, *callables: Iterable[Callable], state: Mapping | None = None) -> None:
        if len(callables) == 1 and isinstance(callables, Sequence):
            callables = callables[0]  # type: ignore
        self._storage = callables  # type: ignore
        if state is None:
            state = {}
        self.state = state
        self.device = self.state.get("device")

    @property
    def _storage(self):
        return self.__storage

    @_storage.setter
    def _storage(self, callables: Sequence):
        if not isinstance(callables, Sequence):
            raise ValueError(f"callables must be a Sequence, bug got {type(callables)}")
        if len(callables) == 0:
            raise ValueError("callables must be a non-empty Sequence.")
        if not callable(callables[0]):
            raise ValueError(f"callables must be a Sequence of Callable, bug got {type(callables[0])}")
        self.__storage = callables

    def __call__(self, *args, **kwargs) -> NestedTensor | Tensor | Sequence[Tensor]:
        from .nested_tensor import NestedTensor

        ret = [call(*args, **kwargs) for call in self._storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            try:
                return torch.stack(ret, dim=0)
            except (ValueError, RuntimeError):
                return NestedTensor(ret, **self.state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret


NestedTensorFuncRegistry = TorchFuncRegistry()


@NestedTensorFuncRegistry.implement(torch.abs)
def abs(tensor: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.abs(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.add)
def add(input: NestedTensor, other: NestedTensor | Tensor, *, alpha: float = 1) -> NestedTensor:
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    return NestedTensor(torch.add(x, y, alpha=alpha) for x, y in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.allclose)
def allclose(
    input: NestedTensor, other: NestedTensor | Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    return all(
        torch.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan) for x, y in zip(input._storage, other._storage)
    )


@NestedTensorFuncRegistry.implement(torch.cat)
def cat(tensors: Tuple[Tensor | NestedTensor, ...], dim: int = 0):
    from .nested_tensor import NestedTensor

    if dim == 0:
        storage = []
        state: Mapping = {}
        for tensor in tensors:
            if isinstance(tensor, NestedTensor):
                storage.extend(tensor._storage)
                if not state:
                    state = tensor._state
            else:
                storage.append(tensor)
        return NestedTensor(storage, **state)

    if not all(isinstance(t, NestedTensor) for t in tensors):
        raise NotImplementedError("NestedTensor cat along non-zero dim requires all inputs to be NestedTensor.")
    first: NestedTensor = tensors[0]  # type: ignore[index]
    if any(len(t) != len(first) for t in tensors):
        raise ValueError("NestedTensor cat along non-zero dim requires the same batch length.")
    dim_adj = dim if dim < 0 else dim - 1
    storage = [
        torch.cat([t._storage[i] for t in tensors], dim=dim_adj) for i in range(len(first))  # type: ignore[index]
    ]
    return NestedTensor(storage, **first._state)


@NestedTensorFuncRegistry.implement(torch.cos)
def cos(tensor: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.cos(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.div)
def div(input: NestedTensor, other: NestedTensor | Tensor, *, rounding_mode: str | None = None) -> NestedTensor:
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    return NestedTensor(torch.div(x, y, rounding_mode=rounding_mode) for x, y in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.eq)
def eq(input: NestedTensor, other: NestedTensor | Tensor) -> Tensor:
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    return NestedTensor(torch.eq(x, y) for x, y in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.equal)
def equal(input: NestedTensor, other: NestedTensor | Tensor) -> bool:
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    return all(torch.equal(x, y) for x, y in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.exp)
def exp(tensor: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.exp(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.isin)
def isin(elements, test_elements, *, assume_unique: bool = False, invert: bool = False):
    from .nested_tensor import NestedTensor

    if isinstance(elements, NestedTensor):
        elements = elements.tensor
    if isinstance(test_elements, NestedTensor):
        test_elements = test_elements.tensor
    return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)


@NestedTensorFuncRegistry.implement(torch.log)
def log(tensor):
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.log(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.matmul)
def matmul(input: NestedTensor, other: NestedTensor | Tensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    return NestedTensor(torch.matmul(x, y) for x, y in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.max)
def max(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    if dim is None:
        return input.max().max()
    return input.max(dim=dim, keepdim=keepdim)


@NestedTensorFuncRegistry.implement(torch.min)
def min(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    if dim is None:
        return input.min().min()
    return input.min(dim=dim, keepdim=keepdim)


@NestedTensorFuncRegistry.implement(torch.mul)
def mul(input: NestedTensor, other: NestedTensor | Tensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    return NestedTensor(torch.mul(x, y) for x, y in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.mean)
def mean(
    input,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    if dim is None:
        return input.mean(dtype=dtype).mean()
    return input.mean(dim=dim, keepdim=keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.permute)
def permute(input: NestedTensor, dims: Sequence[int]) -> NestedTensor:
    return input.permute(dims)


@NestedTensorFuncRegistry.implement(torch.pow)
def pow(input: NestedTensor, exponent: NestedTensor | Tensor | float) -> NestedTensor:
    from .nested_tensor import NestedTensor

    if isinstance(exponent, (int, float)):
        return NestedTensor(torch.pow(t, exponent) for t in input._storage)

    if not isinstance(input, NestedTensor):
        input = exponent.nested_like(input)
    elif not isinstance(exponent, NestedTensor):
        exponent = input.nested_like(exponent)
    return NestedTensor(torch.pow(x, y) for x, y in zip(input._storage, exponent._storage))


@NestedTensorFuncRegistry.implement(torch.clamp)
def clamp(input: NestedTensor, min=None, max=None):
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.clamp(t, min=min, max=max) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.clip)
def clip(input: NestedTensor, min=None, max=None):
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.clamp(t, min=min, max=max) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.relu)
def relu(tensor: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.relu(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.reshape)
def reshape(input: NestedTensor, shape: Sequence[int]) -> NestedTensor:
    return input.reshape(shape)


@NestedTensorFuncRegistry.implement(torch.sigmoid)
def sigmoid(tensor: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.sigmoid(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.sin)
def sin(tensor: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.sin(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.softmax)
def softmax(input: NestedTensor, dim: int, dtype: torch.dtype | None = None) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.softmax(t, dim=dim, dtype=dtype) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.sqrt)
def sqrt(tensor):
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.sqrt(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.stack)
def stack(*args, **kwargs):
    tensors = args[0] if args else ()
    dim = kwargs.get("dim", 0)
    if dim != 0:
        raise NotImplementedError(f"NestedTensor only supports stack when dim=0, but got {dim}")
    if not isinstance(tensors, (tuple, list)) or not tensors:
        raise ValueError("Expected a non-empty sequence of NestedTensor objects.")
    from .nested_tensor import NestedTensor

    if not all(isinstance(t, NestedTensor) for t in tensors):
        raise NotImplementedError("torch.stack for NestedTensor requires all inputs to be NestedTensor.")
    return torch.stack([t.tensor for t in tensors], dim=0)


@NestedTensorFuncRegistry.implement(torch.sub)
def sub(input: NestedTensor, other: NestedTensor | Tensor, *, alpha: float = 1) -> NestedTensor:
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    return NestedTensor(torch.sub(x, y, alpha=alpha) for x, y in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.sum)
def sum(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    if dim is None:
        return input.sum(dtype=dtype).sum()
    return input.sum(dim=dim, keepdim=keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.tanh)
def tanh(tensor: NestedTensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.tanh(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.transpose)
def transpose(input: NestedTensor, dim0: int, dim1: int) -> NestedTensor:
    return input.transpose(dim0, dim1)


@NestedTensorFuncRegistry.implement(torch.flatten)
def flatten(input: NestedTensor, start_dim: int = 0, end_dim: int = -1):
    from .nested_tensor import NestedTensor

    if start_dim == 0:
        return torch.flatten(input.tensor, start_dim=start_dim, end_dim=end_dim)

    def _translate(dim: int) -> int:
        return dim if dim < 0 else dim - 1

    start = _translate(start_dim)
    end = _translate(end_dim)
    return NestedTensor(torch.flatten(t, start_dim=start, end_dim=end) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.unflatten)
def unflatten(input: NestedTensor, dim: int, sizes):
    from .nested_tensor import NestedTensor

    if dim == 0:
        raise NotImplementedError("unflatten on the batch dimension is not supported for NestedTensor.")
    dim = dim if dim < 0 else dim - 1
    return NestedTensor(torch.unflatten(t, dim, sizes) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.squeeze)
def squeeze(input: NestedTensor, dim: int | None = None):
    from .nested_tensor import NestedTensor

    def _translate(d):
        return d if d is None or d < 0 else d - 1

    dim_adj = _translate(dim)
    if dim_adj is None:
        return NestedTensor(t.squeeze() for t in input._storage)
    return NestedTensor(t.squeeze(dim_adj) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.unsqueeze)
def unsqueeze(input: NestedTensor, dim: int):
    from .nested_tensor import NestedTensor

    if dim == 0:
        raise ValueError("Cannot unsqueeze batch dimension for NestedTensor.")
    dim = dim if dim < 0 else dim - 1
    return NestedTensor(t.unsqueeze(dim) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.moveaxis)
def moveaxis(input: NestedTensor, source, destination):
    from .nested_tensor import NestedTensor

    def _translate(d):
        return d if d < 0 else d - 1

    source_adj = [_translate(d) for d in (source if isinstance(source, (tuple, list)) else [source])]
    dest_adj = [_translate(d) for d in (destination if isinstance(destination, (tuple, list)) else [destination])]
    return NestedTensor(torch.moveaxis(t, source_adj, dest_adj) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.swapaxes)
def swapaxes(input: NestedTensor, axis0: int, axis1: int):
    from .nested_tensor import NestedTensor

    if axis0 == 0 or axis1 == 0:
        raise ValueError("Cannot swap the batch dimension for NestedTensor.")
    axis0 = axis0 if axis0 < 0 else axis0 - 1
    axis1 = axis1 if axis1 < 0 else axis1 - 1
    return NestedTensor(torch.swapaxes(t, axis0, axis1) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.swapdims)
def swapdims(input: NestedTensor, dim0: int, dim1: int):
    return torch.swapaxes(input, dim0, dim1)


@NestedTensorFuncRegistry.implement(torch.gather)
def gather(input: NestedTensor, dim: int, index):
    from .nested_tensor import NestedTensor

    if dim == 0:
        raise NotImplementedError("gather on batch dimension is not supported for NestedTensor.")
    dim_adj = dim if dim < 0 else dim - 1
    if isinstance(index, Tensor) and input.shape == index.shape:
        index = input.nested_like(index, strict=False)
    if isinstance(index, NestedTensor):
        return NestedTensor(torch.gather(t, dim_adj, idx) for t, idx in zip(input._storage, index._storage))
    return NestedTensor(torch.gather(t, dim_adj, index) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.scatter)
def scatter(input: NestedTensor, dim: int, index, src):
    from .nested_tensor import NestedTensor

    if dim == 0:
        raise NotImplementedError("scatter on batch dimension is not supported for NestedTensor.")
    dim_adj = dim if dim < 0 else dim - 1
    if isinstance(index, Tensor) and input.shape == index.shape:
        index = input.nested_like(index, strict=False)
    if isinstance(src, Tensor) and input.shape == src.shape:
        src = input.nested_like(src, strict=False)
    if isinstance(index, NestedTensor):
        indices = index._storage
    else:
        indices = [index for _ in input._storage]
    if isinstance(src, NestedTensor):
        srcs = src._storage
    else:
        srcs = [src for _ in input._storage]
    return NestedTensor(torch.scatter(t, dim_adj, idx, s) for t, idx, s in zip(input._storage, indices, srcs))
