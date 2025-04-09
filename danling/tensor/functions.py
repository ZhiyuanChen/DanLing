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


@NestedTensorFuncRegistry.implement(torch.cat)
def cat(tensors: Tuple[Tensor | NestedTensor, ...], dim: int = 0):
    from .nested_tensor import NestedTensor

    if dim != 0:
        raise NotImplementedError(f"NestedTensor only supports cat when dim=0, but got {dim}")
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


@NestedTensorFuncRegistry.implement(torch.sqrt)
def sqrt(tensor):
    from .nested_tensor import NestedTensor

    return NestedTensor(torch.sqrt(t) for t in tensor._storage)


@NestedTensorFuncRegistry.implement(torch.stack)
def stack(*args, **kwargs):
    raise NotImplementedError("NestedTensor does not support stack as of now")
