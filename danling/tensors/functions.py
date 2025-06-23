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

from .ops import (
    _as_tensor_like,
    _binary_op_maybe_tensor,
    _broadcast_storage,
    _concat_apply_same_shape,
    _concat_dim_for_tensor_dim,
    _get_batch_dim,
    _map_storage,
    _normalize_dim,
    _reduce_dim,
    _reduce_dims_masked,
    _reduce_none,
    _stack_or_nested,
    _ternary_op,
    _translate_dim,
    _translate_dims,
    _translate_non_batch_dim,
)

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

    def implement(self, torch_function: Callable, *, inherit_state: bool = True) -> Callable:
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
            wrapped_function = function
            if inherit_state:
                wrapped_function = _wrap_with_state(function)
            self.set(torch_function, wrapped_function)
            return wrapped_function

        return register


class NestedTensorFuncWrapper:  # pylint: disable=R0903
    r"""
    Function Wrapper to handle NestedTensor as input.
    """

    __storage: Sequence[Callable]
    state: Mapping

    def __init__(self, *callables: Iterable[Callable], state: Mapping | None = None) -> None:
        r"""
        Initialize a wrapper around one or more callables.

        Examples:
            >>> from danling.tensors.functions import NestedTensorFuncWrapper
            >>> wrapper = NestedTensorFuncWrapper([lambda x: x + 1])
            >>> wrapper(1)
            2
        """
        if len(callables) == 1 and isinstance(callables[0], Sequence) and not callable(callables[0]):
            callables = tuple(callables[0])  # type: ignore[assignment]
        self._storage = callables  # type: ignore[arg-type]
        if state is None:
            state = {}
        self.state = state
        self.device = self.state.get("device")

    @property
    def _storage(self):
        r"""
        Get or set the stored callables.

        Examples:
            >>> from danling.tensors.functions import NestedTensorFuncWrapper
            >>> wrapper = NestedTensorFuncWrapper([lambda x: x + 1])
            >>> len(wrapper._storage)
            1
        """
        return self.__storage

    @_storage.setter
    def _storage(self, callables: Sequence):
        r"""
        Get or set the stored callables.

        Examples:
            >>> from danling.tensors.functions import NestedTensorFuncWrapper
            >>> wrapper = NestedTensorFuncWrapper([lambda x: x + 1])
            >>> len(wrapper._storage)
            1
        """
        if not isinstance(callables, Sequence):
            raise ValueError(f"callables must be a Sequence, but got {type(callables)}")
        if len(callables) == 0:
            raise ValueError("callables must be a non-empty Sequence.")
        if any(not callable(c) for c in callables):
            bad = next(type(c) for c in callables if not callable(c))
            raise ValueError(f"callables must be a Sequence of Callable, but got {bad}")
        self.__storage = callables

    def __call__(self, *args, **kwargs) -> NestedTensor | Tensor | Sequence[Tensor]:
        r"""
        Call stored functions and combine the results.

        Examples:
            >>> import torch
            >>> from danling.tensors.functions import NestedTensorFuncWrapper
            >>> wrapper = NestedTensorFuncWrapper([lambda x: torch.tensor(x), lambda x: torch.tensor(x + 1)])
            >>> out = wrapper(1)
            >>> torch.equal(out, torch.stack([torch.tensor(1), torch.tensor(2)]))
            True
        """
        from .nested_tensor import NestedTensor

        ret = [call(*args, **kwargs) for call in self._storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            try:
                return torch.stack(ret, dim=0)
            except (ValueError, RuntimeError):
                state = dict(self.state)
                state["device"] = elem.device
                return NestedTensor(ret, **state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret


def _wrap_with_state(function: Callable) -> Callable:
    r"""
    Wrap a function and propagate NestedTensor state from inputs.

    Examples:
        >>> import torch
        >>> from danling.tensors.functions import _wrap_with_state
        >>> fn = _wrap_with_state(torch.abs)
        >>> torch.equal(fn(torch.tensor([-1.0])), torch.tensor([1.0]))
        True
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        return _inherit_nested_state(result, args, kwargs)

    return wrapper


def _inherit_nested_state(result, args, kwargs):
    r"""
    Attach NestedTensor state from an input to the result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.functions import _inherit_nested_state
        >>> nt = NestedTensor(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> out = _inherit_nested_state(torch.add(nt, 1.0), (nt,), {})
        >>> isinstance(out, NestedTensor)
        True
    """
    from .nested_tensor import NestedTensor

    source = next((a for a in args if isinstance(a, NestedTensor)), None)
    if source is None:
        source = next((v for v in kwargs.values() if isinstance(v, NestedTensor)), None)
    if source is None:
        return result

    def attach(obj):
        if isinstance(obj, NestedTensor):
            obj._inherit_state_from(source)
        elif isinstance(obj, tuple):
            return tuple(attach(o) for o in obj)
        elif isinstance(obj, list):
            return [attach(o) for o in obj]
        return obj

    return attach(result)


NestedTensorFuncRegistry = TorchFuncRegistry()


@NestedTensorFuncRegistry.implement(torch.add)
def add(input: NestedTensor, other: NestedTensor | Tensor, *, alpha: float = 1) -> NestedTensor:
    r"""
    Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.
    See also [torch.add][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(torch.add(nt, 1.5), torch.add(nt.tensor, 1.5))
        True
    """
    return _binary_op_maybe_tensor(input, other, lambda x, y: torch.add(x, y, alpha=alpha))


@NestedTensorFuncRegistry.implement(torch.all)
def all_(input: NestedTensor, dim: int | Sequence[int] | None = None, keepdim: bool = False):
    r"""
    Tests if all elements in :attr:`input` evaluate to `True`.
    See also [torch.all][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[True, False], [True, True]]), torch.tensor([[False, True], [True, False]]))
        >>> torch.equal(torch.all(nt, dim=1), torch.all(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.all)
    if isinstance(dim, int):
        return _reduce_dim(input, torch.all, dim, keepdim)

    dims = tuple(dim)
    if len(dims) == 1:
        return _reduce_dim(input, torch.all, dims[0], keepdim)
    return _reduce_dims_masked(input, dims, torch.all, keepdim, fill_value=True)


@NestedTensorFuncRegistry.implement(torch.allclose)
def allclose(
    input: NestedTensor, other: NestedTensor | Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    r"""
    This function checks if :attr:`input` and :attr:`other` satisfy the condition: .. math:: \lvert \text{input}_i -
    \text{other}_i \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other}_i \rvert elementwise, for all
    elements of :attr:`input` and :attr:`other`. The behaviour of this function is analogous to `numpy.allclose
    <https://numpy.org/doc/stable/reference/generated/numpy.allclose.html>`_
    See also [torch.allclose][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(nt, nt.tensor)
        True
    """
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    if len(input) != len(other):
        return False
    return all(
        torch.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan) for x, y in zip(input._storage, other._storage)
    )


@NestedTensorFuncRegistry.implement(torch.any)
def any_(input: NestedTensor, dim: int | Sequence[int] | None = None, keepdim: bool = False):
    r"""
    Tests if any element in :attr:`input` evaluates to `True`.
    See also [torch.any][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[True, False], [True, True]]), torch.tensor([[False, True], [True, False]]))
        >>> torch.equal(torch.any(nt, dim=1), torch.any(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.any)
    if isinstance(dim, int):
        return _reduce_dim(input, torch.any, dim, keepdim)

    dims = tuple(dim)
    if len(dims) == 1:
        return _reduce_dim(input, torch.any, dims[0], keepdim)
    return _reduce_dims_masked(input, dims, torch.any, keepdim, fill_value=False)


@NestedTensorFuncRegistry.implement(torch.prod)
def prod(
    input: NestedTensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    r"""
    Returns the product of all elements in the :attr:`input` tensor.
    See also [torch.prod][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.prod(nt, dim=1), torch.prod(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.prod, dtype=dtype)

    return _reduce_dim(input, torch.prod, dim, keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.addcmul)
def addcmul(input, tensor1, tensor2, *, value=1):
    r"""
    Performs the element-wise multiplication of :attr:`tensor1` by :attr:`tensor2`, multiplies the result by the scalar
    :attr:`value` and adds it to :attr:`input`.
    See also [torch.addcmul][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> t1 = torch.tensor([2.0, 3.0])
        >>> t2 = torch.tensor([4.0, 5.0])
        >>> torch.allclose(torch.addcmul(nt, t1, t2), torch.addcmul(nt.tensor, t1, t2))
        True
    """
    from .nested_tensor import NestedTensor

    ref = next((t for t in (input, tensor1, tensor2) if isinstance(t, NestedTensor)), None)
    if ref is None:
        return torch.addcmul(input, tensor1, tensor2, value=value)
    return _ternary_op(ref, input, tensor1, tensor2, torch.addcmul, value=value)


@NestedTensorFuncRegistry.implement(torch.addcdiv)
def addcdiv(input, tensor1, tensor2, *, value=1):
    r"""
    Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`, multiplies the result by the scalar
    :attr:`value` and adds it to :attr:`input`.
    See also [torch.addcdiv][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> t1 = torch.tensor([2.0, 3.0])
        >>> t2 = torch.tensor([4.0, 5.0])
        >>> torch.allclose(torch.addcdiv(nt, t1, t2), torch.addcdiv(nt.tensor, t1, t2))
        True
    """
    from .nested_tensor import NestedTensor

    ref = next((t for t in (input, tensor1, tensor2) if isinstance(t, NestedTensor)), None)
    if ref is None:
        return torch.addcdiv(input, tensor1, tensor2, value=value)
    return _ternary_op(ref, input, tensor1, tensor2, torch.addcdiv, value=value)


@NestedTensorFuncRegistry.implement(torch.cat)
def cat(tensors: Tuple[Tensor | NestedTensor, ...], dim: int = 0):
    r"""
    Concatenates the given sequence of tensors in :attr:`tensors` in the given dimension.
    See also [torch.cat][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.cat((nt, nt), dim=0)
        >>> ref = torch.cat((nt.tensor, nt.tensor), dim=0)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    ref = next((t for t in tensors if isinstance(t, NestedTensor)), None)
    if ref is None:
        return torch.cat(tensors, dim=dim)

    dim = _normalize_dim(dim, ref.dim())
    batch_dim = _get_batch_dim(ref)

    if dim == batch_dim:
        storage: list = []
        state: Mapping = ref._state
        for tensor in tensors:
            if isinstance(tensor, NestedTensor):
                storage.extend(tensor._storage)
            else:
                storage.append(tensor)
        return NestedTensor(storage, **state)

    if not all(isinstance(t, NestedTensor) for t in tensors):
        raise NotImplementedError("NestedTensor cat along non-batch dim requires all inputs to be NestedTensor.")
    first: NestedTensor = tensors[0]  # type: ignore[index]
    if any(len(t) != len(first) for t in tensors):
        raise ValueError("NestedTensor cat along non-batch dim requires the same batch length.")

    dim_adj = _translate_dim(first, dim)
    storage = [
        torch.cat([t._storage[i] for t in tensors], dim=dim_adj) for i in range(len(first))  # type: ignore[index]
    ]
    return NestedTensor(storage, **first._state)


@NestedTensorFuncRegistry.implement(torch.concat)
def concat(tensors, dim: int = 0):
    r"""
    Alias of :func:`torch.cat`.
    See also [torch.concat][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.concat((nt, nt), dim=0)
        >>> ref = torch.concat((nt.tensor, nt.tensor), dim=0)
        >>> torch.equal(out, ref)
        True
    """
    return torch.cat(tuple(tensors), dim=dim)


@NestedTensorFuncRegistry.implement(torch.concatenate)
def concatenate(tensors, dim: int = 0):
    r"""
    Alias of :func:`torch.cat`.
    See also [torch.concatenate][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.concatenate((nt, nt), dim=0)
        >>> ref = torch.concatenate((nt.tensor, nt.tensor), dim=0)
        >>> torch.equal(out, ref)
        True
    """
    return torch.cat(tuple(tensors), dim=dim)


@NestedTensorFuncRegistry.implement(torch.div)
def div(input: NestedTensor, other: NestedTensor | Tensor, *, rounding_mode: str | None = None) -> NestedTensor:
    r"""
    Divides each element of the input ``input`` by the corresponding element of :attr:`other`.
    See also [torch.div][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(torch.div(nt, 1.5), torch.div(nt.tensor, 1.5))
        True
    """
    return _binary_op_maybe_tensor(input, other, lambda x, y: torch.div(x, y, rounding_mode=rounding_mode))


@NestedTensorFuncRegistry.implement(torch.equal)
def equal(input: NestedTensor, other: NestedTensor | Tensor) -> bool:
    r"""
    ``True`` if two tensors have the same size and elements, ``False`` otherwise.
    See also [torch.equal][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1, 2]), torch.tensor([3, 4]))
        >>> torch.equal(nt, nt.tensor)
        True
    """
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    if len(input) != len(other):
        return False
    return all(torch.equal(x, y) for x, y in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.dropout)
def dropout(input: NestedTensor, p: float = 0.5, train: bool = True):
    r"""
    Applies dropout to the input.
    See also [torch.dropout][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(torch.dropout(nt, p=0.0, train=False), torch.dropout(nt.tensor, p=0.0, train=False))
        True
    """
    return _concat_apply_same_shape(input, lambda t: torch.dropout(t, p=p, train=train))


@NestedTensorFuncRegistry.implement(torch.bernoulli)
def bernoulli(input: NestedTensor, *, generator=None):
    r"""
    Draws binary random numbers (0 or 1) from a Bernoulli distribution.
    See also [torch.bernoulli][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([0.2, 0.8]), torch.tensor([0.6, 0.4, 0.1]))
        >>> out = torch.bernoulli(nt)
        >>> out[0].shape == nt[0].shape and out[1].shape == nt[1].shape
        True
    """
    return _concat_apply_same_shape(input, lambda t: torch.bernoulli(t, generator=generator))


@NestedTensorFuncRegistry.implement(torch.rand_like)
def rand_like(input: NestedTensor, **kwargs):
    r"""
    Returns a tensor with the same size as :attr:`input` that is filled with random numbers from a uniform distribution
    on the interval :math:`[0, 1)`.
    See also [torch.rand_like][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.rand_like(nt)
        >>> out[0].shape == nt[0].shape and out[1].shape == nt[1].shape
        True
    """
    return _map_storage(input, lambda t: torch.rand_like(t, **kwargs))


@NestedTensorFuncRegistry.implement(torch.randn_like)
def randn_like(input: NestedTensor, **kwargs):
    r"""
    Returns a tensor with the same size as :attr:`input` that is filled with random numbers from a normal distribution
    with mean 0 and variance 1. Please refer to :func:`torch.randn` for the sampling process of complex dtypes.
    ``torch.randn_like(input)`` is equivalent to ``torch.randn(input.size(), dtype=input.dtype, layout=input.layout,
    device=input.device)``.
    See also [torch.randn_like][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.randn_like(nt)
        >>> out[0].shape == nt[0].shape and out[1].shape == nt[1].shape
        True
    """
    return _map_storage(input, lambda t: torch.randn_like(t, **kwargs))


@NestedTensorFuncRegistry.implement(torch.empty_like)
def empty_like(input: NestedTensor, **kwargs):
    r"""
    Returns an uninitialized tensor with the same size as :attr:`input`.
    See also [torch.empty_like][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.empty_like(nt)
        >>> out[0].shape == nt[0].shape and out[1].shape == nt[1].shape and out[0].dtype == nt[0].dtype
        True
    """
    return _map_storage(input, lambda t: torch.empty_like(t, **kwargs))


@NestedTensorFuncRegistry.implement(torch.zeros_like)
def zeros_like(input: NestedTensor, **kwargs):
    r"""
    Returns a tensor filled with the scalar value `0`, with the same size as :attr:`input`. ``torch.zeros_like(input)``
    is equivalent to ``torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
    See also [torch.zeros_like][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.zeros_like(nt)
        >>> ref = NestedTensor(torch.zeros_like(nt[0]), torch.zeros_like(nt[1]))
        >>> torch.equal(out, ref)
        True
    """
    return _map_storage(input, lambda t: torch.zeros_like(t, **kwargs))


@NestedTensorFuncRegistry.implement(torch.ones_like)
def ones_like(input: NestedTensor, **kwargs):
    r"""
    Returns a tensor filled with the scalar value `1`, with the same size as :attr:`input`. ``torch.ones_like(input)``
    is equivalent to ``torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
    See also [torch.ones_like][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.ones_like(nt)
        >>> ref = NestedTensor(torch.ones_like(nt[0]), torch.ones_like(nt[1]))
        >>> torch.equal(out, ref)
        True
    """
    return _map_storage(input, lambda t: torch.ones_like(t, **kwargs))


@NestedTensorFuncRegistry.implement(torch.full_like)
def full_like(input: NestedTensor, fill_value, **kwargs):
    r"""
    Returns a tensor with the same size as :attr:`input` filled with :attr:`fill_value`.
    See also [torch.full_like][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.full_like(nt, 2.0)
        >>> ref = NestedTensor(torch.full_like(nt[0], 2.0), torch.full_like(nt[1], 2.0))
        >>> torch.equal(out, ref)
        True
    """
    return _map_storage(input, lambda t: torch.full_like(t, fill_value, **kwargs))


@NestedTensorFuncRegistry.implement(torch.randint_like)
def randint_like(input: NestedTensor, *args, **kwargs):
    r"""
    Returns a tensor with the same shape as Tensor :attr:`input` filled with random integers generated uniformly between
    :attr:`low` (inclusive) and :attr:`high` (exclusive).
    See also [torch.randint_like][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1, 2]), torch.tensor([3, 4, 5]))
        >>> out = torch.randint_like(nt, 0, 5)
        >>> out[0].shape == nt[0].shape and out[1].shape == nt[1].shape
        True
    """
    return _map_storage(input, lambda t: torch.randint_like(t, *args, **kwargs))


@NestedTensorFuncRegistry.implement(torch.nan_to_num)
def nan_to_num(input: NestedTensor, nan=0.0, posinf=None, neginf=None):
    r"""
    Replaces :literal:`NaN`, positive infinity, and negative infinity values in :attr:`input` with the values specified
    by :attr:`nan`, :attr:`posinf`, and :attr:`neginf`, respectively.
    See also [torch.nan_to_num][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, -2.0]), torch.tensor([3.0, -4.0]))
        >>> torch.allclose(torch.nan_to_num(nt), torch.nan_to_num(nt.tensor))
        True
    """
    return _concat_apply_same_shape(input, lambda t: torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf))


@NestedTensorFuncRegistry.implement(torch.isin)
def isin(elements, test_elements, *, assume_unique: bool = False, invert: bool = False):
    r"""
    Tests if each element of :attr:`elements` is in :attr:`test_elements`. Returns a boolean tensor of the same shape as
    :attr:`elements` that is True for elements in :attr:`test_elements` and False otherwise.
    See also [torch.isin][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1, 2]), torch.tensor([3, 4]))
        >>> test = torch.tensor([2, 4])
        >>> torch.equal(torch.isin(nt, test), torch.isin(nt.tensor, test))
        True
    """
    from .nested_tensor import NestedTensor

    if isinstance(elements, NestedTensor):
        elements = elements.tensor
    if isinstance(test_elements, NestedTensor):
        test_elements = test_elements.tensor
    return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)


@NestedTensorFuncRegistry.implement(torch.logsumexp)
def logsumexp(input: NestedTensor, dim: int | Sequence[int], keepdim: bool = False):
    r"""
    Returns the log of summed exponentials of each row of the :attr:`input` tensor in the given dimension :attr:`dim`.
    The computation is numerically stabilized.
    See also [torch.logsumexp][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.logsumexp(nt, dim=1), torch.logsumexp(nt.tensor, dim=1))
        True
    """
    batch_dim = _get_batch_dim(input)
    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        if dim_int == batch_dim:
            output = torch.stack([torch.logsumexp(t.reshape(-1), dim=0) for t in input._storage])
            if keepdim:
                return output.unsqueeze(0 if input.batch_first else 1)
            return output

        dim_adj = _translate_dim(input, dim_int)
        ret = [torch.logsumexp(t, dim=dim_adj, keepdim=keepdim) for t in input._storage]
        return _stack_or_nested(ret, input)

    dims = tuple(dim)
    if len(dims) == 1:
        return torch.logsumexp(input, dim=dims[0], keepdim=keepdim)
    dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
    if batch_dim in dims_norm:
        raise NotImplementedError(
            "logsumexp over the batch dimension and other dims is not supported for NestedTensor."
        )
    dims_adj = _translate_dims(input, dims_norm)
    ret = [torch.logsumexp(t, dim=dims_adj, keepdim=keepdim) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.numel, inherit_state=False)
def numel(input: NestedTensor) -> int:
    r"""
    Returns the total number of elements in the :attr:`input` tensor.
    See also [torch.numel][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1, 2]), torch.tensor([3, 4]))
        >>> torch.numel(nt) == torch.numel(nt.tensor)
        True
    """
    return input.numel()


@NestedTensorFuncRegistry.implement(torch.count_nonzero)
def count_nonzero(input: NestedTensor, dim: int | Sequence[int] | None = None):
    r"""
    Counts the number of non-zero values in the tensor :attr:`input` along the given :attr:`dim`.
    See also [torch.count_nonzero][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([0, 1]), torch.tensor([2, 0]))
        >>> torch.equal(torch.count_nonzero(nt), torch.count_nonzero(nt.tensor))
        True
    """
    if dim is None:
        return torch.stack([torch.count_nonzero(t) for t in input._storage]).sum()

    batch_dim = _get_batch_dim(input)
    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        if dim_int == batch_dim:
            raise NotImplementedError("count_nonzero along the batch dimension is not supported for NestedTensor.")
        dim_adj = _translate_dim(input, dim_int)
        ret = [torch.count_nonzero(t, dim=dim_adj) for t in input._storage]
    else:
        dims = tuple(dim)
        if len(dims) == 1:
            return torch.count_nonzero(input, dim=dims[0])
        dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
        if batch_dim in dims_norm:
            raise NotImplementedError("count_nonzero over the batch dimension and other dims is not supported.")
        dims_adj = _translate_dims(input, dims_norm)
        ret = [torch.count_nonzero(t, dim=dims_adj) for t in input._storage]

    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.nonzero)
def nonzero(input: NestedTensor, *, out=None, as_tuple: bool = False):
    r"""
    .. note:: :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a 2-D tensor where each row
    is the index for a nonzero value.
    See also [torch.nonzero][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([0, 1])
        >>> b = torch.tensor([2, 0, 3])
        >>> nt = NestedTensor(a, b)
        >>> out = torch.nonzero(nt)
        >>> ref = NestedTensor(torch.nonzero(a, as_tuple=False), torch.nonzero(b, as_tuple=False))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.nonzero(..., out=...) is not supported for NestedTensor.")

    if not input._storage:
        if as_tuple:
            return ()
        return NestedTensor([], **input._state)

    if not as_tuple:
        return NestedTensor(torch.nonzero(t, as_tuple=False) for t in input._storage)

    ndims = {t.dim() for t in input._storage}
    if len(ndims) != 1:
        raise NotImplementedError("torch.nonzero(as_tuple=True) requires all tensors in NestedTensor to share ndim.")
    (ndim,) = ndims
    if ndim == 0:
        return ()
    per_dim: list[list[Tensor]] = [[] for _ in range(ndim)]
    for t in input._storage:
        indices = torch.nonzero(t, as_tuple=True)
        for dim, idx in enumerate(indices):
            per_dim[dim].append(idx)
    return tuple(NestedTensor(per_dim[dim], **input._state) for dim in range(ndim))


@NestedTensorFuncRegistry.implement(torch.take)
def take(input: NestedTensor, index, *, out=None):
    r"""
    Returns a new tensor with the elements of :attr:`input` at the given indices.
    See also [torch.take][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1, 2]), torch.tensor([3, 4]))
        >>> index = torch.tensor([0, 3])
        >>> torch.equal(torch.take(nt, index), torch.take(nt.tensor, index))
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.take(..., out=...) is not supported for NestedTensor.")

    if isinstance(index, NestedTensor):
        if len(input) != len(index):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(index)}")
        return NestedTensor(torch.take(t.reshape(-1), i) for t, i in zip(input._storage, index._storage))

    flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
    return torch.take(flat, index)


@NestedTensorFuncRegistry.implement(torch.matmul)
def matmul(input, other, *, out=None):
    r"""
    Matrix product of two tensors.
    See also [torch.matmul][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> other = torch.eye(2)
        >>> torch.allclose(torch.matmul(nt, other), torch.matmul(nt.tensor, other))
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.matmul(..., out=...) is not supported for NestedTensor.")

    if isinstance(input, NestedTensor) and isinstance(other, NestedTensor):
        if len(input) != len(other):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(other)}")
        return NestedTensor((torch.matmul(x, y) for x, y in zip(input, other)), **input._state)

    if isinstance(input, NestedTensor):
        if isinstance(other, Tensor) and input.shape == other.shape:
            other = input.nested_like(other)
            if len(input) != len(other):
                raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(other)}")
            return NestedTensor((torch.matmul(x, y) for x, y in zip(input, other)), **input._state)
        return NestedTensor((torch.matmul(t, other) for t in input), **input._state)

    if isinstance(other, NestedTensor):
        if isinstance(input, Tensor) and other.shape == input.shape:
            input = other.nested_like(input)
            if len(input) != len(other):
                raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(other)}")
            return NestedTensor((torch.matmul(x, y) for x, y in zip(input, other)), **other._state)
        return NestedTensor((torch.matmul(input, t) for t in other), **other._state)

    return torch.matmul(input, other)


@NestedTensorFuncRegistry.implement(torch.mm)
def mm(input, mat2, *, out=None):
    r"""
    Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.
    See also [torch.mm][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        >>> nt = NestedTensor(a, b)
        >>> other = torch.eye(2)
        >>> out = torch.mm(nt, other)
        >>> ref = NestedTensor(torch.mm(a, other), torch.mm(b, other))
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.mm(..., out=...) is not supported for NestedTensor.")

    if isinstance(input, NestedTensor) and isinstance(mat2, NestedTensor):
        if len(input) != len(mat2):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(mat2)}")
        return NestedTensor((torch.mm(x, y) for x, y in zip(input, mat2)), **input._state)

    if isinstance(input, NestedTensor):
        return NestedTensor((torch.mm(t, mat2) for t in input), **input._state)

    if isinstance(mat2, NestedTensor):
        return NestedTensor((torch.mm(input, t) for t in mat2), **mat2._state)

    return torch.mm(input, mat2)


@NestedTensorFuncRegistry.implement(torch.bmm)
def bmm(input, mat2, *, out=None):
    r"""
    Performs a batch matrix-matrix product of matrices stored in :attr:`input` and :attr:`mat2`.
    See also [torch.bmm][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.arange(8.0).reshape(1, 2, 4)
        >>> b = torch.arange(8.0).reshape(1, 4, 2)
        >>> nt_a = NestedTensor(a, a)
        >>> nt_b = NestedTensor(b, b)
        >>> out = torch.bmm(nt_a, nt_b)
        >>> ref = NestedTensor(torch.bmm(a, b), torch.bmm(a, b))
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.bmm(..., out=...) is not supported for NestedTensor.")

    if isinstance(input, NestedTensor) and isinstance(mat2, NestedTensor):
        if len(input) != len(mat2):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(mat2)}")
        return NestedTensor((torch.bmm(x, y) for x, y in zip(input, mat2)), **input._state)

    if isinstance(input, NestedTensor):
        return NestedTensor((torch.bmm(t, mat2) for t in input), **input._state)

    if isinstance(mat2, NestedTensor):
        return NestedTensor((torch.bmm(input, t) for t in mat2), **mat2._state)

    return torch.bmm(input, mat2)


@NestedTensorFuncRegistry.implement(torch.dist)
def dist(input: NestedTensor, other: NestedTensor | Tensor, p=2):
    r"""
    Returns the p-norm of (:attr:`input` - :attr:`other`) The shapes of :attr:`input` and :attr:`other` must be
    :ref:`broadcastable <broadcasting-semantics>`.
    See also [torch.dist][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([1.0, 2.0])
        >>> b = torch.tensor([3.0, 4.0])
        >>> nt = NestedTensor(a, b)
        >>> out = torch.dist(nt, nt)
        >>> ref = torch.stack([torch.dist(a, a), torch.dist(b, b)])
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    if len(input) != len(other):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(other)}")
    if not input._storage:
        return torch.empty((0,), device=input.device)
    return torch.stack([torch.dist(x, y, p=p) for x, y in zip(input._storage, other._storage)])


@NestedTensorFuncRegistry.implement(torch.max)
def max(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the maximum value of all elements in the ``input`` tensor.
    See also [torch.max][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.max(nt, dim=1)
        >>> ref = torch.max(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if dim is None:
        return torch.stack([t.max() for t in input._storage]).max()

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        output = torch.stack([t.max() for t in input._storage])
        if keepdim:
            return output.unsqueeze(0 if input.batch_first else 1)
        return output

    dim_adj = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        result = torch.max(t, dim=dim_adj, keepdim=keepdim)
        values.append(result.values)
        indices.append(result.indices)
    values = _stack_or_nested(values, input)
    indices = _stack_or_nested(indices, input)
    return torch.return_types.max((values, indices))


@NestedTensorFuncRegistry.implement(torch.amax)
def amax(input: NestedTensor, dim: int | Sequence[int] | None = None, keepdim: bool = False):
    r"""
    Returns the maximum value of each slice of the :attr:`input` tensor in the given dimension(s) :attr:`dim`.
    See also [torch.amax][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.amax(nt, dim=1), torch.amax(nt.tensor, dim=1))
        True
    """
    if dim is None:
        output = torch.stack([torch.amax(t) for t in input._storage]).amax()
        if keepdim:
            return output.reshape((1,) * input.dim())
        return output
    batch_dim = _get_batch_dim(input)
    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        if dim_int == batch_dim:
            output = torch.stack([torch.amax(t) for t in input._storage])
            if keepdim:
                return output.unsqueeze(0 if input.batch_first else 1)
            return output

        dim_adj = _translate_dim(input, dim_int)
        ret = [torch.amax(t, dim=dim_adj, keepdim=keepdim) for t in input._storage]
        return _stack_or_nested(ret, input)

    dims = tuple(dim)
    if len(dims) == 1:
        return torch.amax(input, dim=dims[0], keepdim=keepdim)
    dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
    if batch_dim in dims_norm:
        raise NotImplementedError("amax over the batch dimension and other dims is not supported for NestedTensor.")
    dims_adj = _translate_dims(input, dims_norm)
    ret = [torch.amax(t, dim=dims_adj, keepdim=keepdim) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.amin)
def amin(input: NestedTensor, dim: int | Sequence[int] | None = None, keepdim: bool = False):
    r"""
    Returns the minimum value of each slice of the :attr:`input` tensor in the given dimension(s) :attr:`dim`.
    See also [torch.amin][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.amin(nt, dim=1), torch.amin(nt.tensor, dim=1))
        True
    """
    if dim is None:
        output = torch.stack([torch.amin(t) for t in input._storage]).amin()
        if keepdim:
            return output.reshape((1,) * input.dim())
        return output
    batch_dim = _get_batch_dim(input)
    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        if dim_int == batch_dim:
            output = torch.stack([torch.amin(t) for t in input._storage])
            if keepdim:
                return output.unsqueeze(0 if input.batch_first else 1)
            return output

        dim_adj = _translate_dim(input, dim_int)
        ret = [torch.amin(t, dim=dim_adj, keepdim=keepdim) for t in input._storage]
        return _stack_or_nested(ret, input)

    dims = tuple(dim)
    if len(dims) == 1:
        return torch.amin(input, dim=dims[0], keepdim=keepdim)
    dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
    if batch_dim in dims_norm:
        raise NotImplementedError("amin over the batch dimension and other dims is not supported for NestedTensor.")
    dims_adj = _translate_dims(input, dims_norm)
    ret = [torch.amin(t, dim=dims_adj, keepdim=keepdim) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.aminmax)
def aminmax(input: NestedTensor, *, dim: int | None = None, keepdim: bool = False):
    r"""
    Computes the minimum and maximum values of the :attr:`input` tensor.
    See also [torch.aminmax][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_min, out_max = torch.aminmax(nt, dim=1)
        >>> ref_min, ref_max = torch.aminmax(nt.tensor, dim=1)
        >>> torch.equal(out_min, ref_min) and torch.equal(out_max, ref_max)
        True
    """
    if dim is None:
        mins = torch.stack([torch.amin(t) for t in input._storage]).amin()
        maxs = torch.stack([torch.amax(t) for t in input._storage]).amax()
        if keepdim:
            shape = (1,) * input.dim()
            return mins.reshape(shape), maxs.reshape(shape)
        return mins, maxs

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        mins = torch.stack([torch.amin(t) for t in input._storage])
        maxs = torch.stack([torch.amax(t) for t in input._storage])
        if keepdim:
            mins = mins.unsqueeze(0 if input.batch_first else 1)
            maxs = maxs.unsqueeze(0 if input.batch_first else 1)
        return mins, maxs

    dim_adj = _translate_dim(input, dim)
    mins = []
    maxs = []
    for t in input._storage:
        result = torch.aminmax(t, dim=dim_adj, keepdim=keepdim)
        mins.append(result.min)
        maxs.append(result.max)
    return _stack_or_nested(mins, input), _stack_or_nested(maxs, input)


@NestedTensorFuncRegistry.implement(torch.min)
def min(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the minimum value of all elements in the :attr:`input` tensor.
    See also [torch.min][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.min(nt, dim=1)
        >>> ref = torch.min(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if dim is None:
        return torch.stack([t.min() for t in input._storage]).min()

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        output = torch.stack([t.min() for t in input._storage])
        if keepdim:
            return output.unsqueeze(0 if input.batch_first else 1)
        return output

    dim_adj = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        result = torch.min(t, dim=dim_adj, keepdim=keepdim)
        values.append(result.values)
        indices.append(result.indices)
    values = _stack_or_nested(values, input)
    indices = _stack_or_nested(indices, input)
    return torch.return_types.min((values, indices))


@NestedTensorFuncRegistry.implement(torch.mean)
def mean(
    input,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    r"""
    .. note:: If the `input` tensor is empty, ``torch.mean()`` returns ``nan``.
    See also [torch.mean][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.mean(nt, dim=1), torch.mean(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return torch.sum(input, dtype=dtype) / input.numel()

    if isinstance(dim, (list, tuple)):
        if len(dim) == 1:
            dim = dim[0]
        else:
            dims = tuple(_normalize_dim(d, input.dim()) for d in dim)
            tensor = input.tensor
            mask = input.mask
            valid = mask if not input.mask_value else ~mask
            while valid.dim() < tensor.dim():
                valid = valid.unsqueeze(-1)
            data = tensor.to(dtype=dtype) if dtype is not None else tensor
            data = torch.where(valid, data, torch.zeros_like(data))
            summed = torch.sum(data, dim=dims, keepdim=keepdim)
            count = torch.sum(valid, dim=dims, keepdim=keepdim)
            return summed / count.to(dtype=summed.dtype)

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        output = torch.stack([t.mean(dtype=dtype) for t in input._storage])
        if keepdim:
            return output.unsqueeze(0 if input.batch_first else 1)
        return output

    dim_adj = _translate_dim(input, dim)
    ret = [t.mean(dim=dim_adj, keepdim=keepdim, dtype=dtype) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.nanmean)
def nanmean(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    r"""
    Computes the mean of all `non-NaN` elements along the specified dimensions.
    See also [torch.nanmean][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.nanmean(nt, dim=1), torch.nanmean(nt.tensor, dim=1))
        True
    """
    if dim is None:
        total: Tensor | None = None
        count: Tensor | None = None
        for t in input._storage:
            data = t.to(dtype=dtype) if dtype is not None else t
            part_sum = torch.nansum(data)
            part_count = torch.sum(~torch.isnan(data))
            total = part_sum if total is None else total + part_sum
            count = part_count if count is None else count + part_count
        if total is None or count is None:
            return torch.nanmean(input.tensor, dim=None, keepdim=keepdim, dtype=dtype)
        output = total / count.to(dtype=total.dtype)
        if keepdim:
            return output.reshape((1,) * input.dim())
        return output

    batch_dim = _get_batch_dim(input)
    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        if dim_int == batch_dim:
            output = torch.stack([torch.nanmean(t, dim=None, dtype=dtype) for t in input._storage])
            if keepdim:
                return output.unsqueeze(0 if input.batch_first else 1)
            return output

        dim_adj = _translate_dim(input, dim_int)
        ret = [torch.nanmean(t, dim=dim_adj, keepdim=keepdim, dtype=dtype) for t in input._storage]
        return _stack_or_nested(ret, input)

    dims = tuple(dim)
    if len(dims) == 1:
        return torch.nanmean(input, dim=dims[0], keepdim=keepdim, dtype=dtype)
    dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
    if batch_dim in dims_norm:
        raise NotImplementedError("nanmean over the batch dimension and other dims is not supported for NestedTensor.")
    dims_adj = _translate_dims(input, dims_norm)
    ret = [torch.nanmean(t, dim=dims_adj, keepdim=keepdim, dtype=dtype) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.nansum)
def nansum(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    r"""
    Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.
    See also [torch.nansum][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.nansum(nt, dim=1), torch.nansum(nt.tensor, dim=1))
        True
    """
    if dim is None:
        output = torch.stack([torch.nansum(t, dtype=dtype) for t in input._storage]).sum()
        if keepdim:
            return output.reshape((1,) * input.dim())
        return output

    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        batch_dim = _get_batch_dim(input)
        if dim_int == batch_dim:
            output = torch.stack([torch.nansum(t, dtype=dtype) for t in input._storage])
            if keepdim:
                return output.unsqueeze(0 if input.batch_first else 1)
            return output

        dim_adj = _translate_dim(input, dim_int)
        ret = [torch.nansum(t, dim=dim_adj, keepdim=keepdim, dtype=dtype) for t in input._storage]
        return _stack_or_nested(ret, input)

    dims = tuple(dim)
    if len(dims) == 1:
        return torch.nansum(input, dim=dims[0], keepdim=keepdim, dtype=dtype)
    dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
    tensor = input.tensor
    mask = input.mask
    valid = mask if not input.mask_value else ~mask
    while valid.dim() < tensor.dim():
        valid = valid.unsqueeze(-1)
    zeros = torch.zeros_like(tensor)
    data = torch.where(valid, tensor, zeros)
    return torch.nansum(data, dim=dims_norm, keepdim=keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.var)
def var(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
):
    r"""
    Calculates the variance over the dimensions specified by :attr:`dim`. :attr:`dim` can be a single dimension, list of
    dimensions, or ``None`` to reduce over all dimensions.
    See also [torch.var][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.var(nt, dim=1), torch.var(nt.tensor, dim=1))
        True
    """
    if dim is None:
        flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
        output = torch.var(flat, correction=correction)
        if keepdim:
            return output.reshape((1,) * input.dim())
        return output
    batch_dim = _get_batch_dim(input)
    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        if dim_int == batch_dim:
            output = torch.stack([torch.var(t, dim=None, correction=correction) for t in input._storage])
            if keepdim:
                return output.unsqueeze(0 if input.batch_first else 1)
            return output

        dim_adj = _translate_dim(input, dim_int)
        ret = [torch.var(t, dim=dim_adj, correction=correction, keepdim=keepdim) for t in input._storage]
        return _stack_or_nested(ret, input)

    dims = tuple(dim)
    if len(dims) == 1:
        return torch.var(input, dim=dims[0], correction=correction, keepdim=keepdim)
    dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
    if batch_dim in dims_norm:
        raise NotImplementedError("var over the batch dimension and other dims is not supported for NestedTensor.")
    dims_adj = _translate_dims(input, dims_norm)
    ret = [torch.var(t, dim=dims_adj, correction=correction, keepdim=keepdim) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.std)
def std(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
):
    r"""
    Calculates the standard deviation over the dimensions specified by :attr:`dim`.
    See also [torch.std][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.std(nt, dim=1), torch.std(nt.tensor, dim=1))
        True
    """
    if dim is None:
        flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
        output = torch.std(flat, correction=correction)
        if keepdim:
            return output.reshape((1,) * input.dim())
        return output
    batch_dim = _get_batch_dim(input)
    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        if dim_int == batch_dim:
            output = torch.stack([torch.std(t, dim=None, correction=correction) for t in input._storage])
            if keepdim:
                return output.unsqueeze(0 if input.batch_first else 1)
            return output

        dim_adj = _translate_dim(input, dim_int)
        ret = [torch.std(t, dim=dim_adj, correction=correction, keepdim=keepdim) for t in input._storage]
        return _stack_or_nested(ret, input)

    dims = tuple(dim)
    if len(dims) == 1:
        return torch.std(input, dim=dims[0], correction=correction, keepdim=keepdim)
    dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
    if batch_dim in dims_norm:
        raise NotImplementedError("std over the batch dimension and other dims is not supported for NestedTensor.")
    dims_adj = _translate_dims(input, dims_norm)
    ret = [torch.std(t, dim=dims_adj, correction=correction, keepdim=keepdim) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.var_mean)
def var_mean(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
):
    r"""
    Calculates the variance and mean over the dimensions specified by :attr:`dim`.
    See also [torch.var_mean][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out = torch.var_mean(nt, dim=1)
        >>> ref = torch.var_mean(nt.tensor, dim=1)
        >>> torch.allclose(out[0], ref[0]) and torch.allclose(out[1], ref[1])
        True
    """
    if dim is None:
        flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
        variance, mean_value = torch.var_mean(flat, correction=correction)
        if keepdim:
            shape = (1,) * input.dim()
            return variance.reshape(shape), mean_value.reshape(shape)
        return variance, mean_value
    batch_dim = _get_batch_dim(input)
    if isinstance(dim, int):
        dim_int = _normalize_dim(dim, input.dim())
        if dim_int == batch_dim:
            vars_ = []
            means = []
            for t in input._storage:
                v, m = torch.var_mean(t, dim=None, correction=correction)
                vars_.append(v)
                means.append(m)
            vars_tensor = torch.stack(vars_)
            means_tensor = torch.stack(means)
            if keepdim:
                vars_tensor = vars_tensor.unsqueeze(0 if input.batch_first else 1)
                means_tensor = means_tensor.unsqueeze(0 if input.batch_first else 1)
            return vars_tensor, means_tensor

        dim_adj = _translate_dim(input, dim_int)
        vars_ = []
        means = []
        for t in input._storage:
            v, m = torch.var_mean(t, dim=dim_adj, correction=correction, keepdim=keepdim)
            vars_.append(v)
            means.append(m)
        return _stack_or_nested(vars_, input), _stack_or_nested(means, input)

    dims = tuple(dim)
    if len(dims) == 1:
        return torch.var_mean(input, dim=dims[0], correction=correction, keepdim=keepdim)
    dims_norm = tuple(_normalize_dim(d, input.dim()) for d in dims)
    if batch_dim in dims_norm:
        raise NotImplementedError("var_mean over the batch dimension and other dims is not supported for NestedTensor.")
    dims_adj = _translate_dims(input, dims_norm)
    vars_ = []
    means = []
    for t in input._storage:
        v, m = torch.var_mean(t, dim=dims_adj, correction=correction, keepdim=keepdim)
        vars_.append(v)
        means.append(m)
    return _stack_or_nested(vars_, input), _stack_or_nested(means, input)


@NestedTensorFuncRegistry.implement(torch.permute)
def permute(input: NestedTensor, dims: Sequence[int]) -> NestedTensor:
    r"""
    Returns a view of the original tensor :attr:`input` with its dimensions permuted.
    See also [torch.permute][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]))  # noqa: E501
        >>> out = torch.permute(nt, (0, 2, 1))
        >>> ref = torch.permute(nt.tensor, (0, 2, 1))
        >>> torch.equal(out, ref)
        True
    """
    if len(dims) != input.dim():
        raise ValueError(f"Expected {input.dim()} dimensions, got {len(dims)}")

    dim_count = input.dim()
    normalized_dims = tuple(d if d >= 0 else d + dim_count for d in dims)
    batch_dim = _get_batch_dim(input)
    if set(normalized_dims) != set(range(dim_count)):
        raise ValueError(f"Invalid permutation dims {dims} for shape with {dim_count} dims")
    if normalized_dims[batch_dim] != batch_dim:
        raise ValueError("Permuting the batch dimension is not supported for NestedTensor.")

    tensor_dims = [(_translate_dim(input, d)) for d in normalized_dims if d != batch_dim]
    return _map_storage(input, lambda t: t.permute(*tensor_dims))


@NestedTensorFuncRegistry.implement(torch.clamp)
def clamp(input: NestedTensor, min=None, max=None):
    r"""
    Clamps all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]`.
    See also [torch.clamp][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([3.0, -4.0]))
        >>> torch.allclose(torch.clamp(nt, min=0.0, max=1.0), torch.clamp(nt.tensor, min=0.0, max=1.0))
        True
    """
    from .nested_tensor import NestedTensor

    min_storage = _broadcast_storage(input, min) if min is not None else None
    max_storage = _broadcast_storage(input, max) if max is not None else None
    storage = []
    for i, t in enumerate(input._storage):
        min_t = None if min_storage is None else _as_tensor_like(min_storage[i], t)
        max_t = None if max_storage is None else _as_tensor_like(max_storage[i], t)
        storage.append(torch.clamp(t, min=min_t, max=max_t))
    return NestedTensor(storage, **input._state)


@NestedTensorFuncRegistry.implement(torch.clamp_min)
def clamp_min(input: NestedTensor, min):
    r"""
    Clamps all elements in input to be at least min.
    See also [torch.clamp_min][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(torch.clamp_min(nt, 1.5), torch.clamp_min(nt.tensor, 1.5))
        True
    """
    return _binary_op_maybe_tensor(input, min, torch.clamp_min)


@NestedTensorFuncRegistry.implement(torch.clamp_max)
def clamp_max(input: NestedTensor, max):
    r"""
    Clamps all elements in input to be at most max.
    See also [torch.clamp_max][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(torch.clamp_max(nt, 1.5), torch.clamp_max(nt.tensor, 1.5))
        True
    """
    return _binary_op_maybe_tensor(input, max, torch.clamp_max)


@NestedTensorFuncRegistry.implement(torch.clip)
def clip(input: NestedTensor, min=None, max=None):
    r"""
    Alias for :func:`torch.clamp`.
    See also [torch.clip][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([-1.0, 2.0]), torch.tensor([3.0, -4.0]))
        >>> torch.allclose(torch.clip(nt, min=0.0, max=1.0), torch.clip(nt.tensor, min=0.0, max=1.0))
        True
    """
    return torch.clamp(input, min=min, max=max)


@NestedTensorFuncRegistry.implement(torch.reshape)
def reshape(input: NestedTensor, shape: Sequence[int]) -> NestedTensor:
    r"""
    Returns a tensor with the same data and number of elements as :attr:`input`, but with the specified shape. When
    possible, the returned tensor will be a view of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and
    inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing
    behavior.
    See also [torch.reshape][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 6.0, 7.0, 8.0]))
        >>> out = torch.reshape(nt, (2, 2, 2))
        >>> ref = torch.reshape(nt.tensor, (2, 2, 2))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if not input._storage:
        return NestedTensor([], **input._state)
    view_shapes = input._view_shapes(shape)
    reshaped_tensors = [t.reshape(s) for t, s in zip(input._storage, view_shapes)]
    return NestedTensor(reshaped_tensors, **input._state)


@NestedTensorFuncRegistry.implement(torch.softmax)
def softmax(input: NestedTensor, dim: int, dtype: torch.dtype | None = None) -> NestedTensor:
    r"""
    Applies a softmax function over the specified dimension. See also [torch.softmax][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> out = torch.softmax(nt, dim=-1)
        >>> ref = torch.softmax(nt.tensor, dim=-1)
        >>> torch.allclose(out, ref)
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="softmax")
    concat_dim = _concat_dim_for_tensor_dim(input, dim_adj)
    if concat_dim is None:
        return _map_storage(input, lambda t: torch.softmax(t, dim=dim_adj, dtype=dtype))
    return _concat_apply_same_shape(input, lambda t: torch.softmax(t, dim=concat_dim, dtype=dtype))


@NestedTensorFuncRegistry.implement(torch.log_softmax)
def log_softmax(input: NestedTensor, dim: int, dtype: torch.dtype | None = None) -> NestedTensor:
    r"""
    Applies log-softmax over the specified dimension. See also [torch.log_softmax][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> out = torch.log_softmax(nt, dim=-1)
        >>> ref = torch.log_softmax(nt.tensor, dim=-1)
        >>> torch.allclose(out, ref)
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="log_softmax")
    concat_dim = _concat_dim_for_tensor_dim(input, dim_adj)
    if concat_dim is None:
        return _map_storage(input, lambda t: torch.log_softmax(t, dim=dim_adj, dtype=dtype))
    return _concat_apply_same_shape(input, lambda t: torch.log_softmax(t, dim=concat_dim, dtype=dtype))


@NestedTensorFuncRegistry.implement(torch.stack)
def stack(*args, **kwargs):
    r"""
    Concatenates a sequence of tensors along a new dimension.
    See also [torch.stack][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.stack((nt, nt), dim=0)
        >>> ref = NestedTensor(
        ...     torch.stack((nt[0], nt[0]), dim=0),
        ...     torch.stack((nt[1], nt[1]), dim=0),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    tensors = args[0] if args else ()
    dim = kwargs.get("dim", 0)
    if dim != 0:
        raise NotImplementedError(f"NestedTensor only supports stack when dim=0, but got {dim}")
    if not isinstance(tensors, (tuple, list)) or not tensors:
        raise ValueError("Expected a non-empty sequence of NestedTensor objects.")
    from .nested_tensor import NestedTensor

    if not all(isinstance(t, NestedTensor) for t in tensors):
        raise NotImplementedError("torch.stack for NestedTensor requires all inputs to be NestedTensor.")
    first: NestedTensor = tensors[0]
    if any(
        (t.batch_first != first.batch_first)
        or (t.padding_value != first.padding_value)
        or (t.mask_value != first.mask_value)
        for t in tensors[1:]
    ):
        raise ValueError("All NestedTensor inputs must share batch_first, padding_value, and mask_value to stack.")
    if any(len(t) != len(first) for t in tensors[1:]):
        raise ValueError("All NestedTensor inputs must have the same batch length to stack.")
    state = first._build_state(return_dtype=True, return_device=True, return_requires_grad=False)
    storage = [torch.stack([t._storage[i] for t in tensors], dim=0) for i in range(len(first))]
    return NestedTensor(storage, **state)


@NestedTensorFuncRegistry.implement(torch.unbind)
def unbind(input: NestedTensor, dim: int = 0):
    r"""
    Removes a tensor dimension.
    See also [torch.unbind][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.unbind(nt, dim=0)
        >>> torch.equal(out[0], nt[0]) and torch.equal(out[1], nt[1])
        True
    """
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim != batch_dim:
        raise NotImplementedError("torch.unbind for NestedTensor only supports unbinding along the batch dimension.")
    return input._storage


@NestedTensorFuncRegistry.implement(torch.split)
def split(input: NestedTensor, split_size_or_sections, dim: int = 0):
    r"""
    Splits the tensor into chunks. Each chunk is a view of the original tensor.
    See also [torch.split][].

    Supports splitting along both the batch dimension and non-batch dimensions.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor

        Split along batch dimension:
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.split(nt, 1, dim=0)
        >>> ref = (NestedTensor(nt[0]), NestedTensor(nt[1]))
        >>> torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
        True

        Split along feature dimension:
        >>> a = torch.randn(3, 6)
        >>> b = torch.randn(5, 6)
        >>> nt = NestedTensor(a, b)
        >>> parts = torch.split(nt, 2, dim=-1)
        >>> len(parts)
        3
        >>> torch.equal(parts[0][0], a[:, :2]) and torch.equal(parts[0][1], b[:, :2])
        True

        Method-style call:
        >>> parts = nt.split(3, dim=-1)
        >>> len(parts)
        2
        >>> torch.equal(parts[0][0], a[:, :3]) and torch.equal(parts[0][1], b[:, :3])
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)

    # ── Batch dim split ──
    if dim == batch_dim:
        if isinstance(split_size_or_sections, int):
            split_size = split_size_or_sections
            if split_size <= 0:
                raise ValueError("split_size must be a positive integer.")
            storage = input._storage
            return tuple(
                NestedTensor(storage[i : i + split_size], **input._state)  # noqa: E203
                for i in range(0, len(storage), split_size)
            )

        if not isinstance(split_size_or_sections, (list, tuple)):
            raise TypeError(
                f"split_size_or_sections must be int or a sequence of ints, got {type(split_size_or_sections)}"
            )

        storage = input._storage
        chunks = []
        start = 0
        for section in split_size_or_sections:
            if section < 0:
                raise ValueError("split sections must be non-negative.")
            end = start + int(section)
            chunks.append(NestedTensor(storage[start:end], **input._state))
            start = end
        if start != len(storage):
            raise ValueError("split sections do not sum to the NestedTensor batch size.")
        return tuple(chunks)

    # ── Non-batch dim split ──
    storage = input._storage
    if not storage:
        return (NestedTensor([], **input._state),)

    elem_dim = _translate_dim(input, dim)
    split_results = [torch.split(t, split_size_or_sections, dim=elem_dim) for t in storage]

    num_chunks = len(split_results[0])
    return tuple(
        NestedTensor([split_results[i][k] for i in range(len(storage))], **input._state) for k in range(num_chunks)
    )


@NestedTensorFuncRegistry.implement(torch.chunk)
def chunk(input: NestedTensor, chunks: int, dim: int = 0):
    r"""
    Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
    See also [torch.chunk][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor

        Chunk along batch dimension:
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.chunk(nt, 2, dim=0)
        >>> ref = (NestedTensor(nt[0]), NestedTensor(nt[1]))
        >>> torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
        True

        Chunk along feature dimension:
        >>> a = torch.randn(3, 6)
        >>> b = torch.randn(5, 6)
        >>> nt = NestedTensor(a, b)
        >>> parts = torch.chunk(nt, 3, dim=-1)
        >>> len(parts)
        3
        >>> parts[0][0].shape
        torch.Size([3, 2])
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)

    if chunks <= 0:
        raise ValueError("chunks must be a positive integer.")

    # ── Batch dim chunk ──
    if dim == batch_dim:
        storage = input._storage
        if not storage:
            return ()
        chunk_size = (len(storage) + chunks - 1) // chunks
        return tuple(
            NestedTensor(storage[i : i + chunk_size], **input._state)  # noqa: E203
            for i in range(0, len(storage), chunk_size)
        )

    # ── Non-batch dim chunk ──
    storage = input._storage
    if not storage:
        return ()

    elem_dim = _translate_dim(input, dim)
    chunk_results = [torch.chunk(t, chunks, dim=elem_dim) for t in storage]

    num_chunks = len(chunk_results[0])
    return tuple(
        NestedTensor([chunk_results[i][k] for i in range(len(storage))], **input._state) for k in range(num_chunks)
    )


@NestedTensorFuncRegistry.implement(torch.sub)
def sub(input: NestedTensor, other: NestedTensor | Tensor, *, alpha: float = 1) -> NestedTensor:
    r"""
    Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.
    See also [torch.sub][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(torch.sub(nt, 1.5), torch.sub(nt.tensor, 1.5))
        True
    """
    return _binary_op_maybe_tensor(input, other, lambda x, y: torch.sub(x, y, alpha=alpha))


@NestedTensorFuncRegistry.implement(torch.sum)
def sum(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    r"""
    Returns the sum of all elements in the :attr:`input` tensor.
    See also [torch.sum][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.sum(nt, dim=1), torch.sum(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.sum, dtype=dtype)
    if isinstance(dim, int):
        return _reduce_dim(input, torch.sum, dim, keepdim, dtype=dtype)

    dims = tuple(dim)
    if len(dims) == 1:
        return _reduce_dim(input, torch.sum, dims[0], keepdim, dtype=dtype)
    return _reduce_dims_masked(input, dims, torch.sum, keepdim, dtype=dtype, fill_value=0)


@NestedTensorFuncRegistry.implement(torch.transpose)
def transpose(input: NestedTensor, dim0: int, dim1: int) -> NestedTensor:
    r"""
    Returns a tensor that is a transposed version of :attr:`input`.
    See also [torch.transpose][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.transpose(nt, 1, 2)
        >>> ref = torch.transpose(nt.tensor, 1, 2)
        >>> torch.equal(out, ref)
        True
    """
    ndims = input.dim()
    if dim0 < 0:
        dim0 += ndims
    if dim1 < 0:
        dim1 += ndims

    if dim0 < 0 or dim0 >= ndims or dim1 < 0 or dim1 >= ndims:
        raise IndexError(f"Dimension out of range for NestedTensor with {ndims} dims: ({dim0}, {dim1})")

    batch_dim = _get_batch_dim(input)
    if dim0 == batch_dim or dim1 == batch_dim:
        raise ValueError("Cannot transpose the batch dimension for NestedTensor.")
    tensor_dim0 = _translate_dim(input, dim0)
    tensor_dim1 = _translate_dim(input, dim1)
    return _map_storage(input, lambda t: t.transpose(tensor_dim0, tensor_dim1))


@NestedTensorFuncRegistry.implement(torch.where)
def where(condition, input, other):
    r"""
    Return a tensor of elements selected from either :attr:`input` or :attr:`other`, depending on :attr:`condition`.
    See also [torch.where][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> cond = nt > 2
        >>> out = torch.where(cond, nt, 0.0)
        >>> ref = torch.where(cond.tensor, nt.tensor, 0.0)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    def to_nested(value, ref: NestedTensor, *, dtype=None) -> NestedTensor:
        if isinstance(value, NestedTensor):
            return value.to(dtype=dtype) if dtype is not None else value
        if isinstance(value, Tensor) and value.shape == ref.shape:
            return ref.nested_like(value).to(dtype=dtype)
        storage = []
        for t in ref._storage:
            vt = torch.as_tensor(value, device=t.device)
            if dtype is not None:
                vt = vt.to(dtype)
            try:
                vt = vt.expand_as(t)
            except RuntimeError:
                vt = vt.expand(t.shape)
            storage.append(vt)
        return NestedTensor(storage, **ref._state)

    ref = next((v for v in (input, other, condition) if isinstance(v, NestedTensor)), None)
    if ref is None:
        return torch.where(condition, input, other)

    cond_nt = to_nested(condition, ref, dtype=torch.bool)
    input_nt = to_nested(input, ref)
    other_nt = to_nested(other, ref)
    for nested in (cond_nt, input_nt, other_nt):
        if len(nested) != len(ref):
            raise ValueError(f"NestedTensor batch length mismatch: {len(ref)} vs {len(nested)}")
    return NestedTensor(
        (torch.where(c, x, y) for c, x, y in zip(cond_nt._storage, input_nt._storage, other_nt._storage)),
        **ref._state,
    )


@NestedTensorFuncRegistry.implement(torch.flatten)
def flatten(input: NestedTensor, start_dim: int = 0, end_dim: int = -1):
    r"""
    Flattens :attr:`input` by reshaping it into a one-dimensional tensor. If :attr:`start_dim` or :attr:`end_dim` are
    passed, only dimensions starting with :attr:`start_dim` and ending with :attr:`end_dim` are flattened.
    See also [torch.flatten][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out = torch.flatten(nt, start_dim=1)
        >>> ref = torch.flatten(nt.tensor, start_dim=1)
        >>> torch.equal(out, ref)
        True
    """
    ndims = input.dim()
    start = start_dim if start_dim >= 0 else start_dim + ndims
    end = end_dim if end_dim >= 0 else end_dim + ndims
    if start < 0 or end < 0 or start >= ndims or end >= ndims:
        raise IndexError(f"start_dim and end_dim must be in range [0, {ndims}), got ({start_dim}, {end_dim})")
    if start > end:
        raise ValueError(f"start_dim must be <= end_dim, got ({start_dim}, {end_dim})")

    batch_dim = _get_batch_dim(input)
    if start <= batch_dim <= end:
        return torch.flatten(input.tensor, start_dim=start_dim, end_dim=end_dim)

    start_adj = _translate_dim(input, start)
    end_adj = _translate_dim(input, end)
    return _map_storage(input, lambda t: torch.flatten(t, start_dim=start_adj, end_dim=end_adj))


@NestedTensorFuncRegistry.implement(torch.flip)
def flip(input: NestedTensor, dims: Sequence[int]) -> NestedTensor:
    r"""
    Reverse the order of an n-D tensor along given axis in dims.
    See also [torch.flip][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.flip(nt, dims=(1,))
        >>> ref = NestedTensor(
        ...     torch.flip(nt[0], dims=(0,)),
        ...     torch.flip(nt[1], dims=(0,)),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    dims_adj = _translate_dims(input, dims)
    return _map_storage(input, lambda t: torch.flip(t, dims=dims_adj))


@NestedTensorFuncRegistry.implement(torch.roll)
def roll(input: NestedTensor, shifts, dims=None):
    r"""
    Roll the tensor :attr:`input` along the given dimension(s). Elements that are shifted beyond the last position are
    re-introduced at the first position. If :attr:`dims` is `None`, the tensor will be flattened before rolling and then
    restored to the original shape.
    See also [torch.roll][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.roll(nt, shifts=1, dims=1)
        >>> ref = NestedTensor(
        ...     torch.roll(nt[0], shifts=1, dims=0),
        ...     torch.roll(nt[1], shifts=1, dims=0),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    if dims is None:
        return _map_storage(input, lambda t: torch.roll(t, shifts, dims=None))

    dims_adj: int | Tuple[int, ...]
    if isinstance(dims, int):
        dims_adj = _translate_dim(input, dims)
    else:
        dims_adj = _translate_dims(input, dims)
    return _map_storage(input, lambda t: torch.roll(t, shifts, dims=dims_adj))


@NestedTensorFuncRegistry.implement(torch.rot90)
def rot90(input: NestedTensor, k: int = 1, dims: Sequence[int] = (0, 1)) -> NestedTensor:
    r"""
    Rotate an n-D tensor by 90 degrees in the plane specified by dims axis.
    See also [torch.rot90][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.rot90(nt, k=1, dims=(1, 2))
        >>> ref = torch.rot90(nt.tensor, k=1, dims=(1, 2))
        >>> torch.equal(out, ref)
        True
    """
    if len(dims) != 2:
        raise ValueError("rot90 dims must be a sequence of two dimensions.")

    dims_adj = _translate_dims(input, dims)
    return _map_storage(input, lambda t: torch.rot90(t, k=k, dims=dims_adj))


@NestedTensorFuncRegistry.implement(torch.unflatten)
def unflatten(input: NestedTensor, dim: int, sizes):
    r"""
    Expands a dimension of the input tensor over multiple dimensions.
    See also [torch.unflatten][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> out = torch.unflatten(nt, 1, (1, 2))
        >>> ref = torch.unflatten(nt.tensor, 1, (1, 2))
        >>> torch.equal(out, ref)
        True
    """
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim <= batch_dim:
        raise NotImplementedError("unflatten at or before the batch dimension is not supported for NestedTensor.")
    dim_adj = _translate_dim(input, dim)
    return _map_storage(input, lambda t: torch.unflatten(t, dim_adj, sizes))


@NestedTensorFuncRegistry.implement(torch.squeeze)
def squeeze(input: NestedTensor, dim: int | None = None):
    r"""
    Returns a tensor with all specified dimensions of :attr:`input` of size `1` removed.
    See also [torch.squeeze][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0], [5.0]]))
        >>> out = torch.squeeze(nt, dim=2)
        >>> ref = torch.squeeze(nt.tensor, dim=2)
        >>> torch.equal(out, ref)
        True
    """
    if dim is None:
        return _map_storage(input, lambda t: t.squeeze())
    dim_norm = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim_norm <= batch_dim:
        raise ValueError("Cannot squeeze the batch dimension or dimensions before it for NestedTensor.")
    dim_adj = _translate_dim(input, dim_norm)
    return _map_storage(input, lambda t: t.squeeze(dim_adj))


@NestedTensorFuncRegistry.implement(torch.unsqueeze)
def unsqueeze(input: NestedTensor, dim: int):
    r"""
    Returns a new tensor with a dimension of size one inserted at the specified position.
    See also [torch.unsqueeze][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.unsqueeze(nt, dim=2)
        >>> ref = torch.unsqueeze(nt.tensor, dim=2)
        >>> torch.equal(out, ref)
        True
    """
    ndims = input.dim()
    if dim < 0:
        dim += ndims + 1
    if dim < 0 or dim > ndims:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-ndims - 1}, {ndims}], but got {dim})")

    batch_dim = _get_batch_dim(input)
    if dim <= batch_dim:
        raise ValueError("Cannot unsqueeze at or before the batch dimension for NestedTensor.")

    dim_adj = dim - 1
    return _map_storage(input, lambda t: t.unsqueeze(dim_adj))


@NestedTensorFuncRegistry.implement(torch.moveaxis)
def moveaxis(input: NestedTensor, source, destination):
    r"""
    Alias for :func:`torch.movedim`.
    See also [torch.moveaxis][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.moveaxis(nt, 1, 2)
        >>> ref = torch.moveaxis(nt.tensor, 1, 2)
        >>> torch.equal(out, ref)
        True
    """
    ndims = input.dim()
    src_axes = (source,) if isinstance(source, int) else tuple(source)
    dst_axes = (destination,) if isinstance(destination, int) else tuple(destination)
    if len(src_axes) != len(dst_axes):
        raise ValueError("moveaxis: `source` and `destination` must have the same number of dimensions")

    src_norm = [_normalize_dim(d, ndims) for d in src_axes]
    dst_norm = [_normalize_dim(d, ndims) for d in dst_axes]

    if len(set(src_norm)) != len(src_norm) or len(set(dst_norm)) != len(dst_norm):
        raise ValueError("moveaxis: duplicate dimensions are not allowed")

    order = [i for i in range(ndims) if i not in src_norm]
    for dest, src in sorted(zip(dst_norm, src_norm)):
        order.insert(dest, src)

    return input.permute(*order)


@NestedTensorFuncRegistry.implement(torch.swapaxes)
def swapaxes(input: NestedTensor, axis0: int, axis1: int):
    r"""
    Alias for :func:`torch.transpose`.
    See also [torch.swapaxes][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.swapaxes(nt, 1, 2)
        >>> ref = torch.swapaxes(nt.tensor, 1, 2)
        >>> torch.equal(out, ref)
        True
    """
    axis0 = _normalize_dim(axis0, input.dim())
    axis1 = _normalize_dim(axis1, input.dim())
    batch_dim = _get_batch_dim(input)
    if axis0 == batch_dim or axis1 == batch_dim:
        raise ValueError("Cannot swap the batch dimension for NestedTensor.")
    axis0_adj = _translate_dim(input, axis0)
    axis1_adj = _translate_dim(input, axis1)
    return _map_storage(input, lambda t: torch.swapaxes(t, axis0_adj, axis1_adj))


@NestedTensorFuncRegistry.implement(torch.swapdims)
def swapdims(input: NestedTensor, dim0: int, dim1: int):
    r"""
    Alias for :func:`torch.transpose`.
    See also [torch.swapdims][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.swapdims(nt, 1, 2)
        >>> ref = torch.swapdims(nt.tensor, 1, 2)
        >>> torch.equal(out, ref)
        True
    """
    return torch.swapaxes(input, dim0, dim1)


@NestedTensorFuncRegistry.implement(torch.argmax)
def argmax(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the indices of the maximum value of all elements in the :attr:`input` tensor.
    See also [torch.argmax][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.equal(torch.argmax(nt, dim=1), torch.argmax(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return torch.stack([t.reshape(-1).argmax() for t in input._storage])
    if dim < 0:
        dim += input.dim()
    if (input.batch_first and dim == 0) or (not input.batch_first and dim == 1):
        output = torch.stack([t.reshape(-1).argmax() for t in input._storage])
        if keepdim:
            return output.unsqueeze(0 if input.batch_first else 1)
        return output
    if input.batch_first or dim != 0:
        dim -= 1
    ret = [t.argmax(dim=dim, keepdim=keepdim) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.argmin)
def argmin(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the indices of the minimum value(s) of the flattened tensor or along a dimension This is the second value
    returned by :meth:`torch.min`. See its documentation for the exact semantics of this method.
    See also [torch.argmin][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.equal(torch.argmin(nt, dim=1), torch.argmin(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return torch.stack([t.reshape(-1).argmin() for t in input._storage])
    if dim < 0:
        dim += input.dim()
    if (input.batch_first and dim == 0) or (not input.batch_first and dim == 1):
        output = torch.stack([t.reshape(-1).argmin() for t in input._storage])
        if keepdim:
            return output.unsqueeze(0 if input.batch_first else 1)
        return output
    if input.batch_first or dim != 0:
        dim -= 1
    ret = [t.argmin(dim=dim, keepdim=keepdim) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.kthvalue)
def kthvalue(input: NestedTensor, k: int, dim: int = -1, keepdim: bool = False):
    r"""
    Returns a namedtuple ``(values, indices)`` where ``values`` is the :attr:`k` th smallest element of each row of the
    :attr:`input` tensor in the given dimension :attr:`dim`. And ``indices`` is the index location of each element
    found.
    See also [torch.kthvalue][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.kthvalue(nt, k=1, dim=1)
        >>> ref = torch.kthvalue(nt.tensor, k=1, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        values = []
        indices = []
        for t in input._storage:
            v, i = torch.kthvalue(t.reshape(-1), k, dim=0, keepdim=False)
            values.append(v)
            indices.append(i)
        values_tensor = torch.stack(values)
        indices_tensor = torch.stack(indices)
        if keepdim:
            values_tensor = values_tensor.unsqueeze(0 if input.batch_first else 1)
            indices_tensor = indices_tensor.unsqueeze(0 if input.batch_first else 1)
        return values_tensor, indices_tensor

    dim_adj = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        v, i = torch.kthvalue(t, k, dim=dim_adj, keepdim=keepdim)
        values.append(v)
        indices.append(i)
    return _stack_or_nested(values, input), _stack_or_nested(indices, input)


@NestedTensorFuncRegistry.implement(torch.topk)
def topk(input: NestedTensor, k, dim: int | None = None, largest: bool = True, sorted: bool = True):
    r"""
    Returns the :attr:`k` largest elements of the given :attr:`input` tensor along a given dimension.
    See also [torch.topk][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.topk(nt, k=1, dim=1)
        >>> ref = torch.topk(nt.tensor, k=1, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    from .nested_tensor import NestedTensor

    dim = -1 if dim is None else _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise NotImplementedError("topk along the batch dimension is not supported for NestedTensor.")
    dim = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        v, i = torch.topk(t, k, dim=dim, largest=largest, sorted=sorted)
        values.append(v)
        indices.append(i)
    return NestedTensor(values, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(torch.sort)
def sort(input: NestedTensor, dim: int = -1, descending: bool = False, stable: bool | None = None):
    r"""
    Sorts the elements of the :attr:`input` tensor along a given dimension in ascending order by value.
    See also [torch.sort][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.sort(nt, dim=1)
        >>> ref = torch.sort(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise NotImplementedError("sort along the batch dimension is not supported for NestedTensor.")
    dim = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        v, i = torch.sort(t, dim=dim, descending=descending, stable=stable)
        values.append(v)
        indices.append(i)
    return NestedTensor(values, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(torch.argsort)
def argsort(input: NestedTensor, dim: int = -1, descending: bool = False, stable: bool | None = None):
    r"""
    Returns the indices that sort a tensor along a given dimension in ascending order by value.
    See also [torch.argsort][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.equal(torch.argsort(nt, dim=1), torch.argsort(nt.tensor, dim=1))
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise NotImplementedError("argsort along the batch dimension is not supported for NestedTensor.")
    dim = _translate_dim(input, dim)
    stable = False if stable is None else stable
    ret = [torch.argsort(t, dim=dim, descending=descending, stable=stable) for t in input._storage]
    return NestedTensor(ret, **input._state)


@NestedTensorFuncRegistry.implement(torch.median)
def median(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the median of the values in :attr:`input`.
    See also [torch.median][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.median(nt, dim=1)
        >>> ref = torch.median(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if dim is None:
        flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
        return torch.median(flat)

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        values = []
        indices = []
        for t in input._storage:
            v, i = torch.median(t.reshape(-1), dim=0, keepdim=False)
            values.append(v)
            indices.append(i)
        values_tensor = torch.stack(values)
        indices_tensor = torch.stack(indices)
        if keepdim:
            values_tensor = values_tensor.unsqueeze(0 if input.batch_first else 1)
            indices_tensor = indices_tensor.unsqueeze(0 if input.batch_first else 1)
        return values_tensor, indices_tensor

    dim_adj = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        v, i = torch.median(t, dim=dim_adj, keepdim=keepdim)
        values.append(v)
        indices.append(i)
    return _stack_or_nested(values, input), _stack_or_nested(indices, input)


@NestedTensorFuncRegistry.implement(torch.nanmedian)
def nanmedian(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the median of the values in :attr:`input`, ignoring ``NaN`` values.
    See also [torch.nanmedian][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.nanmedian(nt, dim=1)
        >>> ref = torch.nanmedian(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if dim is None:
        flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
        return torch.nanmedian(flat)

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        values = []
        indices = []
        for t in input._storage:
            v, i = torch.nanmedian(t.reshape(-1), dim=0, keepdim=False)
            values.append(v)
            indices.append(i)
        values_tensor = torch.stack(values)
        indices_tensor = torch.stack(indices)
        if keepdim:
            values_tensor = values_tensor.unsqueeze(0 if input.batch_first else 1)
            indices_tensor = indices_tensor.unsqueeze(0 if input.batch_first else 1)
        return values_tensor, indices_tensor

    dim_adj = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        v, i = torch.nanmedian(t, dim=dim_adj, keepdim=keepdim)
        values.append(v)
        indices.append(i)
    return _stack_or_nested(values, input), _stack_or_nested(indices, input)


@NestedTensorFuncRegistry.implement(torch.mode)
def mode(input: NestedTensor, dim: int = -1, keepdim: bool = False):
    r"""
    Returns a namedtuple ``(values, indices)`` where ``values`` is the mode value of each row of the :attr:`input`
    tensor in the given dimension :attr:`dim`, i.e. a value which appears most often in that row, and ``indices`` is the
    index location of each mode value found.
    See also [torch.mode][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.mode(nt, dim=1)
        >>> ref = torch.mode(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        values = []
        indices = []
        for t in input._storage:
            v, i = torch.mode(t.reshape(-1), dim=0, keepdim=False)
            values.append(v)
            indices.append(i)
        values_tensor = torch.stack(values)
        indices_tensor = torch.stack(indices)
        if keepdim:
            values_tensor = values_tensor.unsqueeze(0 if input.batch_first else 1)
            indices_tensor = indices_tensor.unsqueeze(0 if input.batch_first else 1)
        return values_tensor, indices_tensor

    dim_adj = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        v, i = torch.mode(t, dim=dim_adj, keepdim=keepdim)
        values.append(v)
        indices.append(i)
    return _stack_or_nested(values, input), _stack_or_nested(indices, input)


@NestedTensorFuncRegistry.implement(torch.quantile)
def quantile(
    input: NestedTensor,
    q,
    dim: int | None = None,
    keepdim: bool = False,
    interpolation: str = "linear",
):
    r"""
    Computes the q-th quantiles of each row of the :attr:`input` tensor along the dimension :attr:`dim`.
    See also [torch.quantile][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.quantile(nt, q=0.5, dim=1), torch.quantile(nt.tensor, q=0.5, dim=1))
        True
    """
    if dim is None:
        flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
        output = torch.quantile(flat, q, dim=0, keepdim=False, interpolation=interpolation)
        if keepdim:
            output = output.reshape(tuple(output.shape) + (1,) * input.dim())
        return output

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        ret = [
            torch.quantile(t.reshape(-1), q, dim=0, keepdim=False, interpolation=interpolation) for t in input._storage
        ]
        output = torch.stack(ret)
        if keepdim:
            output = output.unsqueeze(0 if input.batch_first else 1)
        return output

    dim_adj = _translate_dim(input, dim)
    ret = [torch.quantile(t, q, dim=dim_adj, keepdim=keepdim, interpolation=interpolation) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.nanquantile)
def nanquantile(input: NestedTensor, q, dim: int | None = None, keepdim: bool = False, interpolation: str = "linear"):
    r"""
    This is a variant of :func:`torch.quantile` that "ignores" ``NaN`` values, computing the quantiles :attr:`q` as if
    ``NaN`` values in :attr:`input` did not exist. If all values in a reduced row are ``NaN`` then the quantiles for
    that reduction will be ``NaN``. See the documentation for :func:`torch.quantile`.
    See also [torch.nanquantile][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.nanquantile(nt, q=0.5, dim=1), torch.nanquantile(nt.tensor, q=0.5, dim=1))
        True
    """
    if dim is None:
        flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
        output = torch.nanquantile(flat, q, dim=0, keepdim=False, interpolation=interpolation)
        if keepdim:
            output = output.reshape(tuple(output.shape) + (1,) * input.dim())
        return output

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        ret = [
            torch.nanquantile(t.reshape(-1), q, dim=0, keepdim=False, interpolation=interpolation)
            for t in input._storage
        ]
        output = torch.stack(ret)
        if keepdim:
            output = output.unsqueeze(0 if input.batch_first else 1)
        return output

    dim_adj = _translate_dim(input, dim)
    ret = [torch.nanquantile(t, q, dim=dim_adj, keepdim=keepdim, interpolation=interpolation) for t in input._storage]
    return _stack_or_nested(ret, input)


@NestedTensorFuncRegistry.implement(torch.cumsum)
def cumsum(input: NestedTensor, dim: int, *, dtype=None):
    r"""
    Returns the cumulative sum of elements of :attr:`input` in the dimension :attr:`dim`.
    See also [torch.cumsum][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.cumsum(nt, dim=1), torch.cumsum(nt.tensor, dim=1))
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="cumsum")
    return _map_storage(input, lambda t: torch.cumsum(t, dim=dim_adj, dtype=dtype))


@NestedTensorFuncRegistry.implement(torch.cumprod)
def cumprod(input: NestedTensor, dim: int, *, dtype=None):
    r"""
    Returns the cumulative product of elements of :attr:`input` in the dimension :attr:`dim`.
    See also [torch.cumprod][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.cumprod(nt, dim=1), torch.cumprod(nt.tensor, dim=1))
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="cumprod")
    return _map_storage(input, lambda t: torch.cumprod(t, dim=dim_adj, dtype=dtype))


@NestedTensorFuncRegistry.implement(torch.logcumsumexp)
def logcumsumexp(input: NestedTensor, dim: int):
    r"""
    Returns the logarithm of the cumulative summation of the exponentiation of elements of :attr:`input` in the
    dimension :attr:`dim`.
    See also [torch.logcumsumexp][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.logcumsumexp(nt, dim=1), torch.logcumsumexp(nt.tensor, dim=1))
        True
    """
    dim_adj = _translate_non_batch_dim(input, dim, name="logcumsumexp")
    return _map_storage(input, lambda t: torch.logcumsumexp(t, dim=dim_adj))


@NestedTensorFuncRegistry.implement(torch.cummax)
def cummax(input: NestedTensor, dim: int):
    r"""
    Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative maximum of elements of :attr:`input`
    in the dimension :attr:`dim`. And ``indices`` is the index location of each maximum value found in the dimension
    :attr:`dim`.
    See also [torch.cummax][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.cummax(nt, dim=1)
        >>> ref = torch.cummax(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise NotImplementedError("cummax along the batch dimension is not supported for NestedTensor.")
    dim_adj = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        v, i = torch.cummax(t, dim=dim_adj)
        values.append(v)
        indices.append(i)
    return NestedTensor(values, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(torch.cummin)
def cummin(input: NestedTensor, dim: int):
    r"""
    Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative minimum of elements of :attr:`input`
    in the dimension :attr:`dim`. And ``indices`` is the index location of each maximum value found in the dimension
    :attr:`dim`.
    See also [torch.cummin][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.cummin(nt, dim=1)
        >>> ref = torch.cummin(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise NotImplementedError("cummin along the batch dimension is not supported for NestedTensor.")
    dim_adj = _translate_dim(input, dim)
    values = []
    indices = []
    for t in input._storage:
        v, i = torch.cummin(t, dim=dim_adj)
        values.append(v)
        indices.append(i)
    return NestedTensor(values, **input._state), NestedTensor(indices, **input._state)


@NestedTensorFuncRegistry.implement(torch.gather)
def gather(input: NestedTensor, dim: int, index):
    r"""
    Gathers values along an axis specified by `dim`.
    See also [torch.gather][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([[0, 1], [1, 0]])
        >>> out = torch.gather(nt, 1, index)
        >>> ref = torch.gather(nt.tensor, 1, index)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    ndims = input.dim()
    dim = dim if dim >= 0 else dim + ndims
    batch_dim = 0 if input.batch_first else 1
    if dim == batch_dim:
        raise NotImplementedError("gather on batch dimension is not supported for NestedTensor.")
    if input.batch_first:
        dim_adj = dim - 1
    else:
        dim_adj = dim if dim < batch_dim else dim - 1
    if isinstance(index, Tensor) and input.shape == index.shape:
        index = input.nested_like(index, strict=False)
    if isinstance(index, NestedTensor):
        if len(input) != len(index):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(index)}")
        return NestedTensor(torch.gather(t, dim_adj, idx) for t, idx in zip(input._storage, index._storage))
    return NestedTensor(torch.gather(t, dim_adj, index) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.scatter)
def scatter(input: NestedTensor, dim: int, index, src):
    r"""
    Out-of-place version of :meth:`torch.Tensor.scatter_`
    See also [torch.scatter][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([[0, 1], [1, 0]])
        >>> out = torch.scatter(nt, 1, index, 0.0)
        >>> ref = torch.scatter(nt.tensor, 1, index, 0.0)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    ndims = input.dim()
    dim = dim if dim >= 0 else dim + ndims
    batch_dim = 0 if input.batch_first else 1
    if dim == batch_dim:
        raise NotImplementedError("scatter on batch dimension is not supported for NestedTensor.")
    if input.batch_first:
        dim_adj = dim - 1
    else:
        dim_adj = dim if dim < batch_dim else dim - 1
    if isinstance(index, Tensor) and input.shape == index.shape:
        index = input.nested_like(index, strict=False)
    if isinstance(src, Tensor) and input.shape == src.shape:
        src = input.nested_like(src, strict=False)
    if isinstance(index, NestedTensor):
        indices = index._storage
        if len(input) != len(index):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(index)}")
    else:
        indices = tuple(index for _ in input._storage)
    if isinstance(src, NestedTensor):
        srcs = src._storage
        if len(input) != len(src):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(src)}")
    else:
        srcs = tuple(src for _ in input._storage)
    return NestedTensor(torch.scatter(t, dim_adj, idx, s) for t, idx, s in zip(input._storage, indices, srcs))


@NestedTensorFuncRegistry.implement(torch.scatter_add)
def scatter_add(input: NestedTensor, dim: int, index, src):
    r"""
    Out-of-place version of :meth:`torch.Tensor.scatter_add_`
    See also [torch.scatter_add][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([[0, 1], [1, 0]])
        >>> src = torch.ones_like(nt.tensor)
        >>> out = torch.scatter_add(nt, 1, index, src)
        >>> ref = torch.scatter_add(nt.tensor, 1, index, src)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    ndims = input.dim()
    dim = dim if dim >= 0 else dim + ndims
    batch_dim = 0 if input.batch_first else 1
    if dim == batch_dim:
        raise NotImplementedError("scatter_add on batch dimension is not supported for NestedTensor.")
    if input.batch_first:
        dim_adj = dim - 1
    else:
        dim_adj = dim if dim < batch_dim else dim - 1
    if isinstance(index, Tensor) and input.shape == index.shape:
        index = input.nested_like(index, strict=False)
    if isinstance(src, Tensor) and input.shape == src.shape:
        src = input.nested_like(src, strict=False)
    if isinstance(index, NestedTensor):
        indices = index._storage
        if len(input) != len(index):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(index)}")
    else:
        indices = tuple(index for _ in input._storage)
    if isinstance(src, NestedTensor):
        srcs = src._storage
        if len(input) != len(src):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(src)}")
    else:
        srcs = tuple(src for _ in input._storage)
    return NestedTensor(torch.scatter_add(t, dim_adj, idx, s) for t, idx, s in zip(input._storage, indices, srcs))


if hasattr(torch, "scatter_reduce"):

    @NestedTensorFuncRegistry.implement(torch.scatter_reduce)
    def scatter_reduce(input: NestedTensor, dim: int, index, src, reduce: str, *, include_self: bool = True):
        r"""
        Out-of-place version of :meth:`torch.Tensor.scatter_reduce_`
        See also [torch.scatter_reduce][].

        Examples:
            >>> import torch
            >>> from danling.tensors import NestedTensor
            >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
            >>> index = torch.tensor([[0, 1], [1, 0]])
            >>> src = torch.ones_like(nt.tensor)
            >>> out = torch.scatter_reduce(nt, 1, index, src, reduce='sum')
            >>> ref = torch.scatter_reduce(nt.tensor, 1, index, src, reduce='sum')
            >>> torch.equal(out, ref)
            True
        """
        from .nested_tensor import NestedTensor

        ndims = input.dim()
        dim = dim if dim >= 0 else dim + ndims
        batch_dim = 0 if input.batch_first else 1
        if dim == batch_dim:
            raise NotImplementedError("scatter_reduce on batch dimension is not supported for NestedTensor.")
        if input.batch_first:
            dim_adj = dim - 1
        else:
            dim_adj = dim if dim < batch_dim else dim - 1
        if isinstance(index, Tensor) and input.shape == index.shape:
            index = input.nested_like(index, strict=False)
        if isinstance(src, Tensor) and input.shape == src.shape:
            src = input.nested_like(src, strict=False)
        if isinstance(index, NestedTensor):
            indices = index._storage
            if len(input) != len(index):
                raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(index)}")
        else:
            indices = tuple(index for _ in input._storage)
        if isinstance(src, NestedTensor):
            srcs = src._storage
            if len(input) != len(src):
                raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(src)}")
        else:
            srcs = tuple(src for _ in input._storage)
        return NestedTensor(
            torch.scatter_reduce(t, dim_adj, idx, s, reduce=reduce, include_self=include_self)
            for t, idx, s in zip(input._storage, indices, srcs)
        )


@NestedTensorFuncRegistry.implement(torch.index_select)
def index_select(input: NestedTensor, dim: int, index: Tensor):
    r"""
    Returns a new tensor which indexes the :attr:`input` tensor along dimension :attr:`dim` using the entries in
    :attr:`index` which is a `LongTensor`.
    See also [torch.index_select][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([0])
        >>> out = torch.index_select(nt, 1, index)
        >>> ref = torch.index_select(nt.tensor, 1, index)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    ndims = input.dim()
    dim = dim if dim >= 0 else dim + ndims
    batch_dim = 0 if input.batch_first else 1
    if dim == batch_dim:
        indices = index.to(dtype=torch.long, device="cpu").tolist()
        return NestedTensor([input._storage[i] for i in indices], **input._state)
    if input.batch_first:
        dim_adj = dim - 1
    else:
        dim_adj = dim if dim < batch_dim else dim - 1
    return NestedTensor(torch.index_select(t, dim_adj, index.to(device=t.device)) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.index_add)
def index_add(input: NestedTensor, dim: int, index, source, *, alpha: float = 1):
    r"""
    Adds the elements of source into input at the specified indices along dim.
    See also [torch.index_add][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([0, 1])
        >>> src = torch.ones_like(nt.tensor)
        >>> out = torch.index_add(nt, 1, index, src)
        >>> ref = torch.index_add(nt.tensor, 1, index, src)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise NotImplementedError("index_add on the batch dimension is not supported for NestedTensor.")
    dim_adj = _translate_dim(input, dim)

    if isinstance(index, Tensor):
        index = index.to(dtype=torch.long)
    if isinstance(source, Tensor) and source.shape == input.shape:
        source = input.nested_like(source, strict=False)

    if isinstance(index, NestedTensor) and len(index) != len(input):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(index)}")
    if isinstance(source, NestedTensor) and len(source) != len(input):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(source)}")

    storage = []
    for i, t in enumerate(input._storage):
        idx = index._storage[i] if isinstance(index, NestedTensor) else index
        src = source._storage[i] if isinstance(source, NestedTensor) else source
        if isinstance(idx, Tensor) and idx.device != t.device:
            idx = idx.to(device=t.device)
        if isinstance(src, Tensor) and src.device != t.device:
            src = src.to(device=t.device)
        storage.append(torch.index_add(t, dim_adj, idx, src, alpha=alpha))
    return NestedTensor(storage, **input._state)


@NestedTensorFuncRegistry.implement(torch.index_copy)
def index_copy(input: NestedTensor, dim: int, index, source):
    r"""
    Copies elements from source into input at the specified indices along dim.
    See also [torch.index_copy][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([0, 1])
        >>> src = torch.ones_like(nt.tensor)
        >>> out = torch.index_copy(nt, 1, index, src)
        >>> ref = torch.index_copy(nt.tensor, 1, index, src)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise NotImplementedError("index_copy on the batch dimension is not supported for NestedTensor.")
    dim_adj = _translate_dim(input, dim)

    if isinstance(index, Tensor):
        index = index.to(dtype=torch.long)
    if isinstance(source, Tensor) and source.shape == input.shape:
        source = input.nested_like(source, strict=False)

    if isinstance(index, NestedTensor) and len(index) != len(input):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(index)}")
    if isinstance(source, NestedTensor) and len(source) != len(input):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(source)}")

    storage = []
    for i, t in enumerate(input._storage):
        idx = index._storage[i] if isinstance(index, NestedTensor) else index
        src = source._storage[i] if isinstance(source, NestedTensor) else source
        if isinstance(idx, Tensor) and idx.device != t.device:
            idx = idx.to(device=t.device)
        if isinstance(src, Tensor) and src.device != t.device:
            src = src.to(device=t.device)
        storage.append(torch.index_copy(t, dim_adj, idx, src))
    return NestedTensor(storage, **input._state)


@NestedTensorFuncRegistry.implement(torch.index_put)
def index_put(input: NestedTensor, indices, values, accumulate: bool = False):
    r"""
    Puts values into the input tensor at the specified indices.
    See also [torch.index_put][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([1.0, 2.0])
        >>> b = torch.tensor([3.0, 4.0, 5.0])
        >>> nt = NestedTensor(a, b)
        >>> idx = torch.tensor([0])
        >>> values = torch.tensor([9.0])
        >>> out = torch.index_put(nt, (idx,), values)
        >>> ref = NestedTensor(torch.index_put(a, (idx,), values), torch.index_put(b, (idx,), values))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if not isinstance(indices, (tuple, list)):
        indices = (indices,)

    if isinstance(values, Tensor) and values.shape == input.shape:
        values = input.nested_like(values, strict=False)

    for idx in indices:
        if isinstance(idx, NestedTensor) and len(idx) != len(input):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(idx)}")
    if isinstance(values, NestedTensor) and len(values) != len(input):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(values)}")

    storage = []
    for i, t in enumerate(input._storage):
        per_tensor_indices = []
        for idx in indices:
            idx_i = idx._storage[i] if isinstance(idx, NestedTensor) else idx
            if isinstance(idx_i, Tensor) and idx_i.device != t.device:
                idx_i = idx_i.to(device=t.device)
            per_tensor_indices.append(idx_i)
        value_i = values._storage[i] if isinstance(values, NestedTensor) else values
        if isinstance(value_i, Tensor) and value_i.device != t.device:
            value_i = value_i.to(device=t.device)
        storage.append(torch.index_put(t, tuple(per_tensor_indices), value_i, accumulate=accumulate))
    return NestedTensor(storage, **input._state)


@NestedTensorFuncRegistry.implement(torch.masked_select)
def masked_select(input: NestedTensor, mask):
    r"""
    Returns a new 1-D tensor which indexes the :attr:`input` tensor according to the boolean mask :attr:`mask` which is
    a `BoolTensor`.
    See also [torch.masked_select][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([1.0, 0.0])
        >>> b = torch.tensor([2.0, 3.0, 4.0])
        >>> nt = NestedTensor(a, b)
        >>> mask = NestedTensor(torch.tensor([True, False]), torch.tensor([True, False, True]))
        >>> out = torch.masked_select(nt, mask)
        >>> ref = NestedTensor(torch.masked_select(a, mask[0]), torch.masked_select(b, mask[1]))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if isinstance(mask, Tensor) and mask.shape == input.shape:
        mask = input.nested_like(mask, strict=False)
    if isinstance(mask, NestedTensor):
        if len(input) != len(mask):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(mask)}")
        return NestedTensor(torch.masked_select(t, m) for t, m in zip(input._storage, mask._storage))
    storage = []
    for t in input._storage:
        m = mask
        if not isinstance(m, Tensor):
            m = torch.as_tensor(m, dtype=torch.bool, device=t.device)
        else:
            m = m.to(device=t.device)
        storage.append(torch.masked_select(t, m))
    return NestedTensor(storage, **input._state)


@NestedTensorFuncRegistry.implement(torch.masked_fill)
def masked_fill(input: NestedTensor, mask, value):
    r"""
    Fills elements of input where mask is True with value.
    See also [torch.masked_fill][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> mask = NestedTensor(torch.tensor([True, False]), torch.tensor([False, True, False]))
        >>> out = torch.masked_fill(nt, mask, 0.0)
        >>> ref = NestedTensor(torch.masked_fill(nt[0], mask[0], 0.0), torch.masked_fill(nt[1], mask[1], 0.0))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if isinstance(mask, Tensor) and mask.shape == input.shape:
        mask = input.nested_like(mask, strict=False)
    if isinstance(mask, NestedTensor):
        if len(input) != len(mask):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(mask)}")
        return NestedTensor(torch.masked_fill(t, m, value) for t, m in zip(input._storage, mask._storage))
    storage = []
    for t in input._storage:
        m = mask
        if not isinstance(m, Tensor):
            m = torch.as_tensor(m, dtype=torch.bool, device=t.device)
        else:
            m = m.to(device=t.device)
        storage.append(torch.masked_fill(t, m, value))
    return NestedTensor(storage, **input._state)


@NestedTensorFuncRegistry.implement(torch.masked_scatter)
def masked_scatter(input: NestedTensor, mask, source):
    r"""
    Copies elements from source into input where mask is True.
    See also [torch.masked_scatter][].

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([1.0, 2.0])
        >>> b = torch.tensor([3.0, 4.0, 5.0])
        >>> nt = NestedTensor(a, b)
        >>> mask = NestedTensor(torch.tensor([True, False]), torch.tensor([False, True, False]))
        >>> src = NestedTensor(torch.tensor([9.0]), torch.tensor([8.0]))
        >>> out = torch.masked_scatter(nt, mask, src)
        >>> ref = NestedTensor(
        ...     torch.masked_scatter(a, mask[0], src[0]),
        ...     torch.masked_scatter(b, mask[1], src[1]),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if isinstance(mask, Tensor) and mask.shape == input.shape:
        mask = input.nested_like(mask, strict=False)
    if isinstance(source, Tensor) and source.shape == input.shape:
        source = input.nested_like(source, strict=False)
    if isinstance(mask, NestedTensor) and len(input) != len(mask):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(mask)}")
    if isinstance(source, NestedTensor) and len(input) != len(source):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(source)}")

    storage = []
    for i, t in enumerate(input._storage):
        m = mask._storage[i] if isinstance(mask, NestedTensor) else mask
        s = source._storage[i] if isinstance(source, NestedTensor) else source
        if isinstance(m, Tensor) and m.device != t.device:
            m = m.to(device=t.device)
        if isinstance(s, Tensor) and s.device != t.device:
            s = s.to(device=t.device)
        storage.append(torch.masked_scatter(t, m, s))
    return NestedTensor(storage, **input._state)


if hasattr(torch, "take_along_dim"):

    @NestedTensorFuncRegistry.implement(torch.take_along_dim)
    def take_along_dim(input: NestedTensor, indices, dim=None):
        r"""
        Selects values from :attr:`input` at the 1-dimensional indices from :attr:`indices` along the given :attr:`dim`.
        See also [torch.take_along_dim][].

        Examples:
            >>> import torch
            >>> from danling.tensors import NestedTensor
            >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
            >>> indices = torch.tensor([[0, 1], [1, 0]])
            >>> out = torch.take_along_dim(nt, indices, dim=1)
            >>> ref = torch.take_along_dim(nt.tensor, indices, dim=1)
            >>> torch.equal(out, ref)
            True
        """
        from .nested_tensor import NestedTensor

        if dim is None:
            if isinstance(indices, Tensor) and indices.shape == input.shape:
                indices = input.nested_like(indices, strict=False)
            if isinstance(indices, NestedTensor):
                if len(input) != len(indices):
                    raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(indices)}")
                return NestedTensor(
                    torch.take_along_dim(t, i, dim=None) for t, i in zip(input._storage, indices._storage)
                )
            return NestedTensor(torch.take_along_dim(t, indices.to(device=t.device), dim=None) for t in input._storage)

        ndims = input.dim()
        dim = dim if dim >= 0 else dim + ndims
        batch_dim = 0 if input.batch_first else 1
        if dim == batch_dim:
            raise NotImplementedError("take_along_dim on batch dimension is not supported for NestedTensor.")
        if input.batch_first:
            dim_adj = dim - 1
        else:
            dim_adj = dim if dim < batch_dim else dim - 1
        if isinstance(indices, Tensor) and indices.shape == input.shape:
            indices = input.nested_like(indices, strict=False)
        if isinstance(indices, NestedTensor):
            if len(input) != len(indices):
                raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(indices)}")
            return NestedTensor(
                torch.take_along_dim(t, i, dim=dim_adj) for t, i in zip(input._storage, indices._storage)
            )
        return NestedTensor(torch.take_along_dim(t, indices.to(device=t.device), dim=dim_adj) for t in input._storage)


# ---------------------------------------------------------------------------
# Bulk elementwise registrations
# ---------------------------------------------------------------------------
# These register elementwise ops in the NestedTensorFuncRegistry, which is
# caught early in __torch_function__ — bypassing the aten decomposition →
# __torch_dispatch__ path (~14x speedup for elementwise ops on CPU).
#
# Unary ops: apply directly to packed _values via _from_values.
# Binary ops: use _binary_op_maybe_tensor (handles NT+NT, NT+scalar,
#   and NT+matching-shape-Tensor via nested_like conversion; already has
#   a fast path for same-layout NT+NT via _from_values internally).

_UNARY_OPS = [
    torch.abs,
    torch.neg,
    torch.sign,
    torch.ceil,
    torch.floor,
    torch.round,
    torch.trunc,
    torch.frac,
    torch.reciprocal,
    torch.sqrt,
    torch.rsqrt,
    torch.exp,
    torch.exp2,
    torch.expm1,
    torch.log,
    torch.log2,
    torch.log10,
    torch.log1p,
    torch.sin,
    torch.cos,
    torch.tan,
    torch.asin,
    torch.acos,
    torch.atan,
    torch.sinh,
    torch.cosh,
    torch.tanh,
    torch.asinh,
    torch.acosh,
    torch.atanh,
    torch.sigmoid,
    torch.logit,
    torch.relu,
    torch.isnan,
    torch.isinf,
    torch.isfinite,
    torch.logical_not,
    torch.erf,
    torch.erfc,
    torch.erfinv,
    torch.positive,
    torch.bitwise_not,
    torch.sgn,
]

_BINARY_OPS = [
    torch.true_divide,
    torch.floor_divide,
    torch.remainder,
    torch.eq,
    torch.ne,
    torch.gt,
    torch.ge,
    torch.lt,
    torch.le,
    torch.logical_and,
    torch.logical_or,
    torch.logical_xor,
    torch.bitwise_and,
    torch.bitwise_or,
    torch.bitwise_xor,
    torch.bitwise_left_shift,
    torch.bitwise_right_shift,
    torch.maximum,
    torch.minimum,
    torch.mul,
    torch.pow,
    torch.atan2,
    torch.fmod,
]


for _op in _UNARY_OPS:
    if _op not in NestedTensorFuncRegistry:

        @NestedTensorFuncRegistry.implement(_op)
        def _unary_impl(input, *args, _fn=_op, **kwargs):
            from .dispatch import _from_values

            return _from_values(input, _fn(input._values, *args, **kwargs))


for _op in _BINARY_OPS:
    if _op not in NestedTensorFuncRegistry:

        @NestedTensorFuncRegistry.implement(_op)
        def _binary_impl(input, other, _fn=_op):
            return _binary_op_maybe_tensor(input, other, _fn)
