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
from typing import TYPE_CHECKING, Callable, Iterable, List, Mapping, Sequence, Tuple

import torch
from chanfig import Registry
from torch import Tensor
from torch.nn import functional as F

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
            raise ValueError(f"Torch function {torch_function.__name__} already implemented by {self[torch_function]}.")

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

    def __call__(self, *args, **kwargs) -> NestedTensor | Sequence[Tensor]:
        from .nested_tensor import NestedTensor
        from .tensor import PNTensor

        ret = [call(*args, **kwargs) for call in self._storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            try:
                return PNTensor(ret)
            except ValueError:
                return NestedTensor(ret, **self.state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret


NestedTensorFuncRegistry = TorchFuncRegistry()


@NestedTensorFuncRegistry.implement(torch.all)
def all(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    return input.all(dim=dim, keepdim=keepdim)


@NestedTensorFuncRegistry.implement(torch.allclose)
def allclose(
    input: NestedTensor | Tensor,
    other: NestedTensor | Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    equal_nan: bool = False,
):
    r"""
    Check if all elements in NestedTensor are close to the corresponding elements in another tensor.

    Args:
        input: NestedTensor
        other: Tensor or NestedTensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        equal_nan: Whether to consider NaN values as equal

    Returns:
        bool: True if all elements are close, False otherwise
    """
    from .nested_tensor import NestedTensor

    if isinstance(input, Tensor) and isinstance(other, NestedTensor):
        input = NestedTensor.from_tensor_mask(input, other.mask)

    if isinstance(input, NestedTensor) and isinstance(other, Tensor):
        other = NestedTensor.from_tensor_mask(other, input.mask)

    return all(torch.allclose(a, b, rtol, atol, equal_nan) for a, b in zip(input._storage, other._storage))


@NestedTensorFuncRegistry.implement(torch.cat)
def cat(tensors: Tuple[NestedTensor | Tensor, ...], dim: int = 0):
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

    return NestedTensor((torch.log(t) for t in tensor._storage), **tensor._state)


@NestedTensorFuncRegistry.implement(torch.log1p)
def log1p(tensor):
    from .nested_tensor import NestedTensor

    return NestedTensor((torch.log1p(t) for t in tensor._storage), **tensor._state)


@NestedTensorFuncRegistry.implement(torch.max)
def max(
    input: NestedTensor,
    dim: int | None = None,
    keepdim: bool = False,
):
    return input.max(dim=dim, keepdim=keepdim)


@NestedTensorFuncRegistry.implement(torch.mean)
def mean(
    input: NestedTensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    return input.mean(dim=dim, keepdim=keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.min)
def min(
    input: NestedTensor,
    dim: int | None = None,
    keepdim: bool = False,
):
    return input.min(dim=dim, keepdim=keepdim)


@NestedTensorFuncRegistry.implement(torch.sqrt)
def sqrt(tensor):
    from .nested_tensor import NestedTensor

    return NestedTensor((torch.sqrt(t) for t in tensor._storage), **tensor._state)


@NestedTensorFuncRegistry.implement(torch.stack)
def stack(*args, **kwargs):
    raise NotImplementedError("NestedTensor does not support stack as of now")


@NestedTensorFuncRegistry.implement(torch.sum)
def sum(input: NestedTensor, dim: int | None = None, keepdim: bool = False, dtype: torch.dtype | None = None):
    r"""
    Sum of all elements in the NestedTensor, or along specified dimension.

    Args:
        input (NestedTensor): Input tensor
        dim (int, optional): Dimension to reduce
        keepdim (bool): Whether to keep the reduced dimension
        dtype (torch.dtype, optional): Output data type

    Returns:
        NestedTensor or Tensor: Sum result
    """
    from .nested_tensor import NestedTensor

    if dim is None:
        return sum([t.sum() for t in input._storage])
    if (input.batch_first and dim == 0) or (not input.batch_first and dim == 1):
        return torch.stack([t.sum() for t in input._storage])
    return NestedTensor([t.sum(dim=dim, keepdim=keepdim, dtype=dtype) for t in input._storage], **input._state)


@NestedTensorFuncRegistry.implement(torch.nn.functional.avg_pool2d)
def avg_pool2d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, int],
    stride: int | Tuple[int, int] | None = None,
    padding: int | Tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    r"""
    Applies 2D average pooling over a NestedTensor with varying spatial dimensions.

    This allows pooling of images with different sizes in a batch.

    Args:
        input (NestedTensor): Input tensor
        kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override:
            Same parameters as torch.nn.functional.avg_pool2d

    Returns:
        NestedTensor: Result of applying avg_pool2d to each tensor
    """
    from .nested_tensor import NestedTensor

    return NestedTensor(
        (
            F.avg_pool2d(
                t,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )
            for t in input._storage
        ),
        **input._state,
    )


@NestedTensorFuncRegistry.implement(torch.nn.functional.batch_norm)
def batch_norm(
    input: NestedTensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    r"""
    Applies Batch Normalization to each tensor in a NestedTensor.

    This implementation supports variable-sized inputs, which is particularly useful
    when working with batches of images or feature maps of different spatial dimensions.

    During training, each tensor in the NestedTensor is normalized independently. During
    inference, the running statistics (running_mean and running_var) are used for normalization.

    Args:
        input (NestedTensor): Input tensor (N, C, H, W) or (N, C, L)
        running_mean (Tensor): Running mean of shape (C)
        running_var (Tensor): Running variance of shape (C)
        weight (Tensor, optional): Scale parameter of shape (C)
        bias (Tensor, optional): Shift parameter of shape (C)
        training (bool): Whether to use batch statistics (training mode) or running statistics (inference mode)
        momentum (float): Momentum factor for running statistics updates
        eps (float): Small constant for numerical stability

    Returns:
        NestedTensor: Batch normalized tensor with the same structure as input
    """
    from .nested_tensor import NestedTensor

    if training:
        # In training mode, we need to calculate batch statistics for each tensor
        # (mean and var) and update running statistics if provided
        results = []
        for tensor in input._storage:
            # For each tensor, compute the mean and variance across spatial dimensions (H,W)
            # and across batch dimension within this specific tensor
            dim = (0,) + tuple(range(2, tensor.dim()))  # (0, 2, 3) for 4D tensors

            # Compute mean and variance for this tensor
            tensor_mean = tensor.mean(dim=dim, keepdim=True)
            tensor_var = tensor.var(dim=dim, unbiased=False, keepdim=True)

            # Update running statistics if provided
            if running_mean is not None and running_var is not None:
                with torch.no_grad():
                    # Compute unbatched mean and var over all elements in this tensor
                    # across batch and spatial dimensions
                    unbatched_mean = tensor.mean(dim=dim)
                    unbatched_var = tensor.var(dim=dim, unbiased=False)

                    # Update running statistics weighted by batch size proportionally
                    batch_size = tensor.size(0) if tensor.dim() > 0 else 1
                    running_mean.mul_(1 - momentum).add_(unbatched_mean.detach() * momentum)
                    running_var.mul_(1 - momentum).add_(unbatched_var.detach() * momentum)

            # Apply normalization
            normalized = (tensor - tensor_mean) / torch.sqrt(tensor_var + eps)

            # Apply scale and shift if provided
            if weight is not None and bias is not None:
                # Reshape weight and bias to match the tensor dimensions
                weight_shape = [1, -1] + [1] * (tensor.dim() - 2)
                bias_shape = [1, -1] + [1] * (tensor.dim() - 2)

                normalized = normalized * weight.view(*weight_shape) + bias.view(*bias_shape)

            results.append(normalized)
    else:
        # In inference mode, use the running statistics for all tensors
        results = []
        for tensor in input._storage:
            # Reshape running stats to match the tensor dimensions
            mean_shape = [1, -1] + [1] * (tensor.dim() - 2)
            var_shape = [1, -1] + [1] * (tensor.dim() - 2)

            # Apply normalization with running statistics
            normalized = (tensor - running_mean.view(*mean_shape)) / torch.sqrt(running_var.view(*var_shape) + eps)

            # Apply scale and shift if provided
            if weight is not None and bias is not None:
                weight_shape = [1, -1] + [1] * (tensor.dim() - 2)
                bias_shape = [1, -1] + [1] * (tensor.dim() - 2)

                normalized = normalized * weight.view(*weight_shape) + bias.view(*bias_shape)

            results.append(normalized)

    return NestedTensor(results, **input._state)


@NestedTensorFuncRegistry.implement(torch.nn.functional.conv2d)
def conv2d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple = 1,
    padding: int | tuple = 0,
    dilation: int | tuple = 1,
    groups: int = 1,
):
    r"""
    Applies 2D convolution to each tensor in a NestedTensor.

    This supports batches of images with different spatial dimensions.

    Args:
        input (NestedTensor): Input tensor
        weight, bias, stride, padding, dilation, groups:
            Same parameters as torch.nn.functional.conv2d

    Returns:
        NestedTensor: Result of applying conv2d to each tensor
    """
    from .nested_tensor import NestedTensor

    return NestedTensor(
        (
            F.conv2d(
                t,
                weight=weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            for t in input._storage
        ),
        **input._state,
    )


@NestedTensorFuncRegistry.implement(torch.nn.functional.dropout)
def dropout(input: NestedTensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    r"""
    Applies dropout to each tensor in a NestedTensor.

    Args:
        input (NestedTensor): Input tensor
        p (float): Dropout probability
        training (bool): If True, applies dropout; otherwise, identity
        inplace (bool): If True, modifies the input tensor in-place

    Returns:
        NestedTensor: Result with dropout applied
    """
    from .nested_tensor import NestedTensor

    if inplace and training:
        for tensor in input._storage:
            F.dropout(tensor, p=p, training=True, inplace=True)
        return input

    return NestedTensor(
        (F.dropout(tensor, p=p, training=training, inplace=False) for tensor in input._storage), **input._state
    )


@NestedTensorFuncRegistry.implement(torch.nn.functional.embedding)
def embedding(
    input: NestedTensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    r"""
    Applies embedding lookup on NestedTensor.

    Args:
        input: NestedTensor of indices
        weight: Embedding weights
        Other parameters: Same as torch.nn.functional.embedding

    Returns:
        NestedTensor: Result with embeddings
    """
    from .nested_tensor import NestedTensor

    return NestedTensor(
        [
            F.embedding(tensor, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
            for tensor in input._storage
        ],
        **input._state,
    )


@NestedTensorFuncRegistry.implement(torch.nn.functional.gelu)
def gelu(input: NestedTensor, approximate: str = "none"):
    r"""
    Applies GELU activation to each tensor in a NestedTensor.

    Args:
        input (NestedTensor): Input tensor
        approximate (str): Approximation type ('none' or 'tanh')

    Returns:
        NestedTensor: Result with GELU applied
    """
    from .nested_tensor import NestedTensor

    return NestedTensor((F.gelu(tensor, approximate=approximate) for tensor in input._storage), **input._state)


@NestedTensorFuncRegistry.implement(torch.nn.functional.layer_norm)
def layer_norm(
    input: NestedTensor,
    normalized_shape: List[int] | int | torch.Size,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
):
    r"""
    Applies layer normalization to each tensor in a NestedTensor.

    This is useful for normalizing variable length sequences.

    Args:
        input (NestedTensor): Input tensor
        normalized_shape (int or list or torch.Size): Input shape from an expected input
        weight (Tensor, optional): Scale factors
        bias (Tensor, optional): Shift factors
        eps (float): Small constant for numerical stability

    Returns:
        NestedTensor: Tensor with layer normalization applied
    """
    from .nested_tensor import NestedTensor

    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    elif isinstance(normalized_shape, torch.Size):
        normalized_shape = list(normalized_shape)

    results = []
    for tensor in input._storage:
        if len(normalized_shape) == 1:
            if tensor.dim() > 0:
                tensor_normalized_shape = [min(normalized_shape[0], tensor.size(-1))]
            else:
                tensor_normalized_shape = normalized_shape
        else:
            tensor_normalized_shape = []
            for i, dim_size in enumerate(normalized_shape):
                if i < tensor.dim():
                    tensor_normalized_shape.append(min(dim_size, tensor.size(-(i + 1))))
                else:
                    break
            tensor_normalized_shape = tensor_normalized_shape[::-1]

        norm_tensor = F.layer_norm(tensor, tensor_normalized_shape, weight, bias, eps)
        results.append(norm_tensor)

    return NestedTensor(results, **input._state)


@NestedTensorFuncRegistry.implement(torch.nn.functional.linear)
def linear(input: NestedTensor, weight: Tensor, bias: Tensor | None = None):
    r"""
    Applies a linear transformation to NestedTensor input.

    This is optimized to work efficiently with the concat representation
    when possible to avoid padding overhead.

    Args:
        input: NestedTensor input
        weight: Weight matrix
        bias: Optional bias

    Returns:
        NestedTensor: Result of linear transformation
    """
    from .nested_tensor import NestedTensor

    if input.ndim == 3:
        return NestedTensor(F.linear(input.torch, weight, bias).unbind(), **input._state)
    return NestedTensor([F.linear(tensor, weight, bias) for tensor in input._storage], **input._state)


@NestedTensorFuncRegistry.implement(torch.matmul)
def matmul(input: NestedTensor, other: NestedTensor | Tensor):
    r"""
    Performs matrix multiplication between NestedTensor and another tensor.

    Args:
        input: NestedTensor
        other: Tensor or NestedTensor

    Returns:
        NestedTensor: Result of matrix multiplication
    """
    from .nested_tensor import NestedTensor

    if isinstance(other, NestedTensor):
        if len(input._storage) != len(other._storage):
            raise ValueError(f"NestedTensor batch sizes don't match: {len(input._storage)} vs {len(other._storage)}")

        return NestedTensor([torch.matmul(a, b) for a, b in zip(input._storage, other._storage)], **input._state)

    return NestedTensor([torch.matmul(tensor, other) for tensor in input._storage], **input._state)


@NestedTensorFuncRegistry.implement(torch.nn.functional.max_pool2d)
def max_pool2d(
    input: NestedTensor,
    kernel_size: int | Tuple[int, int],
    stride: int | Tuple[int, int] | None = None,
    padding: int | Tuple[int, int] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
):
    r"""
    Applies 2D max pooling over a NestedTensor with varying spatial dimensions.

    This allows pooling of images with different sizes in a batch.

    Args:
        input (NestedTensor): Input tensor
        kernel_size, stride, padding, dilation, ceil_mode, return_indices:
            Same parameters as torch.nn.functional.max_pool2d

    Returns:
        NestedTensor or tuple: Result of applying max_pool2d to each tensor,
                               or tuple of (result, indices) if return_indices=True
    """
    from .nested_tensor import NestedTensor

    if return_indices:
        outputs = []
        indices_list = []

        for t in input._storage:
            output, indices = F.max_pool2d(
                t,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                return_indices=True,
            )
            outputs.append(output)
            indices_list.append(indices)

        outputs_nt = NestedTensor(outputs, **input._state)
        indices_nt = NestedTensor(indices_list, **input._state)

        return outputs_nt, indices_nt

    return NestedTensor(
        (
            F.max_pool2d(
                t,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                return_indices=False,
            )
            for t in input._storage
        ),
        **input._state,
    )


@NestedTensorFuncRegistry.implement(torch.nn.functional.pad)
def pad(input: NestedTensor, pad: tuple, mode: str = "constant", value: float = 0):
    r"""
    Pad each tensor in the NestedTensor.

    This is useful for operations like convolutions that need specific padding.

    Args:
        input (NestedTensor): Input tensor
        pad (tuple): m-elements tuple where m/2 <= input dimension
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'
        value (float, optional): Fill value for 'constant' padding

    Returns:
        NestedTensor: Padded tensor
    """
    from .nested_tensor import NestedTensor

    return NestedTensor([F.pad(tensor, pad, mode, value) for tensor in input._storage], **input._state)


@NestedTensorFuncRegistry.implement(torch.nn.functional.relu)
def relu(input: NestedTensor, inplace: bool = False):
    r"""
    Applies ReLU activation to each tensor in a NestedTensor.

    Args:
        input (NestedTensor): Input tensor
        inplace (bool): If True, modifies the input tensor in-place

    Returns:
        NestedTensor: Result with ReLU applied
    """
    from .nested_tensor import NestedTensor

    if inplace:
        for tensor in input._storage:
            F.relu(tensor, inplace=True)
        return input

    return NestedTensor((F.relu(tensor) for tensor in input._storage), **input._state)


# @NestedTensorFuncRegistry.implement(torch.nn.functional.scaled_dot_product_attention)
# def scaled_dot_product_attention(
#     query: NestedTensor,
#     key: NestedTensor,
#     value: NestedTensor,
#     attn_mask: Tensor | None = None,
#     dropout_p: float = 0.0,
#     is_causal: bool = False,
#     scale: float | None = None,
# ):
#     r"""
#     Implements the scaled dot product attention mechanism for NestedTensor inputs.

#     Particularly useful for efficient attention computation with variable-length sequences.

#     Args:
#         query, key, value: NestedTensor query, key, and value tensors
#         attn_mask: Optional mask to apply
#         dropout_p: Dropout probability
#         is_causal: Whether to apply causal masking
#         scale: Optional scaling factor

#     Returns:
#         NestedTensor: Attention output
#     """
#     from .nested_tensor import NestedTensor

#     if attn_mask is None and isinstance(query, NestedTensor) and is_causal:
#         # Use the specialized causal mask for variable-length sequences
#         attn_mask = query.causal_mask()

#     # Process each sequence separately to avoid padding influence
#     if isinstance(query, NestedTensor) and isinstance(key, NestedTensor) and isinstance(value, NestedTensor):
#         if len(query._storage) != len(key._storage) or len(query._storage) != len(value._storage):
#             raise ValueError("Query, key, and value must have the same batch size")

#         results = []
#         for q, k, v in zip(query._storage, key._storage, value._storage):
#             # Compute attention separately for each item in batch
#             seq_result = F.scaled_dot_product_attention(
#                 q.unsqueeze(0),
#                 k.unsqueeze(0),
#                 v.unsqueeze(0),
#                 attn_mask=attn_mask,
#                 dropout_p=dropout_p,
#                 is_causal=is_causal,
#                 scale=scale,
#             )
#             results.append(seq_result.squeeze(0))

#         return NestedTensor(results, **value._state)

#     # When not all inputs are NestedTensor, do standard conversion
#     if isinstance(query, NestedTensor):
#         q_tensor = query.tensor
#     else:
#         q_tensor = query

#     if isinstance(key, NestedTensor):
#         k_tensor = key.tensor
#     else:
#         k_tensor = key

#     if isinstance(value, NestedTensor):
#         v_tensor = value.tensor
#         state = value._state
#     else:
#         v_tensor = value
#         state = {}  # Default state if no NestedTensor provided

#     result = F.scaled_dot_product_attention(
#         q_tensor, k_tensor, v_tensor, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
#     )

#     # Preserve the nested structure if value was a NestedTensor
#     if isinstance(value, NestedTensor):
#         # We need to decompose the result back into individual sequences
#         return NestedTensor([result[i] for i in range(result.size(0))], **state)
#     return result


@NestedTensorFuncRegistry.implement(torch.nn.functional.softmax)
def softmax(input: NestedTensor, dim: int, dtype: torch.dtype | None = None, **kwargs):
    r"""
    Applies softmax to a NestedTensor along the specified dimension.

    Args:
        input (NestedTensor): Input tensor
        dim (int): Dimension along which to apply softmax
        dtype (torch.dtype, optional): Output data type

    Returns:
        NestedTensor: Tensor with softmax applied
    """
    from .nested_tensor import NestedTensor

    return NestedTensor((F.softmax(t, dim=dim, dtype=dtype, **kwargs) for t in input._storage), **input._state)
