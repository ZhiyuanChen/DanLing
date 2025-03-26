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

import warnings
from functools import wraps
from typing import Callable

import torch
from chanfig import Registry
from torch import Tensor

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "float8": torch.float8_e4m3fn,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}

try:
    _ = torch.bfloat16
    _ = torch.zeros(1, dtype=torch.bfloat16)
    DTYPE_MAP["bfloat16"] = torch.bfloat16
    BFLOAT16_AVAILABLE = True
except (RuntimeError, AttributeError):
    BFLOAT16_AVAILABLE = False

HALF_MIN = torch.finfo(torch.float16).min
HALF_MAX = torch.finfo(torch.float16).max
CHAR_MIN = torch.iinfo(torch.int8).min
CHAR_MAX = torch.iinfo(torch.int8).max
SHORT_MIN = torch.iinfo(torch.int16).min
SHORT_MAX = torch.iinfo(torch.int16).max
INT_MIN = torch.iinfo(torch.int32).min
INT_MAX = torch.iinfo(torch.int32).max


def convert_tensor_precision(tensor: Tensor, precision: str | None = "auto") -> Tensor:
    """
    Determine the appropriate dtype for a tensor based on its values and requested precision.

    Args:
        tensor: The input tensor
        precision: Requested precision level or "auto" for automatic detection or None to keep original

    Returns:
        The appropriate torch.dtype
    """
    if precision is None:
        return tensor

    if precision in DTYPE_MAP:
        if precision == "float8":
            warnings.warn("float8 has limited support, use with caution.", RuntimeWarning, stacklevel=2)
        return tensor.to(dtype=DTYPE_MAP[precision])

    maximum, minimum = tensor.max(), tensor.min()
    if tensor.is_floating_point():
        if maximum <= HALF_MAX and minimum >= HALF_MIN:
            return tensor.to(dtype=torch.float16)
        return torch.bfloat16 if BFLOAT16_AVAILABLE else torch.float32
    else:
        if maximum <= CHAR_MAX and minimum >= CHAR_MIN:
            return tensor.to(dtype=torch.int8)
        if maximum <= SHORT_MAX and minimum >= SHORT_MIN:
            return tensor.to(dtype=torch.int16)
        return tensor.to(dtype=torch.int32)
    return tensor


class TorchFuncRegistry(Registry):  # pylint: disable=too-few-public-methods
    """
    `TorchFuncRegistry` for extending PyTorch Tensor.
    """

    def implement(self, torch_function: Callable) -> Callable:
        r"""
        Implement an implementation for a torch function.

        Args:
            function: The torch function to implement.

        Returns:
            function: The registered function.

        Raises:
            ValueError: If the function with the same name already registered and `TorchFuncRegistry.override=False`.

        Examples:
            >>> import torch
            >>> registry = TorchFuncRegistry("test")
            >>> @registry.implement(torch.mean)
            ... def mean(input):
            ...     raise input.mean()
            >>> registry  # doctest: +ELLIPSIS
            TorchFuncRegistry(
              (<built-in method mean of type object at ...>): <function mean at ...>
            )
        """

        if torch_function in self and not self.override:
            raise ValueError(f"Torch function {torch_function.__name__} already registered.")

        @wraps(self.register)
        def register(function):
            self.set(torch_function, function)
            return function

        return register
