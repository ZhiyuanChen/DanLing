# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from functools import wraps
from typing import Callable

from chanfig import Registry


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
