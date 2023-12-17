from functools import wraps
from typing import Callable

from chanfig import Registry


class TorchFuncRegistry(Registry):
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
            >>> @registry.implement(torch.mean)  # pylint: disable=E1101
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
