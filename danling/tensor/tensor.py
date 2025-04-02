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

from typing import Any

import torch
from torch import Tensor


def tensor(data: Any, dtype=None, device=None, requires_grad: bool = False, pin_memory: bool = False) -> PNTensor:
    """
    Create a PNTensor from data, similar to torch.tensor() but returning a PNTensor.

    This function is a convenient way to create PNTensor objects that will be
    automatically collated into NestedTensor when used with PyTorch DataLoader.
    The interface mirrors torch.tensor() to make it easy to switch between regular
    tensors and PNTensors.

    Args:
        data: Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, etc.
        dtype: Desired data type of the returned tensor.
        device: Device on which to place the tensor.
        requires_grad: If autograd should record operations on the returned tensor.
        pin_memory: If True, the tensor will be allocated in pinned memory.

    Returns:
        PNTensor: A tensor wrapper that will be automatically collated into NestedTensor

    Examples:
        >>> from danling.tensor import tensor
        >>> t = tensor([1, 2, 3])
        >>> t
        PNTensor([1, 2, 3])
    """
    return PNTensor(torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory))


class PNTensor(Tensor):
    r"""
    A tensor wrapper that enables automatic collation into NestedTensor with PyTorch DataLoader.

    `PNTensor` (Potential Nested Tensor) seamlessly bridges the gap between individual tensors
    and batched `NestedTensor` objects in PyTorch workflows. It's designed specifically to work
    with PyTorch's DataLoader collation mechanism, allowing datasets to return variable-length
    tensors that will automatically be combined into a `NestedTensor` when batched.

    The class provides three properties that mirror those of NestedTensor:
    - `.tensor`: The tensor itself (self)
    - `.mask`: A tensor of ones with the same shape as self
    - `.concat`: The tensor itself (self)

    Attributes:
        Inherits all attributes from torch.Tensor

    Methods:
        Inherits all methods from torch.Tensor

    Examples:
        Basic usage with PyTorch DataLoader:

        >>> from torch.utils.data import Dataset, DataLoader
        >>> from danling.tensor import PNTensor
        >>> class VariableLengthDataset(Dataset):
        ...     def __init__(self, data):
        ...         self.data = data
        ...     def __len__(self):
        ...         return len(self.data)
        ...     def __getitem__(self, idx):
        ...         return PNTensor(self.data[idx])
        >>> # Create a dataset with variable-length sequences
        >>> dataset = VariableLengthDataset([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
        >>> dataloader = DataLoader(dataset, batch_size=3)
        >>> # The DataLoader automatically produces NestedTensor batches
        >>> batch = next(iter(dataloader))
        >>> batch
        NestedTensor([[1., 2., 3., 0.],
                [4., 5., 0., 0.],
                [6., 7., 8., 9.]])

        Using PNTensor directly:

        >>> tensor = PNTensor([1, 2, 3])
        >>> tensor
        PNTensor([1., 2., 3.])
        >>> tensor.tensor
        PNTensor([1., 2., 3.])
        >>> tensor.mask
        PNTensor([True, True, True])
    """

    @property
    def tensor(self) -> Tensor:
        r"""
        Identical to `self`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor([1, 2, 3])
            >>> bool((tensor == pn_tensor).all())
            True
            >>> bool((tensor == pn_tensor.tensor).all())
            True
        """

        return self

    @property
    def mask(self) -> Tensor:
        r"""
        Identical to `torch.ones_like(self)`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor([1, 2, 3])
            >>> bool((pn_tensor.mask == torch.ones_like(pn_tensor)).all().item())
            True
        """

        return torch.ones_like(self, dtype=torch.bool)

    @property
    def contact(self) -> Tensor:
        r"""
        Identical to `self`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor([1, 2, 3])
            >>> bool((tensor == pn_tensor).all())
            True
            >>> bool((tensor == pn_tensor.contact).all())
            True
        """

        return self

    def new_empty(self, *args, **kwargs):
        return PNTensor(super().new_empty(*args, **kwargs))
