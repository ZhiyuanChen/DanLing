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

r"""
Variable-length tensor utilities built around [NestedTensor][].

This package provides [NestedTensor][] — a [torch.Tensor][] subclass
that stores variable-length tensors in a packed representation and dispatches
PyTorch operations through a three-tier system:

1. **Aten dispatch** (``aten_functions``): fastest, operates on packed ``_values``
2. **Torch function dispatch** (``torch_functions``): handles ``torch.*`` ops
   needing dimension translation or per-element logic
3. **NN function dispatch** (``nn_functions``): handles ``torch.nn.functional.*``
   ops like convolution, pooling, and attention

See ``README.md`` in this directory for full architecture details.
"""

from __future__ import annotations

from typing import Callable

from torch.utils.data._utils.collate import default_collate_fn_map

from . import nn_functions, torch_functions  # noqa: F401
from .nested_tensor import NestedTensor
from .nn_functions import create_flex_block_mask
from .ops import NestedTensorAtenRegistry, NestedTensorFuncRegistry, TorchFuncRegistry
from .pn_tensor import PNTensor, tensor

__all__ = [
    "NestedTensor",
    "PNTensor",
    "tensor",
    "collate_pn_tensor_fn",
    "register_pn_tensor_collate",
    "unregister_pn_tensor_collate",
    "TorchFuncRegistry",
    "NestedTensorFuncRegistry",
    "NestedTensorAtenRegistry",
    "create_flex_block_mask",
]


def collate_pn_tensor_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None):
    r"""Collate PNTensor elements into a NestedTensor for DataLoader."""
    return NestedTensor(batch)


def register_pn_tensor_collate(collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None) -> None:
    r"""Register PNTensor collation into a collate map (default: PyTorch global map)."""
    target = default_collate_fn_map if collate_fn_map is None else collate_fn_map
    target[PNTensor] = collate_pn_tensor_fn


def unregister_pn_tensor_collate(collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None) -> None:
    r"""Remove PNTensor collation from a collate map (default: PyTorch global map)."""
    target = default_collate_fn_map if collate_fn_map is None else collate_fn_map
    target.pop(PNTensor, None)
