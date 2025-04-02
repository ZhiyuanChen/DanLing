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

from typing import Callable

from torch.utils.data._utils.collate import default_collate_fn_map

from .functions import TorchFuncRegistry
from .nested_tensor import NestedTensor
from .tensor import PNTensor, tensor
from .utils import mask_tensor, pad_tensor, tensor_mask

__all__ = ["NestedTensor", "PNTensor", "tensor", "TorchFuncRegistry", "tensor_mask", "pad_tensor", "mask_tensor"]


def collate_pn_tensor_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None):
    return NestedTensor(batch)


default_collate_fn_map[PNTensor] = collate_pn_tensor_fn
