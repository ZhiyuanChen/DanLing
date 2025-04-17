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

from typing import Any

import torch
from chanfig import FlatDict, NestedDict


def to_device(data: Any, device: torch.device):
    r"""Move data to device."""
    if isinstance(data, list):
        return [to_device(i, device) for i in data]
    if isinstance(data, tuple):
        return tuple(to_device(i, device) for i in data)
    if isinstance(data, NestedDict):
        return NestedDict({k: to_device(v, device) for k, v in data.all_items()})
    if isinstance(data, dict):
        return FlatDict({k: to_device(v, device) for k, v in data.items()})
    if hasattr(data, "to"):
        return data.to(device)
    return data
