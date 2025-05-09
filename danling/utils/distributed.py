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

from warnings import warn

from torch import distributed as dist


def get_world_size(group: dist.ProcessGroup | None = None) -> int:
    r"""Return the number of processes in the current process group."""
    if dist.is_available():
        if dist.is_initialized():
            return dist.get_world_size(group)
        warn("Distributed process group is not initialized, returning 1")
    return 1
