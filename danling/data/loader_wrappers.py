# DanLing
# Copyright (C) 2022-Present  DanLing
#
# This file is part of DanLing.
#
# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0
#
# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from typing import Any


class StepProxyLoader:
    """Proxy loader that preserves step count while yielding placeholder batches."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self.batch_sampler = getattr(loader, "batch_sampler", None)
        self.sampler = getattr(loader, "sampler", None)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        for _ in range(len(self._loader)):
            yield None
