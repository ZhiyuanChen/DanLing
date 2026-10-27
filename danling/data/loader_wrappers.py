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

from collections.abc import Mapping
from typing import Any


class StepProxyLoader:
    """Proxy loader that preserves step count while yielding placeholder batches."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self.batch_sampler = getattr(loader, "batch_sampler", None)
        self.sampler = getattr(loader, "sampler", None)
        self._position = 0
        self._exhausted = False

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        if self._exhausted:
            self._position = 0
            self._exhausted = False
        while self._position < len(self._loader):
            self._position += 1
            yield None
        self._exhausted = True

    def state_dict(self):
        state_dict_fn = getattr(self._loader, "state_dict", None)
        return {
            "loader": state_dict_fn() if callable(state_dict_fn) else None,
            "position": self._position,
            "exhausted": self._exhausted,
        }

    def load_state_dict(self, state_dict) -> None:
        load_state_dict_fn = getattr(self._loader, "load_state_dict", None)
        if isinstance(state_dict, Mapping) and "position" in state_dict and "loader" in state_dict:
            self._position = int(state_dict["position"])
            self._exhausted = bool(state_dict["exhausted"])
            state_dict = state_dict["loader"]
        else:
            # Older checkpoints stored only the underlying loader's cursor.
            self._position = int(state_dict.get("_num_yielded", 0))
            self._exhausted = bool(state_dict.get("_iterator_finished", False))
        if callable(load_state_dict_fn) and state_dict is not None:
            load_state_dict_fn(state_dict)
