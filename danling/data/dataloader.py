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

from chanfig import FlatDict


class DataLoaderDict(FlatDict):
    """Stateful dataloader mapping used by runners."""

    def state_dict(self) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for split, loader in self.items():
            state_dict_fn = getattr(loader, "state_dict", None)
            if not callable(state_dict_fn):
                continue
            state = state_dict_fn()
            if state is None:
                continue
            states[str(split)] = state
        return states

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        for split, loader_state in state_dict.items():
            if split not in self:
                continue
            load_state_dict_fn = getattr(self[split], "load_state_dict", None)
            if not callable(load_state_dict_fn):
                continue
            load_state_dict_fn(loader_state)
