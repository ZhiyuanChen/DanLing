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

from typing import Callable


class flist(list):
    r"""Python `list` that support `__format__` and `to`."""

    def to(self, *args, **kwargs):
        return flist(i.to(*args, **kwargs) for i in self)

    def __format__(self, *args, **kwargs):
        return " ".join([x.__format__(*args, **kwargs) for x in self])


class defaultlist(flist):
    default_factory: Callable

    def __init__(self, default_factory: Callable):
        self.default_factory = default_factory

    def _fill(self, index):
        while len(self) <= index:
            self.append(self.default_factory())

    def __setitem__(self, index, value):
        self._fill(index)
        super().__setitem__(self, index, value)

    def __getitem__(self, index):
        self._fill(index)
        return super().__getitem__(self, index)
