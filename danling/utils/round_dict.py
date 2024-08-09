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

from collections.abc import Mapping
from math import isnan
from numbers import Integral, Real
from typing import Any, Generic, Union

from chanfig import NestedDict
from numpy import ndarray
from torch import Tensor
from typing_extensions import Self, TypeVar

from .lists import flist

K = TypeVar("K", bound=str, default=str)
V = TypeVar("V", float, int, flist, Union[float, flist], Union[float, int], default=Union[float, flist])


class RoundDict(NestedDict, Generic[K, V]):
    convert_mapping = True

    @staticmethod
    def _normalize_leaf(value: Any) -> Any:
        if isinstance(value, Tensor):
            if value.numel() == 1:
                return value.item()
            return flist(value.detach().cpu().reshape(-1).tolist())
        if isinstance(value, ndarray):
            if value.size == 1:
                return value.item()
            return flist(value.reshape(-1).tolist())
        if isinstance(value, (list, tuple)) and not isinstance(value, flist):
            return flist(value)
        return value

    @classmethod
    def _round_leaf(cls, value: Any, ndigits: int = 4) -> Any:
        value = cls._normalize_leaf(value)
        if isinstance(value, flist):
            return flist(
                round(float(item), ndigits) if isinstance(item, Real) and not isinstance(item, bool) else item
                for item in value
            )
        if isinstance(value, Real) and not isinstance(value, bool):
            return round(float(value), ndigits)
        return value

    def round(self, ndigits: int = 4) -> Self:
        for key, value in self.all_items():
            self[key] = self._round_leaf(value, ndigits=ndigits)
        return self

    def __round__(self, ndigits: int = 4) -> Self:
        out = self.empty_like()
        for key, value in self.all_items():
            out[key] = self._round_leaf(value, ndigits=ndigits)
        return out

    @classmethod
    def _format_scalars(cls, result: Mapping[str, Any], format_spec: str = ".4f") -> str:
        parts: list[str] = []
        for key, value in result.items():
            if isinstance(value, Mapping):
                continue
            value = cls._normalize_leaf(value)
            padding = 1
            if isinstance(value, flist):
                value = format(value, format_spec)
            elif isinstance(value, Real) and not isinstance(value, bool):
                if isinstance(value, Integral):
                    value = int(value)
                else:
                    scalar = float(value)
                    is_negative = scalar < 0 if not isnan(scalar) else False
                    value = format(scalar, format_spec) if not isnan(scalar) else "  NaN  "
                    padding = padding if is_negative else padding + 1
            parts.append(f"{key}:{' ' * padding}{value}")
        return "\t".join(parts)

    @classmethod
    def _format_mapping(cls, result: Mapping[str, Any], format_spec: str = ".4f", depth: int = 0) -> str:
        if not isinstance(result, RoundDict):
            result = RoundDict(result).round(4)
        longest_key = max((len(str(key)) for key in result.keys()), default=0)
        repr_list = [cls._format_scalars(result, format_spec=format_spec)]
        for key, value in result.items():
            if isinstance(value, Mapping):
                initials = " " * (longest_key - len(str(key))) + "\t" * depth
                repr_list.append(f"{initials}{key}: {cls._format_mapping(value, format_spec, depth + 1)}")
        return "\n".join(repr_list)

    def __format__(self, format_spec: str) -> str:
        format_spec = format_spec or ".4f"
        return self._format_mapping(self, format_spec=format_spec)
