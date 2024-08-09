# DanLing
# Copyright (C) 2022-Present  DanLing
#
# This file is part of DanLing.
#
# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0
#
# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

import numpy as np
import torch

from danling.utils import RoundDict


def test_round_dict_round_methods():
    values = RoundDict({"a": 1.23456, "b": 9.87654})
    rounded_in_place = values.round(2)
    assert rounded_in_place is values
    assert values["a"] == 1.23
    assert values["b"] == 9.88

    original = RoundDict({"x": 3.14159, "y": 2.71828})
    rounded_copy = round(original, 3)
    assert rounded_copy["x"] == 3.142
    assert rounded_copy["y"] == 2.718
    assert original["x"] == 3.14159


def test_round_dict_round_tensor_vector_to_flist():
    values = RoundDict({"vec": torch.tensor([1.23456, 2.34567])})
    values.round(3)
    assert values["vec"] == [1.235, 2.346]
    formatted = format(values, ".3f")
    assert "[1.235, 2.346]" not in formatted
    assert "1.235 2.346" in formatted


def test_round_dict_format_handles_tensor_and_ndarray_values():
    values = RoundDict(
        {
            "scalar": torch.tensor(1.23456),
            "tensor_vec": torch.tensor([1.23456, 2.34567]),
            "array_vec": np.array([3.45678, 4.56789]),
        }
    )
    formatted = format(values, ".2f")
    assert "scalar" in formatted
    assert "1.23" in formatted
    assert "2.35" in formatted
    assert "3.46 4.57" in formatted
