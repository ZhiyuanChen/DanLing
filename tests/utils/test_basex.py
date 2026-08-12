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

import pytest

from danling.utils import base60


class TestBase60:
    @pytest.mark.parametrize("value", (0, 59, 60, 3599))
    def test_round_trip(self, value: int) -> None:
        assert base60.decode(base60.encode(value)) == value

    def test_alphabet(self) -> None:
        assert "".join(base60.encode(value) for value in range(60)) == (
            "0123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
