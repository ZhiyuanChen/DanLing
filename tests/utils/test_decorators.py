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

from danling.utils import decorators


class TestCatch:
    def test_error(self):
        @decorators.catch()
        def func():
            raise Exception("test")

        func()

    @pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
    def test_control_flow(self, error):
        @decorators.catch()
        def func():
            raise error("test")

        with pytest.raises(error):
            func()

    @pytest.mark.parametrize(
        "options",
        [{"error": FileExistsError}, {"exclude": FileNotFoundError}],
        ids=["uncaught", "excluded"],
    )
    def test_propagates(self, options):
        @decorators.catch(**options)
        def func():
            raise FileNotFoundError("test")

        with pytest.raises(FileNotFoundError):
            func()
