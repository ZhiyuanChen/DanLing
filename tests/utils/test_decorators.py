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


def test_catch_error():
    @decorators.catch()
    def func():
        raise Exception("test")

    func()


def test_catch_interrupt():
    @decorators.catch()
    def func():
        raise KeyboardInterrupt("test")

    with pytest.raises(KeyboardInterrupt):
        func()


def test_catch_exit():
    @decorators.catch()
    def func():
        raise SystemExit("test")

    with pytest.raises(SystemExit):
        func()


def test_catch_raise():
    @decorators.catch(FileExistsError)
    def func():
        raise FileNotFoundError("test")

    with pytest.raises(FileNotFoundError):
        func()


def test_catch_exclude():
    @decorators.catch(exclude=FileNotFoundError)
    def func():
        raise FileNotFoundError("test")

    with pytest.raises(FileNotFoundError):
        func()
