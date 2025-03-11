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

import os

from danling.utils import descriptors


class Class:
    __test__ = False

    @descriptors.cached_ensure_dir
    def cached_ensure_dir(self):
        return "temp"

    @descriptors.ensure_dir
    def ensure_dir(self):
        return "temp"

    @descriptors.cached_ensure_parent_dir
    def cached_ensure_parent_dir(self):
        return os.path.join("temp", "test")

    @descriptors.ensure_parent_dir
    def ensure_parent_dir(self):
        return os.path.join("temp", "test")


class Test:

    def test_ensure_dir(self):
        obj = Class()
        assert os.path.exists(obj.ensure_dir)
        os.rmdir(obj.cached_ensure_dir)
        # Run twice for cached version to ensure caching won't break `makedirs`
        assert os.path.exists(obj.cached_ensure_dir)
        os.rmdir(obj.ensure_dir)
        assert os.path.exists(obj.cached_ensure_dir)
        os.rmdir(obj.ensure_dir)

    def test_ensure_parent_dir(self):
        obj = Class()
        parent_dir = os.path.dirname(obj.ensure_parent_dir)
        assert os.path.exists(parent_dir)
        os.rmdir(parent_dir)
        # Run twice for cached version to ensure caching won't break `makedirs`
        parent_dir = os.path.dirname(obj.cached_ensure_parent_dir)
        assert os.path.exists(parent_dir)
        os.rmdir(parent_dir)
        parent_dir = os.path.dirname(obj.cached_ensure_parent_dir)
        assert os.path.exists(parent_dir)
        os.rmdir(parent_dir)
