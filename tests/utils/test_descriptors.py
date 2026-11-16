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

from pathlib import Path

import pytest

from danling.utils import descriptors


class Class:
    __test__ = False

    def __init__(self, root: Path):
        self.root = root

    @descriptors.cached_ensure_dir
    def cached_ensure_dir(self):
        return str(self.root / "temp")

    @descriptors.ensure_dir
    def ensure_dir(self):
        return str(self.root / "temp")

    @descriptors.cached_ensure_parent_dir
    def cached_ensure_parent_dir(self):
        return str(self.root / "temp" / "test")

    @descriptors.ensure_parent_dir
    def ensure_parent_dir(self):
        return str(self.root / "temp" / "test")


class TestPaths:
    @pytest.mark.parametrize(
        "attribute,parent",
        [
            ("ensure_dir", False),
            ("cached_ensure_dir", False),
            ("ensure_parent_dir", True),
            ("cached_ensure_parent_dir", True),
        ],
    )
    def test_recreates_directory(self, tmp_path, attribute, parent):
        paths = Class(tmp_path)
        for _ in range(2):
            value = Path(getattr(paths, attribute))
            directory = value.parent if parent else value
            assert directory.is_dir()
            directory.rmdir()

    @pytest.mark.parametrize("descriptor", [descriptors.ensure_parent_dir, descriptors.cached_ensure_parent_dir])
    def test_relative_path(self, descriptor):
        class Paths:
            def __init__(self):
                self.name = "result.json"

            file = descriptor(lambda self: self.name)

        assert Paths.file is not None
        assert Paths().file == "result.json"
