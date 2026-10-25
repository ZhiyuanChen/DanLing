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


def test_ensure_dir(tmp_path):
    obj = Class(tmp_path)
    assert Path(obj.ensure_dir).exists()
    Path(obj.cached_ensure_dir).rmdir()
    # Run twice for cached version to ensure caching won't break `makedirs`
    assert Path(obj.cached_ensure_dir).exists()
    Path(obj.ensure_dir).rmdir()
    assert Path(obj.cached_ensure_dir).exists()
    Path(obj.ensure_dir).rmdir()


def test_ensure_parent_dir(tmp_path):
    obj = Class(tmp_path)
    parent_dir = Path(obj.ensure_parent_dir).parent
    assert parent_dir.exists()
    parent_dir.rmdir()
    # Run twice for cached version to ensure caching won't break `makedirs`
    parent_dir = Path(obj.cached_ensure_parent_dir).parent
    assert parent_dir.exists()
    parent_dir.rmdir()
    parent_dir = Path(obj.cached_ensure_parent_dir).parent
    assert parent_dir.exists()
    parent_dir.rmdir()


def test_directory_properties_support_class_access_and_relative_basename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Paths:
        @descriptors.ensure_parent_dir
        def file(self):
            return "result.json"

        @descriptors.cached_ensure_parent_dir
        def cached_file(self):
            return "cached.json"

    assert Paths.file is not None
    assert Paths.cached_file is not None
    paths = Paths()
    assert paths.file == "result.json"
    assert paths.cached_file == "cached.json"
