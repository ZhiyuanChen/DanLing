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

import os
from zipfile import ZipFile

import numpy as np
import pytest
from chanfig import FlatDict

from danling.utils.io import load, save


@pytest.mark.parametrize("extension", ["pt", "npy", "json", "yaml", "pkl"])
@pytest.mark.parametrize("path_type", ["path", "str", "bytes"])
def test_path_roundtrip(tmp_path, extension, path_type):
    path = tmp_path / f"measurements-数据.{extension}"
    file = path if path_type == "path" else str(path) if path_type == "str" else os.fsencode(path)
    value = [1.0, 2.0, 3.0]
    assert save(value, file) is file
    assert path.is_file()
    assert list(load(file)) == value


@pytest.mark.parametrize("extension", ["npy", "numpy", "npz", "NPY", "NUMPY", "NPZ"])
@pytest.mark.parametrize("path_type", ["path", "str", "bytes"])
def test_numpy_exact_path(tmp_path, extension, path_type):
    path = tmp_path / f"measurements-数据.{extension}"
    file = path if path_type == "path" else str(path) if path_type == "str" else os.fsencode(path)
    value = np.array([[1, 2], [3, 4]], dtype=np.int16)
    assert save(value, file, allow_pickle=False) is file
    assert list(tmp_path.iterdir()) == [path]
    if extension.lower() == "npz":
        assert path.read_bytes().startswith(b"PK\x03\x04")
        with ZipFile(path) as archive:
            assert archive.namelist() == ["arr_0.npy"]
            assert archive.read("arr_0.npy").startswith(b"\x93NUMPY")
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == ["arr_0"]
            np.testing.assert_array_equal(archive["arr_0"], value, strict=True)
    else:
        assert path.read_bytes().startswith(b"\x93NUMPY")
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), value, strict=True)


def test_npz_extra_arrays(tmp_path):
    path = tmp_path / "values.npz"
    first = np.array([1, 2], dtype=np.int8)
    second = np.array([3, 4], dtype=np.int16)
    named = np.array([5, 6], dtype=np.float32)
    save(first, path, second, named=named, allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"arr_0", "arr_1", "named"}
        np.testing.assert_array_equal(archive["arr_0"], first, strict=True)
        np.testing.assert_array_equal(archive["arr_1"], second, strict=True)
        np.testing.assert_array_equal(archive["named"], named, strict=True)


def test_load_numpy_archive(tmp_path):
    path = tmp_path / "values.NPZ"
    value = np.array([1.5, 2.5], dtype=np.float32)
    with path.open("wb") as stream:
        np.savez(stream, value)
    archive = load(os.fsencode(path), allow_pickle=False)
    with archive:
        assert isinstance(archive, np.lib.npyio.NpzFile)
        np.testing.assert_array_equal(archive["arr_0"], value, strict=True)
    assert archive.zip is None


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_flatdict_options(tmp_path, extension):
    path = tmp_path / f"saved.{extension}"
    expected = tmp_path / f"native.{extension}"
    value = FlatDict({"z": {"b": 2, "a": 1}, "a": "data"})
    save(value, path, indent=4, sort_keys=True)
    getattr(value, extension)(expected, indent=4, sort_keys=True)
    assert path.read_text() == expected.read_text()
