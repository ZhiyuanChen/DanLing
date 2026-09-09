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

import json
import os
import pickle
from typing import Any

import yaml
from chanfig import FlatDict

from danling.typing import File, PathStr

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


PYTORCH = ("pt", "pth")
NUMPY = ("numpy", "npy", "npz")
JSON = ("json",)
YAML = ("yaml", "yml")
CSV = ("csv", "tsv")
PANDAS = ("pandas", "pd")
PARQUET = ("parquet", "pq")
PICKLE = ("pickle", "pkl")
H5 = ("h5", "hdf5")
EXCEL = ("xlsx", "xls")
XML = ("xml",)
SQL = ("sql",)
PANDAS_SUPPORTED = sum([JSON, YAML, CSV, PANDAS, PARQUET, PICKLE, H5, EXCEL, XML, SQL], ())


def save(obj: Any, file: PathStr, *args: Any, **kwargs: Any) -> File:
    r"""
    Save an object using the backend selected by its filename extension.

    NumPy suffixes are case-insensitive and write to the exact requested path.
    ``.npy`` and ``.numpy`` use NPY encoding. ``.npz`` uses ``numpy.savez``:
    ``obj`` is stored as ``arr_0``, additional positional arrays as ``arr_1``,
    ``arr_2``, etc., and keyword arguments follow NumPy's native contract.

    Args:
        obj: Object accepted by the selected backend.
        file: String, bytes or path-like filename. Bytes paths use the filesystem encoding.
        *args: Positional arguments forwarded to the backend.
        **kwargs: Keyword arguments forwarded to the backend.

    Returns:
        File: The original filename object supplied by the caller.
    """
    path = os.fsdecode(file)
    extension = os.path.splitext(path)[-1].lower()[1:]
    if extension in PYTORCH:
        if not TORCH_AVAILABLE:
            raise ImportError(f"Trying to save {obj} to {file!r} but torch is not installed.")
        torch.save(obj, path, *args, **kwargs)
    elif extension in NUMPY:
        if not NUMPY_AVAILABLE:
            raise ImportError(f"Trying to save {obj} to {file!r} but numpy is not installed.")
        with open(path, "wb") as array_file:
            if extension == "npz":
                numpy.savez(array_file, obj, *args, **kwargs)
            else:
                numpy.save(array_file, obj, *args, **kwargs)
    elif extension in PANDAS:
        if not PANDAS_AVAILABLE:
            raise ImportError(f"Trying to save {obj} to {file!r} but pandas is not installed.")
        pandas.to_pickle(obj, path, *args, **kwargs)
    elif extension in PARQUET:
        if isinstance(obj, pandas.DataFrame):
            obj.to_parquet(path, *args, **kwargs)
        elif not PYARROW_AVAILABLE:
            raise ImportError(f"Trying to save {obj} to {file!r} but pyarrow is not installed.")
        else:
            pyarrow.parquet.write_table(obj, path, *args, **kwargs)
    elif extension in CSV:
        if isinstance(obj, pandas.DataFrame):
            obj.to_csv(path, *args, **kwargs)
        else:
            raise NotImplementedError(f"Trying to save {obj} to {file!r} but is not supported")
    elif extension in JSON:
        if isinstance(obj, FlatDict):
            obj.json(path, *args, **kwargs)
        else:
            with open(path, "w") as fp:
                json.dump(obj, fp, *args, **kwargs)  # type: ignore[arg-type]
    elif extension in YAML:
        if isinstance(obj, FlatDict):
            obj.yaml(path, *args, **kwargs)
        else:
            with open(path, "w") as fp:
                yaml.dump(obj, fp, *args, **kwargs)  # type: ignore[arg-type, call-overload]
    elif extension in PICKLE:
        with open(path, "wb") as fp:
            pickle.dump(obj, fp, *args, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Tying to save {obj} to {file!r} with unsupported extension={extension!r}")
    return file


def load(file: PathStr, *args: Any, **kwargs: Any) -> Any:
    r"""
    Load an object using the backend selected by its filename extension.

    NumPy files return the native ``numpy.load`` result. In particular, an NPZ
    archive returns an ``NpzFile`` that the caller must close, preferably with
    a ``with`` statement. NPY files return an array (or a memory map when requested).

    Args:
        file: String, bytes or path-like filename. Bytes paths use the filesystem encoding.
        *args: Positional arguments forwarded to the backend.
        **kwargs: Keyword arguments forwarded to the backend.

    Returns:
        Any: The object returned by the selected backend.

    Raises:
        ValueError: The filename is not a file or has an unsupported extension.

    Examples:
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> import numpy as np
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "values.npz"
        ...     _ = save(np.array([1, 2]), path)
        ...     with load(path) as archive:
        ...         print(archive["arr_0"].tolist())
        [1, 2]
    """
    if not os.path.isfile(file):
        raise ValueError(f"Trying to load {file!r} but it is not a file.")
    path = os.fsdecode(file)
    extension = os.path.splitext(path)[-1].lower()[1:]
    if extension in PYTORCH:
        if not TORCH_AVAILABLE:
            raise ImportError(f"Trying to load {file!r} but torch is not installed.")
        return torch.load(path, *args, **kwargs)
    if extension in NUMPY:
        if not NUMPY_AVAILABLE:
            raise ImportError(f"Trying to load {file!r} but numpy is not installed.")
        return numpy.load(path, *args, **kwargs)
    if extension in JSON:
        with open(path) as fp:
            return json.load(fp, *args, **kwargs)  # type: ignore[arg-type]
    if extension in YAML:
        with open(path) as fp:
            kwargs.setdefault("Loader", yaml.FullLoader)  # type: ignore[arg-type]
            return yaml.load(fp, *args, **kwargs)  # type: ignore[arg-type]
    if extension in PICKLE:
        with open(path, "rb") as fp:
            return pickle.load(fp, *args, **kwargs)  # type: ignore[arg-type]
    if extension in PANDAS_SUPPORTED:
        return load_pandas(path, *args, **kwargs)
    raise ValueError(f"Tying to load {file!r} with unsupported extension={extension!r}")


def load_pandas(file: PathStr, *args: Any, **kwargs: Any) -> Any:
    r"""
    Load any pandas data file with supported extensions.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(f"Trying to load {file!r} but pandas is not installed.")
    if not os.path.isfile(file):
        raise ValueError(f"Trying to load {file!r} but it is not a file.")
    path = os.fsdecode(file)
    extension = os.path.splitext(path)[-1].lower()[1:]
    if extension in PANDAS or extension in PICKLE:
        return pandas.read_pickle(path, *args, **kwargs)
    if extension in PARQUET:
        return pandas.read_parquet(path, *args, **kwargs)
    if extension in H5:
        return pandas.read_hdf(path, *args, **kwargs)
    if extension in CSV:
        return pandas.read_csv(path, *args, **kwargs)
    if extension in JSON:
        return pandas.read_json(path, *args, **kwargs)
    if extension in EXCEL:
        return pandas.read_excel(path, *args, **kwargs)
    if extension in XML:
        return pandas.read_xml(path, *args, **kwargs)
    if extension in SQL:
        return pandas.read_sql(path, *args, **kwargs)
    raise ValueError(f"Tying to load {file!r} with unsupported extension={extension!r}")


def is_json_serializable(obj: Any) -> bool:
    r"""
    Check if `obj` is JSON serializable.
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
