# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

import json
import os
import pickle
from typing import Any, Dict, List

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


def save(obj: Any, file: PathStr, *args: List[Any], **kwargs: Dict[str, Any]) -> File:
    r"""
    Save any file with supported extensions.
    """
    extension = os.path.splitext(file)[-1].lower()[1:]
    if extension in PYTORCH:
        if not TORCH_AVAILABLE:
            raise ImportError(f"Trying to save {obj} to {file!r} but torch is not installed.")
        torch.save(obj, file, *args, **kwargs)
    elif extension in NUMPY:
        if not NUMPY_AVAILABLE:
            raise ImportError(f"Trying to save {obj} to {file!r} but numpy is not installed.")
        numpy.save(file, obj, *args, **kwargs)
    elif extension in PANDAS:
        if not PANDAS_AVAILABLE:
            raise ImportError(f"Trying to save {obj} to {file!r} but pandas is not installed.")
        pandas.to_pickle(obj, file, *args, **kwargs)
    elif extension in PARQUET:
        if isinstance(obj, pandas.DataFrame):
            obj.to_parquet(file, *args, **kwargs)
        elif not PYARROW_AVAILABLE:
            raise ImportError(f"Trying to save {obj} to {file!r} but pyarrow is not installed.")
        else:
            pyarrow.parquet.write_table(obj, file, *args, **kwargs)
    elif extension in CSV:
        if isinstance(obj, pandas.DataFrame):
            obj.to_csv(file, *args, **kwargs)
        else:
            raise NotImplementedError(f"Trying to save {obj} to {file!r} but is not supported")
    elif extension in JSON:
        if isinstance(obj, FlatDict):
            obj.json(file)
        else:
            with open(file, "w") as fp:
                json.dump(obj, fp, *args, **kwargs)  # type: ignore
    elif extension in YAML:
        if isinstance(obj, FlatDict):
            obj.yaml(file)
        else:
            with open(file, "w") as fp:
                yaml.dump(obj, fp, *args, **kwargs)  # type: ignore
    elif extension in PICKLE:
        with open(file, "wb") as fp:
            pickle.dump(obj, fp, *args, **kwargs)  # type: ignore
    else:
        raise ValueError(f"Tying to save {obj} to {file!r} with unsupported extension={extension!r}")
    return file


def load(file: PathStr, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
    r"""
    Load any file with supported extensions.
    """
    if not os.path.isfile(file):
        raise ValueError(f"Trying to load {file!r} but it is not a file.")
    extension = os.path.splitext(file)[-1].lower()[1:]
    if extension in PYTORCH:
        if not TORCH_AVAILABLE:
            raise ImportError(f"Trying to load {file!r} but torch is not installed.")
        return torch.load(file, *args, **kwargs)
    if extension in NUMPY:
        if not NUMPY_AVAILABLE:
            raise ImportError(f"Trying to load {file!r} but numpy is not installed.")
        return numpy.load(file, *args, **kwargs)
    if extension in JSON:
        with open(file) as fp:
            return json.load(fp, *args, **kwargs)  # type: ignore
    if extension in YAML:
        with open(file) as fp:
            return yaml.load(fp, *args, **kwargs)  # type: ignore
    if extension in PICKLE:
        with open(file, "rb") as fp:
            return pickle.load(fp, *args, **kwargs)  # type: ignore
    if extension in PANDAS_SUPPORTED:
        return load_pandas(file, *args, **kwargs)
    raise ValueError(f"Tying to load {file!r} with unsupported extension={extension!r}")


def load_pandas(file: PathStr, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
    r"""
    Load any pandas data file with supported extensions.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(f"Trying to load {file!r} but pandas is not installed.")
    if not os.path.isfile(file):
        raise ValueError(f"Trying to load {file!r} but it is not a file.")
    extension = os.path.splitext(file)[-1].lower()[1:]
    if extension in PANDAS or extension in PICKLE:
        return pandas.read_pickle(file, *args, **kwargs)
    if extension in PARQUET:
        return pandas.read_parquet(file, *args, **kwargs)
    if extension in H5:
        return pandas.read_hdf(file, *args, **kwargs)
    if extension in CSV:
        return pandas.read_csv(file, *args, **kwargs)
    if extension in JSON:
        return pandas.read_json(file, *args, **kwargs)
    if extension in EXCEL:
        return pandas.read_excel(file, *args, **kwargs)
    if extension in XML:
        return pandas.read_xml(file, *args, **kwargs)
    if extension in SQL:
        return pandas.read_sql(file, *args, **kwargs)
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
