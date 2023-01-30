import os
from json import dump as json_dump
from json import dumps as json_dumps
from json import load as json_load
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from typing import Any, Dict, List

from danling.typing import File

try:
    from torch import load as torch_load
    from torch import save as torch_save
except ImportError:
    torch_save = None
    torch_load = None
try:
    from numpy import load as numpy_load
    from numpy import save as numpy_save
except ImportError:
    numpy_save = None
    numpy_load = None
try:
    from pandas import read_csv
except ImportError:
    read_csv = None

JSON = ("json",)
PYTORCH = ("pt", "pth")
CSV = ("csv",)
NUMPY = ("numpy", "npy", "npz")
PICKLE = ("pickle", "pkl")


def save(obj: Any, file: File, *args: List[Any], **kwargs: Dict[str, Any]) -> File:
    r"""
    Save any file with supported extensions.
    """
    extension = os.path.splitext(file)[-1].lower()[1:]
    if extension in PYTORCH:
        if torch_save is None:
            raise ImportError(f"Trying to save {obj} to {file} but torch is not installed.")
        torch_save(obj, file, *args, **kwargs)
    elif extension in NUMPY:
        if numpy_save is None:
            raise ImportError(f"Trying to save {obj} to {file} but numpy is not installed.")
        numpy_save(file, obj, *args, **kwargs)
    elif extension in CSV:
        obj.to_csv(file, *args, **kwargs)
    elif extension in JSON:
        with open(file, "w") as fp:  # pylint: disable=W1514, C0103
            json_dump(obj, fp, *args, **kwargs)  # type: ignore
    elif extension in PICKLE:
        with open(file, "wb") as fp:  # pylint: disable=C0103
            pickle_dump(obj, fp, *args, **kwargs)  # type: ignore
    else:
        raise ValueError(f"Tying to save {obj} to {file} with unsupported extension={extension}")
    return file


def load(file: File, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
    r"""
    Load any file with supported extensions.
    """
    if not os.path.isfile(file):
        raise ValueError(f"Trying to load {file} but it is not a file.")
    extension = os.path.splitext(file)[-1].lower()[1:]
    if extension in PYTORCH:
        if torch_load is None:
            raise ImportError(f"Trying to load {file} but torch is not installed.")
        return torch_load(file)
    if extension in NUMPY:
        if numpy_load is None:
            raise ImportError(f"Trying to load {file} but numpy is not installed.")
        return numpy_load(file, allow_pickle=True)
    if extension in CSV:
        if read_csv is None:
            raise ImportError(f"Trying to load {file} but pandas is not installed.")
        return read_csv(file, *args, **kwargs)
    if extension in JSON:
        with open(file, "r") as fp:  # pylint: disable=W1514, C0103
            return json_load(fp, *args, **kwargs)  # type: ignore
    if extension in PICKLE:
        with open(file, "rb") as fp:  # pylint: disable=C0103
            return pickle_load(fp, *args, **kwargs)  # type: ignore
    raise ValueError(f"Tying to load {file} with unsupported extension={extension}")


def is_json_serializable(obj: Any) -> bool:
    r"""
    Check if `obj` is JSON serializable.
    """
    try:
        json_dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
