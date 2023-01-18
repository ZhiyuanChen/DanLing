import os
from json import dumps as json_dumps
from json import load as json_load
from pickle import load as pickle_load
from typing import Any, Dict, List

try:
    from torch import load as torch_load
except ImportError:
    torch_load = None
try:
    from numpy import load as numpy_load
except ImportError:
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


def load(path: str, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
    """
    Load any file with supported extensions.
    """
    if not os.path.isfile(path):
        raise ValueError(f"Trying to load {path} but it is not a file.")
    extension = os.path.splitext(path)[-1].lower()[1:]
    if extension in PYTORCH:
        if torch_load is None:
            raise ImportError(f"Trying to load {path} but torch is not installed.")
        return torch_load(path)
    if extension in NUMPY:
        if numpy_load is None:
            raise ImportError(f"Trying to load {path} but numpy is not installed.")
        return numpy_load(path, allow_pickle=True)
    if extension in CSV:
        if read_csv is None:
            raise ImportError(f"Trying to load {path} but pandas is not installed.")
        return read_csv(path, *args, **kwargs)
    if extension in JSON:
        with open(path, "r") as fp:  # pylint: disable=W1514, C0103
            return json_load(fp, *args, **kwargs)  # type: ignore
    if extension in PICKLE:
        with open(path, "rb") as fp:  # pylint: disable=C0103
            return pickle_load(fp, *args, **kwargs)  # type: ignore
    raise ValueError(f"Tying to load {path} with unsupported extension={extension}")


def is_json_serializable(obj: Any) -> bool:
    """
    Check if `obj` is JSON serializable.
    """
    try:
        json_dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
