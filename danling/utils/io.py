import json
import os
import pickle
from typing import Any, Dict, List

JSON = ("json",)
PYTORCH = ("pt", "pth")
CSV = ("csv",)
NUMPY = ("numpy", "npy", "npz")
PICKLE = ("pickle", "pkl")


def load(path: str, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
    """
    Load everything
    """
    if not os.path.isfile(path):
        raise ValueError(f"Trying to load {path} but it is not a file.")
    extension = os.path.splitext(path)[-1].lower()[1:]
    if extension in PYTORCH:
        from torch import load

        return load(path)
    elif extension in NUMPY:
        import numpy as np

        return np.load(path, allow_pickle=True)
    elif extension in CSV:
        from pandas import read_csv

        return read_csv(path, *args, **kwargs)
    elif extension in JSON:
        with open(path, "r") as f:
            return json.load(f, *args, **kwargs)
    elif extension in PICKLE:
        with open(path, "rb") as f:
            return pickle.load(f, *args, **kwargs)
    else:
        raise ValueError(f"Tying to load {path} with unsupported extension={extension}")


def is_json_serializable(obj: Any) -> bool:
    """
    Check if obj is JSON serializable
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
