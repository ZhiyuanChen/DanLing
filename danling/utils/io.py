import json
import os
import pickle
from typing import Any, Dict, List

import torch


def load(path: str, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
    """
    Load everything
    """
    if not os.path.isfile(path):
        raise ValueError(f"Trying to load {path} but it is not a file.")
    extension = os.path.splitext(path)[-1].lower()
    if extension == ".json":
        with open(path, "r") as f:
            result = json.load(f, *args, **kwargs)
    elif extension == ".csv":
        import pandas as pd

        result = pd.read_csv(path, *args, **kwargs)
    elif extension == ".pth":
        result = torch.load(path)
    elif extension == ".npy" or extension == ".npz":
        import numpy as np

        result = np.load(path, allow_pickle=True)
    elif extension == ".pkl":
        try:
            try:
                with open(path, "r") as f:
                    result = pickle.load(f, *args, **kwargs)
            except UnicodeDecodeError:
                with open(path, "rb") as f:
                    result = pickle.load(f, *args, **kwargs)
        except:
            import pandas as pd

            result = pd.read_pickle(path, *args, **kwargs)
    else:
        raise ValueError(f"Tying to load {path} with unsupported extension")
    return result


def is_json_serializable(obj: Any) -> bool:
    """
    Check if obj is JSON serializable
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
