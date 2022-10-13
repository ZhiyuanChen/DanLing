from functools import wraps
from os import makedirs
from os.path import abspath


def on_main_process(func):
    """
    Run func on main process
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_main_process or not self.distributed:
            return func(self, *args, **kwargs)

    return wrapper


def on_local_main_process(func):
    """
    Run func on local main process
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_local_main_process or not self.distributed:
            return func(self, *args, **kwargs)

    return wrapper


def ensure_dir(func):
    """
    Ensure directory exists
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        path = abspath(func(*args, **kwargs))
        makedirs(path, exist_ok=True)
        return path

    return wrapper
