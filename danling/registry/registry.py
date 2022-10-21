from functools import wraps
from typing import Callable, Optional

from chanfig import NestedDict


class Registry(NestedDict):
    """
    Registry for components
    """

    def __init__(self, name):
        super().__init__()
        self.setattr("__name__", name)

    def register(self, value: Optional[Callable] = None, name: Optional[str] = None):
        # Register.register()
        if name is not None:
            self.set(name, value)

        # @Register.register()
        @wraps(self.register)
        def register(value, name=None):
            if name is None:
                name = value.__name__
            self.set(name, value)
            return value

        # @Register.register
        if callable(value) and name is None:
            return register(value)

        return lambda x: register(x, value)

    lookup = NestedDict.get


GlobalRegistry = Registry("Global")
