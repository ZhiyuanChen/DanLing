from functools import wraps
from inspect import isfunction


def flexible_decorator(maybe_decorator=None):
    def decorator(func):
        @wraps(decorator)
        def wrapper(*args, **kwargs):
            if len(args) == 1 and isfunction(args[0]):
                return func(**kwargs)(args[0])
            return func(*args, **kwargs)

        return wrapper

    if maybe_decorator is None:
        return decorator
    return decorator(maybe_decorator)
