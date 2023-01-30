from functools import lru_cache, wraps
from inspect import isfunction
from os import makedirs
from os.path import abspath
from sys import stderr
from traceback import format_exc
from typing import Callable, Optional
from weakref import ref


def flexible_decorator(maybe_decorator: Optional[Callable] = None):
    """
    Decorator to allow bracket-less when no arguments are passed.

    Examples:
    For decorator defined as follows:

    ```python
    >>> @flexible_decorator
    >>> def decorator(*args, **kwargs):
    ...     pass
    ```

    The following two are equivalent:

    ```python
    >>> @decorator
    >>> def func(*args, **kwargs):
    ...     pass
    ```

    ```python
    >>> @decorator()
    >>> def func(*args, **kwargs):
    ...     pass
    ```

    """

    def decorator(func: Callable):
        @wraps(decorator)
        def wrapper(*args, **kwargs):
            if len(args) == 1 and isfunction(args[0]):
                return func(**kwargs)(args[0])
            return func(*args, **kwargs)

        return wrapper

    if maybe_decorator is None:
        return decorator
    return decorator(maybe_decorator)


@flexible_decorator
def catch(error: Exception = Exception, exclude: Optional[Exception] = None, print_args: bool = False):
    """
    Decorator to catch `error` except for `exclude`.
    Detailed traceback will be printed to `stderr`.

    `catch` is extremely useful for unfatal errors.
    For example, `Runner` saves checkpoint regularly, however, this might break running if the space is full.
    Decorating `save` method with `catch` will allow you to catch these errors and continue your running.

    Args:
        error:
        exclude:
        print_args: Whether to print the arguments passed to the function.
    """

    def decorator(func, error: Exception = Exception, exclude: Optional[Exception] = None, print_args: bool = False):
        @wraps(func)
        def wrapper(*args, **kwargs):  # pylint: disable=R1710
            try:
                return func(*args, **kwargs)
            except error as exc:  # pylint: disable=W0703
                if exclude is not None and isinstance(exc, exclude):
                    raise exc
                message = format_exc()
                message += f"\nencoutered when calling {func}"
                if print_args:
                    message += (f"with args {args} and kwargs {kwargs}",)
                print(message, file=stderr, force=True)

        return wrapper

    return lambda func: decorator(func, error, exclude, print_args)


def method_cache(*cache_args, **lru_kwargs):
    r"""
    Decorator to cache the result of an instance method.

    `functools.lru_cache` uses a strong reference to the instance,
    which will make the instance immortal and break the garbage collection.

    `method_cache` uses a weak reference to the instance and works fine.

    https://rednafi.github.io/reflections/dont-wrap-instance-methods-with-functoolslru_cache-decorator-in-python.html
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self_ref = ref(self)

            @wraps(func)
            @lru_cache(*cache_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_ref(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapper

    return decorator


def ensure_dir(func):
    """
    Decorator to ensure a directory property exists.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        path = abspath(func(*args, **kwargs))
        makedirs(path, exist_ok=True)
        return path

    return wrapper
