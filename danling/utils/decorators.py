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

from __future__ import annotations

import traceback
from functools import lru_cache, wraps
from inspect import isfunction
from os import makedirs
from os.path import abspath
from sys import stderr
from typing import Callable, Optional
from weakref import ref

from danling.typing import Exceptions


def flexible_decorator(maybe_decorator: Optional[Callable] = None):
    r"""
    Meta decorator to allow bracket-less decorator when no arguments are passed.

    Examples:
        For decorator defined as follows:

        >>> @flexible_decorator
        ... def decorator(*args, **kwargs):
        ...     def wrapper(func, *args, **kwargs):
        ...         pass
        ...     return wrapper

        The following two are equivalent:

        >>> @decorator
        ... def func(*args, **kwargs):
        ...     pass

        >>> @decorator()
        ... def func(*args, **kwargs):
        ...     pass
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 1 and isfunction(args[0]):
                return func(**kwargs)(args[0])
            return func(*args, **kwargs)

        return wrapper

    if maybe_decorator is None:
        return decorator
    return decorator(maybe_decorator)


def print_exc(exc, func, args, kwargs, verbosity: int = 40):  # pylint: disable=W0613
    r"""
    Print exception raised by `func` with `args` and `kwargs` to `stderr`.
    This function serves as the default callback for catch.

    Args:
        verbosity: What level of traceback to print.
            0-: No traceback.
            0-10: Full information of arguments and key word arguments.
            10-20: Stack trace to function calls.
            40+: Function name and error messages.
    """

    if verbosity >= 0:
        message = traceback.format_exc()
        message += f"\nencoutered when calling {func}"
        if verbosity <= 20:
            message += "\n\nstack:\n" + "\n".join(traceback.format_stack()[:-2])
        if verbosity <= 10:
            message += "\n" + f"args: {args}\nkwargs: {kwargs}"
        try:
            print(message, file=stderr, force=True)  # type: ignore
        except TypeError:
            print(message, file=stderr)


@flexible_decorator
def catch(  # pylint: disable=keyword-arg-before-vararg
    error: Exceptions = Exception,
    exclude: Exceptions | None = None,
    callback: Callable = print_exc,
    *callback_args,
    **callback_kwargs,
):
    r"""
    Decorator to catch `error` except for `exclude`.
    Detailed traceback will be printed to `stderr`.

    `catch` is extremely useful for unfatal errors.
    For example, `Runner` saves checkpoint regularly, however, this might break running if the space is full.
    Decorating `save` method with `catch` will allow you to catch these errors and continue your running.

    Args:
        error: Exceptions to be caught.
        exclude: Exceptions to be excluded.
        callback: Callback to be called when an error occurs.
            The first four arguments to `callback` are `exc`, `func`, `args`, `kwargs`.
            Additional arguments should be passed with `*callback_args` and `**callback_kwargs`.
        callback_args: Arguments to be passed to `callback`.
        callback_kwargs: Keyword arguments to be passed to `callback`.

    Examples:
        >>> def file_not_found(*args, **kwargs):
        ...     raise FileNotFoundError
        >>> func = file_not_found
        >>> func()
        Traceback (most recent call last):
        FileNotFoundError
        >>> func = catch(OSError)(file_not_found)
        >>> func()
        >>> func = catch(IOError)(file_not_found)
        >>> func()
        >>> func = catch(ZeroDivisionError)(file_not_found)
        >>> func()
        Traceback (most recent call last):
        FileNotFoundError
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
            try:
                return func(*args, **kwargs)
            except error as exc:  # pylint: disable=broad-exception-caught
                if exclude is not None and isinstance(exc, exclude):
                    raise exc
                callback(exc, func, args, kwargs, *callback_args, **callback_kwargs)

        return wrapper

    decorator.__doc__ = catch.__doc__

    return decorator


@flexible_decorator
def method_cache(maxsize: int | None = 128, typed: bool = False):
    r"""
    Decorator to cache the result of an instance method.

    `functools.lru_cache` uses a strong reference to the instance,
    which will make the instance immortal and break the garbage collection.

    `method_cache` uses a weak reference to the instance to resolve this issue.

    See Also:
        https://rednafi.github.io/reflections/dont-wrap-instance-methods-with-functoolslru_cache-decorator-in-python.html
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self_ref = ref(self)

            @wraps(func)
            @lru_cache(maxsize=maxsize, typed=typed)
            def cached_method(*args, **kwargs):
                return func(self_ref(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapper

    return decorator


def ensure_dir(func):
    r"""
    Decorator to ensure a directory property exists.

    Note:
        Please avoid using this with `cached_property`.

    Examples:
        >>> @property
        ... @ensure_dir
        ... def dir(self) -> str:
        ...     return os.path.join("path", "to", "dir")
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        path = abspath(func(*args, **kwargs))
        makedirs(path, exist_ok=True)
        return path

    return wrapper
