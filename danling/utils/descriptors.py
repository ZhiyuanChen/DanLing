# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.


from __future__ import annotations

from os import makedirs
from os.path import dirname
from threading import RLock
from typing import Any, Callable

_NOT_FOUND = object()


class cached_property(property):
    r"""
    A thread-safe cached property.
    """

    fget: Callable[[Any], Any]
    fset: Callable[[Any, Any], None]
    fdel: Callable[[Any], None]
    doc: str | None
    lock: RLock
    name: str

    def __init__(
        self,
        fget: Callable[[Any], Any] | None = None,
        fset: Callable[[Any, Any], None] | None = None,
        fdel: Callable[[Any], None] | None = None,
        doc: str | None = None,
    ) -> None:
        super().__init__(fget, fset, fdel, doc)
        self.lock = RLock()
        self.name = self.fget.__name__

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        if not hasattr(instance, "__cache__"):
            try:
                instance.__cache__ = {}
            except AttributeError:
                raise TypeError(f"Cannot create `__cache__` for {type(instance).__name__!r} instance")

        cache = instance.__cache__
        name = self.name

        val = cache.get(name, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                val = cache.get(name, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.fget(instance)
                    try:
                        cache[name] = val
                    except TypeError:
                        raise TypeError(f"Cannot assign value to `__cache__` for {type(instance).__name__!r} instance")
        return val


class ensure_dir(property):
    r"""
    Ensure a directory property exists.

    Examples:
        >>> @ensure_dir
        ... def dir(self) -> str:
        ...     return os.path.join("path", "to", "dir")
    """

    def __get__(self, instance, owner=None):
        val = super().__get__(instance, owner)
        makedirs(val, exist_ok=True)
        return val


class ensure_parent_dir(property):
    r"""
    Ensure the parent directory of a property exists.

    Examples:
        >>> @ensure_parent_dir
        ... def file(self) -> str:
        ...     return os.path.join("path", "to", "file")
    """

    def __get__(self, instance, owner=None):
        val = super().__get__(instance, owner)
        makedirs(dirname(val), exist_ok=True)
        return val


class cached_ensure_dir(cached_property):
    r"""
    Ensure a directory property exists, with caching.

    Examples:
        >>> @cached_ensure_dir
        ... def dir(self) -> str:
        ...     return os.path.join("path", "to", "dir")
    """

    def __get__(self, instance, owner=None):
        val = super().__get__(instance, owner)
        makedirs(val, exist_ok=True)
        return val


class cached_ensure_parent_dir(cached_property):
    r"""
    Ensure the parent directory of a property exists, with caching.

    Examples:
        >>> @ensure_parent_dir
        ... def file(self) -> str:
        ...     return os.path.join("path", "to", "file")
    """

    def __get__(self, instance, owner=None):
        val = super().__get__(instance, owner)
        makedirs(dirname(val), exist_ok=True)
        return val
