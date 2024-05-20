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

import sys
from contextlib import contextmanager
from typing import Optional

from danling.typing import Exceptions

try:
    import ipdb as pdb
except ImportError:
    import pdb


@contextmanager
def debug(
    enable: bool = True,
    error: Exceptions = Exception,
    exclude: Optional[Exceptions] = None,
):
    """
    Contextmanager to enter debug mode on `error` except for `exclude`.

    `debug` is intended to be used to catch the error and enter debug mode.
    Since it is mainly for development purposed, we include an `enable` args so that it can be deactivated.

    Args:
        enable: Whether to enable the contextmanager.
            Defaults to `True`.
        error: The error to catch.
            Defaults to `Exception`.
        exclude: The error to exclude.
            Defaults to `None`.
    """

    if not enable:
        yield
        return
    try:
        yield
    except error as exc:  # pylint: disable=broad-exception-caught
        if exclude is not None and isinstance(exc, exclude):
            raise exc
        _, m, tb = sys.exc_info()
        print(repr(m), file=sys.stderr)
        pdb.post_mortem(tb)
    finally:
        pass
