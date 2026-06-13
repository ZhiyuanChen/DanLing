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

# Makes `tests` importable for helper modules.


import pytest

from tests.tensors.utils import FLOAT_DTYPES, available_devices

collect_ignore = ["nn_overrides.py"]


@pytest.fixture(autouse=True)
def restore_builtin_print():
    """Isolate tests from the runner's global ``builtins.print`` monkeypatch.

    A runner installs a print router on ``builtins.print`` and only restores it on
    ``close()``. A test that builds a runner without closing it would leak an active
    router that swallows later tests' ``print`` output into a stale logger (this is
    how ``test_profiler_close_reports_artifacts`` saw empty stdout in the full suite).
    Snapshotting and restoring ``builtins.print`` around every test keeps them isolated.
    """
    import builtins

    saved = builtins.print
    try:
        yield
    finally:
        builtins.print = saved


@pytest.fixture(autouse=True)
def seed_all():
    import random

    import torch

    random.seed(1016)
    torch.manual_seed(1016)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1016)


@pytest.fixture(params=available_devices())
def device(request):
    return request.param


@pytest.fixture(params=FLOAT_DTYPES)
def float_dtype(request):
    import torch

    dtype = request.param
    device = request.getfixturevalue("device")
    if dtype in (torch.float16, torch.bfloat16) and device.type != "cuda":
        pytest.skip(f"{dtype} unsupported on CPU for these ops")
    return dtype
