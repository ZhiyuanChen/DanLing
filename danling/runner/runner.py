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

from lazy_imports import try_import

from .base_runner import BaseRunner
from .config import Config
from .torch_runner import TorchRunner

with try_import() as ac:
    import accelerate  # noqa: F401

    from .accelerate_runner import AccelerateRunner

with try_import() as ds:
    import deepspeed  # noqa: F401

    from .deepspeed_runner import DeepSpeedRunner


class Runner(BaseRunner):
    r"""
    Dynamic runner class that selects the appropriate platform based on configuration.

    This runner dynamically modifies the `__class__` attribute to adapt to the platform.

    It's safe (and recommended) to inherit from this class to extend the Runner.

    Valid platform options are:

    - "auto" (default)
    - "torch"
    - "accelerate"
    - "deepspeed"

    Examples:
        >>> config = Config({"platform": "accelerate"})
        >>> runner = Runner(config)

    See Also:
        - [`BaseRunner`][danling.runner.BaseRunner]: Base class for all runners.
        - [`TorchRunner`][danling.runner.TorchRunner]: PyTorch runner.
        - [`AccelerateRunner`][danling.runner.AccelerateRunner]: PyTorch runner with Accelerate.
        - [`DeepSpeedRunner`][danling.runner.DeepSpeedRunner]: PyTorch runner with DeepSpeed.
    """

    def __new__(cls, config: Config) -> Runner:
        platform = config.get("platform", "auto").lower()

        if platform == "auto":
            platform = "deepspeed" if ds.is_successful() else "torch"

        if platform == "accelerate":
            ac.check()
            cls = type("AccelerateRunner", (cls, AccelerateRunner), {})
        elif platform == "deepspeed":
            ds.check()
            cls = type("DeepSpeedRunner", (cls, DeepSpeedRunner), {})
        elif platform == "torch":
            cls = type("TorchRunner", (cls, TorchRunner), {})
        else:
            raise ValueError(f"Unknown platform: {platform}. Valid options are: torch, accelerate, deepspeed")

        return super().__new__(cls)
