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
    from .accelerate_runner import AccelerateRunner

with try_import() as ds:
    from .deepspeed_runner import DeepSpeedRunner


class Runner(BaseRunner):
    r"""
    Dynamic runner class that selects the appropriate platform based on configuration.

    This runner dynamically changes its class to combine with the appropriate platform
    (torch, accelerate, or deepspeed) based on the 'platform' configuration option.

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
        - [`TorchRunner`][danling.runner.TorchRunner]: Runner for PyTorch.
        - [`AccelerateRunner`][danling.runner.AccelerateRunner]: Runner for Accelerate.
        - [`DeepSpeedRunner`][danling.runner.DeepSpeedRunner]: Runner for DeepSpeed.
    """

    def __init__(self, config: Config) -> None:
        platform = config.get("platform", "auto").lower()

        if platform == "auto":
            platform = "deepspeed" if ds.is_successful() else "torch"

        if platform == "accelerate":
            ac.check()
            self.__class__ = type("AccelerateRunner", (self.__class__, AccelerateRunner), {})
        elif platform == "deepspeed":
            ds.check()
            self.__class__ = type("DeepSpeedRunner", (self.__class__, DeepSpeedRunner), {})
        elif platform == "torch":
            self.__class__ = type("TorchRunner", (self.__class__, TorchRunner), {})
        else:
            raise ValueError(f"Unknown platform: {platform}. Valid options are: torch, accelerate, deepspeed")

        super().__init__(config)
