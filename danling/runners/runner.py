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
    Dynamic platform-selecting runner that serves as the primary entry point for DanLing.

    The Runner class automatically selects the most appropriate distributed training platform
    based on your configuration and available packages. It dynamically modifies its class
    at initialization to transform into the chosen platform's runner implementation.

    Key features:

    * **Automatic platform selection**: Chooses the best available backend
    * **Dynamic class transformation**: Becomes a TorchRunner, DeepSpeedRunner, or AccelerateRunner
    * **Unified interface**: Provides consistent API across all platforms

    Platform selection logic:

    1. If `config.platform` is "auto" (default):
       - Uses DeepSpeed if available
       - Falls back to PyTorch otherwise
    2. If `config.platform` is explicitly set to "torch", "deepspeed", or "accelerate":
       - Uses the specified platform
       - Raises an error if the required packages are not installed

    Usage Examples:

    ```python
    # Automatic platform selection
    config = Config({"platform": "auto"})
    runner = Runner(config)  # Will use DeepSpeed if available, otherwise PyTorch

    # Explicit platform selection
    config = Config({"platform": "accelerate"})
    runner = Runner(config)  # Will use Accelerate
    ```

    Extension Guidelines:

    * **DO inherit from Runner** when you want to add functionality that should work across all platforms:
      ```python
      class MyRunner(Runner):
          def __init__(self, config):
              super().__init__(config)
              # Your initialization code

          def my_method(self):
              # Custom functionality
              pass
      ```

    * **DON'T inherit from a specific platform runner** (like TorchRunner) unless you're implementing
      a new distributed training framework.

    Args:
        config: Configuration object containing runner settings. The `platform` key
               determines which backend implementation will be used.

    Raises:
        ValueError: If an unknown platform is specified or required packages are missing.

    See Also:
        - [`BaseRunner`][danling.runners.BaseRunner]: Base class with core functionality.
        - [`TorchRunner`][danling.runners.TorchRunner]: PyTorch DDP implementation.
        - [`DeepSpeedRunner`][danling.runners.DeepSpeedRunner]: DeepSpeed implementation.
        - [`AccelerateRunner`][danling.runners.AccelerateRunner]: HuggingFace Accelerate implementation.
    """

    def __new__(cls, config: Config) -> Runner:
        platform = config.get("platform", "auto").lower()

        if platform == "auto":
            platform = "deepspeed" if ds.is_successful() else "torch"
        config["platform"] = platform

        if platform == "accelerate":
            ac.check()
            return super().__new__(type("AccelerateRunner", (cls, AccelerateRunner), {}))
        if platform == "deepspeed":
            ds.check()
            return super().__new__(type("DeepSpeedRunner", (cls, DeepSpeedRunner), {}))
        if platform == "torch":
            return super().__new__(type("TorchRunner", (cls, TorchRunner), {}))

        raise ValueError(f"Unknown platform: {platform}. Valid options are: torch, accelerate, deepspeed")
