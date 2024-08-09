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

"""
Runner entrypoints and backend implementations.

The public hierarchy is intentionally small:

```text
Runner  (dynamic dispatcher: picks stack from RunnerConfig)
  └── BaseRunner  (shared config, state, workspace, checkpoint, logging)
        └── TorchRunner  (PyTorch/DDP train/eval/infer loops)
              ├── GraphRunner  (experimental graph-backed train step)
              ├── DeepSpeedRunner  (DeepSpeed engine + ZeRO checkpoint layout)
              └── ParallelRunner   (FSDP2, pipeline, tensor/context/expert axes)
```

Use `Runner(config)` when user code should select the backend from
`config.stack`. Inherit from `Runner` when downstream projects need the same
dynamic backend dispatch. Inherit from a concrete runner (`TorchRunner`,
`DeepSpeedRunner`, or `ParallelRunner`) when the backend is fixed.

Runner docs focus on override contracts rather than doctests. The important
questions are lifecycle-oriented: which hook is called when, which attributes
are already bound, which side effects the runner owns, and which actions an
override must leave to the surrounding loop.
"""

from .base_runner import BaseRunner
from .config import RunnerConfig
from .deepspeed_runner import DeepSpeedRunner
from .graph_runner import GraphRunner
from .mixins import Fp8Mixin
from .parallel_runner import ParallelRunner
from .runner import Runner
from .state import RunnerState
from .torch_runner import TorchRunner
from .utils import on_local_main_process, on_main_process
from .workspace import RunnerWorkspace

__all__ = [
    "RunnerConfig",
    "Runner",
    "BaseRunner",
    "TorchRunner",
    "GraphRunner",
    "DeepSpeedRunner",
    "Fp8Mixin",
    "ParallelRunner",
    "RunnerState",
    "RunnerWorkspace",
    "on_main_process",
    "on_local_main_process",
]
