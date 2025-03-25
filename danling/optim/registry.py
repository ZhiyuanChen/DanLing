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

from contextlib import suppress

import torch
from chanfig import Registry as Registry_
from packaging.version import Version, parse
from torch import optim

try:
    import deepspeed as ds
except ImportError:
    ds = None

torch_version = parse(torch.__version__)


class Registry(Registry_):
    case_sensitive = False

    def build(self, *args, **kwargs) -> torch.optim.Optimizer:
        return super().build(*args, **kwargs)


OPTIMIZERS = Registry()

OPTIMIZERS.register(optim.SGD, "sgd")
OPTIMIZERS.register(optim.ASGD, "asgd")
OPTIMIZERS.register(optim.Adam, "torch_adam")
OPTIMIZERS.register(optim.AdamW, "torch_adamw")
OPTIMIZERS.register(optim.Adadelta, "adadelta")
OPTIMIZERS.register(optim.Adagrad, "adagrad")
OPTIMIZERS.register(optim.Adamax, "adamax")
OPTIMIZERS.register(optim.LBFGS, "lbfgs")
OPTIMIZERS.register(optim.NAdam, "nadam")
OPTIMIZERS.register(optim.RAdam, "radam")
OPTIMIZERS.register(optim.RMSprop, "rmsprop")
OPTIMIZERS.register(optim.Rprop, "rprop")

if torch_version >= Version("2.5.0"):
    OPTIMIZERS.register(optim.Adafactor, "adafactor")

if ds is not None:
    OPTIMIZERS.register(ds.ops.adagrad.DeepSpeedCPUAdagrad, "ds_adagrad")

    OPTIMIZERS.register(ds.ops.adam.FusedAdam, "adam")
    OPTIMIZERS.register(ds.ops.adam.FusedAdam, "adamw")
    OPTIMIZERS.register(ds.ops.adam.FusedAdam, "ds_adam")
    OPTIMIZERS.register(ds.ops.adam.FusedAdam, "ds_adamw")
    with suppress(AttributeError):
        OPTIMIZERS.register(ds.ops.adam.DeepSpeedCPUAdam, "cpu_adam")
        OPTIMIZERS.register(ds.ops.adam.DeepSpeedCPUAdam, "cpu_adamw")
        OPTIMIZERS.register(ds.ops.adam.DeepSpeedCPUAdam, "cpuadam")
        OPTIMIZERS.register(ds.ops.adam.DeepSpeedCPUAdam, "cpuadamw")

    OPTIMIZERS.register(ds.ops.lamb.FusedLamb, "lamb")
    with suppress(AttributeError):
        OPTIMIZERS.register(ds.ops.lamb.DeepSpeedCPULamb, "cpulamb")
        OPTIMIZERS.register(ds.ops.lamb.DeepSpeedCPULamb, "cpu_lamb")

    OPTIMIZERS.register(ds.ops.lion.FusedLion, "lion")
    with suppress(AttributeError):
        OPTIMIZERS.register(ds.ops.lion.DeepSpeedCPULion, "cpulion")
        OPTIMIZERS.register(ds.ops.lion.DeepSpeedCPULion, "cpu_lion")
else:
    OPTIMIZERS.register(optim.Adam, "adam")
    OPTIMIZERS.register(optim.AdamW, "adamw")
