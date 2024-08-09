# DanLing
# Copyright (C) 2022-Present  DanLing
#
# This file is part of DanLing.
#
# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0
#
# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

import os
from math import ceil
from warnings import warn

import torch
from chanfig import NestedDict


def apply_auto_tuning(
    config,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    cpu_count: int | None = None,
    cuda_available: bool | None = None,
) -> bool:
    auto_tune_cfg = config.get("auto_tune")
    if not auto_tune_cfg:
        return False

    enabled = bool(auto_tune_cfg.get("enabled", False))
    if not enabled or bool(auto_tune_cfg.get("applied", False)):
        return False

    dataloader_cfg = config.setdefault("dataloader", NestedDict())
    local_batch_size = dataloader_cfg.get("batch_size")
    if local_batch_size is None:
        raise ValueError("auto_tune.enabled requires `config.dataloader.batch_size`")
    if int(local_batch_size) <= 0:
        raise ValueError(f"Invalid dataloader.batch_size: {local_batch_size}")
    local_batch_size = int(local_batch_size)

    if local_world_size is None:
        local_world_size = max(int(os.getenv("LOCAL_WORLD_SIZE", "1")), 1)
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if world_size is None:
        world_size = max(int(os.getenv("WORLD_SIZE", "1")), 1)
    else:
        world_size = max(int(world_size), 1)

    if auto_tune_cfg.get("num_workers") == "auto" and dataloader_cfg.get("num_workers") is None:
        max_num_workers = int(auto_tune_cfg.get("max_num_workers", 16))
        num_workers = max(1, min(max_num_workers, cpu_count // local_world_size))
        dataloader_cfg.num_workers = num_workers
        if num_workers > 0 and dataloader_cfg.get("persistent_workers") is None:
            dataloader_cfg.persistent_workers = True
    if cuda_available and dataloader_cfg.get("pin_memory") is None:
        dataloader_cfg.pin_memory = True

    target_global_batch_size = auto_tune_cfg.get("target_global_batch_size")
    if target_global_batch_size is not None:
        target_global_batch_size = int(target_global_batch_size)
        if target_global_batch_size <= 0:
            raise ValueError(f"Invalid target_global_batch_size: {target_global_batch_size}")
        micro_batch_global = local_batch_size * world_size
        accum_steps = max(1, ceil(target_global_batch_size / micro_batch_global))
        config.accum_steps = accum_steps
        if target_global_batch_size % micro_batch_global != 0:
            warn(
                "target_global_batch_size is not divisible by local_batch_size * world_size; "
                "rounded up accum_steps to satisfy minimum global batch target.",
                category=RuntimeWarning,
                stacklevel=2,
            )

    if bool(auto_tune_cfg.get("scale_lr", False)):
        if "optim" not in config or "lr" not in config.optim:
            raise ValueError("auto_tune.scale_lr=True requires `config.optim.lr`")
        base_global_batch_size = auto_tune_cfg.get("base_global_batch_size")
        if base_global_batch_size is None:
            raise ValueError("auto_tune.scale_lr=True requires `auto_tune.base_global_batch_size`")
        base_global_batch_size = int(base_global_batch_size)
        if base_global_batch_size <= 0:
            raise ValueError(f"Invalid base_global_batch_size: {base_global_batch_size}")

        accum_steps = int(config.get("accum_steps", 1))
        effective_global_batch_size = local_batch_size * world_size * accum_steps
        lr_scale = effective_global_batch_size / base_global_batch_size
        lr_scale_method = str(auto_tune_cfg.get("lr_scale_method", "linear")).lower()
        if lr_scale_method == "sqrt":
            lr_scale = lr_scale**0.5
        elif lr_scale_method != "linear":
            raise ValueError(f"Unknown lr_scale_method: {lr_scale_method}")

        config.optim.lr = float(config.optim.lr) * float(lr_scale)
        auto_tune_cfg.lr_scale_factor = float(lr_scale)
        auto_tune_cfg.effective_global_batch_size = int(effective_global_batch_size)

    auto_tune_cfg.applied = True
    return True
