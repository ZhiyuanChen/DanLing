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

from collections.abc import Mapping
from typing import Any

import torch

from .utils import get_precision


def normalize_policy_dtype(dtype: object) -> object:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).strip().lower().replace("-", "_")
    if normalized in {"float", "float32", "fp32"}:
        return torch.float32
    return get_precision(normalized)


def build_mixed_precision_policy(
    *,
    policy: object,
    mixed_precision_policy_cls: type | None,
    label: str,
) -> object | None:
    if policy is None:
        return None
    if mixed_precision_policy_cls is not None and isinstance(policy, mixed_precision_policy_cls):
        return policy
    if not isinstance(policy, Mapping):
        raise ValueError(f"{label} must be a mapping or MixedPrecisionPolicy instance")
    if mixed_precision_policy_cls is None:
        raise RuntimeError(f"cannot build {label}: MixedPrecisionPolicy is unavailable")

    policy_kwargs = dict(policy)
    for key in ("param_dtype", "reduce_dtype", "output_dtype"):
        if key in policy_kwargs:
            policy_kwargs[key] = normalize_policy_dtype(policy_kwargs[key])
    return mixed_precision_policy_cls(**policy_kwargs)


def build_offload_policy(
    *,
    policy: object,
    cpu_offload_policy_cls: type | None,
    label: str,
) -> object | None:
    if policy is None:
        return None
    if cpu_offload_policy_cls is not None and isinstance(policy, cpu_offload_policy_cls):
        return policy
    if not isinstance(policy, Mapping):
        raise ValueError(f"{label} must be a mapping or CPUOffloadPolicy instance")
    if cpu_offload_policy_cls is None:
        raise RuntimeError(f"cannot build {label}: CPUOffloadPolicy is unavailable")
    return cpu_offload_policy_cls(**dict(policy))


def build_fsdp2_kwargs(
    *,
    config: Mapping[str, Any],
    mesh: object,
    mixed_precision_policy: object | None,
    offload_policy: object | None,
    config_name: str,
    supported_keys: set[str],
    support_hint: str,
) -> dict[str, Any]:
    unsupported = sorted(key for key in config.keys() if key not in supported_keys)
    if unsupported:
        unsupported_text = ", ".join(unsupported)
        raise ValueError(
            f"unsupported {config_name} configuration for FSDP2: "
            f"{unsupported_text}. Use FSDP2 keys such as {support_hint}."
        )

    reshard_after_forward = config.get("reshard_after_forward")
    if reshard_after_forward is None:
        reshard_after_forward = True

    fsdp_kwargs: dict[str, Any] = {
        "mesh": mesh,
        "reshard_after_forward": reshard_after_forward,
    }

    shard_placement_fn = config.get("shard_placement_fn")
    if shard_placement_fn is not None:
        fsdp_kwargs["shard_placement_fn"] = shard_placement_fn

    if mixed_precision_policy is not None:
        fsdp_kwargs["mp_policy"] = mixed_precision_policy

    if offload_policy is not None:
        fsdp_kwargs["offload_policy"] = offload_policy

    ignored_params = config.get("ignored_params")
    if ignored_params is not None:
        fsdp_kwargs["ignored_params"] = set(ignored_params)

    return fsdp_kwargs
