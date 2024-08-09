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

import pytest
import torch

from danling.runners.fsdp import build_fsdp2_kwargs, normalize_policy_dtype


def test_normalize_policy_dtype_accepts_common_precision_aliases() -> None:
    assert normalize_policy_dtype(None) is None
    assert normalize_policy_dtype(torch.float16) is torch.float16
    assert normalize_policy_dtype("fp32") is torch.float32
    assert normalize_policy_dtype("bf16") is torch.bfloat16


def test_build_fsdp2_kwargs_translates_supported_runtime_options() -> None:
    ignored_param = torch.nn.Parameter(torch.ones(1))

    kwargs = build_fsdp2_kwargs(
        config={
            "enabled": True,
            "mode": "full_shard",
            "reshard_after_forward": False,
            "ignored_params": [ignored_param],
        },
        mesh="mesh",
        mixed_precision_policy="mp",
        offload_policy="offload",
        config_name="fsdp",
        supported_keys={"enabled", "mode", "reshard_after_forward", "ignored_params"},
        support_hint="mesh/reshard_after_forward/mp_policy/offload_policy",
    )

    assert kwargs["mesh"] == "mesh"
    assert kwargs["reshard_after_forward"] is False
    assert kwargs["mp_policy"] == "mp"
    assert kwargs["offload_policy"] == "offload"
    assert kwargs["ignored_params"] == {ignored_param}


def test_build_fsdp2_kwargs_rejects_unsupported_options() -> None:
    with pytest.raises(ValueError, match="unsupported fsdp configuration"):
        build_fsdp2_kwargs(
            config={"process_group": object()},
            mesh="mesh",
            mixed_precision_policy=None,
            offload_policy=None,
            config_name="fsdp",
            supported_keys=set(),
            support_hint="mesh/reshard_after_forward/mp_policy/offload_policy",
        )
