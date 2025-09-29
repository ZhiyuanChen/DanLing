# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

import random

import pytest
import torch

from danling.metrics.factory import binary_metrics
from danling.metrics.multitask import MultiTaskMetrics


def test_multitask_nested_sequence_payload_updates():
    random.seed(0)
    torch.random.manual_seed(0)
    metrics = MultiTaskMetrics()
    metrics.group = MultiTaskMetrics()
    metrics.group.a = binary_metrics(mode="stream")
    metrics.group.b = binary_metrics(mode="stream")

    metrics.update(
        {
            "group": {
                "a": (torch.randn(8), torch.randint(2, (8,))),
                "b": (torch.randn(8), torch.randint(2, (8,))),
            }
        }
    )

    assert "group" in metrics.avg
    assert {"acc", "f1"} <= set(metrics.avg["group"]["a"].keys())
    assert {"acc", "f1"} <= set(metrics.avg["group"]["b"].keys())


def test_multitask_set_invalid_does_not_mutate():
    metrics = MultiTaskMetrics()
    assert list(metrics.keys()) == []

    with pytest.raises(ValueError):
        metrics.set("invalid", 1)

    assert list(metrics.keys()) == []
