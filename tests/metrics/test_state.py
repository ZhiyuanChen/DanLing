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

from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

from danling.metrics.state import MetricState


class TestCollectRequirements:
    def test_requires_nonempty(self):
        with pytest.raises(ValueError, match="At least one metric function is required."):
            MetricState.collect_requirements([], require_nonempty=True)

    def test_detects_conflicting_scalars(self):
        metrics = [
            SimpleNamespace(
                confmat=True,
                preds_targets=False,
                task="binary",
                num_classes=None,
                num_labels=None,
                threshold=0.5,
                ignore_index=None,
            ),
            SimpleNamespace(
                confmat=True,
                preds_targets=False,
                task="binary",
                num_classes=None,
                num_labels=None,
                threshold=0.3,
                ignore_index=None,
            ),
        ]

        with pytest.raises(ValueError, match="Conflicting metric requirement for threshold"):
            MetricState.collect_requirements(metrics)

    def test_requires_task_for_confmat(self):
        metric = SimpleNamespace(
            confmat=True,
            preds_targets=False,
            task=None,
            num_classes=None,
            num_labels=None,
            threshold=None,
            ignore_index=None,
        )

        with pytest.raises(ValueError, match="Confusion matrix computation requires a task to be specified."):
            MetricState.collect_requirements([metric])


class TestComputeConfmat:
    def test_short_circuits_when_not_applicable(self):
        requirements = {"confmat": True, "task": "binary"}
        assert MetricState.compute_confmat([0, 1], [0, 1], requirements) is None
        assert MetricState.compute_confmat(torch.tensor([0, 1]), torch.tensor([0, 1]), {"confmat": False}) is None

    def test_requires_task(self):
        with pytest.raises(ValueError, match="Confusion matrix computation requires a task to be specified."):
            MetricState.compute_confmat(torch.tensor([0, 1]), torch.tensor([0, 1]), {"confmat": True, "task": None})

    def test_binary(self):
        preds = torch.tensor([0, 1])
        targets = torch.tensor([0, 1])
        requirements = {
            "confmat": True,
            "task": "binary",
            "num_classes": None,
            "num_labels": None,
            "threshold": 0.2,
            "ignore_index": -1,
        }

        assert_close(MetricState.compute_confmat(preds, targets, requirements), torch.tensor([[1, 0], [0, 1]]))


class TestFromRequirements:
    def test_populates_confmat(self):
        state = MetricState.from_requirements(
            torch.tensor([0, 1]), torch.tensor([0, 1]), {"confmat": True, "task": "binary"}
        )

        assert_close(state.confmat, torch.tensor([[1, 0], [0, 1]]))
