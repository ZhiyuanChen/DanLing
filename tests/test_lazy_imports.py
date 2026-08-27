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

import subprocess
import sys
import textwrap


def run_isolated(code: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


def test_tensor_import_does_not_load_metrics_or_runners() -> None:
    run_isolated("""
        import sys

        from danling.tensors import NestedTensor

        assert NestedTensor.__module__ == "danling.tensors.nested_tensor"
        assert "danling.metrics" not in sys.modules
        assert "danling.runners" not in sys.modules
        assert "torchmetrics" not in sys.modules
        assert "transformers" not in sys.modules
        """)


def test_top_level_exports_are_lazy_and_cached() -> None:
    run_isolated("""
        import sys

        import danling

        expected = [
            "RunnerConfig",
            "Runner",
            "BaseRunner",
            "RunnerState",
            "OPTIMIZERS",
            "SCHEDULERS",
            "LRScheduler",
            "TorchRunner",
            "DeepSpeedRunner",
            "ParallelRunner",
            "METRICS",
            "GlobalMetrics",
            "MultiTaskMetrics",
            "MetricMeter",
            "StreamMetrics",
            "AverageMeter",
            "AverageMeters",
            "NestedTensor",
            "PNTensor",
            "tensor",
            "to_device",
            "save",
            "load",
            "load_pandas",
            "catch",
            "debug",
            "flexible_decorator",
            "method_cache",
            "ensure_dir",
            "is_json_serializable",
        ]

        assert danling.__all__ == expected
        assert set(expected) <= set(dir(danling))
        assert "NestedTensor" not in vars(danling)
        assert "BaseRunner" not in vars(danling)
        assert "danling.tensors" not in sys.modules
        assert "danling.metrics" not in sys.modules
        assert "danling.runners" not in sys.modules

        nested_tensor = danling.NestedTensor

        assert nested_tensor.__module__ == "danling.tensors.nested_tensor"
        assert vars(danling)["NestedTensor"] is nested_tensor
        assert "danling.tensors" in sys.modules
        assert "danling.metrics" not in sys.modules
        assert "danling.runners" not in sys.modules
        assert "torchmetrics" not in sys.modules
        assert "transformers" not in sys.modules
        """)


def test_representative_runner_export_remains_available() -> None:
    run_isolated("""
        import danling

        base_runner = danling.BaseRunner

        assert base_runner.__module__ == "danling.runners.base_runner"
        assert vars(danling)["BaseRunner"] is base_runner
        assert danling.runners.BaseRunner is base_runner
        """)
