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


def test_tensor_import_works_without_metrics_or_runner_dependencies() -> None:
    run_isolated("""
        import builtins

        original_import = builtins.__import__

        def import_without_optional_dependencies(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"torchmetrics", "transformers"}:
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_optional_dependencies

        from danling.tensors import NestedTensor
        import torch

        nested = NestedTensor([torch.ones(2), torch.ones(3)])
        assert nested.shape == (2, 3)
        """)


def test_representative_top_level_exports_remain_available() -> None:
    run_isolated("""
        import danling
        from danling import BaseRunner, NestedTensor
        import torch

        nested = NestedTensor([torch.ones(2), torch.ones(3)])
        assert nested.shape == (2, 3)
        assert BaseRunner is danling.runners.BaseRunner
        """)
