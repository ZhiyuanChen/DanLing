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

from danling.runners.compile import Compiler
from danling.runners.config import CompileConfig


class TestCompileConfig:

    def test_compiler_uses_single_enable_flag(self) -> None:
        compiler = Compiler(CompileConfig({"enable": True}))

        assert compiler.enabled is True

    def test_disabled_by_default(self) -> None:
        assert Compiler(CompileConfig()).enabled is False
        assert Compiler(CompileConfig({"enable": False})).enabled is False

    def test_kwargs_forward_runtime_options(self) -> None:
        compiler = Compiler(
            CompileConfig(
                {
                    "backend": "inductor",
                    "mode": "max-autotune",
                    "fullgraph": True,
                    "dynamic": False,
                    "options": {"trace": {"enabled": True}},
                }
            )
        )

        assert compiler.kwargs == {
            "backend": "inductor",
            "mode": "max-autotune",
            "fullgraph": True,
            "dynamic": False,
            "options": {"trace": {"enabled": True}},
        }

    def test_kwargs_reject_non_mapping_options(self) -> None:
        with pytest.raises(TypeError, match="options"):
            CompileConfig({"options": ["trace"]})


class TestDdpOptimizerContext:

    def test_restores_dynamo_config(self) -> None:
        dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
        if dynamo_config is None or not hasattr(dynamo_config, "optimize_ddp"):
            pytest.skip("torch._dynamo.config.optimize_ddp is not available")

        previous = dynamo_config.optimize_ddp
        compiler = Compiler(CompileConfig({"enable": True, "optimize_ddp": "python_reducer"}))

        with compiler.ddp_optimizer():
            assert dynamo_config.optimize_ddp == "python_reducer"

        assert dynamo_config.optimize_ddp == previous

    def test_noops_when_compile_is_disabled(self) -> None:
        dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
        if dynamo_config is None or not hasattr(dynamo_config, "optimize_ddp"):
            pytest.skip("torch._dynamo.config.optimize_ddp is not available")

        previous = dynamo_config.optimize_ddp
        compiler = Compiler(CompileConfig({"enable": False, "optimize_ddp": "python_reducer"}))

        with compiler.ddp_optimizer():
            assert dynamo_config.optimize_ddp == previous

        assert dynamo_config.optimize_ddp == previous
