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


@pytest.fixture
def dynamo_config():
    config = torch._dynamo.config
    if not hasattr(config, "optimize_ddp"):
        pytest.skip("DDP optimizer configuration is unavailable")
    return config


class TestCompiler:
    @pytest.mark.parametrize("settings,enabled", [({}, False), ({"enabled": False}, False), ({"enabled": True}, True)])
    def test_enabled(self, settings, enabled):
        assert Compiler(CompileConfig(settings)).enabled is enabled

    def test_options(self):
        settings = {
            "backend": "inductor",
            "mode": "max-autotune",
            "fullgraph": True,
            "dynamic": False,
            "options": {"trace": {"enabled": True}},
        }
        assert Compiler(CompileConfig(settings)).kwargs == settings

    def test_artifact_fingerprint(self):
        compiler = Compiler(
            CompileConfig(
                {"enabled": True, "precompile_artifact_dir": "/tmp/danling-compile", "memory_policy": "cpu_offload_all"}
            )
        )
        assert compiler.precompile_artifact_dir == "/tmp/danling-compile"
        assert compiler.memory_policy == "cpu_offload_all"
        assert compiler.artifact_fingerprint({"parallel": {"shard": 2}}) == compiler.artifact_fingerprint(
            {"parallel": {"shard": 2}}
        )
        assert compiler.artifact_fingerprint({"parallel": {"shard": 2}}) != compiler.artifact_fingerprint(
            {"parallel": {"shard": 4}}
        )


class TestDdpOptimizerContext:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_context_restores_setting(self, dynamo_config, enabled):
        previous = dynamo_config.optimize_ddp
        compiler = Compiler(CompileConfig({"enabled": enabled, "optimize_ddp": "python_reducer"}))
        with compiler.ddp_optimizer():
            assert dynamo_config.optimize_ddp == ("python_reducer" if enabled else previous)
        assert dynamo_config.optimize_ddp == previous

    @pytest.mark.parametrize("grouped", [False, True])
    def test_lazy_context(self, dynamo_config, grouped, isolated_compiler):
        from copy import deepcopy
        from uuid import uuid4

        observed = []

        def backend(graph, _inputs):
            observed.append(dynamo_config.optimize_ddp)
            return graph.forward

        name = "danling_context_" + uuid4().hex
        torch._dynamo.register_backend(backend, name=name)
        previous = dynamo_config.optimize_ddp
        target = "python_reducer" if previous != "python_reducer" else False
        compiler = Compiler(CompileConfig({"enabled": True, "backend": name, "optimize_ddp": target}))
        modules = [torch.nn.Linear(2, 2) for _ in range(2 if grouped else 1)]
        references = deepcopy(modules)
        hooks = []
        modules[0].register_forward_pre_hook(lambda *_: hooks.append("pre"))
        modules[0].register_forward_hook(lambda *_: hooks.append("post"))
        compiled = compiler.compile_modules(modules) if grouped else (compiler.compile(modules[0]),)
        for module, reference in zip(compiled, references):
            inputs = torch.ones(2, requires_grad=True)
            ref_inputs = inputs.detach().clone().requires_grad_()
            output, expected = module(inputs), reference(ref_inputs)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(
                torch.autograd.grad(output.sum(), (inputs, *module.parameters())),
                torch.autograd.grad(expected.sum(), (ref_inputs, *reference.parameters())),
            )
        assert observed and all(value == target for value in observed)
        assert hooks == ["pre", "post"]
        assert dynamo_config.optimize_ddp == previous
