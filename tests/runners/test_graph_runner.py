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

import json
from pathlib import Path

import pytest
import torch
from torch import nn, optim

from danling.runners import GraphRunner, TorchRunner


class TinyTorchRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(2, 1, bias=False)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class TinyGraphRunner(GraphRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(2, 1, bias=False)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class CountingGraphRunner(TinyGraphRunner):
    def __init__(self, config):
        self.iter_optimizer_parameter_calls = 0
        super().__init__(config)

    def iter_optimizer_parameters(self):
        self.iter_optimizer_parameter_calls += 1
        yield from super().iter_optimizer_parameters()


class BuildCountingGraphRunner(TinyGraphRunner):
    def __init__(self, config):
        self.build_graph_train_step_calls = 0
        super().__init__(config)

    def build_graph_train_step(self, params):
        self.build_graph_train_step_calls += 1
        return super().build_graph_train_step(params)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "target": torch.tensor([[1.0], [2.0]]),
    }


def test_graph_runner_train_step_matches_eager_update() -> None:
    eager = TinyTorchRunner({"logging.enabled": False})
    graph = TinyGraphRunner({"logging.enabled": False})
    try:
        assert eager.model is not None
        assert graph.model is not None
        with torch.no_grad():
            eager.model.weight.copy_(torch.tensor([[0.25, -0.5]]))
            graph.model.weight.copy_(eager.model.weight)

        eager.train_step(_batch())
        graph.train_step(_batch())

        assert graph.train_state.global_step == eager.train_state.global_step == 1
        assert torch.allclose(graph.model.weight, eager.model.weight)
    finally:
        eager.close()
        graph.close()


def test_graph_runner_builds_step_key_from_single_parameter_iteration() -> None:
    runner = CountingGraphRunner({"logging.enabled": False})
    try:
        runner.train_step(_batch())

        assert runner.iter_optimizer_parameter_calls == 1
    finally:
        runner.close()


def test_graph_runner_reuses_built_graph_train_step() -> None:
    runner = BuildCountingGraphRunner({"logging.enabled": False})
    try:
        runner.train_step(_batch())
        runner.train_step(_batch())

        assert runner.build_graph_train_step_calls == 1
    finally:
        runner.close()


def test_graph_runner_rejects_non_default_memory_policy() -> None:
    with pytest.raises(NotImplementedError, match="memory_policy"):
        TinyGraphRunner({"logging.enabled": False, "compile": {"memory_policy": "budget_limited_offload"}})


def test_graph_runner_requires_compile_for_cache_artifacts() -> None:
    with pytest.raises(ValueError, match="precompile_artifact_dir"):
        TinyGraphRunner({"logging.enabled": False, "compile": {"precompile_artifact_dir": "/tmp/danling-graph-cache"}})


def test_graph_runner_persists_torch_compile_cache_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compiled_calls: list[dict[str, object]] = []

    def fake_compile(fn, **kwargs):
        compiled_calls.append(dict(kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"cache-bytes", object()))

    runner = TinyGraphRunner(
        {
            "logging.enabled": False,
            "compile": {
                "enabled": True,
                "backend": "eager",
                "precompile_artifact_dir": str(tmp_path),
            },
        }
    )
    try:
        runner.train_step(_batch())

        artifact_path = runner.graph_cache_artifact_path()
        assert artifact_path is not None
        assert artifact_path.read_bytes() == b"cache-bytes"
        metadata_path = runner.graph_cache_metadata_path(artifact_path)
        assert metadata_path is not None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["format"] == "danling.graph_cache.v1"
        assert metadata["fingerprint"] == runner.graph_artifact_fingerprint()
        assert metadata["compile"]["backend"] == "eager"
        assert compiled_calls == [{"backend": "eager"}]
    finally:
        runner.close()


def test_graph_runner_loads_existing_torch_compile_cache_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[bytes] = []

    def fake_compile(fn, **kwargs):
        del kwargs
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda payload: loaded.append(payload))
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)

    runner = TinyGraphRunner(
        {
            "logging.enabled": False,
            "compile": {
                "enabled": True,
                "backend": "eager",
                "precompile_artifact_dir": str(tmp_path),
            },
        }
    )
    try:
        artifact_path = runner.graph_cache_artifact_path()
        assert artifact_path is not None
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"existing-cache")

        runner.train_step(_batch())

        assert loaded == [b"existing-cache"]
    finally:
        runner.close()


def test_graph_runner_skips_existing_cache_with_mismatched_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[bytes] = []

    def fake_compile(fn, **kwargs):
        del kwargs
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda payload: loaded.append(payload))
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)

    runner = TinyGraphRunner(
        {
            "logging.enabled": False,
            "compile": {
                "enabled": True,
                "backend": "eager",
                "precompile_artifact_dir": str(tmp_path),
            },
        }
    )
    try:
        artifact_path = runner.graph_cache_artifact_path()
        assert artifact_path is not None
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"stale-cache")
        metadata_path = runner.graph_cache_metadata_path(artifact_path)
        assert metadata_path is not None
        metadata_path.write_text(json.dumps({"fingerprint": "stale"}), encoding="utf-8")

        with pytest.warns(RuntimeWarning, match="metadata fingerprint"):
            runner.train_step(_batch())

        assert loaded == []
    finally:
        runner.close()
