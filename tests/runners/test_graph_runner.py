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


class EagerRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(2, 1, bias=False)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


class Runner(GraphRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(2, 1, bias=False)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "target": torch.tensor([[1.0], [2.0]]),
    }


def test_graph_runner_train_step_matches_eager_update() -> None:
    eager = EagerRunner({"logging.enabled": False})
    graph = Runner({"logging.enabled": False})
    try:
        assert eager.model is not None
        assert graph.model is not None
        with torch.no_grad():
            eager.model.weight.copy_(torch.tensor([[0.25, -0.5]]))
            graph.model.weight.copy_(eager.model.weight)

        for step in range(1, 4):
            eager.train_step(_batch())
            graph.train_step(_batch())
            assert graph.train_state.global_step == eager.train_state.global_step == step
            torch.testing.assert_close(graph.model.weight, eager.model.weight)
    finally:
        eager.close()
        graph.close()


@pytest.mark.parametrize("empty_first", [False, True])
def test_graph_runner_zero_weight_discards_nonfinite_vjp(empty_first) -> None:
    runner = Runner({"device": "cpu", "logging.enabled": False, "accum_steps": 2, "logging.interval": 0})
    try:
        runner.model.double()
        initial = runner.model.weight.detach().clone()
        valid = {
            "input": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            "target": torch.zeros(1, 1, dtype=torch.float64),
            "loss_normalizer": 0.123456789123,
        }
        empty = {
            "input": torch.empty(0, 2, dtype=torch.float64),
            "target": torch.empty(0, 1, dtype=torch.float64),
            "loss_normalizer": 0.0,
        }
        gradient = torch.autograd.grad(
            runner.criterion(runner.model(valid["input"]), valid["target"]), runner.model.weight
        )[0]
        runner.dataloaders["train"] = [empty, valid] if empty_first else [valid, empty]
        result = runner.train_epoch()
        torch.testing.assert_close(runner.model.weight, initial - 0.1 * gradient, rtol=1e-12, atol=1e-12)
        assert runner.train_state.global_step == 1
        assert torch.isfinite(torch.tensor(result["loss"]))
    finally:
        runner.close()


def test_graph_runner_replaces_nonfinite_loss_without_accumulating_gradients() -> None:
    runner = Runner({"logging.enabled": False, "skip_nonfinite_loss": True})
    try:
        assert runner.model is not None
        initial = runner.model.weight.detach().clone()

        _prediction, loss = runner.train_step(
            {
                "input": torch.full((2, 2), float("inf")),
                "target": torch.zeros(2, 1),
            }
        )

        assert loss is not None and loss.item() == 0
        torch.testing.assert_close(runner.model.weight, initial)
        assert runner.train_state.global_step == 1
    finally:
        runner.close()


def test_graph_runner_rejects_non_default_memory_policy() -> None:
    with pytest.raises(NotImplementedError, match="memory_policy"):
        Runner({"logging.enabled": False, "compile": {"memory_policy": "budget_limited_offload"}})


def test_graph_runner_requires_compile_for_cache_artifacts() -> None:
    with pytest.raises(ValueError, match="precompile_artifact_dir"):
        Runner({"logging.enabled": False, "compile": {"precompile_artifact_dir": "/tmp/danling-graph-cache"}})


@pytest.fixture
def cache_runner(tmp_path: Path, runner_factory, monkeypatch):
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)
    return runner_factory(
        {"device": "cpu", "compile": {"enabled": True, "backend": "eager", "precompile_artifact_dir": str(tmp_path)}},
        runner_type=Runner,
    )


class TestCache:
    def test_save(self, cache_runner, monkeypatch):
        monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"cache-bytes", object()))
        cache_runner.train_step(_batch())
        artifact_path = cache_runner.graph_cache_artifact_path()
        assert artifact_path.read_bytes() == b"cache-bytes"
        metadata_path = cache_runner.graph_cache_metadata_path(artifact_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["format"] == "danling.graph_cache.v1"
        assert metadata["fingerprint"] == cache_runner.graph_artifact_fingerprint()
        assert metadata["compile"]["backend"] == "eager"

    def test_load(self, cache_runner, monkeypatch):
        loaded = []
        monkeypatch.setattr(torch.compiler, "load_cache_artifacts", loaded.append)
        artifact_path = cache_runner.graph_cache_artifact_path()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"existing-cache")
        cache_runner.train_step(_batch())
        assert loaded == [b"existing-cache"]

    def test_stale(self, cache_runner, monkeypatch):
        loaded = []
        monkeypatch.setattr(torch.compiler, "load_cache_artifacts", loaded.append)
        artifact_path = cache_runner.graph_cache_artifact_path()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"stale-cache")
        metadata_path = cache_runner.graph_cache_metadata_path(artifact_path)
        metadata_path.write_text(json.dumps({"fingerprint": "stale"}), encoding="utf-8")
        with pytest.warns(RuntimeWarning, match="metadata fingerprint"):
            cache_runner.train_step(_batch())
        assert loaded == []
