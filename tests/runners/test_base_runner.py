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

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from danling.metrics import AverageMeter
from danling.runners.base_runner import BaseRunner


class MinimalRunner(BaseRunner):
    pass


class SequencingRunner(BaseRunner):
    def __init__(self, config):
        self.calls: list[str] = []
        super().__init__(config)

    def load_state_dict(self, checkpoint):
        self.calls.append("state")
        super().load_state_dict(checkpoint)

    def load_model(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs
        self.calls.append("model")

    def load_optimizer(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs
        self.calls.append("optimizer")

    def load_scheduler(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs
        self.calls.append("scheduler")

    def load_dataloaders(self, state_dict):
        del state_dict
        self.calls.append("dataloaders")


class StreamingLoader:
    def __iter__(self):
        return iter(())


class _ToggleCloseCheckpointManager:
    def __init__(self) -> None:
        self.drained = False
        self.calls: list[float | None] = []

    def close(self, timeout: float | None = None) -> bool:
        self.calls.append(timeout)
        return self.drained


class _RecordingCheckpointManager:
    is_collective = False

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def save_checkpoint(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))

    def close(self, timeout: float | None = None) -> bool:
        del timeout
        return True


def _config(tmp_path: Path, **kwargs):
    config = {
        "log": False,
        "workspace_root": str(tmp_path),
        "lineage": "lineage-a",
        "experiment": "experiment-a",
    }
    config.update(kwargs)
    return config


def _config_hash(runner: MinimalRunner) -> str:
    return format(hash(runner.config) & ((1 << 48) - 1), "012x")


def test_base_runner_log_interval_defaults_to_1024_for_unsized_loader() -> None:
    runner = MinimalRunner({"log": False})
    try:
        runner.dataloaders["train"] = StreamingLoader()
        assert runner.log_interval == 1024
    finally:
        runner.close()


def test_base_runner_sorts_configured_splits(tmp_path: Path) -> None:
    runner = MinimalRunner(
        _config(
            tmp_path,
            train_splits=["z", "a", "z", "m"],
            evaluate_splits=["test", "val", "test"],
        )
    )
    try:
        assert runner.train_splits == ["a", "m", "z"]
        assert runner.evaluate_splits == ["test", "val"]
    finally:
        runner.close()


def test_base_runner_close_timeout_does_not_teardown_resources(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path))
    manager = _ToggleCloseCheckpointManager()
    writer_calls: list[str] = []

    class RecordingWriter:
        def flush(self) -> None:
            writer_calls.append("flush")

        def close(self) -> None:
            writer_calls.append("close")

    runner.checkpoint_manager = manager  # type: ignore[assignment]
    runner.writer = RecordingWriter()

    with pytest.warns(RuntimeWarning, match="timed out while draining async checkpoints"):
        assert runner.close(timeout=0.0) is False

    assert manager.calls == [0.0]
    assert writer_calls == []
    assert runner.writer is not None

    manager.drained = True
    assert runner.close(timeout=1.0) is True
    assert manager.calls == [0.0, 1.0]
    assert writer_calls == ["flush", "close"]
    assert runner.writer is None


def test_base_runner_write_result_flattens_nested_metrics(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path))
    writes: list[tuple[str, Any, str, int]] = []

    def capture(name: str, score: float, split: str, steps: int) -> None:
        writes.append((name, score, split, steps))

    class RecordingWriter:
        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    loss = AverageMeter()
    loss.update(4.0)
    nested_loss = AverageMeter()
    nested_loss.update(2.5)

    try:
        runner.writer = RecordingWriter()
        runner.train_state.global_step = 7
        runner.write_score = capture  # type: ignore[method-assign]
        runner.write_result(
            {
                "loss": loss,
                "metrics": {
                    "acc": 0.5,
                    "per_class": [0.1, 0.2],
                    "nested": {"loss": nested_loss},
                },
                "vector": (1.0, 2.0),
            },
            "train",
        )
    finally:
        runner.close()

    assert writes == [
        ("loss", 4.0, "train", 7),
        ("metrics/acc", 0.5, "train", 7),
        ("metrics/per_class/0", 0.1, "train", 7),
        ("metrics/per_class/1", 0.2, "train", 7),
        ("metrics/nested/loss", 2.5, "train", 7),
        ("vector/0", 1.0, "train", 7),
        ("vector/1", 2.0, "train", 7),
    ]


def test_base_runner_write_result_skips_flattening_without_sinks(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path))

    def fail_flatten(_result):
        raise AssertionError("flatten_result should not run when no metric sink is configured")

    try:
        runner.flatten_result = fail_flatten  # type: ignore[method-assign]
        runner.write_result({"loss": 1.0}, "train")
    finally:
        runner.close()


def test_base_runner_wandb_logs_flattened_result_once(tmp_path: Path) -> None:
    wandb_calls: list[dict[str, object]] = []
    log_calls: list[tuple[dict[str, object], int]] = []
    finish_calls: list[str] = []

    class RecordingRun:
        def log(self, payload: dict[str, object], step: int) -> None:
            log_calls.append((payload, step))

        def finish(self) -> None:
            finish_calls.append("finish")

    def record_init(**kwargs):
        wandb_calls.append(kwargs)
        return RecordingRun()

    previous_wandb = sys.modules.get("wandb")
    wandb_module = ModuleType("wandb")
    wandb_module.init = record_init  # type: ignore[attr-defined]
    sys.modules["wandb"] = wandb_module

    try:
        runner = MinimalRunner(
            _config(
                tmp_path,
                wandb={
                    "enabled": True,
                    "tags": ["mnist", "debug"],
                },
            )
        )
        writes: list[tuple[str, Any, str, int]] = []

        def capture(name: str, score: float, split: str, steps: int) -> None:
            writes.append((name, score, split, steps))

        assert len(wandb_calls) == 1
        init_kwargs = wandb_calls[0]
        assert init_kwargs["project"] == "lineage-a"
        assert init_kwargs["group"] == "experiment-a"
        assert init_kwargs["name"] == runner.id
        assert init_kwargs["dir"] == runner.workspace.dir
        assert init_kwargs["tags"] == ["mnist", "debug"]
        assert init_kwargs["config"] == runner.config.dict()

        runner.train_state.global_step = 5
        runner.write_score = capture  # type: ignore[method-assign]
        runner.write_result({"loss": 1.0, "metrics": {"acc": 0.75, "topk": [0.8, 0.9]}}, "train")
        assert writes == [
            ("loss", 1.0, "train", 5),
            ("metrics/acc", 0.75, "train", 5),
            ("metrics/topk/0", 0.8, "train", 5),
            ("metrics/topk/1", 0.9, "train", 5),
        ]
        assert log_calls == [
            (
                {
                    "train/loss": 1.0,
                    "train/metrics/acc": 0.75,
                    "train/metrics/topk/0": 0.8,
                    "train/metrics/topk/1": 0.9,
                },
                5,
            )
        ]
    finally:
        runner.close()
        if previous_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = previous_wandb

    assert finish_calls == ["finish"]


def test_base_runner_load_checkpoint_restores_in_expected_order() -> None:
    runner = SequencingRunner({"log": False})
    try:
        runner.load_checkpoint(
            {
                "runner": {"log": False},
                "state": {"train": {"global_step": 1, "epoch": 0}},
                "model": {"w": 1},
                "optimizer": {"opt": 1},
                "scheduler": {"sched": 1},
                "dataloaders": {"train": {"cursor": 1}},
            }
        )
        assert runner.calls == ["state", "model", "optimizer", "scheduler", "dataloaders"]
    finally:
        runner.close()


def test_base_runner_load_checkpoint_excludes_configured_components() -> None:
    runner = SequencingRunner(
        {
            "log": False,
            "checkpoint": {
                "exclude_from_loading": ["model", "optimizer", "lr_scheduler", "data_loader"],
            },
        }
    )
    try:
        runner.load_checkpoint(
            {
                "runner": {"log": False},
                "state": {"train": {"global_step": 1, "epoch": 0}},
                "model": {"w": 1},
                "optimizer": {"opt": 1},
                "scheduler": {"sched": 1},
                "dataloaders": {"train": {"cursor": 1}},
            }
        )
        assert runner.calls == ["state"]
    finally:
        runner.close()


def test_base_runner_load_checkpoint_excludes_nested_state_key() -> None:
    runner = MinimalRunner({"log": False, "checkpoint": {"exclude_from_loading": ["state.train"]}})
    try:
        runner.load_checkpoint(
            {
                "runner": {"log": False},
                "state": {"train": {"global_step": 9, "epoch": 3}},
            }
        )
        assert runner.train_state.global_step == 0
        assert runner.train_state.epoch == 0
    finally:
        runner.close()


def test_base_runner_load_checkpoint_warns_when_runner_config_validation_is_excluded() -> None:
    runner = MinimalRunner({"log": False, "checkpoint": {"exclude_from_loading": ["runner"]}})
    try:
        with pytest.warns(RuntimeWarning, match="semantic runner config validation"):
            runner.load_checkpoint(
                {
                    "runner": {"log": False, "steps": 123},
                    "state": {"train": {"global_step": 4, "epoch": 0}},
                }
            )
        assert runner.train_state.global_step == 4
    finally:
        runner.close()


def test_base_runner_save_seed_checkpoint_forces_last_step_save() -> None:
    runner = MinimalRunner({"log": False})
    manager = _RecordingCheckpointManager()
    try:
        runner.checkpoint_manager = manager  # type: ignore[assignment]
        runner.save_seed_checkpoint()

        assert manager.calls == [
            {
                "name": "seed",
                "epochs": 0,
                "save_best": False,
                "last_step": True,
                "force": True,
            }
        ]
    finally:
        runner.close()


def test_base_runner_from_checkpoint_path_restores_full_state(tmp_path: Path) -> None:
    source = MinimalRunner(_config(tmp_path, seed=123))
    checkpoint_path = tmp_path / "runner-checkpoint.pth"
    try:
        source.set_seed()
        source.train_state.global_step = 7
        source.train_state.epoch = 3
        source.elastic_state.restart_count = 2
        checkpoint = dict(source.state_dict())
        checkpoint["runner"]["auto_resume"] = True
        checkpoint["runner"]["pretrained"] = "stale-pretrained"
        torch.save(checkpoint, checkpoint_path)
    finally:
        source.close()

    restored = MinimalRunner.from_checkpoint(checkpoint_path)
    try:
        assert restored.train_state.global_step == 7
        assert restored.train_state.epoch == 3
        assert restored.elastic_state.restart_count == 2
        assert restored.config.resume == str(checkpoint_path)
        assert restored.config.auto_resume is False
        assert restored.config.pretrained is None
        assert restored.rng_state.python is not None
    finally:
        restored.close()


def test_base_runner_load_state_dict_ignores_resume_and_pretrained_sources(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, resume="latest-a", pretrained="model-a"))
    try:
        checkpoint_runner = runner.config.dict()
        checkpoint_runner["resume"] = "latest-b"
        checkpoint_runner["pretrained"] = "model-b"

        runner.load_state_dict({"runner": checkpoint_runner, "state": {}})
    finally:
        runner.close()


def test_base_runner_load_state_dict_ignores_heartbeat_policy(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, heartbeat={"enabled": False, "interval_seconds": 60.0}))
    try:
        checkpoint_runner = runner.config.dict()
        checkpoint_runner["heartbeat"]["enabled"] = True
        checkpoint_runner["heartbeat"]["interval_seconds"] = 15.0
        checkpoint_runner["heartbeat"]["dir_name"] = "hb"

        runner.load_state_dict({"runner": checkpoint_runner, "state": {}})
    finally:
        runner.close()


def test_base_runner_load_state_dict_rejects_checkpoint_backend_change(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, checkpoint={"backend": "dcp"}))
    try:
        checkpoint_runner = runner.config.dict()
        checkpoint_runner["checkpoint"]["backend"] = "file"

        with pytest.raises(ValueError, match="checkpoint"):
            runner.load_state_dict({"runner": checkpoint_runner, "state": {}})
    finally:
        runner.close()
