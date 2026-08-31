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
from typing import Any

import pytest
import torch

from danling.metrics import AverageMeter
from danling.runners.base_runner import BaseRunner
from danling.runners.config import RunnerConfig


class MinimalRunner(BaseRunner):
    pass


class CheckpointRunner(BaseRunner):
    def load_model(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs

    def load_optimizer(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs

    def load_scheduler(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs

    def load_dataloaders(self, state_dict):
        del state_dict


class StreamingLoader:
    def __iter__(self):
        return iter(())


class _ToggleCloseCheckpointManager:
    def __init__(self) -> None:
        self.drained = False

    def close(self, timeout: float | None = None) -> bool:
        del timeout
        return self.drained


def _config(tmp_path: Path, **kwargs):
    config = {
        "logging.enabled": False,
        "workspace.root": str(tmp_path),
        "workspace.lineage": "lineage-a",
        "workspace.experiment": "experiment-a",
    }
    config.update(kwargs)
    return config


def _config_hash(runner: MinimalRunner) -> str:
    return format(hash(runner.config) & ((1 << 48) - 1), "012x")


def test_base_runner_log_interval_defaults_to_1024_for_unsized_loader() -> None:
    runner = MinimalRunner({"logging.enabled": False})
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


def test_base_runner_close_timeout_keeps_resources_available(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path))
    manager = _ToggleCloseCheckpointManager()

    class RecordingWriter:
        closed = False

        def flush(self) -> None:
            return

        def close(self) -> None:
            self.closed = True

    runner.checkpoint_manager = manager  # type: ignore[assignment]
    writer = RecordingWriter()
    runner.writer = writer

    with pytest.warns(RuntimeWarning, match="timed out while draining async checkpoints"):
        assert runner.close(timeout=0.0) is False

    assert writer.closed is False
    assert runner.writer is not None

    manager.drained = True
    assert runner.close(timeout=1.0) is True
    assert writer.closed is True
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


@pytest.mark.parametrize("use_bytes", [False, True])
def test_base_runner_reads_config_from_dcp_directory(tmp_path: Path, use_bytes: bool) -> None:
    RunnerConfig({"logging.enabled": False, "seed": 123}).yaml(tmp_path / "runner.yaml")
    checkpoint = bytes(tmp_path) if use_bytes else tmp_path

    config = BaseRunner.read_config(checkpoint)

    assert config.seed == 123


def test_base_runner_rejects_dcp_directory_without_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing 'runner.yaml'"):
        BaseRunner.read_config(tmp_path)


def test_base_runner_wandb_accepts_nested_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("wandb.sdk", reason="W&B support is an optional dependency")
    import wandb  # pylint: disable=import-outside-toplevel

    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    runner: MinimalRunner | None = None
    run_dir: Path | None = None
    try:
        runner = MinimalRunner(
            _config(
                tmp_path,
                wandb={
                    "enabled": True,
                    "mode": "offline",
                    "save_code": False,
                    "sync_tensorboard": False,
                },
            )
        )
        assert runner.wandb is not None
        run_dir = Path(runner.wandb.dir).parent
        runner.train_state.global_step = 5
        runner.write_result({"loss": 1.0, "metrics": {"acc": 0.75, "topk": [0.8, 0.9]}}, "train")
    finally:
        if runner is not None:
            runner.close()
        if wandb.run is not None:
            wandb.finish()

    assert run_dir is not None
    assert list(run_dir.glob("run-*.wandb"))


def test_base_runner_mlflow_logs_flattened_result_once(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous_tracking_uri = mlflow.get_tracking_uri()
    runner: MinimalRunner | None = None
    try:
        runner = MinimalRunner(
            _config(
                tmp_path,
                mlflow={
                    "enabled": True,
                    "tracking_uri": tracking_uri,
                    "experiment_name": "experiment-mlflow",
                    "run_name": "debug-run",
                    "tags": {"stage": "debug"},
                    "description": "smoke run",
                },
            )
        )
        runner.train_state.global_step = 5
        runner.write_result({"loss": 1.0, "metrics": {"acc": 0.75, "topk": [0.8, 0.9]}}, "train")
    finally:
        if runner is not None:
            runner.close()
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("experiment-mlflow")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags["mlflow.runName"] == "debug-run"
    assert run.data.tags["stage"] == "debug"
    assert run.data.tags["mlflow.note.content"] == "smoke run"
    assert run.data.metrics == {
        "train/loss": 1.0,
        "train/metrics/acc": 0.75,
        "train/metrics/topk/0": 0.8,
        "train/metrics/topk/1": 0.9,
    }


def test_base_runner_load_checkpoint_reports_restore_summary(capsys: pytest.CaptureFixture[str]) -> None:
    runner = CheckpointRunner({"logging.enabled": False})
    runner.optimizer = object()
    runner.scheduler = object()
    try:
        runner.load_checkpoint(
            {
                "runner": {"logging.enabled": False},
                "state": {"train": {"global_step": 13, "epoch": 4}},
                "model": {"w": 1},
                "optimizer": {"opt": 1},
                "scheduler": {"sched": 1},
            }
        )

        output = capsys.readouterr().out
        assert "restore:" in output
        assert "kind=checkpoint" in output
        assert "source=<mapping>" in output
        assert "step=13" in output
        assert "epoch=4" in output
        assert "optimizer=restored" in output
        assert "scheduler=restored" in output
    finally:
        runner.close()


def test_base_runner_prints_concise_failure_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = MinimalRunner(_config(tmp_path))
    try:
        runner.log_crash_summary(RuntimeError("boom"))

        output = capsys.readouterr().out
        assert "run failed:" in output
        assert "error=RuntimeError: boom" in output
        assert "rank=0/1" in output
        assert f"workspace={runner.workspace.dir}" in output
    finally:
        runner.close()


def test_base_runner_save_replaces_existing_file(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path))
    target = tmp_path / "payload.json"
    target.write_text('{"value": 0}', encoding="utf-8")

    try:
        runner.save({"value": 1}, target, indent=2)

        assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    finally:
        runner.close()


def test_base_runner_from_checkpoint_path_restores_full_state(tmp_path: Path) -> None:
    source = MinimalRunner(_config(tmp_path, seed=1016))
    checkpoint_path = tmp_path / "runner-checkpoint.pth"
    try:
        source.set_seed()
        source.train_state.global_step = 7
        source.train_state.epoch = 3
        source.elastic_state.restart_count = 2
        checkpoint = dict(source.state_dict())
        checkpoint["runner"]["resume"] = True
        checkpoint["runner"]["pretrained"] = "stale-pretrained"
        torch.save(checkpoint, checkpoint_path)
    finally:
        source.close()

    restored = MinimalRunner.from_checkpoint(checkpoint_path)
    try:
        assert restored.train_state.global_step == 7
        assert restored.train_state.epoch == 3
        assert restored.elastic_state.restart_count == 2
        assert restored.config.checkpoint == str(checkpoint_path)
        assert restored.config.resume is False
        assert restored.config.pretrained is None
    finally:
        restored.close()


def test_base_runner_resume_keeps_current_checkpoint_sources(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, checkpoint="latest-a", pretrained="model-a"))
    try:
        checkpoint_runner = runner.config.dict()
        checkpoint_runner["checkpoint"] = "latest-b"
        checkpoint_runner["pretrained"] = "model-b"

        runner.load_state_dict({"runner": checkpoint_runner, "state": {}})
        assert runner.config.checkpoint == "latest-a"
        assert runner.config.pretrained == "model-a"
    finally:
        runner.close()


def test_base_runner_resume_keeps_current_heartbeat_policy(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, heartbeat={"enabled": False, "interval_seconds": 60.0}))
    try:
        checkpoint_runner = runner.config.dict()
        checkpoint_runner["heartbeat"]["enabled"] = True
        checkpoint_runner["heartbeat"]["interval_seconds"] = 15.0
        checkpoint_runner["heartbeat"]["dir"] = "hb"

        runner.load_state_dict({"runner": checkpoint_runner, "state": {}})
        assert runner.config.heartbeat.enabled is False
        assert runner.config.heartbeat.interval_seconds == 60.0
    finally:
        runner.close()


def test_base_runner_resume_rejects_checkpoint_backend_changes(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, ckpt={"backend": "dcp"}))
    try:
        checkpoint_runner = runner.config.dict()
        checkpoint_runner["ckpt"]["backend"] = "file"

        with pytest.raises(ValueError, match="ckpt"):
            runner.load_state_dict({"runner": checkpoint_runner, "state": {}})
    finally:
        runner.close()
