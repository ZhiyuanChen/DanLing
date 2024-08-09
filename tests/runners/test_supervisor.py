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

import gc as python_gc
import json
import signal
from pathlib import Path

import pytest

from danling.runners.base_runner import BaseRunner
from danling.runners.supervisor import RunnerSupervisor


class MinimalRunner(BaseRunner):
    pass


def _config(tmp_path: Path, **kwargs):
    config = {
        "log": False,
        "workspace_root": str(tmp_path),
        "lineage": "lineage-a",
        "experiment": "experiment-a",
    }
    config.update(kwargs)
    return config


def test_runner_supervisor_gc_pacing_disables_and_restores_automatic_gc(tmp_path: Path) -> None:
    originally_enabled = python_gc.isenabled()
    python_gc.enable()

    runner = MinimalRunner(_config(tmp_path, gc={"interval": 4}))
    try:
        assert python_gc.isenabled() is False
    finally:
        runner.close()
        if originally_enabled:
            python_gc.enable()
        else:
            python_gc.disable()

    assert python_gc.isenabled() is originally_enabled


def test_runner_supervisor_writes_heartbeat_file_on_startup(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, heartbeat={"enabled": True, "interval_seconds": 30.0}, steps=8))
    heartbeat_path = Path(runner.workspace.dir) / "heartbeats" / f"rank-{runner.rank:05d}.json"
    try:
        startup_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        assert startup_payload["status"] == "starting"
        assert startup_payload["event"] == "startup"
        assert startup_payload["rank"] == runner.rank
        assert startup_payload["steps"] == 8
    finally:
        runner.close()


def test_runner_supervisor_writes_heartbeat_file_on_close(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, heartbeat={"enabled": True, "interval_seconds": 30.0}, steps=8))
    heartbeat_path = Path(runner.workspace.dir) / "heartbeats" / f"rank-{runner.rank:05d}.json"

    runner.close()

    shutdown_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert shutdown_payload["status"] == "closed"
    assert shutdown_payload["event"] == "shutdown"


def test_runner_supervisor_heartbeat_payload_tracks_progress(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, heartbeat={"enabled": True, "interval_seconds": 30.0}, steps=8))
    try:
        runner.train_state.global_step = 3
        runner.train_state.epoch = 1
        runner.split = "train"
        runner.mode = "train"
        runner.supervisor.mark_heartbeat_progress()
        runner.supervisor.write_heartbeat(status="running", event="progress")

        heartbeat_path = Path(runner.workspace.dir) / "heartbeats" / f"rank-{runner.rank:05d}.json"
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        assert payload["status"] == "running"
        assert payload["event"] == "progress"
        assert payload["global_step"] == 3
        assert payload["epoch"] == 1
        assert payload["split"] == "train"
        assert payload["mode"] == "train"
        assert payload["progress"] == pytest.approx(3 / 8)
        assert payload["seconds_since_progress"] >= 0.0
    finally:
        runner.close()


def test_runner_supervisor_heartbeat_omits_unknown_step_budget(tmp_path: Path) -> None:
    class StreamingLoader:
        def __iter__(self):
            return iter(())

    runner = MinimalRunner(
        _config(
            tmp_path,
            epochs=2,
            train_splits=["train"],
            heartbeat={"enabled": True, "interval_seconds": 30.0},
        )
    )
    try:
        runner.dataloaders["train"] = StreamingLoader()
        payload = runner.supervisor.heartbeat_payload(status="running", event="progress")
        assert "steps" not in payload
        assert payload["epochs"] == 2
    finally:
        runner.close()


def test_runner_supervisor_sigterm_saves_and_drains_checkpoints() -> None:
    class Config:
        def get(self, key, default=None):
            if key == "checkpoint.wait_timeout":
                return 7.5
            return default

    class RecordingRunner:
        config = Config()

        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def prepare_for_shutdown_checkpoint(self) -> None:
            self.calls.append(("prepare", None))

        def save_checkpoint(self, **kwargs) -> None:
            self.calls.append(("save", kwargs))

        def close(self, timeout=None) -> bool:
            self.calls.append(("close", timeout))
            return True

    runner = RecordingRunner()
    supervisor = RunnerSupervisor(runner)

    supervisor.request_shutdown(signal.SIGTERM, None)

    with pytest.raises(SystemExit) as exc_info:
        supervisor.maybe_handle_termination_signal()

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert runner.calls == [
        ("prepare", None),
        ("save", {"save_best": False, "force": True}),
        ("close", 7.5),
    ]
