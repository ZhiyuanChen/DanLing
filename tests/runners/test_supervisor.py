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
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from danling.runners.checkpoints import TorchDistributedCheckpointManager
from danling.runners.supervisor import RunnerSupervisor
from tests.runners.distributed import can_bind_localhost, find_free_port


def test_runner_supervisor_gc_pacing_disables_and_restores_automatic_gc(tmp_path: Path, runner_factory) -> None:
    originally_enabled = python_gc.isenabled()
    python_gc.enable()

    runner = runner_factory({"gc": {"interval": 4}})
    try:
        assert python_gc.isenabled() is False
    finally:
        runner.close()
        if originally_enabled:
            python_gc.enable()
        else:
            python_gc.disable()

    assert python_gc.isenabled() is originally_enabled


def test_runner_supervisor_writes_heartbeat_file_on_startup(tmp_path: Path, runner_factory) -> None:
    runner = runner_factory({"heartbeat": {"enabled": True, "interval_seconds": 30.0}, "steps": 8})
    heartbeat_path = Path(runner.workspace.dir) / "heartbeats" / f"rank-{runner.rank:05d}.json"
    startup_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert startup_payload["status"] == "starting"
    assert startup_payload["event"] == "startup"
    assert startup_payload["rank"] == runner.rank
    assert startup_payload["steps"] == 8


def test_runner_supervisor_writes_heartbeat_file_on_close(tmp_path: Path, runner_factory) -> None:
    runner = runner_factory({"heartbeat": {"enabled": True, "interval_seconds": 30.0}, "steps": 8})
    heartbeat_path = Path(runner.workspace.dir) / "heartbeats" / f"rank-{runner.rank:05d}.json"

    runner.close()

    shutdown_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert shutdown_payload["status"] == "closed"
    assert shutdown_payload["event"] == "shutdown"


def test_runner_supervisor_heartbeat_reports_progress(tmp_path: Path, runner_factory) -> None:
    runner = runner_factory({"heartbeat": {"enabled": True, "interval_seconds": 30.0}, "steps": 8})
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


def test_runner_supervisor_heartbeat_omits_unknown_step_budget(tmp_path: Path, runner_factory) -> None:
    class StreamingLoader:
        def __iter__(self):
            return iter(())

    runner = runner_factory(
        {"epochs": 2, "train_splits": ["train"], "heartbeat": {"enabled": True, "interval_seconds": 30.0}}
    )
    runner.dataloaders["train"] = StreamingLoader()
    payload = runner.supervisor.heartbeat_payload(status="running", event="progress")
    assert "steps" not in payload
    assert payload["epochs"] == 2


def test_runner_supervisor_sigterm_saves_and_drains_checkpoints() -> None:
    class Config:
        def get(self, key, default=None):
            if key == "ckpt.wait_timeout_seconds":
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


def _require_gloo_subprocess_runtime() -> None:
    if not hasattr(signal, "SIGTERM"):
        pytest.skip("SIGTERM is unavailable in this environment.")
    if not can_bind_localhost():
        pytest.skip("Local TCP sockets are unavailable in this environment.")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo distributed support is unavailable in this PyTorch build.")


def _launch_sigterm_worker(
    *,
    repo_root: Path,
    run_dir: Path,
    rank: int,
    world_size: int,
    master_port: int,
    steps: int,
    signal_after_steps: int,
    signal_after_micro_steps: int,
    accum_steps: int,
    signal_rank: int,
    resume: bool,
    status_name: str,
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    pythonpath = os.fsdecode(repo_root)
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join((pythonpath, env["PYTHONPATH"]))
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "PYTHONPATH": pythonpath,
            "DANLING_SIGNAL_AFTER_STEPS": str(signal_after_steps),
            "DANLING_SIGNAL_AFTER_MICRO_STEPS": str(signal_after_micro_steps),
            "DANLING_SIGNAL_RANK": str(signal_rank),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    cmd = [
        sys.executable,
        str(repo_root / "tests" / "runners" / "integration" / "workers" / "sigterm_torch_worker.py"),
        "--run-dir",
        str(run_dir),
        "--steps",
        str(steps),
        "--accum-steps",
        str(accum_steps),
        "--signal-after-steps",
        str(signal_after_steps),
        "--status-name",
        status_name,
    ]
    if resume:
        cmd.append("--resume")
    return subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _run_sigterm_workers(
    *,
    repo_root: Path,
    run_dir: Path,
    steps: int,
    signal_after_steps: int,
    resume: bool,
    status_name: str,
    expected_returncode: int,
    signal_after_micro_steps: int = 0,
    accum_steps: int = 1,
    signal_rank: int = -1,
) -> list[str]:
    world_size = 2
    master_port = find_free_port()
    procs = [
        _launch_sigterm_worker(
            repo_root=repo_root,
            run_dir=run_dir,
            rank=rank,
            world_size=world_size,
            master_port=master_port,
            steps=steps,
            signal_after_steps=signal_after_steps,
            signal_after_micro_steps=signal_after_micro_steps,
            accum_steps=accum_steps,
            signal_rank=signal_rank,
            resume=resume,
            status_name=status_name,
        )
        for rank in range(world_size)
    ]
    outputs: list[str] = []
    try:
        for process in procs:
            stdout, _ = process.communicate(timeout=120)
            outputs.append(stdout)
        failed = [
            (rank, process.returncode, outputs[rank])
            for rank, process in enumerate(procs)
            if process.returncode != expected_returncode
        ]
        if failed:
            failure_text = "\n\n".join(
                f"rank={rank} returncode={returncode}\n{stdout}" for rank, returncode, stdout in failed
            )
            raise AssertionError(f"SIGTERM DDP workers exited unexpectedly:\n{failure_text}")
        return outputs
    finally:
        for process in procs:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


class TestIntegration:
    @pytest.mark.parametrize("accum_steps", [1, 3])
    def test_sigterm_resume(self, tmp_path: Path, accum_steps: int) -> None:
        _require_gloo_subprocess_runtime()
        repo_root = Path(__file__).resolve().parents[2]
        run_dir = tmp_path / "sigterm-ddp"
        expected_sigterm_code = 128 + signal.SIGTERM

        _run_sigterm_workers(
            repo_root=repo_root,
            run_dir=run_dir,
            steps=4,
            signal_after_steps=2 if accum_steps == 1 else 0,
            signal_after_micro_steps=1 if accum_steps > 1 else 0,
            accum_steps=accum_steps,
            signal_rank=0 if accum_steps > 1 else -1,
            resume=False,
            status_name="preempted",
            expected_returncode=expected_sigterm_code,
        )

        checkpoint_dir = run_dir / "checkpoints"
        latest_target = Path(TorchDistributedCheckpointManager.resolve_checkpoint_path(checkpoint_dir / "latest"))
        assert (latest_target / ".metadata").is_file()

        preempted_status = json.loads((run_dir / "sigterm-preempted.json").read_text(encoding="utf-8"))
        assert preempted_status["exit_code"] == expected_sigterm_code
        assert preempted_status["global_step"] == (2 if accum_steps == 1 else 1)

        _run_sigterm_workers(
            repo_root=repo_root,
            run_dir=run_dir,
            steps=4,
            signal_after_steps=0,
            accum_steps=accum_steps,
            resume=True,
            status_name="resumed",
            expected_returncode=0,
        )

        resumed_status = json.loads((run_dir / "sigterm-resumed.json").read_text(encoding="utf-8"))
        assert resumed_status["exit_code"] == 0
        assert resumed_status["global_step"] == 4
        for rank in range(2):
            before = json.loads((run_dir / f"sigterm-preempted-rank{rank}.json").read_text(encoding="utf-8"))
            after = json.loads((run_dir / f"sigterm-resumed-rank{rank}.json").read_text(encoding="utf-8"))
            assert before["consumed_samples"] + after["consumed_samples"] == list(
                range(rank + 1, 4 * accum_steps * 2 + 1, 2)
            )
