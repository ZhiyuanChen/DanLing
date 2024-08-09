# DanLing
# Copyright (C) 2022-Present  DanLing

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from danling.runners.checkpoints import TorchDistributedCheckpointManager
from tests.runners.distributed import can_bind_localhost, find_free_port


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
    auto_resume: bool,
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
        "--signal-after-steps",
        str(signal_after_steps),
        "--status-name",
        status_name,
    ]
    if auto_resume:
        cmd.append("--auto-resume")
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
    auto_resume: bool,
    status_name: str,
    expected_returncode: int,
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
            auto_resume=auto_resume,
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


def test_runner_supervisor_sigterm_saves_dcp_checkpoint_and_resume(tmp_path: Path) -> None:
    _require_gloo_subprocess_runtime()
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = tmp_path / "sigterm-ddp"
    expected_sigterm_code = 128 + signal.SIGTERM

    _run_sigterm_workers(
        repo_root=repo_root,
        run_dir=run_dir,
        steps=4,
        signal_after_steps=2,
        auto_resume=False,
        status_name="preempted",
        expected_returncode=expected_sigterm_code,
    )

    checkpoint_dir = run_dir / "checkpoints"
    latest_target = Path(TorchDistributedCheckpointManager.resolve_checkpoint_path(checkpoint_dir / "latest"))
    assert (latest_target / ".metadata").is_file()

    preempted_status = json.loads((run_dir / "sigterm-preempted.json").read_text(encoding="utf-8"))
    assert preempted_status["exit_code"] == expected_sigterm_code
    assert preempted_status["global_step"] == 2

    _run_sigterm_workers(
        repo_root=repo_root,
        run_dir=run_dir,
        steps=4,
        signal_after_steps=0,
        auto_resume=True,
        status_name="resumed",
        expected_returncode=0,
    )

    resumed_status = json.loads((run_dir / "sigterm-resumed.json").read_text(encoding="utf-8"))
    assert resumed_status["exit_code"] == 0
    assert resumed_status["global_step"] == 4
