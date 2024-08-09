# DanLing
# Copyright (C) 2022-Present  DanLing

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from danling.runners.checkpoints import TorchDistributedCheckpointManager
from tests.runners.distributed import can_bind_localhost, find_free_port

LIGHTHOUSE_PORT = 29510


def _port_is_available(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _require_torchft_fsdp_runtime() -> None:
    if importlib.util.find_spec("torchft") is None:
        pytest.skip("torchft is not installed")
    if shutil.which("torchft_lighthouse") is None:
        pytest.skip("torchft_lighthouse is unavailable")
    if not can_bind_localhost():
        pytest.skip("Local TCP sockets are unavailable in this environment.")
    if not _port_is_available(LIGHTHOUSE_PORT):
        pytest.skip(f"Port {LIGHTHOUSE_PORT} is unavailable for torchft_lighthouse.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this environment.")
    if torch.cuda.device_count() < 4:
        pytest.skip(f"Expected at least 4 CUDA devices, found {torch.cuda.device_count()}.")
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        pytest.skip("NCCL distributed support is unavailable in this PyTorch build.")


def _require_torchft_ddp_runtime() -> None:
    if importlib.util.find_spec("torchft") is None:
        pytest.skip("torchft is not installed")
    if shutil.which("torchft_lighthouse") is None:
        pytest.skip("torchft_lighthouse is unavailable")
    if not can_bind_localhost():
        pytest.skip("Local TCP sockets are unavailable in this environment.")
    if not _port_is_available(LIGHTHOUSE_PORT):
        pytest.skip(f"Port {LIGHTHOUSE_PORT} is unavailable for torchft_lighthouse.")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo distributed support is unavailable in this PyTorch build.")


def _start_lighthouse(repo_root: Path) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["RUST_BACKTRACE"] = "1"
    process = subprocess.Popen(
        [
            "torchft_lighthouse",
            "--min_replicas",
            "1",
            "--quorum_tick_ms",
            "100",
            "--join_timeout_ms",
            "10000",
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(1.0)
    if process.poll() is not None:
        output = "" if process.stdout is None else process.stdout.read()
        raise RuntimeError(f"torchft_lighthouse exited early:\n{output}")
    return process


def _launch_replica_group(
    *,
    repo_root: Path,
    run_dir: Path,
    replica_id: int,
    visible_devices: str,
    master_port: int,
    steps: int,
    auto_resume: bool,
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["TORCHFT_LIGHTHOUSE"] = f"http://127.0.0.1:{LIGHTHOUSE_PORT}"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        f"--master_port={master_port}",
        str(repo_root / "tests" / "runners" / "integration" / "workers" / "fault_tolerance_worker.py"),
        "--run-dir",
        str(run_dir),
        "--replica-id",
        str(replica_id),
        "--group-size",
        "2",
        "--steps",
        str(steps),
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


def _run_replica_groups(*, repo_root: Path, run_dir: Path, steps: int, auto_resume: bool) -> None:
    procs = [
        _launch_replica_group(
            repo_root=repo_root,
            run_dir=run_dir,
            replica_id=0,
            visible_devices="0,1",
            master_port=find_free_port(),
            steps=steps,
            auto_resume=auto_resume,
        ),
        _launch_replica_group(
            repo_root=repo_root,
            run_dir=run_dir,
            replica_id=1,
            visible_devices="2,3",
            master_port=find_free_port(),
            steps=steps,
            auto_resume=auto_resume,
        ),
    ]
    outputs: list[str] = []
    try:
        for process in procs:
            stdout, _ = process.communicate(timeout=300)
            outputs.append(stdout)
        failed = [
            (index, process.returncode, outputs[index]) for index, process in enumerate(procs) if process.returncode
        ]
        if failed:
            failure_text = "\n\n".join(
                f"replica_group={index} returncode={returncode}\n{stdout}" for index, returncode, stdout in failed
            )
            raise AssertionError(f"fault-tolerance replica groups failed:\n{failure_text}")
    finally:
        for process in procs:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def _launch_torch_replica_group(
    *,
    repo_root: Path,
    run_dir: Path,
    replica_id: int,
    master_port: int,
    steps: int,
    auto_resume: bool,
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["TORCHFT_LIGHTHOUSE"] = f"http://127.0.0.1:{LIGHTHOUSE_PORT}"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        f"--master_port={master_port}",
        str(repo_root / "tests" / "runners" / "integration" / "workers" / "fault_tolerance_torch_worker.py"),
        "--run-dir",
        str(run_dir),
        "--replica-id",
        str(replica_id),
        "--group-size",
        "2",
        "--steps",
        str(steps),
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


def _run_torch_replica_groups(*, repo_root: Path, run_dir: Path, steps: int, auto_resume: bool) -> None:
    procs = [
        _launch_torch_replica_group(
            repo_root=repo_root,
            run_dir=run_dir,
            replica_id=0,
            master_port=find_free_port(),
            steps=steps,
            auto_resume=auto_resume,
        ),
        _launch_torch_replica_group(
            repo_root=repo_root,
            run_dir=run_dir,
            replica_id=1,
            master_port=find_free_port(),
            steps=steps,
            auto_resume=auto_resume,
        ),
    ]
    outputs: list[str] = []
    try:
        for process in procs:
            stdout, _ = process.communicate(timeout=300)
            outputs.append(stdout)
        failed = [
            (index, process.returncode, outputs[index]) for index, process in enumerate(procs) if process.returncode
        ]
        if failed:
            failure_text = "\n\n".join(
                f"replica_group={index} returncode={returncode}\n{stdout}" for index, returncode, stdout in failed
            )
            raise AssertionError(f"fault-tolerance replica groups failed:\n{failure_text}")
    finally:
        for process in procs:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def test_fault_tolerance_fsdp_save_and_resume_across_replica_groups(tmp_path: Path) -> None:
    _require_torchft_fsdp_runtime()
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = tmp_path / "ft-fsdp-smoke"

    lighthouse = _start_lighthouse(repo_root)
    try:
        _run_replica_groups(repo_root=repo_root, run_dir=run_dir, steps=1, auto_resume=False)
        _run_replica_groups(repo_root=repo_root, run_dir=run_dir, steps=2, auto_resume=True)
    finally:
        if lighthouse.poll() is None:
            lighthouse.terminate()
            lighthouse.wait(timeout=10)

    checkpoint_dir = run_dir / "checkpoints"
    latest_target = Path(TorchDistributedCheckpointManager.resolve_checkpoint_path(checkpoint_dir / "latest"))
    assert (latest_target / ".metadata").is_file()
    assert any((checkpoint_dir / "ft-replica-0").iterdir())
    assert any((checkpoint_dir / "ft-replica-1").iterdir())

    for replica_id in (0, 1):
        status_path = run_dir / f"ft-status-replica-{replica_id}.json"
        assert status_path.is_file()
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["global_step"] == 2
        assert status["cursor"] == 2


def test_fault_tolerance_torch_runner_save_and_resume_across_replica_groups(tmp_path: Path) -> None:
    _require_torchft_ddp_runtime()
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = tmp_path / "ft-ddp-smoke"

    lighthouse = _start_lighthouse(repo_root)
    try:
        _run_torch_replica_groups(repo_root=repo_root, run_dir=run_dir, steps=1, auto_resume=False)
        _run_torch_replica_groups(repo_root=repo_root, run_dir=run_dir, steps=2, auto_resume=True)
    finally:
        if lighthouse.poll() is None:
            lighthouse.terminate()
            lighthouse.wait(timeout=10)

    checkpoint_dir = run_dir / "checkpoints"
    latest_target = Path(TorchDistributedCheckpointManager.resolve_checkpoint_path(checkpoint_dir / "latest"))
    assert (latest_target / ".metadata").is_file()
    assert any((checkpoint_dir / "ft-replica-0").iterdir())
    assert any((checkpoint_dir / "ft-replica-1").iterdir())

    for replica_id in (0, 1):
        status_path = run_dir / f"ft-status-replica-{replica_id}.json"
        assert status_path.is_file()
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["global_step"] == 2
        assert status["cursor"] == 2


def test_fault_tolerance_torch_runner_soaks_multiple_restarts(tmp_path: Path) -> None:
    _require_torchft_ddp_runtime()
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = tmp_path / "ft-ddp-restart-soak"

    lighthouse = _start_lighthouse(repo_root)
    try:
        _run_torch_replica_groups(repo_root=repo_root, run_dir=run_dir, steps=1, auto_resume=False)
        for steps in (2, 3, 4):
            _run_torch_replica_groups(repo_root=repo_root, run_dir=run_dir, steps=steps, auto_resume=True)
    finally:
        if lighthouse.poll() is None:
            lighthouse.terminate()
            lighthouse.wait(timeout=10)

    checkpoint_dir = run_dir / "checkpoints"
    latest_target = Path(TorchDistributedCheckpointManager.resolve_checkpoint_path(checkpoint_dir / "latest"))
    assert (latest_target / ".metadata").is_file()
    assert any((checkpoint_dir / "ft-replica-0").iterdir())
    assert any((checkpoint_dir / "ft-replica-1").iterdir())

    for replica_id in (0, 1):
        status_path = run_dir / f"ft-status-replica-{replica_id}.json"
        assert status_path.is_file()
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["global_step"] == 4
        assert status["cursor"] == 4
