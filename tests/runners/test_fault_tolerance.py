# DanLing
# Copyright (C) 2022-Present  DanLing

from __future__ import annotations

import importlib.util
from collections import deque

import pytest
import torch
from torch import nn, optim

from danling.runners import TorchRunner


class TinyFaultToleranceRunner(TorchRunner):
    def init_distributed(self) -> None:
        return

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Linear(4, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


def test_fault_tolerance_is_disabled_by_default() -> None:
    runner = TinyFaultToleranceRunner({"logging.enabled": False})
    try:
        assert runner.fault_tolerance is not None
        assert runner.fault_tolerance.enabled is False
        assert runner.fault_tolerance.replicate_process_group is None
        assert runner.fault_tolerance.data_parallel_info(4, 1) == (4, 1)
    finally:
        runner.close()


def test_fault_tolerance_requires_torchft_when_enabled() -> None:
    if importlib.util.find_spec("torchft") is not None:
        pytest.skip("torchft is installed")

    with pytest.raises(ImportError, match="torchft"):
        TinyFaultToleranceRunner({"logging.enabled": False, "ft": {"enabled": True}})


class _QuorumManager:
    """Deterministic accept/reject peer for the runner's public update contract."""

    def __init__(self, decisions):
        self.decisions = deque(decisions)
        self.starts = 0
        self.votes = 0
        self.errors = []

    def start_quorum(self):
        self.starts += 1

    def should_commit(self):
        self.votes += 1
        return self.decisions.popleft()

    def report_error(self, error):
        self.errors.append(error)

    def state_dict(self):
        return {"step": self.votes}

    def load_state_dict(self, state):
        self.votes = state["step"]


def test_rejected_quorum_preserves_optimizer_scheduler_ema_and_update_count(tmp_path) -> None:
    runner = TinyFaultToleranceRunner({"logging.enabled": False, "workspace.dir": str(tmp_path)})
    manager = _QuorumManager([False, True])
    runner.fault_tolerance._manager = manager
    runner.scheduler = optim.lr_scheduler.StepLR(runner.optimizer, step_size=1, gamma=0.5)
    runner._bind_optimizer_container()
    ema_updates = []
    runner.update_ema = lambda: ema_updates.append(runner.train_state.global_step)
    inputs, targets = torch.ones(2, 4), torch.zeros(2, 2)
    before = [parameter.detach().clone() for parameter in runner.model.parameters()]
    try:
        runner.train_step((inputs, targets))
        assert runner.train_state.global_step == 0
        assert runner.scheduler.last_epoch == 0
        assert not ema_updates
        for parameter, expected in zip(runner.model.parameters(), before):
            torch.testing.assert_close(parameter, expected)
            assert parameter.grad is None
        runner.train_step((inputs, targets))
        assert runner.train_state.global_step == 1
        assert runner.scheduler.last_epoch == 1
        assert len(ema_updates) == 1
        assert any(
            not torch.equal(parameter, expected) for parameter, expected in zip(runner.model.parameters(), before)
        )
        assert manager.starts == manager.votes == 2
        checkpoint = runner.state_dict()
        manager.votes = 0
        runner.load_state_dict(checkpoint)
        assert manager.votes == 2
    finally:
        runner.close()


def test_nonfinite_microbatch_rejects_entire_fault_tolerant_accumulation(tmp_path) -> None:
    runner = TinyFaultToleranceRunner(
        {"logging.enabled": False, "workspace.dir": str(tmp_path), "accum_steps": 2, "skip_nonfinite_loss": True}
    )
    manager = _QuorumManager([True])
    runner.fault_tolerance._manager = manager
    before = [parameter.detach().clone() for parameter in runner.model.parameters()]
    try:
        runner.train_step((torch.ones(2, 4), torch.zeros(2, 2)))
        runner.train_step((torch.ones(2, 4), torch.full((2, 2), float("nan"))))
        assert runner.train_state.global_step == 0
        assert manager.starts == manager.votes == 1
        assert len(manager.errors) == 1
        for parameter, expected in zip(runner.model.parameters(), before):
            torch.testing.assert_close(parameter, expected)
            assert parameter.grad is None
    finally:
        runner.close()


def test_zero_weight_window_finishes_quorum_without_advancing_updates(tmp_path) -> None:
    runner = TinyFaultToleranceRunner({"logging.enabled": False, "workspace.dir": str(tmp_path)})
    manager = _QuorumManager([True])
    runner.fault_tolerance._manager = manager
    runner.dataloaders["train"] = [{"input": torch.ones(2, 4), "target": torch.zeros(2, 2), "loss_normalizer": 0.0}]
    before = [parameter.detach().clone() for parameter in runner.model.parameters()]
    try:
        runner.train_epoch("train")
        assert runner.train_state.global_step == 0
        assert manager.starts == manager.votes == 1
        assert not manager.errors
        for parameter, expected in zip(runner.model.parameters(), before):
            torch.testing.assert_close(parameter, expected)
    finally:
        runner.close()
