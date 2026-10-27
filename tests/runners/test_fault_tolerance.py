# DanLing
# Copyright (C) 2022-Present  DanLing

from __future__ import annotations

import importlib.util
from collections import deque
from copy import deepcopy

import pytest
import torch
from torch import nn, optim
from torchdata.stateful_dataloader import StatefulDataLoader

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


def _resume_loader(values):
    return StatefulDataLoader(
        [(torch.full((4,), float(value)), torch.zeros(2)) for value in values], batch_size=1, shuffle=False
    )


@pytest.mark.parametrize("heal_on_start", [1, 2])
def test_healing_at_step_budget_does_not_fetch_or_train_again(tmp_path, heal_on_start) -> None:
    config = {"device": "cpu", "logging.enabled": False, "steps": 2, "ckpt.enabled": False}
    donor = TinyFaultToleranceRunner({**config, "workspace.dir": str(tmp_path / "donor")})
    receiver = TinyFaultToleranceRunner({**config, "workspace.dir": str(tmp_path / "receiver")})
    try:
        donor.dataloaders["train"] = _resume_loader(range(1, 5))
        receiver.dataloaders["train"] = _resume_loader(range(1, 5))
        iterator = iter(donor.dataloaders["train"])
        donor.train_step(next(iterator))
        donor.train_step(next(iterator))
        state = deepcopy(donor.fault_tolerance._state_dict())

        class HealingManager(_QuorumManager):
            def start_quorum(self):
                super().start_quorum()
                if self.starts == heal_on_start:
                    receiver.fault_tolerance._load_state_dict(state)
                    self.votes = donor.train_state.global_step

        manager = HealingManager([True, True])
        receiver.fault_tolerance._manager = manager
        receiver.train_steps(["train"], [])
        assert receiver.train_state.global_step == 2
        assert manager.votes == 2
        assert receiver.dataloaders["train"].state_dict()["_num_yielded"] == 2
        for actual, expected in zip(receiver.model.parameters(), donor.model.parameters()):
            torch.testing.assert_close(actual, expected)
    finally:
        receiver.close()
        donor.close()


def test_healing_replans_restored_split_target_before_fetching(tmp_path) -> None:
    class RecordingRunner(TinyFaultToleranceRunner):
        def __init__(self, config):
            super().__init__(config)
            self.consumed = []

        def train_step(self, data):
            self.consumed.append((self.split, int(data[0][0, 0].item())))
            return super().train_step(data)

    config = {"device": "cpu", "logging.enabled": False, "steps": 4, "ckpt.enabled": False}
    donor = RecordingRunner({**config, "workspace.dir": str(tmp_path / "donor")})
    receiver = RecordingRunner({**config, "workspace.dir": str(tmp_path / "receiver")})
    try:
        for runner in (donor, receiver):
            runner.dataloaders["a"] = _resume_loader(range(1, 7))
            runner.dataloaders["b"] = _resume_loader(range(11, 17))
        b_iterator = iter(donor.dataloaders["b"])
        donor.train_step(next(b_iterator))
        donor.train_state.epoch = 2
        donor.train_state.step_mode_split_index = 1
        donor.train_state.step_mode_target = 3
        donor.train_state.sampler_epochs = {"a": 0, "b": 0}
        state = deepcopy(donor.fault_tolerance._state_dict())

        class HealingManager(_QuorumManager):
            def start_quorum(self):
                super().start_quorum()
                if self.starts == 1:
                    receiver.fault_tolerance._load_state_dict(state)
                    self.votes = 1

        manager = HealingManager([True] * 4)
        receiver.fault_tolerance._manager = manager
        receiver.train_steps(["a", "b"], [])
        assert receiver.consumed == [("b", 12), ("b", 13), ("a", 1)]
        assert receiver.train_state.global_step == manager.votes == 4
        assert receiver.dataloaders["a"].state_dict()["_num_yielded"] == 1
        assert receiver.dataloaders["b"].state_dict()["_num_yielded"] == 3
        donor.train_step(next(b_iterator))
        donor.train_step(next(b_iterator))
        donor.train_step(next(iter(donor.dataloaders["a"])))
        for actual, expected in zip(receiver.model.parameters(), donor.model.parameters()):
            torch.testing.assert_close(actual, expected)
    finally:
        receiver.close()
        donor.close()


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
        assert manager.votes == 1
        assert not manager.errors
        for parameter, expected in zip(runner.model.parameters(), before):
            torch.testing.assert_close(parameter, expected)
    finally:
        runner.close()
