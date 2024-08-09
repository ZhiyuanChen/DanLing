# DanLing
# Copyright (C) 2022-Present  DanLing

from __future__ import annotations

import importlib.util

import pytest
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
    runner = TinyFaultToleranceRunner({"log": False})
    try:
        assert runner.ft is not None
        assert runner.ft.enabled is False
        assert runner.ft.replicate_process_group is None
        assert runner.ft.data_parallel_info(4, 1) == (4, 1)
    finally:
        runner.close()


def test_fault_tolerance_requires_torchft_when_enabled() -> None:
    if importlib.util.find_spec("torchft") is not None:
        pytest.skip("torchft is installed")

    with pytest.raises(ImportError, match="torchft"):
        TinyFaultToleranceRunner({"log": False, "ft": {"enabled": True}})
