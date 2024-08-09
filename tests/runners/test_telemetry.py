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

from collections.abc import MutableMapping
from typing import Any

import pytest
import torch

from danling.runners.telemetry import LoopTelemetry, RuntimeTelemetrySnapshot
from danling.tensors import NestedTensor


class _RecordingMeter:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))


class _TelemetryMeters:
    def __init__(self) -> None:
        self.time = _RecordingMeter()


class _TelemetryRunner:
    def __init__(
        self,
        *,
        log_interval: int = 0,
        loop_times: list[float] | None = None,
    ) -> None:
        self.distributed = False
        self.device = torch.device("cpu")
        self.log_interval = log_interval
        self.loop_times = list(loop_times or [])
        self.meters = _TelemetryMeters()
        self.reset_calls: list[torch.device] = []
        self.step_log_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def all_reduce_device(self) -> torch.device:
        return torch.device("cpu")

    def all_reduce(self, tensor: torch.Tensor, *, op=None) -> torch.Tensor:
        del op
        return tensor

    def loop_time(self, *, sync: bool = False) -> float:
        del sync
        if self.loop_times:
            return self.loop_times.pop(0)
        return 0.0

    def get_step_result(self) -> MutableMapping[str, Any]:
        return {}

    def reduce_loss_for_logging(self, loss: torch.Tensor | None, loss_n: int | None) -> torch.Tensor | None:
        del loss_n
        return loss

    def step_log(self, *args: Any, **kwargs: Any) -> None:
        self.step_log_calls.append((args, kwargs))


class RecordingLoopTelemetry(LoopTelemetry):
    peak_memory: tuple[float | None, float | None] = (None, None)

    def reset_peak_memory_stats(self) -> None:
        self.runner.reset_calls.append(self.runner.device)

    def current_peak_memory_mb(self) -> tuple[float | None, float | None]:
        return self.peak_memory


def _emit_training_log() -> tuple[_TelemetryRunner, RecordingLoopTelemetry, dict[str, Any]]:
    runner = _TelemetryRunner(loop_times=[2.0])
    telemetry = RecordingLoopTelemetry(runner, start_time=0.0)
    telemetry.peak_memory = (9.0, 13.0)
    telemetry.observe(
        iteration=0,
        data=torch.ones(5, 2),
        current_time=0.5,
    )
    telemetry.observe(
        iteration=1,
        data=NestedTensor([torch.ones(25)]),
        current_time=1.0,
    )

    telemetry.emit_log(
        split="train",
        iteration=1,
        length=10,
        loss=torch.tensor(2.0),
        loss_n=5,
    )

    args, kwargs = runner.step_log_calls[0]
    assert args[:3] == ("train", 1, 10)
    return runner, telemetry, kwargs["result"]


class TestLoopTelemetryAggregation:

    def test_merge_result_adds_known_rates(self) -> None:
        telemetry = LoopTelemetry(_TelemetryRunner(), start_time=0.0)
        result: dict[str, float] = {}

        telemetry.merge_result(
            result,
            elapsed_seconds=2.0,
            steps=4,
            snapshot=RuntimeTelemetrySnapshot(
                sample_count=10,
                sample_known=True,
                token_count=100,
                token_known=True,
                peak_allocated_mb=None,
                peak_reserved_mb=None,
            ),
        )

        assert result == {
            "time": pytest.approx(0.5),
            "samples_per_s": pytest.approx(5.0),
            "tokens_per_s": pytest.approx(50.0),
        }

    def test_merge_result_adds_peak_memory(self) -> None:
        telemetry = LoopTelemetry(_TelemetryRunner(), start_time=0.0)
        result: dict[str, float] = {}

        telemetry.merge_result(
            result,
            elapsed_seconds=2.0,
            steps=4,
            snapshot=RuntimeTelemetrySnapshot(
                sample_count=0,
                sample_known=False,
                token_count=0,
                token_known=False,
                peak_allocated_mb=7.0,
                peak_reserved_mb=11.0,
            ),
        )

        assert result == {
            "time": pytest.approx(0.5),
            "max_memory_allocated_mb": 7.0,
            "max_memory_reserved_mb": 11.0,
        }

    def test_loop_telemetry_consumes_interval_without_resetting_total(self) -> None:
        telemetry = LoopTelemetry(_TelemetryRunner(), start_time=0.0)
        telemetry.observe(
            iteration=0,
            data=torch.ones(4, 2),
            current_time=0.5,
            peak_allocated_mb=5.0,
            peak_reserved_mb=8.0,
        )
        telemetry.observe(
            iteration=1,
            data=NestedTensor([torch.ones(32)]),
            current_time=1.0,
            peak_allocated_mb=7.0,
            peak_reserved_mb=None,
        )

        steps, snapshot = telemetry.consume_interval(iteration=1)

        assert steps == 2
        assert snapshot.sample_count == 4
        assert snapshot.token_count == 32
        assert snapshot.peak_allocated_mb == 7.0
        assert snapshot.peak_reserved_mb == 8.0
        assert telemetry.interval.sample_known is False
        assert telemetry.total.sample_count == 4
        assert telemetry.total.token_count == 32
        assert telemetry.total.peak_allocated_mb == 7.0

    def test_finalize_result_uses_total_progress(self) -> None:
        telemetry = LoopTelemetry(_TelemetryRunner(), start_time=0.0)
        telemetry.observe(
            iteration=0,
            data=NestedTensor([torch.ones(9)]),
            current_time=1.0,
            peak_allocated_mb=4.0,
            peak_reserved_mb=6.0,
        )

        result = telemetry.finalize_result({}, elapsed_seconds=3.0)

        assert result["time"] == pytest.approx(3.0)
        assert "samples_per_s" not in result
        assert result["tokens_per_s"] == pytest.approx(3.0)
        assert result["max_memory_allocated_mb"] == 4.0
        assert result["max_memory_reserved_mb"] == 6.0


class TestLoopTelemetryRuntime:

    def test_constructor_resets_peak_memory_only_when_logging(self) -> None:
        logging_runner = _TelemetryRunner(log_interval=1)
        telemetry = RecordingLoopTelemetry(logging_runner, start_time=7.0)

        assert telemetry.start_time == 7.0
        assert logging_runner.reset_calls == [logging_runner.device]

        quiet_runner = _TelemetryRunner(log_interval=0)
        RecordingLoopTelemetry(quiet_runner, start_time=8.0)
        assert quiet_runner.reset_calls == []

    def test_emit_log_records_loss_rates(self) -> None:
        _, _, result = _emit_training_log()

        assert result["loss"] == pytest.approx(2.0)
        assert result["samples_per_s"] == pytest.approx(2.5)
        assert result["tokens_per_s"] == pytest.approx(12.5)
        assert result["max_memory_allocated_mb"] == 9.0
        assert result["max_memory_reserved_mb"] == 13.0

    def test_emit_log_resets_interval(self) -> None:
        runner, telemetry, _ = _emit_training_log()

        assert telemetry.log_time == 2.0
        assert telemetry.interval.sample_known is False
        assert runner.reset_calls == [runner.device]

    def test_cpu_memory_helpers_are_noops_without_cuda_runtime(self) -> None:
        device = torch.device("cpu")

        LoopTelemetry.reset_device_peak_memory_stats(device)
        assert LoopTelemetry.current_device_peak_memory_mb(device) == (None, None)
