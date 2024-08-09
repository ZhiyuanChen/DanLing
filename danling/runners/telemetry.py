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

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import distributed as dist

from danling.tensors import NestedTensor


def _max_optional(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return max(lhs, rhs)


@dataclass
class RuntimeTelemetrySnapshot:
    """
    Mutable runtime counters for either a logging interval or a whole loop.

    ``sample_known`` and ``token_known`` distinguish "zero observed" from
    "this loop cannot infer that counter". Unknown observations do not add
    zeros, so throughput fields are only emitted when the loop has seen at
    least one semantically valid count.
    """

    sample_count: int = 0
    sample_known: bool = False
    token_count: int = 0
    token_known: bool = False
    peak_allocated_mb: float | None = None
    peak_reserved_mb: float | None = None

    def observe_progress(
        self,
        *,
        sample_count: int,
        sample_known: bool,
        token_count: int,
        token_known: bool,
    ) -> None:
        if sample_known:
            self.sample_count += sample_count
            self.sample_known = True
        if token_known:
            self.token_count += token_count
            self.token_known = True

    def observe_peaks(self, *, peak_allocated_mb: float | None, peak_reserved_mb: float | None) -> None:
        self.peak_allocated_mb = _max_optional(self.peak_allocated_mb, peak_allocated_mb)
        self.peak_reserved_mb = _max_optional(self.peak_reserved_mb, peak_reserved_mb)


class LoopTelemetry:
    """
    Runner-bound telemetry state for one train/evaluate loop.

    Contract:
        - Construct exactly once per logical loop after runner mode/split and
          meters are initialized.
        - Call ``observe`` once for every consumed batch.
        - Call ``emit_log`` at log boundaries; it consumes the current interval
          and writes through ``runner.step_log``.
        - Call ``finalize_result`` once at loop end to merge total throughput
          and peak-memory fields into the returned result.

    ``LoopTelemetry`` owns runtime telemetry and per-batch time-meter updates.
    It deliberately does not mutate optimizer state, checkpoint state, or
    train/eval progress counters.
    """

    def __init__(self, runner: Any, *, start_time: float | None = None) -> None:
        self.runner = runner
        if start_time is None:
            start_time = runner.loop_time()
        self.start_time = start_time
        self.step_time = start_time
        self.log_time = start_time
        self.reduce_device = runner.all_reduce_device()
        self.processed_batches = 0
        self.last_iteration: int | None = None
        self.last_print_iteration = -1
        self.total = RuntimeTelemetrySnapshot()
        self.interval = RuntimeTelemetrySnapshot()
        if runner.log_interval > 0:
            self.reset_peak_memory_stats()

    def observe(
        self,
        *,
        iteration: int,
        data: Any,
        current_time: float,
        peak_allocated_mb: float | None = None,
        peak_reserved_mb: float | None = None,
    ) -> None:
        sample_count, sample_known, token_count, token_known = self.infer_batch_counts(data)
        self.last_iteration = iteration
        self.processed_batches += 1
        self.runner.meters.time.update(current_time - self.step_time)
        self.step_time = current_time
        self.total.observe_progress(
            sample_count=sample_count,
            sample_known=sample_known,
            token_count=token_count,
            token_known=token_known,
        )
        self.interval.observe_progress(
            sample_count=sample_count,
            sample_known=sample_known,
            token_count=token_count,
            token_known=token_known,
        )
        self.total.observe_peaks(
            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,
        )
        self.interval.observe_peaks(
            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,
        )

    def consume_interval(self, *, iteration: int) -> tuple[int, RuntimeTelemetrySnapshot]:
        steps = iteration - self.last_print_iteration
        snapshot = RuntimeTelemetrySnapshot(
            sample_count=self.interval.sample_count,
            sample_known=self.interval.sample_known,
            token_count=self.interval.token_count,
            token_known=self.interval.token_known,
            peak_allocated_mb=self.interval.peak_allocated_mb,
            peak_reserved_mb=self.interval.peak_reserved_mb,
        )
        self.interval = RuntimeTelemetrySnapshot()
        self.last_print_iteration = iteration
        return steps, snapshot

    def merge_result(
        self,
        result,
        *,
        elapsed_seconds: float,
        steps: int,
        snapshot: RuntimeTelemetrySnapshot | None = None,
    ) -> MutableMapping[str, Any]:
        elapsed = float(elapsed_seconds)
        if snapshot is None:
            snapshot = RuntimeTelemetrySnapshot()
        sample_count = snapshot.sample_count
        sample_known = snapshot.sample_known
        token_count = snapshot.token_count
        token_known = snapshot.token_known
        peak_allocated_mb = snapshot.peak_allocated_mb
        peak_reserved_mb = snapshot.peak_reserved_mb

        if self.runner.distributed and dist.is_available() and dist.is_initialized():
            elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=self.reduce_device)
            self.runner.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor.item())

            count_tensor = torch.tensor(
                [
                    float(sample_count),
                    float(sample_known),
                    float(token_count),
                    float(token_known),
                ],
                dtype=torch.float64,
                device=self.reduce_device,
            )
            self.runner.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            sample_count = int(round(float(count_tensor[0].item())))
            sample_known = count_tensor[1].item() > 0
            token_count = int(round(float(count_tensor[2].item())))
            token_known = count_tensor[3].item() > 0

            peak_tensor = torch.tensor(
                [
                    0.0 if peak_allocated_mb is None else float(peak_allocated_mb),
                    1.0 if peak_allocated_mb is not None else 0.0,
                    0.0 if peak_reserved_mb is None else float(peak_reserved_mb),
                    1.0 if peak_reserved_mb is not None else 0.0,
                ],
                dtype=torch.float64,
                device=self.reduce_device,
            )
            self.runner.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
            peak_allocated_mb = float(peak_tensor[0].item()) if peak_tensor[1].item() > 0 else None
            peak_reserved_mb = float(peak_tensor[2].item()) if peak_tensor[3].item() > 0 else None

        if steps > 0:
            result["time"] = elapsed / steps
        if sample_known and elapsed > 0:
            result["samples_per_s"] = sample_count / elapsed
        if token_known and elapsed > 0:
            result["tokens_per_s"] = token_count / elapsed
        if peak_allocated_mb is not None:
            result["max_memory_allocated_mb"] = peak_allocated_mb
        if peak_reserved_mb is not None:
            result["max_memory_reserved_mb"] = peak_reserved_mb
        return result

    def emit_log(
        self,
        *,
        split: str,
        iteration: int,
        length: int | None,
        loss: torch.Tensor | None,
        loss_n: int | None,
        display_iteration: int | None = None,
        reset_peak_stats: bool = True,
    ) -> None:
        steps, snapshot = self.consume_interval(iteration=iteration)
        result = self.runner.get_step_result()
        reduced_loss = self.runner.reduce_loss_for_logging(loss, loss_n)
        if reduced_loss is not None:
            result["loss"] = reduced_loss.item()
        peak_allocated_mb, peak_reserved_mb = self.current_peak_memory_mb()
        snapshot.observe_peaks(
            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,
        )
        self.total.observe_peaks(
            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,
        )
        log_current_time = self.runner.loop_time(sync=True)
        self.merge_result(
            result,
            elapsed_seconds=log_current_time - self.log_time,
            steps=steps,
            snapshot=snapshot,
        )
        self.log_time = log_current_time
        self.runner.step_log(
            split, iteration if display_iteration is None else display_iteration, length, result=result
        )
        if reset_peak_stats:
            self.reset_peak_memory_stats()

    def finalize_result(
        self,
        result,
        *,
        elapsed_seconds: float | None = None,
        steps: int | None = None,
    ) -> MutableMapping[str, Any]:
        if elapsed_seconds is None:
            elapsed_seconds = self.runner.loop_time(sync=True) - self.start_time
        return self.merge_result(
            result,
            elapsed_seconds=elapsed_seconds,
            steps=self.processed_batches if steps is None else steps,
            snapshot=self.total,
        )

    @staticmethod
    def reset_device_peak_memory_stats(device: torch.device) -> None:
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def current_device_peak_memory_mb(device: torch.device) -> tuple[float | None, float | None]:
        if not (torch.cuda.is_available() and device.type == "cuda"):
            return None, None
        scale = 1024**2
        return (
            float(torch.cuda.max_memory_allocated(device)) / scale,
            float(torch.cuda.max_memory_reserved(device)) / scale,
        )

    def reset_peak_memory_stats(self) -> None:
        self.reset_device_peak_memory_stats(self.runner.device)

    def current_peak_memory_mb(self) -> tuple[float | None, float | None]:
        return self.current_device_peak_memory_mb(self.runner.device)

    @staticmethod
    def find_nested_tensor(data: Any) -> NestedTensor | None:
        if isinstance(data, NestedTensor):
            return data
        if isinstance(data, Mapping):
            for value in data.values():
                nested = LoopTelemetry.find_nested_tensor(value)
                if nested is not None:
                    return nested
            return None
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            for value in data:
                nested = LoopTelemetry.find_nested_tensor(value)
                if nested is not None:
                    return nested
        return None

    @staticmethod
    def find_batch_tensor(data: Any) -> torch.Tensor | None:
        if torch.is_tensor(data) and data.ndim > 0:
            return data
        if isinstance(data, Mapping):
            for value in data.values():
                tensor = LoopTelemetry.find_batch_tensor(value)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            for value in data:
                tensor = LoopTelemetry.find_batch_tensor(value)
                if tensor is not None:
                    return tensor
        return None

    def infer_batch_counts(self, data: Any) -> tuple[int, bool, int, bool]:
        if not getattr(self.runner, "reports_batch_telemetry", True):
            return 0, False, 0, False

        nested = self.find_nested_tensor(data)
        if nested is not None:
            return 0, False, int(nested.numel()), True

        tensor = self.find_batch_tensor(data)
        if tensor is None:
            return 0, False, 0, False
        return int(tensor.shape[0]), True, 0, False
