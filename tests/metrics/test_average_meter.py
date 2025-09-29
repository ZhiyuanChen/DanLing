# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from math import isnan

import pytest
import torch
from torch.testing import assert_close

from danling.metrics.average_meter import AverageMeter, AverageMeters

from .utils import process_group, run_distributed


def _distributed_average_meter_scalar_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        meter = AverageMeter(device="cpu", distributed=True)
        if rank == 0:
            meter.update(2.0, n=2)
            meter.update(4.0, n=1)
            assert meter.val == pytest.approx(4.0)
            assert meter.count == 3
        else:
            meter.update(1.0, n=1)
            meter.update(5.0, n=5)
            assert meter.val == pytest.approx(5.0)
            assert meter.count == 6

        assert meter.batch() == pytest.approx(29.0 / 6.0)
        assert meter.average() == pytest.approx(34.0 / 9.0)


def _distributed_average_meter_tensor_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        meter = AverageMeter(device="cpu", distributed=True)
        if rank == 0:
            meter.update(torch.tensor([1.0, 2.0]), n=2)
            meter.update(torch.tensor([3.0, 5.0]), n=1)
            assert_close(meter.val, torch.tensor([3.0, 5.0], dtype=torch.float64))
            assert meter.count == 3
        else:
            meter.update(torch.tensor([0.0, 1.0]), n=1)
            meter.update(torch.tensor([6.0, 7.0]), n=2)
            assert_close(meter.val, torch.tensor([6.0, 7.0], dtype=torch.float64))
            assert meter.count == 3

        assert_close(meter.batch(), torch.tensor([5.0, 19.0 / 3.0], dtype=torch.float64))
        assert_close(meter.average(), torch.tensor([17.0 / 6.0, 4.0], dtype=torch.float64))


def _distributed_average_meter_zero_count_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        meter = AverageMeter(device="cpu", distributed=True)
        meter.update(1.0, n=0)

        assert isnan(meter.batch())
        assert isnan(meter.average())


def _distributed_average_meter_inconsistent_tensor_metadata_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        meter = AverageMeter(device="cpu", distributed=True)
        if rank == 0:
            meter.update(torch.tensor([1.0, 2.0]))
        else:
            meter.update(torch.tensor([1.0, 2.0, 3.0]))
        meter.batch()


def _distributed_average_meters_mixed_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        meters = AverageMeters(loss=AverageMeter(device="cpu"), vector=AverageMeter(device="cpu"))
        if rank == 0:
            meters["loss"].update(2.0, n=2)
            meters["vector"].update(torch.tensor([1.0, 2.0]), n=2)
        else:
            meters["loss"].update(4.0, n=1)
            meters["vector"].update(torch.tensor([3.0, 5.0]), n=1)

        batches = meters.batch()
        averages = meters.average()

        assert batches["loss"] == pytest.approx(8.0 / 3.0)
        assert averages["loss"] == pytest.approx(8.0 / 3.0)
        assert_close(batches["vector"], torch.tensor([5.0 / 3.0, 3.0], dtype=torch.float64))
        assert_close(averages["vector"], torch.tensor([5.0 / 3.0, 3.0], dtype=torch.float64))


class TestAverageMeter:
    def test_tracks_basic_statistics(self):
        meter = AverageMeter()
        assert isnan(meter.val)
        assert isnan(meter.avg)

    def test_accepts_scalar_tensor_updates(self):
        meter = AverageMeter()
        value = torch.tensor(2.5)

        meter.update(value, n=2)

        assert meter.device == value.device
        assert not isinstance(meter.val, torch.Tensor)
        assert not isinstance(meter.avg, torch.Tensor)
        assert meter.val == pytest.approx(2.5)
        assert meter.sum == pytest.approx(5.0)
        assert meter.count == 2
        assert meter.avg == pytest.approx(2.5)

    def test_tracks_tensor_updates_elementwise(self):
        meter = AverageMeter()
        value = torch.tensor([1.0, 2.0])
        meter.update(value, n=3)
        assert meter.device == value.device
        assert_close(meter.val, torch.tensor([1.0, 2.0], dtype=torch.float64))
        assert_close(meter.sum, torch.tensor([3.0, 6.0], dtype=torch.float64))
        assert meter.count == 3
        assert_close(meter.avg, torch.tensor([1.0, 2.0], dtype=torch.float64))
        assert "[" in f"{meter:.2f}"
        assert "AverageMeter" in repr(meter)

        meter.reset(device="cpu")
        assert meter.device == torch.device("cpu")
        assert isnan(meter.val)
        assert isnan(meter.avg)

    def test_preserves_empty_tensor_shape(self):
        meter = AverageMeter()
        meter.update(torch.tensor([float("nan"), float("nan")]), n=0)

        assert meter.count == 0
        assert meter.n == 0
        assert isinstance(meter.val, torch.Tensor)
        assert isinstance(meter.avg, torch.Tensor)
        assert meter.val.shape == torch.Size([2])
        assert meter.avg.shape == torch.Size([2])
        assert torch.isnan(meter.val).all()
        assert torch.isnan(meter.avg).all()
        assert_close(meter.sum, torch.zeros(2, dtype=torch.float64))

    def test_rejects_mixed_scalar_and_tensor_updates(self):
        meter = AverageMeter()
        meter.update(torch.tensor([1.0, 2.0]))

        with pytest.raises(ValueError, match="mix tensor and scalar values"):
            meter.update(1.0)

    def test_rejects_tensor_shape_changes(self):
        meter = AverageMeter()
        meter.update(torch.tensor([1.0, 2.0]))

        with pytest.raises(ValueError, match="consistent tensor shapes"):
            meter.update(torch.tensor([[1.0, 2.0]]))

    def test_zero_count_scalar_preserves_tensor_state(self):
        meter = AverageMeter()
        meter.update(torch.tensor([1.0, 2.0]), n=2)

        meter.update(3.0, n=0)

        assert meter.n == 0
        assert meter.count == 2
        assert_close(meter.val, torch.full((2,), 3.0, dtype=torch.float64))
        assert_close(meter.sum, torch.tensor([2.0, 4.0], dtype=torch.float64))
        assert_close(meter.avg, torch.tensor([1.0, 2.0], dtype=torch.float64))

    def test_batch_is_local(self):
        meter = AverageMeter(device="cpu", distributed=False)
        meter.update(2.0, n=2)
        meter.update(4.0, n=1)

        assert meter.bat == 4.0
        assert meter.batch() == 4.0

    def test_average_is_local(self):
        meter = AverageMeter(device="cpu", distributed=False)
        meter.update(2.0, n=2)
        meter.update(4.0, n=1)

        assert meter.average() == pytest.approx(8.0 / 3.0)

    def test_sync_across_ranks(self):
        run_distributed(_distributed_average_meter_scalar_worker, world_size=2)

    def test_tensor_sync_across_ranks(self):
        run_distributed(_distributed_average_meter_tensor_worker, world_size=2)

    def test_rejects_inconsistent_metadata_across_ranks(self):
        with pytest.raises(Exception, match="inconsistent tensor metadata across ranks"):
            run_distributed(_distributed_average_meter_inconsistent_tensor_metadata_worker, world_size=2)

    def test_distributed_zero_count_returns_nan(self):
        run_distributed(_distributed_average_meter_zero_count_worker, world_size=2)

    def test_format_uses_local_value_and_average(self):
        meter = AverageMeter(device="cpu", distributed=False)
        meter.update(2.0, n=2)
        meter.update(4.0, n=1)

        assert f"{meter:.2f}" == "4.00 (2.67)"


class TestAverageMeters:
    def test_update_accepts_mapping_values(self):
        meters = AverageMeters()
        payload = {"loss": torch.tensor(0.5)}
        meters.update(payload)
        assert set(payload.keys()) == {"loss"}
        assert_close(payload["loss"], torch.tensor(0.5))
        assert meters.val["loss"] == pytest.approx(0.5)

    def test_update_accepts_keyword_values(self):
        meters = AverageMeters()
        meters.update(acc=1.0)
        assert meters.val["acc"] == pytest.approx(1.0)

        meters.update(loss=0.7)
        assert meters.sum["loss"] == pytest.approx(0.7)
        assert meters.n["loss"] == 1
        assert meters.n["acc"] == 1
        assert meters.count["loss"] == 1
        assert meters.count["acc"] == 1

    def test_update_rejects_multiple_positional_arguments(self):
        meters = AverageMeters()
        with pytest.raises(ValueError, match="Expected only one positional argument"):
            meters.update({}, {})

    def test_sync_across_ranks(self):
        run_distributed(_distributed_average_meters_mixed_worker, world_size=2)
