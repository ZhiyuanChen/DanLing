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

import danling.metrics.average_meter as average_meter_module
from danling.metrics.average_meter import AverageMeter, AverageMeters


def test_average_meter_tracks_basic_statistics():
    meter = AverageMeter()
    assert isnan(meter.val)
    assert isnan(meter.avg)

    value = torch.tensor(2.0)
    meter.update(value, n=3)
    assert meter.device == value.device
    assert meter.val == 2.0
    assert meter.sum == 6.0
    assert meter.count == 3
    assert meter.avg == 2.0
    assert f"{meter:.2f}" == "2.00 (2.00)"
    assert "AverageMeter" in repr(meter)

    meter.reset(device="cpu")
    assert meter.device == torch.device("cpu")
    assert isnan(meter.val)
    assert isnan(meter.avg)


def test_average_meter_non_distributed_batch_and_average(monkeypatch):
    meter = AverageMeter(device="cpu")
    meter.update(2.0, n=2)
    meter.update(4.0, n=1)

    monkeypatch.setattr(average_meter_module, "get_world_size", lambda: 1)
    assert meter.bat == 4.0
    assert meter.batch() == 4.0
    assert meter.average() == pytest.approx(8.0 / 3.0)


def test_average_meter_distributed_batch_and_average_paths(monkeypatch):
    meter = AverageMeter(device="cpu")
    meter.update(3.0, n=2)
    monkeypatch.setattr(average_meter_module, "get_world_size", lambda: 2)

    reduced_tensors = []

    def fake_all_reduce(tensor):
        reduced_tensors.append(tensor.clone())

    monkeypatch.setattr(average_meter_module.dist, "all_reduce", fake_all_reduce)

    assert meter.batch() == 3.0
    assert meter.average() == 3.0
    assert len(reduced_tensors) == 2
    assert_close(reduced_tensors[0], torch.tensor([6.0, 2.0], dtype=torch.float64))
    assert_close(reduced_tensors[1], torch.tensor([6.0, 2.0], dtype=torch.float64))


def test_average_meter_distributed_zero_count_returns_nan(monkeypatch):
    meter = AverageMeter(device="cpu")
    meter.update(1.0, n=1)
    monkeypatch.setattr(average_meter_module, "get_world_size", lambda: 2)

    def zero_all_reduce(tensor):
        tensor.zero_()

    monkeypatch.setattr(average_meter_module.dist, "all_reduce", zero_all_reduce)
    assert isnan(meter.batch())
    assert isnan(meter.average())


def test_average_meter_infer_device_from_distributed_backend(monkeypatch):
    meter = AverageMeter()
    monkeypatch.setattr(average_meter_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(average_meter_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(average_meter_module.dist, "get_backend", lambda: "gloo")

    device = meter._infer_device()
    assert device.type == "cpu"
    assert meter.device == device


def test_average_meter_infer_device_warns_on_unknown_backend(monkeypatch):
    meter = AverageMeter()
    monkeypatch.setattr(average_meter_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(average_meter_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(average_meter_module.dist, "get_backend", lambda: "unknown")

    with pytest.warns(UserWarning, match="Failed to infer device"):
        device = meter._infer_device()

    assert device.type == "cpu"
    assert meter.device == device


def test_average_meter_infer_device_without_dist_defaults_to_cpu(monkeypatch):
    meter = AverageMeter()
    monkeypatch.setattr(average_meter_module.dist, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    device = meter._infer_device()
    assert device.type == "cpu"
    assert meter.device == device


def test_average_meter_infer_device_uses_cuda_for_distributed_nccl(monkeypatch):
    meter = AverageMeter()
    monkeypatch.setattr(average_meter_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(average_meter_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(average_meter_module.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def raise_runtime_error():
        raise RuntimeError("no current cuda device")

    monkeypatch.setattr(torch.cuda, "current_device", raise_runtime_error)
    device = meter._infer_device()
    assert device.type == "cuda"
    assert device.index == 0


def test_average_meter_infer_device_uses_cuda_without_dist(monkeypatch):
    meter = AverageMeter()
    monkeypatch.setattr(average_meter_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(average_meter_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def raise_assertion_error():
        raise AssertionError("cuda context unavailable")

    monkeypatch.setattr(torch.cuda, "current_device", raise_assertion_error)
    device = meter._infer_device()
    assert device.type == "cuda"
    assert device.index == 0


def test_average_meter_infer_device_uses_mps_without_dist(monkeypatch):
    if getattr(torch.backends, "mps", None) is None:
        pytest.skip("MPS backend is unavailable in this PyTorch build")

    meter = AverageMeter()
    monkeypatch.setattr(average_meter_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(average_meter_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = meter._infer_device()
    assert device.type == "mps"
    assert meter.device == device


def test_average_meters_update_and_validation():
    meters = AverageMeters()
    payload = {"loss": torch.tensor(0.5)}
    meters.update(payload, acc=1.0)
    assert meters.val["loss"] == pytest.approx(0.5)
    assert meters.val["acc"] == pytest.approx(1.0)

    meters.update(loss=0.7)
    assert meters.sum["loss"] == pytest.approx(1.2)
    assert meters.count["loss"] == 2
    assert meters.count["acc"] == 1

    with pytest.raises(ValueError, match="Expected only one positional argument"):
        meters.update({}, {})

    with pytest.raises(ValueError, match="Expected values to be int or float"):
        meters.update(loss="bad")
