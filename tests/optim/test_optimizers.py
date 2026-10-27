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

import pytest
import torch
import torch.nn as nn
from packaging import version
from torch import optim

try:
    import deepspeed as ds
except ImportError:
    ds = None

from danling.optim import OptimizerContainer
from danling.optim.registry import OPTIMIZERS
from tests.runners.distributed import process_group, require_gloo, run_distributed


def _global_gradient_clipping_worker(rank, world_size):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    with process_group("gloo", rank, world_size):
        # Pipeline stages own separate parameters, including stages with no grads.
        parameter = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        parameter.grad = torch.tensor([3.0 if rank == 0 else 4.0], dtype=torch.float64)
        container = OptimizerContainer(optim.SGD([parameter], lr=0.1))
        norm = container.clip_gradients(max_grad_norm=1.0, pipeline_group=torch.distributed.group.WORLD)
        assert norm == pytest.approx(5.0)
        torch.testing.assert_close(
            parameter.grad, torch.tensor([0.6 if rank == 0 else 0.8], dtype=torch.float64), atol=1e-6, rtol=1e-6
        )
        parameter.grad = None if rank == 0 else torch.tensor([4.0], dtype=torch.float64)
        assert container.clip_gradients(
            max_grad_norm=1.0, pipeline_group=torch.distributed.group.WORLD, device=torch.device("cpu")
        ) == pytest.approx(4.0)

        # Parameters can be sharded over distinct meshes on the same ranks.
        first_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("first",))
        second_mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("replica", "second"))
        first = nn.Parameter(distribute_tensor(torch.zeros(2, dtype=torch.float64), first_mesh, (Shard(0),)))
        second = nn.Parameter(distribute_tensor(torch.zeros(2, dtype=torch.float64), second_mesh, (Shard(0), Shard(0))))
        first.grad = distribute_tensor(torch.tensor([3.0, 0.0], dtype=torch.float64), first_mesh, (Shard(0),))
        second.grad = distribute_tensor(
            torch.tensor([4.0, 0.0], dtype=torch.float64), second_mesh, (Shard(0), Shard(0))
        )
        container = OptimizerContainer(optim.SGD([first, second], lr=0.1))
        assert container.clip_gradients(max_grad_norm=1.0) == pytest.approx(5.0)
        torch.testing.assert_close(
            first.grad.full_tensor(), torch.tensor([0.6, 0.0], dtype=torch.float64), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            second.grad.full_tensor(), torch.tensor([0.8, 0.0], dtype=torch.float64), atol=1e-6, rtol=1e-6
        )
        second = nn.Parameter(distribute_tensor(torch.zeros(2, dtype=torch.float64), first_mesh, (Replicate(),)))
        first.grad = distribute_tensor(torch.tensor([3.0, 0.0], dtype=torch.float64), first_mesh, (Shard(0),))
        second.grad = distribute_tensor(torch.tensor([4.0, 0.0], dtype=torch.float64), first_mesh, (Replicate(),))
        container = OptimizerContainer(optim.SGD([first, second], lr=0.1))
        assert container.clip_gradients(max_grad_norm=1.0) == pytest.approx(5.0)
        torch.testing.assert_close(
            first.grad.full_tensor(), torch.tensor([0.6, 0.0], dtype=torch.float64), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            second.grad.full_tensor(), torch.tensor([0.8, 0.0], dtype=torch.float64), atol=1e-6, rtol=1e-6
        )
        if torch.cuda.is_available():
            local = nn.Parameter(torch.zeros(1, dtype=torch.float64))
            off_device = nn.Parameter(torch.zeros(1, dtype=torch.float64, device="cuda"))
            local.grad = torch.tensor([3.0], dtype=torch.float64)
            off_device.grad = torch.tensor([4.0], dtype=torch.float64, device="cuda")
            container = OptimizerContainer(optim.SGD([local, off_device], lr=0.1))
            norm = container.clip_gradients(
                max_grad_norm=1.0, pipeline_group=torch.distributed.group.WORLD, device=torch.device("cpu")
            )
            assert norm == pytest.approx(50**0.5)
            torch.testing.assert_close(local.grad, torch.tensor([3 / norm], dtype=torch.float64), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(
                off_device.grad.cpu(), torch.tensor([4 / norm], dtype=torch.float64), atol=1e-6, rtol=1e-6
            )


def test_OPTIMIZERS():
    model = nn.Linear(10, 1)

    assert isinstance(OPTIMIZERS.build(model.parameters(), type="sgd", lr=0.01), optim.SGD)
    assert isinstance(
        OPTIMIZERS.build(model.parameters(), type="torch_adamw", lr=0.001, weight_decay=0.01), optim.AdamW
    )
    assert isinstance(OPTIMIZERS.build(model.parameters(), type="torch_adam", lr=0.001), optim.Adam)
    if version.parse(torch.__version__) >= version.parse("2.5.0"):
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="adafactor", lr=0.001), optim.Adafactor)


def test_optimizer_container_returns_step_result_with_grad_norm():
    model = nn.Linear(2, 1, bias=False)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    container = OptimizerContainer(optimizer)

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    result = container.step(max_grad_norm=1.0)

    assert result.stepped is True
    assert result.grad_norm is not None
    assert result.grad_norm > 0


def test_optimizer_container_clips_one_norm_across_pipeline_stages_and_distinct_meshes():
    require_gloo()
    run_distributed(_global_gradient_clipping_worker)


def test_optimizer_container_skip_nonfinite_returns_step_result_without_update():
    model = nn.Linear(2, 1, bias=False)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    container = OptimizerContainer(optimizer)
    initial = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("inf"))

    result = container.step(skip_nonfinite_grad=True)

    assert result.stepped is False
    assert result.grad_norm is None
    torch.testing.assert_close(model.weight, initial)


@pytest.mark.skipif(ds is None, reason="deepspeed is not installed")
def test_deepspeed_optimizers():
    model = nn.Linear(10, 1)

    if torch.cuda.device_count() > 0:
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="ds_adam", lr=0.001), ds.ops.adam.FusedAdam)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="ds_adamw", lr=0.001), ds.ops.adam.FusedAdam)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="lamb", lr=0.001), ds.ops.lamb.FusedLamb)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="lion", lr=0.001), ds.ops.lion.FusedLion)
    else:
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="ds_adam", lr=0.001), ds.ops.adam.DeepSpeedCPUAdam)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="ds_adamw", lr=0.001), ds.ops.adam.DeepSpeedCPUAdam)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="lamb", lr=0.001), ds.ops.lamb.DeepSpeedCPULamb)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="lion", lr=0.001), ds.ops.lion.DeepSpeedCPULion)

    assert isinstance(OPTIMIZERS.build(model.parameters(), type="adam", lr=0.001), optim.Adam)
    assert isinstance(OPTIMIZERS.build(model.parameters(), type="adamw", lr=0.001), optim.AdamW)
    assert isinstance(OPTIMIZERS.build(model.parameters(), type="lamb", lr=0.001), ds.ops.lamb.DeepSpeedCPULamb)
    assert isinstance(OPTIMIZERS.build(model.parameters(), type="lion", lr=0.001), ds.ops.lion.DeepSpeedCPULion)


def test_optimizer_container_checks_added_parameter_groups():
    first, added = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
    container = OptimizerContainer(optim.SGD([first], lr=0.1))
    first.grad = torch.zeros_like(first)
    assert container.has_nan_inf_grad() is False
    container.optimizer.add_param_group({"params": [added]})
    added.grad = torch.full_like(added, float("inf"))
    assert container.step(skip_nonfinite_grad=True).stepped is False
    torch.testing.assert_close(added, torch.ones(1))


def test_optimizer_container_accepts_finite_sparse_gradients():
    embedding = nn.Embedding(4, 2, sparse=True)
    container = OptimizerContainer(optim.SGD(embedding.parameters(), lr=0.1))
    embedding(torch.tensor([1, 2])).sum().backward()
    assert container.step(skip_nonfinite_grad=True).stepped is True
