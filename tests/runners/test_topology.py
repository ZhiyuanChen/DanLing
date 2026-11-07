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

import pytest
import torch
from torch import distributed as dist

from danling.runners.topology import ParallelContext, ParallelTopology
from tests.runners.distributed import configure_distributed_env, process_group, require_gloo, run_distributed


def _context_sum_worker(rank: int, world_size: int) -> None:
    configure_distributed_env(rank, world_size)
    with process_group("gloo", rank, world_size):
        groups = {}
        for name, ranks in (
            ("context", (0, 1)),
            ("context", (2, 3)),
            ("replicate", (0, 2)),
            ("replicate", (1, 3)),
        ):
            group = dist.new_group(ranks=list(ranks), backend="gloo")
            if rank in ranks:
                groups[name] = group
        topology = ParallelTopology(
            world_size=world_size,
            rank=rank,
            axes={"replicate": 2, "context": 2},
            domains={"data": ("replicate",), "loss": ("replicate", "context")},
        )
        parallel = ParallelContext(topology, groups=groups)
        for lengths in ((1, 4), (3, 2)):
            data_rank = topology.axis_rank("replicate")
            context_rank = topology.axis_rank("context")
            weight1 = torch.nn.Parameter(torch.arange(12, dtype=torch.float64).reshape(3, 4) / 19)
            weight2 = torch.nn.Parameter(torch.arange(8, dtype=torch.float64).reshape(4, 2) / 13)
            reference1 = weight1.detach().clone().requires_grad_()
            reference2 = weight2.detach().clone().requires_grad_()
            inputs = [torch.arange(n * 3, dtype=torch.float64).reshape(n, 3) / 11 for n in lengths]
            value = inputs[data_rank].clone().requires_grad_()
            rows = torch.arange(context_rank, value.shape[0], 2)
            local = value.index_select(0, rows) @ weight1
            packed = local.new_zeros((value.shape[0], 4)).index_copy(0, rows, local)
            packed = packed.T.contiguous().T
            before = packed.detach().clone()
            joined = parallel.all_reduce(packed, domain="context", autograd=True)
            torch.testing.assert_close(packed, before, rtol=0, atol=0)
            expected = value @ reference1
            torch.testing.assert_close(joined, expected, rtol=0, atol=0)
            loss = (joined @ weight2).tanh().square().mean()
            grad1, grad2, grad_input = torch.autograd.grad(loss, (weight1, weight2, value))
            local_reference_loss = (expected @ reference2).tanh().square().mean()
            reference_input = torch.autograd.grad(local_reference_loss, value, retain_graph=True)[0]
            parallel.all_reduce(grad_input, domain="context")
            torch.testing.assert_close(grad_input / 2, reference_input, rtol=1e-12, atol=1e-12)
            reference_loss = torch.stack([(x @ reference1 @ reference2).tanh().square().mean() for x in inputs]).mean()
            expected_grads = torch.autograd.grad(reference_loss, (reference1, reference2))
            for parameter, gradient, reference_gradient in zip(
                (weight1, weight2), (grad1, grad2), expected_grads, strict=True
            ):
                parallel.all_reduce(gradient, domain="loss")
                parameter.grad = gradient / world_size
                torch.testing.assert_close(parameter.grad, reference_gradient, rtol=1e-12, atol=1e-12)
            torch.optim.SGD((weight1, weight2), lr=0.125).step()
            for actual, original, gradient in zip(
                (weight1, weight2), (reference1, reference2), expected_grads, strict=True
            ):
                torch.testing.assert_close(actual, original.detach() - 0.125 * gradient, rtol=1e-12, atol=1e-12)


def test_context_sum_preserves_ragged_queries_vjps_and_replica_update() -> None:
    require_gloo(world_size=4)
    run_distributed(_context_sum_worker, world_size=4)


def test_autograd_sum_singleton_and_unsupported_reduction() -> None:
    topology = ParallelTopology(world_size=1, rank=0, axes={"context": 1})
    parallel = ParallelContext(topology)
    value = torch.tensor([2.0], requires_grad=True)
    assert parallel.all_reduce(value, domain="context") is value
    result = parallel.all_reduce(value, domain="context", autograd=True)
    (gradient,) = torch.autograd.grad(result.square().sum(), value)
    torch.testing.assert_close(gradient, torch.tensor([4.0]))
    with torch.no_grad():
        result.add_(1)
    torch.testing.assert_close(value, torch.tensor([2.0]))
    with pytest.raises(ValueError, match="SUM only"):
        parallel.all_reduce(value, domain="context", op=dist.ReduceOp.MAX, autograd=True)


class TestParallelTopology:

    @pytest.mark.parametrize(
        "rank,expected",
        [
            (0, (0, 0, 0)),
            (1, (0, 0, 1)),
            (2, (0, 1, 0)),
            (7, (0, 3, 1)),
            (8, (1, 0, 0)),
            (15, (1, 3, 1)),
        ],
    )
    def test_rank_to_ordered_axes(self, rank: int, expected: tuple[int, int, int]) -> None:
        topology = ParallelTopology(
            world_size=16,
            rank=rank,
            axes={"replicate": 2, "pipeline": 4, "tensor": 2},
        )

        assert (
            topology.axis_rank("replicate"),
            topology.axis_rank("pipeline"),
            topology.axis_rank("tensor"),
        ) == expected
        assert topology.mesh_shape == (2, 4, 2)
        assert topology.axis_names == ("replicate", "pipeline", "tensor")
        assert topology.domain_degree("replicate") == 2
        assert topology.domain_rank("replicate") == expected[0]

    def test_multi_axis_domain_rank(self) -> None:
        topology = ParallelTopology(
            world_size=16,
            rank=13,
            axes={"replicate": 2, "shard": 2, "pipeline": 2, "tensor": 2},
            domains={"data": ("replicate", "shard"), "fsdp": ("replicate", "shard")},
        )

        assert topology.axis_rank("replicate") == 1
        assert topology.axis_rank("shard") == 1
        assert topology.domain_degree("data") == 4
        assert topology.domain_rank("data") == 3

    def test_full_shard_data_domain(self) -> None:
        topology = ParallelTopology(
            world_size=8,
            rank=5,
            axes={
                "replicate": 1,
                "shard": 2,
                "context": 1,
                "pipeline": 2,
                "tensor": 2,
                "expert": 1,
                "expert_tensor": 1,
            },
            domains={
                "data": ("replicate", "shard"),
                "batch": ("replicate", "shard"),
                "loss": ("replicate", "shard", "context"),
                "fsdp": ("replicate", "shard", "context"),
            },
        )

        assert topology.axis_names == (
            "replicate",
            "shard",
            "context",
            "pipeline",
            "tensor",
            "expert",
            "expert_tensor",
        )
        assert topology.domain_axes("data") == ("replicate", "shard")
        assert topology.domain_degree("data") == 2
        assert topology.axis_degree("shard") == 2
        assert topology.axis_rank("shard") == 1
        assert topology.axis_degree("replicate") == 1
        assert topology.axis_rank("replicate") == 0

    def test_hybrid_shard_data_domain(self) -> None:
        topology = ParallelTopology(
            world_size=8,
            rank=5,
            axes={
                "replicate": 2,
                "shard": 1,
                "context": 1,
                "pipeline": 2,
                "tensor": 2,
                "expert": 1,
                "expert_tensor": 1,
            },
            domains={"data": ("replicate", "shard"), "fsdp": ("replicate", "shard", "context")},
        )

        assert topology.domain_axes("data") == ("replicate", "shard")
        assert topology.domain_degree("data") == 2
        assert topology.domain_rank("data") == 1
        assert topology.axis_degree("replicate") == 2
        assert topology.axis_rank("replicate") == 1
        assert topology.axis_degree("shard") == 1
        assert topology.axis_rank("shard") == 0

    def test_coordinates_to_rank(self) -> None:
        topology = ParallelTopology(
            world_size=32,
            rank=0,
            axes={"replicate": 2, "shard": 2, "context": 2, "pipeline": 4, "tensor": 1},
        )

        assert topology.rank_from_coordinates({"pipeline": 3}) == 3
        assert topology.rank_from_coordinates({"replicate": 1, "shard": 1, "pipeline": 3}) == 27

        with pytest.raises(ValueError, match="pipeline=4"):
            topology.rank_from_coordinates({"pipeline": 4})

        with pytest.raises(ValueError, match="unknown axes"):
            topology.rank_from_coordinates({"data": 0})

    def test_absent_axes_with_explicit_defaults(self) -> None:
        topology = ParallelTopology(
            world_size=16,
            rank=13,
            axes={"shard": 4, "pipeline": 2, "tensor": 2},
            domains={"data": ("shard",), "fsdp": ("shard",)},
        )

        assert topology.axis_rank("shard") == 3
        assert topology.axis_degree("replicate", default=1) == 1
        assert topology.axis_rank("replicate", default=0) == 0

    def test_auto_fills_single_axis_degree(self) -> None:
        topology = ParallelTopology(
            world_size=16,
            rank=13,
            axes={"replicate": 2, "shard": -1, "pipeline": 2, "tensor": 2},
            domains={"data": ("replicate", "shard")},
        )

        assert topology.axis_degree("shard") == 2
        assert topology.mesh_shape == (2, 2, 2, 2)
        assert topology.axis_rank("replicate") == 1
        assert topology.axis_rank("shard") == 1
        assert topology.axis_rank("pipeline") == 0
        assert topology.axis_rank("tensor") == 1
        assert topology.domain_degree("data") == 4
        assert topology.domain_rank("data") == 3

    @pytest.mark.parametrize(
        "world_size,axes",
        [
            (8, {"replicate": 0, "pipeline": 2, "tensor": 2}),
            (8, {"replicate": -2, "pipeline": 2, "tensor": 2}),
            (8, {"replicate": 1, "pipeline": 0, "tensor": 2}),
            (10, {"replicate": 2, "pipeline": 2, "tensor": 2}),
            (10, {"replicate": 2, "shard": -1, "pipeline": 4}),
        ],
    )
    def test_rejects_invalid_axis_layout(self, world_size: int, axes: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="invalid parallel topology"):
            ParallelTopology(world_size=world_size, rank=0, axes=axes)

    def test_rejects_multiple_auto_filled_axes(self) -> None:
        with pytest.raises(ValueError, match="at most one axis"):
            ParallelTopology(
                world_size=8,
                rank=0,
                axes={"replicate": -1, "shard": -1, "pipeline": 2},
            )

    @pytest.mark.parametrize(
        ("domains", "message"),
        [
            ({"replicated_data": ("replicate", "data")}, "unknown axes"),
            ({"replicated_data": ()}, "at least one axis"),
            ({"replicated_data": ("shard", "shard")}, "duplicate axes"),
            ({"shard": ("tensor",)}, "cannot redefine an axis name"),
        ],
    )
    def test_rejects_invalid_domain_axes(self, domains, message) -> None:
        with pytest.raises(ValueError, match=message):
            ParallelTopology(
                world_size=4,
                rank=0,
                axes={"shard": 2, "tensor": 2},
                domains=domains,
            )
