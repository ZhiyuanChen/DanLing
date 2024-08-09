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

from danling.runners.topology import ParallelTopology


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

    @pytest.mark.parametrize(
        "world_size,axes",
        [
            (8, {"replicate": 0, "pipeline": 2, "tensor": 2}),
            (8, {"replicate": 1, "pipeline": 0, "tensor": 2}),
            (10, {"replicate": 2, "pipeline": 2, "tensor": 2}),
        ],
    )
    def test_rejects_invalid_axis_layout(self, world_size: int, axes: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="invalid parallel topology"):
            ParallelTopology(world_size=world_size, rank=0, axes=axes)

    def test_rejects_unknown_domain_axes(self) -> None:
        with pytest.raises(ValueError, match="unknown axes"):
            ParallelTopology(
                world_size=4,
                rank=0,
                axes={"shard": 2, "tensor": 2},
                domains={"replicated_data": ("replicate", "data")},
            )
