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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Any

from torch import distributed as dist


@dataclass
class ParallelTopology:
    """Rank coordinates for an ordered parallel-axis layout."""

    world_size: int
    rank: int
    axes: Mapping[str, int]
    domains: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    ranks: Mapping[str, int] = field(default_factory=dict)

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        axes: Mapping[str, int],
        domains: Mapping[str, Sequence[str]] | None = None,
        label: str = "parallel topology",
    ) -> None:
        world_size = int(world_size)
        rank = int(rank)
        if world_size < 1:
            raise ValueError(f"invalid {label}: WORLD_SIZE must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid {label}: rank must be in [0, {world_size}), got {rank}")

        axis_degrees = {str(axis): int(degree) for axis, degree in axes.items()}
        if not axis_degrees:
            raise ValueError(f"invalid {label}: axes must not be empty")
        for axis, degree in axis_degrees.items():
            if degree < 1:
                raise ValueError(f"invalid {label}: axis degrees must be positive integers, got {axis}={degree}")

        topology_size = prod(axis_degrees.values())
        if topology_size != world_size:
            axis_product = " * ".join(f"{axis}({degree})" for axis, degree in axis_degrees.items())
            raise ValueError(f"invalid {label}: WORLD_SIZE({world_size}) must equal {axis_product} = {topology_size}")

        axis_ranks: dict[str, int] = {}
        stride = topology_size
        for axis, degree in axis_degrees.items():
            stride //= degree
            axis_ranks[axis] = (rank // stride) % degree

        domain_axes: dict[str, tuple[str, ...]] = {axis: (axis,) for axis in axis_degrees}
        if domains is not None:
            for domain, axes_ in domains.items():
                normalized_axes = tuple(str(axis) for axis in axes_)
                if not normalized_axes:
                    raise ValueError(f"invalid {label}: domain {domain!r} must contain at least one axis")
                missing = [axis for axis in normalized_axes if axis not in axis_degrees]
                if missing:
                    raise ValueError(f"invalid {label}: domain {domain!r} references unknown axes {missing}")
                domain_axes[str(domain)] = normalized_axes

        self.world_size = world_size
        self.rank = rank
        self.axes = axis_degrees
        self.domains = domain_axes
        self.ranks = axis_ranks

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(self.axes)

    @property
    def mesh_shape(self) -> tuple[int, ...]:
        return tuple(self.axes.values())

    def axis_degree(self, axis: str, default: int | None = None) -> int:
        if axis in self.axes:
            return self.axes[axis]
        if default is not None:
            return default
        raise KeyError(axis)

    def axis_rank(self, axis: str, default: int | None = None) -> int:
        if axis in self.ranks:
            return self.ranks[axis]
        if default is not None:
            return default
        raise KeyError(axis)

    def domain_axes(self, domain: str) -> tuple[str, ...]:
        return self.domains[domain]

    def domain_degree(self, domain: str) -> int:
        return prod(self.axes[axis] for axis in self.domain_axes(domain))

    def domain_rank(self, domain: str) -> int:
        rank = 0
        for axis in self.domain_axes(domain):
            rank = rank * self.axes[axis] + self.ranks[axis]
        return rank

    def rank_from_coordinates(self, coordinates: Mapping[str, int]) -> int:
        unknown = [axis for axis in coordinates if axis not in self.axes]
        if unknown:
            raise ValueError(f"invalid parallel topology coordinates: unknown axes {unknown}")

        rank = 0
        stride = self.world_size
        for axis, degree in self.axes.items():
            stride //= degree
            coordinate = int(coordinates.get(axis, 0))
            if coordinate < 0 or coordinate >= degree:
                raise ValueError(f"invalid parallel topology coordinates: {axis}={coordinate} is outside [0, {degree})")
            rank += coordinate * stride
        return rank


@dataclass
class ParallelContext:
    """Runtime handles for a `ParallelTopology`."""

    topology: ParallelTopology
    device_mesh: Any | None = None
    groups: dict[str, Any | None] = field(default_factory=dict)

    def __init__(
        self,
        topology: ParallelTopology,
        *,
        device_mesh: Any | None = None,
        groups: Mapping[str, Any | None] | None = None,
    ) -> None:
        self.topology = topology
        self.device_mesh = device_mesh
        self.groups = dict(groups or {})

    def group(self, axis: str) -> Any | None:
        return self.groups.get(axis)

    def all_reduce(self, tensor, *, domain: str = "data", op=dist.ReduceOp.SUM):
        for axis in self.topology.domain_axes(domain):
            if self.topology.axis_degree(axis) > 1:
                group = self.group(axis)
                if group is None:
                    raise RuntimeError(
                        f"cannot all-reduce domain {domain!r}: process group for axis {axis!r} is not initialized"
                    )
                dist.all_reduce(tensor, op=op, group=group)
        return tensor
