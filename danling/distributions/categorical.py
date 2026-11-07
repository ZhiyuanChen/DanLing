# DanLing
# Copyright (C) 2022-Present  DanLing
#
# This file is part of DanLing.
#
# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0
#
# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

r"""Categorical sampling for dense and packed ragged tensors."""

from __future__ import annotations

import torch
from torch.distributions import Categorical as TorchCategorical
from torch.distributions import constraints

from ..tensors import NestedTensor
from ..tensors.torch_functions import _packed_flatten_elements


class _RealVector(constraints.Constraint):
    r"""Validate each real event vector, including an empty batch of vectors."""

    event_dim = 1

    def check(self, value):
        if value.dim() < 1:
            raise ValueError(f"Expected value.dim() >= 1 but got {value.dim()}")
        return constraints.real.check(value).all(-1)


class Categorical(TorchCategorical):
    r"""A PyTorch Categorical distribution that preserves NestedTensor batch layouts.

    The constructor and distribution methods retain the PyTorch interface.
    Ragged samples use native ``torch.multinomial`` on real probability rows.
    Sampled indices have zero display padding and retain the valid-entry mask.
    The current NestedTensor codec supports no sample prefix, or one sample
    axis before a batch-leading input. Ragged sampling currently runs eagerly;
    dense parameters use native sampling.
    """

    arg_constraints = {"probs": constraints.simplex, "logits": _RealVector()}

    def sample(self, sample_shape=()) -> torch.Tensor:
        probs = self.probs
        if not isinstance(probs, NestedTensor):
            return super().sample(sample_shape)
        sample_shape = torch.Size(sample_shape)
        batch_dim = 0 if probs.batch_first or probs.dim() == 1 else 1
        if batch_dim == probs.dim() - 1:
            raise NotImplementedError("Categorical sampling requires categories outside the NestedTensor batch axis")
        if len(sample_shape) > 1 or sample_shape and not probs.batch_first:
            raise NotImplementedError(
                "Categorical sample axes would place the NestedTensor batch axis beyond position one"
            )

        categories = self._num_events
        sizes = probs.element_sizes()
        if not bool(torch.all(sizes[:, -1] == categories)):
            raise NotImplementedError("Categorical sampling requires a common category count across samples")

        # Flatten real probability rows in logical per-sample order. The original
        # probabilities remain the explicit layout reference throughout sampling.
        if probs.packed_dim_order == tuple(range(sizes.shape[1])):
            flat = probs.concat.reshape(-1)
        else:
            flat, _ = _packed_flatten_elements(probs)
        draws = torch.multinomial(flat.reshape(-1, categories), sample_shape.numel(), True)
        row_shapes = sizes[:, :-1]
        lengths = row_shapes.prod(1)
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        row_rank = row_shapes.shape[1]
        physical_maxima = probs._max_physical_dims()[:-1]

        if sample_shape:
            shape_tensor = torch.cat((row_shapes.new_full((len(probs), 1), sample_shape[0]), row_shapes), dim=1)
            ragged_dims = tuple(range(1, row_rank + 1))
            permutation = (*ragged_dims, 0)
            outer_size = torch.Size((*sample_shape, len(probs), *physical_maxima))
            values = draws
            batch_first = False
        else:
            shape_tensor = row_shapes
            ragged_dims = tuple(range(row_rank))
            permutation = ragged_dims
            outer_size = probs.shape[:-1]
            values = draws.squeeze(-1)
            batch_first = probs.batch_first

        return type(probs)._from_packed_raw(
            values,
            offsets,
            shape_tensor,
            permutation=permutation,
            ragged_dims=ragged_dims,
            outer_size=outer_size,
            batch_first=batch_first,
            padding_value=0,
            mask_value=probs.mask_value,
            pin_memory=probs._pin_memory,
            validate=False,
        )
