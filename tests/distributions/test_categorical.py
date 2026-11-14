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

import math

import pytest
import torch

from danling.distributions import Categorical
from danling.tensors import NestedTensor


class TestCategorical:
    @pytest.mark.parametrize("parameter", ["logits", "probs"])
    @pytest.mark.parametrize("sample_shape", [(), (2, 3)])
    def test_dense(self, device, parameter, sample_shape):
        values = torch.randn(1, 3, device=device)
        if parameter == "probs":
            values = values.exp()
        actual = Categorical(**{parameter: values}, validate_args=True).expand((2, 4))
        assert isinstance(actual, Categorical)
        reference = torch.distributions.Categorical(**{parameter: values}, validate_args=True).expand((2, 4))
        torch.manual_seed(47)
        expected = reference.sample(sample_shape)
        expected_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        torch.manual_seed(47)
        torch.testing.assert_close(actual.sample(sample_shape), expected)
        actual_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    @pytest.mark.parametrize("parameter", ["logits", "probs"])
    @pytest.mark.parametrize(
        "shapes,ragged_dims,sample_shape,batch_first",
        [
            (((2, 4, 3), (4, 4, 3)), (0,), (), True),
            (((4, 2, 3), (4, 4, 3)), (1,), (), True),
            (((2, 4, 3), (4, 4, 3)), (0,), (2,), True),
            (((4, 2, 3), (4, 4, 3)), (1,), (3,), True),
            (((0, 3), (3, 3)), (0,), (2,), True),
            (((1, 2, 3), (2, 4, 3)), (0, 1), (2,), True),
            (((3,), (3,)), (0,), (), True),
            (((3,), (3,)), (0,), (2,), True),
            (((0, 3), (0, 3)), (0,), (2,), True),
            (((2, 3), (4, 3)), (0,), (), False),
        ],
    )
    def test_sampling(self, device, parameter, shapes, ragged_dims, sample_shape, batch_first):
        parts = [torch.randn(shape, device=device) for shape in shapes]
        if parameter == "probs":
            parts = [part.exp() for part in parts]
        nested = NestedTensor(parts, ragged_dims=ragged_dims, batch_first=batch_first)
        distribution = Categorical(**{parameter: nested}, validate_args=True)
        packed = torch.cat([part.reshape(-1, shape[-1]) for part, shape in zip(parts, shapes)])
        reference = (
            torch.distributions.Categorical(**{parameter: packed}, validate_args=True) if packed.shape[0] else None
        )

        torch.manual_seed(47)
        expected = (
            reference.sample(sample_shape)
            if reference is not None
            else torch.multinomial(packed, math.prod(sample_shape), True).T.reshape((*sample_shape, 0))
        )
        expected_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        torch.manual_seed(47)
        actual = distribution.sample(sample_shape)
        actual_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        rows = [math.prod(shape[:-1]) for shape in shapes]
        references = tuple(
            part.reshape((*sample_shape, *shape[:-1]))
            for part, shape in zip(expected.split(rows, dim=len(sample_shape)), shapes)
        )
        batch_axis = len(sample_shape) + (0 if batch_first else 1)
        torch.testing.assert_close(actual.unbind(batch_axis), references)
        torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    @pytest.mark.parametrize("parameter", ["logits", "probs"])
    @pytest.mark.parametrize("sample_shape", [(), (2,)])
    def test_empty(self, device, parameter, sample_shape):
        nested = NestedTensor([torch.ones(2, 3, device=device)], ragged_dims=(0,))[:0]
        distribution = Categorical(**{parameter: nested}, validate_args=True)
        torch.manual_seed(47)
        expected = torch.multinomial(torch.empty(0, 3, device=device), math.prod(sample_shape), True).T
        expected_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        torch.manual_seed(47)
        actual = distribution.sample(sample_shape)
        actual_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        assert actual.shape == (*sample_shape, 0, 2)
        assert actual.dtype == expected.dtype
        assert actual.unbind(len(sample_shape)) == ()
        torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
        log_prob = distribution.log_prob(actual)
        assert log_prob.shape == actual.shape
        assert log_prob.unbind(len(sample_shape)) == ()

    @pytest.mark.parametrize("padding", [float("-inf"), float("nan")])
    def test_padding(self, device, padding):
        distribution = Categorical(
            logits=NestedTensor(
                [torch.randn(length, 3, device=device) for length in (2, 4)],
                ragged_dims=(0,),
                padding_value=padding,
            ),
            validate_args=True,
        )
        actual = distribution.sample((2,))
        assert actual.tensor.dtype == torch.long
        torch.testing.assert_close(actual.tensor[~actual.mask], torch.zeros(4, dtype=torch.long, device=device))

    @pytest.mark.parametrize("batch_first,sample_shape", [(True, (2, 3)), (False, (2,))])
    def test_sample_axes(self, device, batch_first, sample_shape):
        distribution = Categorical(
            logits=NestedTensor(
                [torch.randn(length, 3, device=device) for length in (2, 4)],
                ragged_dims=(0,),
                batch_first=batch_first,
            ),
            validate_args=True,
        )
        before = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        with pytest.raises(NotImplementedError, match="batch axis"):
            distribution.sample(sample_shape)
        after = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "shapes,dims,sample_shape,batch_first",
        [
            (((3,), (3,)), (0,), (), True),
            (((3,), (3,)), (0,), (3,), True),
            (((4, 2, 3), (4, 4, 3)), (1,), (), True),
            (((4, 2, 3), (4, 4, 3)), (1,), (3,), True),
            (((4, 2, 3), (4, 4, 3)), (1,), (), False),
        ],
    )
    def test_log_prob(self, device, shapes, dims, sample_shape, batch_first):
        parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        distribution = Categorical(
            logits=NestedTensor(parts, ragged_dims=dims, batch_first=batch_first), validate_args=True
        )
        reference = torch.distributions.Categorical(
            logits=torch.cat([part.reshape(-1, 3) for part in references]), validate_args=True
        )
        torch.manual_seed(47)
        expected = reference.log_prob(reference.sample(sample_shape))
        torch.manual_seed(47)
        actual = distribution.log_prob(distribution.sample(sample_shape))
        rows = [math.prod(shape[:-1]) for shape in shapes]
        expected_parts = tuple(
            part.reshape((*sample_shape, *source.shape[:-1]))
            for part, source in zip(expected.split(rows, dim=len(sample_shape)), parts)
        )
        assert actual.shape == (*sample_shape, *distribution.batch_shape)
        assert actual.batch_first is (batch_first and not sample_shape)
        torch.testing.assert_close(actual.unbind(len(sample_shape) + (not batch_first)), expected_parts)
        torch.testing.assert_close(
            torch.autograd.grad(actual.values().sum(), parts), torch.autograd.grad(expected.sum(), references)
        )

    @pytest.mark.parametrize("parameter,invalid", [("logits", float("nan")), ("probs", float("nan")), ("probs", -1.0)])
    def test_validation(self, device, parameter, invalid):
        values = torch.ones(2, 3, device=device)
        values[0, 0] = invalid
        with pytest.raises(ValueError, match="invalid values"):
            Categorical(**{parameter: NestedTensor([values], ragged_dims=(0,))}, validate_args=True)

    @pytest.mark.parametrize(
        "lengths,width,sample_shape",
        [
            ((2, 3), 4, ()),
            ((2, 4), 4, ()),
            ((17, 34), 4, ()),
            ((2, 3), 4, (2,)),
            ((2, 3), 4, (3,)),
            ((1, 2), None, (2,)),
        ],
    )
    def test_labels(self, device, lengths, width, sample_shape):
        labels = [torch.arange(length, device=device).remainder(3) for length in lengths]
        if width is not None:
            labels = [label[:, None].expand(-1, width) for label in labels]
        parts = [
            torch.full((*label.shape, 3), -1000.0, device=device).scatter(-1, label[..., None], 0) for label in labels
        ]
        output = Categorical(logits=NestedTensor(parts, ragged_dims=(0,))).sample(sample_shape)
        references = [torch.distributions.Categorical(logits=part).sample(sample_shape) for part in parts]
        torch.testing.assert_close(output.unbind(len(sample_shape)), tuple(references))
        assert output.batch_first == (not sample_shape)
