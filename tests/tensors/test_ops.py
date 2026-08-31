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

r"""Public behaviors shared by NestedTensor operations."""

import pytest
import torch
from torch.nn import functional as F

from danling.tensors import NestedTensor

NT = NestedTensor


class TestDropoutValidation:

    def test_dropout_probability_error_types(self):
        nt = NT(
            [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0]),
            ]
        )

        with pytest.raises(RuntimeError, match="between 0 and 1"):
            torch.dropout(nt, p=-0.1, train=False)

        with pytest.raises(ValueError, match="between 0 and 1"):
            F.dropout(nt, p=-0.1, training=False)


class TestDenseBinaryOperands:
    r"""Dense operands that pair with a NestedTensor without materializing padding."""

    def test_dense_operand_matching_concat_shape_is_elementwise(self):
        nt = NT([torch.randn(2, 3), torch.randn(4, 3)])
        dense = torch.randn_like(nt.concat)

        output = nt * dense

        torch.testing.assert_close(output.concat, nt.concat * dense)
        assert [tuple(element.shape) for element in output] == [(2, 3), (4, 3)]

    def test_per_element_dense_broadcast(self):
        # (B, 1, ragged_N, C) against a (B, S, 1, C) term: each element pairs with its own slice.
        nt = NT([torch.randn(1, 2, 3), torch.randn(1, 4, 3)])
        dense = torch.randn(2, 5, 1, 3)

        output = nt + dense
        assert isinstance(output, NestedTensor)
        assert len(output) == 2
        for index, element in enumerate(output):
            torch.testing.assert_close(element, nt[index] + dense[index])

    def test_gather_keeps_grad_after_no_grad_iteration(self):
        values = torch.randn(5, requires_grad=True)
        nt = NT([values[:2], values[2:]])
        with torch.no_grad():
            tuple(nt)

        index = NT([torch.tensor([1, 0]), torch.tensor([2, 1, 0])])
        output = torch.gather(nt, 1, index)
        output.concat.sum().backward()

        torch.testing.assert_close(output.concat, torch.cat((values[:2].flip(0), values[2:].flip(0))))
        torch.testing.assert_close(values.grad, torch.ones_like(values))
