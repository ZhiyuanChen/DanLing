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

r"""Tests for ``danling.tensors.pn_tensor`` — PNTensor properties and collation."""

import torch

from danling.tensors import (
    NestedTensor,
    PNTensor,
    collate_pn_tensor_fn,
    register_pn_tensor_collate,
)
from danling.tensors import tensor as pn_tensor
from danling.tensors import (
    unregister_pn_tensor_collate,
)
from tests.tensors.utils import assert_close


class TestCollateRegistration:

    def test_collate_registration_helpers_on_custom_map(self):
        custom_map = {}
        register_pn_tensor_collate(custom_map)
        assert custom_map[PNTensor] is collate_pn_tensor_fn

        batch = [pn_tensor([1, 2, 3]), pn_tensor([4, 5])]
        collated = custom_map[PNTensor](batch, collate_fn_map=custom_map)
        assert isinstance(collated, NestedTensor)
        assert_close(collated[0], torch.tensor([1, 2, 3]))
        assert_close(collated[1], torch.tensor([4, 5]))

        unregister_pn_tensor_collate(custom_map)
        assert PNTensor not in custom_map


class TestPNTensorProperties:

    def test_concat_property_returns_self(self):
        pn = pn_tensor([1, 2, 3])
        assert_close(pn.concat, pn)

    def test_mask_property_is_all_true(self):
        pn = pn_tensor([1, 2, 3])
        assert_close(pn.mask, torch.ones_like(pn, dtype=torch.bool))

    def test_new_empty_returns_pn_tensor(self):
        pn = pn_tensor([1, 2, 3])
        output = pn.new_empty((2, 2))
        assert isinstance(output, PNTensor)
        assert output.shape == (2, 2)

    def test_tensor_property_returns_self(self):
        pn = pn_tensor([1, 2, 3])
        assert isinstance(pn, PNTensor)
        assert pn.tensor is pn
