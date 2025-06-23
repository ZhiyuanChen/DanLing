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

import torch

from danling.tensors import PNTensor
from danling.tensors import tensor as pn_tensor
from tests.tensors.utils import assert_close


def test_tensor_property_returns_self():
    pn = pn_tensor([1, 2, 3])
    assert isinstance(pn, PNTensor)
    assert pn.tensor is pn


def test_mask_property_is_all_true():
    pn = pn_tensor([1, 2, 3])
    assert_close(pn.mask, torch.ones_like(pn, dtype=torch.bool))


def test_concat_property_returns_self():
    pn = pn_tensor([1, 2, 3])
    assert_close(pn.concat, pn)


def test_new_empty_returns_pn_tensor():
    pn = pn_tensor([1, 2, 3])
    output = pn.new_empty((2, 2))
    assert isinstance(output, PNTensor)
    assert output.shape == (2, 2)
