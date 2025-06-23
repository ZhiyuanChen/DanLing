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

from danling.tensors.utils import mask_tensor, pad_tensor, tensor_mask
from tests.tensors.utils import assert_close


def test_tensor_mask_default():
    tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    tensor, mask = tensor_mask(tensors, torch.Size([2, 3]))
    assert_close(tensor, torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert_close(mask, torch.tensor([[True, True, True], [True, True, False]]))


def test_tensor_mask_with_mask_value_true():
    tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    tensor, mask = tensor_mask(tensors, torch.Size([2, 3]), padding_value=-1, mask_value=True)
    assert_close(tensor, torch.tensor([[1, 2, 3], [4, 5, -1]]))
    assert_close(mask, torch.tensor([[False, False, False], [False, False, True]]))


def test_pad_tensor_matches_tensor_mask():
    tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0]])]
    size = torch.Size([2, 2, 2])
    for batch_first in [True, False]:
        padded = pad_tensor(tensors, size, batch_first=batch_first, padding_value=-1.0)
        tensor, _ = tensor_mask(tensors, size, batch_first=batch_first, padding_value=-1.0)
        assert_close(padded, tensor)


def test_mask_tensor_matches_tensor_mask():
    tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0]])]
    size = torch.Size([2, 2, 2])
    for batch_first in [True, False]:
        mask = mask_tensor(tensors, size, batch_first=batch_first, mask_value=False)
        _, reference = tensor_mask(tensors, size, batch_first=batch_first, mask_value=False)
        assert_close(mask, reference)


def test_mask_tensor_squeeze_channel_flag():
    tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0]])]
    size = torch.Size([2, 2, 2])
    squeezed = mask_tensor(tensors, size, squeeze_channel=True)
    unsqueezed = mask_tensor(tensors, size, squeeze_channel=False)
    assert squeezed.shape == torch.Size([2, 2])
    assert unsqueezed.shape == torch.Size([2, 2, 2])
    assert_close(unsqueezed[..., 0], squeezed)
    assert_close(unsqueezed[..., 1], squeezed)
