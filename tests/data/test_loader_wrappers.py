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
from torchdata.stateful_dataloader import StatefulDataLoader

from danling.data import StepProxyLoader


def test_resume_without_loading():
    reads = []

    class Dataset:
        def __len__(self):
            return 8

        def __getitem__(self, index):
            reads.append(index)
            return index

    loader = StatefulDataLoader(Dataset(), batch_size=2, shuffle=False)
    proxy = StepProxyLoader(loader)
    iterator = iter(proxy)
    assert next(iterator) is None
    assert next(iterator) is None
    state = proxy.state_dict()
    assert reads == []

    restored_loader = StatefulDataLoader(Dataset(), batch_size=2, shuffle=False)
    restored = StepProxyLoader(restored_loader)
    restored.load_state_dict(state)
    assert list(restored) == [None, None]
    assert list(restored) == [None] * 4
    assert reads == []
    torch.testing.assert_close(next(iter(restored_loader)), torch.tensor([0, 1]))
