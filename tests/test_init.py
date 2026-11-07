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


class TestImports:
    def test_exports(self, run_isolated):
        run_isolated("""
            import danling
            from danling import BaseRunner, NestedTensor
            import torch

            nested = NestedTensor([torch.ones(2), torch.ones(3)])
            assert nested.shape == (2, 3)
            assert BaseRunner is danling.runners.BaseRunner
        """)
