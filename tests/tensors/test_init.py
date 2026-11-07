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
    def test_optional_dependencies(self, run_isolated):
        run_isolated("""
            import builtins

            original_import = builtins.__import__

            def import_without_optional_dependencies(name, *args, **kwargs):
                if name.split(".", 1)[0] in {"torchmetrics", "transformers"}:
                    raise ModuleNotFoundError(name)
                return original_import(name, *args, **kwargs)

            builtins.__import__ = import_without_optional_dependencies

            from danling.tensors import NestedTensor
            import torch

            nested = NestedTensor([torch.ones(2), torch.ones(3)])
            assert nested.shape == (2, 3)
        """)
