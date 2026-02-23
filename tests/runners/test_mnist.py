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

import importlib.util
from pathlib import Path

import pytest

TORCH_MNIST_PATH = Path(__file__).resolve().parents[2] / "examples" / "vision" / "torch_mnist.py"


def _load_torch_mnist_module():
    spec = importlib.util.spec_from_file_location("danling_examples_vision_torch_mnist", TORCH_MNIST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {TORCH_MNIST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def torch_mnist_module():
    pytest.importorskip("torchvision")
    return _load_torch_mnist_module()


@pytest.fixture(scope="module")
def mnist_data_root(tmp_path_factory):
    return tmp_path_factory.mktemp("mnist_data")


@pytest.fixture
def runner(torch_mnist_module, mnist_data_root):
    config = torch_mnist_module.MNISTConfig().boot()
    config.dataset.root = str(mnist_data_root)
    return torch_mnist_module.MNISTRunner(config)


def test_train(runner):
    runner.train()


def test_evaluate(runner):
    runner.evaluate(["val"])
