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

from __future__ import annotations

import os

import pytest

from danling.runners.base_runner import BaseRunner


@pytest.fixture
def runner_config(tmp_path):
    return {
        "logging.enabled": False,
        "workspace.root": str(tmp_path),
        "workspace.lineage": "lineage-a",
        "workspace.experiment": "experiment-a",
    }


@pytest.fixture
def runner_factory(runner_config):
    runners = []

    def create(config=None, *, runner_type=BaseRunner):
        runner = runner_type({**runner_config, **(config or {})})
        runners.append(runner)
        return runner

    yield create
    for runner in reversed(runners):
        runner.close()


@pytest.fixture
def runner_environment():
    previous = os.environ.copy()

    def configure(**values):
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)

    try:
        yield configure
    finally:
        os.environ.clear()
        os.environ.update(previous)
