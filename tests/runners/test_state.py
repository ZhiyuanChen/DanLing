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

from danling.runners.state import RunnerTrainState


def test_runner_train_state_persists_only_maintained_progress_fields() -> None:
    state = RunnerTrainState(global_step=3, micro_step=7, epoch=2)

    assert state.state_dict() == {"global_step": 3, "micro_step": 7, "epoch": 2}


def test_runner_train_state_ignores_stale_counter_keys_on_load() -> None:
    state = RunnerTrainState()

    state.load_state_dict(
        {
            "global_step": 5,
            "micro_step": 9,
            "epoch": 4,
            "tokens_seen": 1024,
            "samples_seen": 32,
        }
    )

    assert state.global_step == 5
    assert state.micro_step == 9
    assert state.epoch == 4
    assert not hasattr(state, "tokens_seen")
    assert not hasattr(state, "samples_seen")
