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

RUN_NAME = "Run"
EXPERIMENT_NAME = "DanLing"
EXPERIMENT_ID = "xxxxxxxxxxxxxxxx"
SEED = 1016
IGNORED_NAMES_IN_METRICS = ("index", "epochs", "steps")
IGNORED_NAMES_IN_HASH = {
    "timestamp",
    "epoch",
    "step",
    "results",
    "score_split",
    "score",
    "log_interval",
    "save_interval",
    "tensorboard",
    "checkpoint",
    "auto_resume",
    "experiment_id",
}
