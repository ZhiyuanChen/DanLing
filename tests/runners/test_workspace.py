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

import builtins
import logging
import os
from pathlib import Path

import pytest
from git import Actor, Repo

from danling.runners.base_runner import BaseRunner
from danling.runners.workspace import ColorFormatter


class MetadataRunner(BaseRunner):
    def __post_init__(self) -> None:
        self.config.post_init_only = "finalized"
        super().__post_init__()


class FailingRunner(BaseRunner):
    def __post_init__(self) -> None:
        raise RuntimeError("post init exploded")


def test_directory_identity(tmp_path: Path, runner_factory):
    runner = runner_factory()
    directory = Path(runner.workspace.dir)
    assert directory.is_relative_to(tmp_path)
    assert directory.name == runner.id
    assert directory.parent.name.startswith("lineage-a")
    assert runner.name == "lineage-a-experiment-a"
    assert runner_factory().id == runner.id
    assert runner_factory({"seed": 42}).id != runner.id


def test_log_directory(runner_factory):
    runner = runner_factory()
    assert Path(runner.workspace.log_file).parent == Path(runner.workspace.dir) / "logs"
    assert Path(runner.workspace.log_file).suffix == ".log"


def test_uses_logging_file_override(tmp_path: Path, runner_factory) -> None:
    log_file = tmp_path / "custom.log"
    runner = runner_factory({"logging.file": str(log_file)})
    assert runner.workspace.log_file == str(log_file)


def test_checkpoint_directory(runner_factory):
    runner = runner_factory({"ckpt.dir": "states"})
    assert Path(runner.workspace.checkpoint_dir) == Path(runner.workspace.dir) / "states"


@pytest.fixture
def repository(tmp_path):
    repository = Repo.init(tmp_path / "repository")
    tracked = Path(repository.working_tree_dir) / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    repository.index.add([str(tracked)])
    actor = Actor("Workspace Test", "workspace@example.invalid")
    repository.index.commit("Initial content", author=actor, committer=actor)
    tracked.write_text("changed\n", encoding="utf-8")
    previous = Path.cwd()
    os.chdir(repository.working_tree_dir)
    try:
        yield repository
    finally:
        os.chdir(previous)


def test_git_metadata(repository, runner_factory):
    runner = runner_factory({"epochs": 1})
    metadata_dir = Path(runner.workspace.dir) / "metadata"
    for name in ("config.full.yaml", "config.canonical.yaml", "git.yaml", "git.diff"):
        assert (metadata_dir / name).is_file()
    git_diff = (metadata_dir / "git.diff").read_text(encoding="utf-8")
    assert "-original" in git_diff
    assert "+changed" in git_diff


def test_finalized_metadata(runner_factory):
    runner = runner_factory(runner_type=MetadataRunner)
    full_config = (Path(runner.workspace.dir) / "metadata" / "config.full.yaml").read_text(encoding="utf-8")
    assert "post_init_only: finalized" in full_config


def test_color_formatter_emits_ansi_sequences() -> None:
    formatter = ColorFormatter("%(message)s")
    logger = logging.getLogger("danling.test.workspace")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 0, "hello color", (), None)

    formatted = formatter.format(record)

    assert "\x1b[" in formatted
    assert "hello color" in formatted
    assert "[INFO]" in formatted


def test_logging_keeps_file_output_plain(tmp_path: Path, runner_factory) -> None:
    runner = runner_factory({"logging.enabled": True})
    log_file = Path(runner.workspace.log_file)
    assert runner.logger is not None
    runner.logger.info("hello log file")
    runner.close()

    file_log = log_file.read_text(encoding="utf-8")
    assert "hello log file" in file_log
    assert "\x1b[" not in file_log


@pytest.mark.parametrize("failure", [False, True])
def test_print_restoration(tmp_path, capsys, runner_factory, failure):
    if failure:
        with pytest.raises(RuntimeError, match="post init exploded"):
            runner_factory({"logging.enabled": True}, runner_type=FailingRunner)
    else:
        first = runner_factory({"logging.enabled": True, "workspace.root": str(tmp_path / "a")})
        second = runner_factory({"logging.enabled": True, "workspace.root": str(tmp_path / "b"), "seed": 42})
        second_log = Path(second.workspace.log_file)
        first.close()
        builtins.print("inner still active")
        second.close()
        assert "inner still active" in second_log.read_text(encoding="utf-8")
    builtins.print("restored")
    assert capsys.readouterr().out.splitlines().count("restored") == 1
