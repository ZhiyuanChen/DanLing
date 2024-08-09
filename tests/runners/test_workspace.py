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
from pathlib import Path

import pytest

from danling.runners.base_runner import BaseRunner
from danling.runners.utils import get_git_hash
from danling.runners.workspace import ColorFormatter


class MinimalRunner(BaseRunner):
    pass


class PostInitMetadataRunner(BaseRunner):
    def __post_init__(self) -> None:
        self.config.post_init_only = "finalized"
        super().__post_init__()


class FailingPostInitRunner(BaseRunner):
    def __post_init__(self) -> None:
        raise RuntimeError("post init exploded")


def _config(tmp_path: Path, **kwargs):
    config = {
        "log": False,
        "workspace_root": str(tmp_path),
        "lineage": "lineage-a",
        "experiment": "experiment-a",
    }
    config.update(kwargs)
    return config


def _expected_base_dir(tmp_path: Path, lineage: str) -> Path:
    git_hash = get_git_hash()
    if git_hash is None:
        return tmp_path / lineage
    return tmp_path / f"{lineage}-{git_hash}"


def _config_hash(runner: MinimalRunner) -> str:
    return format(hash(runner.config) & ((1 << 48) - 1), "012x")


def _expected_id(runner: MinimalRunner) -> str:
    git_hash = get_git_hash()
    config_hash = _config_hash(runner)
    if git_hash is None:
        return config_hash
    return f"{git_hash}-{config_hash}"


def test_runner_workspace_dir_layout(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path))
    try:
        expected_id = _expected_id(runner)
        expected_dir = _expected_base_dir(tmp_path, "lineage-a") / expected_id
        assert runner.id == expected_id
        assert runner.workspace.dir == str(expected_dir)
        assert runner.name == "lineage-a-experiment-a"
    finally:
        runner.close()


def test_runner_workspace_log_file_layout(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path))
    try:
        expected_dir = _expected_base_dir(tmp_path, "lineage-a") / _expected_id(runner)
        assert runner.workspace.log_file == str(expected_dir / "logs" / f"{runner.timestamp}.log")
    finally:
        runner.close()


def test_runner_workspace_writes_metadata_files(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, epochs=1))
    try:
        metadata_dir = _expected_base_dir(tmp_path, "lineage-a") / _expected_id(runner) / "metadata"
        assert (metadata_dir / "config.full.yaml").exists()
        assert (metadata_dir / "config.canonical.yaml").exists()
        assert (metadata_dir / "git.yaml").exists()
        assert (metadata_dir / "git.diff").exists()
    finally:
        runner.close()


def test_runner_workspace_saves_metadata_after_post_init(tmp_path: Path) -> None:
    runner = PostInitMetadataRunner(_config(tmp_path))
    try:
        metadata_dir = _expected_base_dir(tmp_path, "lineage-a") / _expected_id(runner) / "metadata"
        full_config = (metadata_dir / "config.full.yaml").read_text(encoding="utf-8")
        assert "post_init_only: finalized" in full_config
    finally:
        runner.close()


def test_color_formatter_emits_ansi_sequences() -> None:
    formatter = ColorFormatter("%(message)s")
    logger = logging.getLogger("danling.test.workspace")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 0, "hello color", (), None)

    formatted = formatter.format(record)

    assert "\x1b[" in formatted
    assert "hello color" in formatted
    assert "[INFO]" in formatted


def test_runner_workspace_logging_keeps_file_output_plain(tmp_path: Path) -> None:
    runner = MinimalRunner(_config(tmp_path, log=True))
    log_file = Path(runner.workspace.log_file)
    try:
        assert runner.logger is not None
        runner.logger.info("hello log file")
    finally:
        runner.close()

    file_log = log_file.read_text(encoding="utf-8")
    assert "hello log file" in file_log
    assert "\x1b[" not in file_log


def test_runner_workspace_nested_print_patch_restores_safely(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_print(*args, sep=" ", end="\n", file=None, flush=False, **kwargs):
        del file, flush, kwargs
        calls.append(sep.join(str(arg) for arg in args) + end.rstrip("\n"))

    monkeypatch.setattr(builtins, "print", fake_print)

    runner_a = MinimalRunner(_config(tmp_path / "a"))
    runner_b = MinimalRunner(_config(tmp_path / "b"))
    try:
        runner_a.close()
        builtins.print("inner still active")
        runner_b.close()
        builtins.print("restored")
    finally:
        if getattr(runner_a, "workspace", None) is not None:
            runner_a.close()
        if getattr(runner_b, "workspace", None) is not None:
            runner_b.close()

    assert calls == ["inner still active", "restored"]
    assert builtins.print is fake_print


def test_runner_workspace_restores_print_after_post_init_failure(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_print(*args, sep=" ", end="\n", file=None, flush=False, **kwargs):
        del file, flush, kwargs
        calls.append(sep.join(str(arg) for arg in args) + end.rstrip("\n"))

    monkeypatch.setattr(builtins, "print", fake_print)

    with pytest.raises(RuntimeError, match="post init exploded"):
        FailingPostInitRunner(_config(tmp_path))

    builtins.print("restored after failure")

    assert calls == ["restored after failure"]
    assert builtins.print is fake_print
