from __future__ import annotations

from pathlib import Path

from danling.runners.base_runner import BaseRunner
from danling.runners.utils import get_git_hash


class DummyRunner(BaseRunner):
    pass


class SequencingRunner(BaseRunner):
    def __init__(self, config):
        self.calls: list[str] = []
        super().__init__(config)

    def load_state_dict(self, checkpoint):
        self.calls.append("state")
        super().load_state_dict(checkpoint)

    def load_model(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs
        self.calls.append("model")

    def load_optimizer(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs
        self.calls.append("optimizer")

    def load_scheduler(self, state_dict, *args, **kwargs):
        del state_dict, args, kwargs
        self.calls.append("scheduler")


class StreamingLoader:
    def __iter__(self):
        return iter(())


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


def _config_hash(runner: DummyRunner) -> str:
    return format(hash(runner.config) & ((1 << 48) - 1), "012x")


def test_base_runner_dir_and_log_file_layout(tmp_path: Path) -> None:
    runner = DummyRunner(_config(tmp_path))
    try:
        expected_dir = _expected_base_dir(tmp_path, "lineage-a") / f"{runner.experiment}-{_config_hash(runner)}"
        assert runner.dir == str(expected_dir)
        assert runner.log_file == str(expected_dir / "logs" / f"{runner.id}.log")
        assert runner.name == "lineage-a-experiment-a"
    finally:
        runner.close()


def test_base_runner_writes_metadata_files(tmp_path: Path) -> None:
    runner = DummyRunner(_config(tmp_path, epochs=1))
    try:
        metadata_dir = (
            _expected_base_dir(tmp_path, "lineage-a") / f"{runner.experiment}-{_config_hash(runner)}" / "metadata"
        )
        assert (metadata_dir / "config.full.yaml").exists()
        assert (metadata_dir / "config.canonical.yaml").exists()
        assert (metadata_dir / "git.yaml").exists()
        assert (metadata_dir / "git.diff").exists()
    finally:
        runner.close()


def test_base_runner_log_interval_defaults_to_1024_for_unsized_loader() -> None:
    runner = DummyRunner({"log": False})
    try:
        runner.dataloaders["train"] = StreamingLoader()
        assert runner.log_interval == 1024
    finally:
        runner.close()


def test_base_runner_load_checkpoint_restores_in_expected_order() -> None:
    runner = SequencingRunner({"log": False})
    try:
        runner.load_checkpoint(
            {
                "runner": {"log": False},
                "state": {"train": {"global_step": 1, "epoch": 0}},
                "model": {"w": 1},
                "optimizer": {"opt": 1},
                "scheduler": {"sched": 1},
            }
        )
        assert runner.calls == ["state", "model", "optimizer", "scheduler"]
    finally:
        runner.close()
