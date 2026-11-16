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

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from danling.runners import ParallelRunner, TorchRunner
from danling.runners.checkpoints.torch_distributed import (
    DCP_PINNED_MEMORY_STAGING_AVAILABLE,
    TorchDistributedCheckpointManager,
)
from danling.runners.topology import ParallelContext
from tests.runners.checkpoints.utils import (
    _assert_checkpoint_failure,
    _checkpoint_targets,
    _CheckpointRunner,
    _dataloader_state_position,
    _dcp_async_dedicated_process_group_failure_worker,
    _make_stateful_dataloader,
    _pointer_target,
    _rank_local_resume_worker,
    _retaining_dcp_manager,
    _save_best_dcp_checkpoint,
    _stateful_dataloader_position,
    _target_path,
)
from tests.runners.distributed import require_gloo, run_distributed

pytestmark = pytest.mark.filterwarnings("ignore:torch.distributed is disabled.*:UserWarning")


def test_rank_rng_and_loader_resume(tmp_path: Path) -> None:
    require_gloo()
    run_distributed(_rank_local_resume_worker, worker_args=(str(tmp_path), False))
    manager = TorchDistributedCheckpointManager(_CheckpointRunner(tmp_path))
    try:
        with pytest.raises(ValueError, match="rank-local RNG/dataloader state.*world size 1"):
            manager.load_checkpoint("latest")
    finally:
        manager.close()


@pytest.mark.parametrize("backend", ["file", "dcp"])
def test_resume_results_and_best(tmp_path, backend):
    class HistoryRunner(TorchRunner):
        def __init__(self, config):
            super().__init__(config)
            self.model = torch.nn.Linear(2, 2)

    config = {
        "device": "cpu",
        "logging.enabled": False,
        "workspace.dir": str(tmp_path),
        "score": {"split": "val", "metric": "loss"},
        "ckpt": {"backend": backend, "async_mode": "disabled", "interval": None},
    }
    source, restored = HistoryRunner(config), HistoryRunner(config)
    try:
        source.train_state.global_step = 2
        source.append_result({"val": {"loss": 0.1}}, index=2)
        source.save_checkpoint(force=True)
        source.train_state.global_step = 9
        metric = torch.tensor([0.7, 0.8], dtype=torch.float64)
        source.append_result({"val": {"loss": 0.5, "new_metric": metric}}, index=9)
        source.save_checkpoint(force=True)
        latest = Path(source.workspace.checkpoint_dir) / "latest.pth" if backend == "file" else "latest"
        restored.load_checkpoint(latest)
        assert list(restored.results) == [2, 9]
        assert restored.results[2] == source.results[2]
        assert restored.best_index == 2
        assert restored.best_score == 0.1
        torch.testing.assert_close(restored.results[9]["val"]["new_metric"], metric)
        restored.append_result({"val": {"loss": 0.8}}, index=12)
        restored.train_state.global_step = 12
        restored.save_checkpoint(force=True)
        best_path = Path(restored.workspace.checkpoint_dir) / "best.pth" if backend == "file" else "best"
        best = restored.read_checkpoint(best_path)
        assert best["state"]["train"]["global_step"] == 2
        assert best["results"] == {2: {"val": {"loss": 0.1}}}
    finally:
        source.close()
        restored.close()


@pytest.mark.parametrize("with_results", [False, True])
def test_legacy_evaluation_progress(tmp_path, with_results):
    class LegacyRunner(TorchRunner):
        legacy_payload = False

        def __init__(self, config):
            super().__init__(config)
            self.model = torch.nn.Linear(2, 2)

        def state_dict(self, cls=dict):
            state = super().state_dict(cls)
            if self.legacy_payload:
                state["state"]["train"].pop("last_evaluation_step")
                if not with_results:
                    state.pop("results")
            return state

    config = {
        "device": "cpu",
        "logging.enabled": False,
        "workspace.dir": str(tmp_path),
        "ckpt": {"backend": "dcp", "async_mode": "disabled"},
    }
    source, restored = LegacyRunner(config), LegacyRunner(config)
    try:
        source.legacy_payload = True
        source.train_state.global_step = 4
        source.append_result({"val": {"loss": 0.1}}, index=4)
        source.save_checkpoint(force=True)
        restored.load_checkpoint("latest")
        assert restored.train_state.global_step == 4
        assert restored.train_state.last_evaluation_step is None
        assert restored.results == (source.results if with_results else {})
    finally:
        source.close()
        restored.close()


@pytest.mark.parametrize("optimizer_cls", [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD])
def test_export_preserves_optimizer_update(optimizer_cls):
    model = torch.nn.Linear(2, 2, dtype=torch.float64)
    reference = deepcopy(model)
    optimizer = optimizer_cls(model.parameters(), lr=0.1)
    control = optimizer_cls(reference.parameters(), lr=0.1)
    events = []
    optimizer.register_step_pre_hook(lambda *args: events.append("step"))
    optimizer.register_state_dict_pre_hook(lambda *args: events.append("export"))
    optimizer.register_state_dict_post_hook(lambda *args: events.append("exported"))

    TorchDistributedCheckpointManager.export_model_optimizer_state(model=model, optimizer=optimizer)
    assert events == ["export", "exported"]
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
    value = torch.ones(1, 2, dtype=torch.float64)
    model(value).square().sum().backward()
    reference(value).square().sum().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    TorchDistributedCheckpointManager.export_model_optimizer_state(model=model, optimizer=optimizer)
    assert all(parameter.grad is gradient for parameter, gradient in zip(model.parameters(), gradients))
    optimizer.step()
    control.step()
    assert events == ["export", "exported", "export", "exported", "step"]
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("native_fqns", [False, True], ids=["raw-ids", "native-fqns"])
@pytest.mark.parametrize(
    "optimizer_cls,updates",
    [
        (torch.optim.Adam, 0),
        (torch.optim.Adam, 2),
        (torch.optim.AdamW, 0),
        (torch.optim.AdamW, 2),
        (torch.optim.SGD, 0),
    ],
)
def test_optimizer_round_trip_update(tmp_path, native_fqns, optimizer_cls, updates):
    base = ParallelRunner if native_fqns else TorchRunner

    class CheckpointRunner(base):
        def init_distributed(self):
            if native_fqns:
                self.topology = self.build_topology()
                self.parallel = ParallelContext(self.topology)

        @property
        def world_size(self):
            return 2 if native_fqns else 1

        def __init__(self, config):
            super().__init__(config)
            self.model = torch.nn.Linear(2, 2, dtype=torch.float64)
            self.optimizer = optimizer_cls(self.model.parameters(), lr=0.1)

    config = {
        "device": "cpu",
        "logging.enabled": False,
        "workspace.dir": str(tmp_path),
        "ckpt": {"backend": "dcp", "async_mode": "disabled", "interval": 1},
        "parallel.axes.shard": 2 if native_fqns else 1,
    }
    source, control, restored, healed = (CheckpointRunner(config) for _ in range(4))
    value = torch.ones(1, 2, dtype=torch.float64)

    def update(runner, parameter=None):
        output = getattr(runner.model, parameter) if parameter is not None else runner.model(value)
        output.square().sum().backward()
        assert runner.optimizer_step()

    try:
        control.model.load_state_dict(source.model.state_dict())
        for _ in range(updates):
            update(source, parameter="weight")
            update(control, parameter="weight")
        source.save_checkpoint(force=True)
        snapshot = deepcopy(source.state_dict())
        if not updates:
            assert not source.optimizer.state
        if updates:
            for destination in (restored, healed):
                update(destination, parameter="bias")
                assert destination.model.bias in destination.optimizer.state
                assert destination.model.weight not in destination.optimizer.state
        else:
            assert not restored.optimizer.state
        restored.load_checkpoint("latest")
        healed.load_checkpoint(snapshot)
        for destination in (restored, healed):
            for original, loaded in zip(source.model.parameters(), destination.model.parameters()):
                torch.testing.assert_close(original, loaded)
                for key, expected in source.optimizer.state.get(original, {}).items():
                    torch.testing.assert_close(destination.optimizer.state[loaded][key], expected)
            if not updates:
                assert not destination.optimizer.state
            else:
                assert destination.model.bias not in destination.optimizer.state
        update(source)
        update(restored)
        update(healed)
        update(control)
        for destination in (source, restored, healed):
            for actual, expected in zip(destination.model.parameters(), control.model.parameters()):
                torch.testing.assert_close(actual, expected)
    finally:
        source.close()
        control.close()
        restored.close()
        healed.close()


def test_async_checkpoint_writes_latest(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.async_mode": "async", "ckpt.interval": 1})
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint()
    assert manager.close(timeout=10.0) is True
    target = _target_path(tmp_path, "latest")
    assert (target / ".metadata").exists()
    assert (target / "runner.yaml").exists()


def test_config_sidecar_round_trip(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "name": "dcp-sidecar-test",
            "ckpt.interval": 1,
        },
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)

    target = _target_path(tmp_path, "latest")
    assert (target / "runner.yaml").exists()
    config = TorchDistributedCheckpointManager.read_config(tmp_path / "latest")
    assert config.name == "dcp-sidecar-test"
    assert config.get("ckpt.backend") == "dcp"
    assert manager.close(timeout=1.0) is True


@pytest.mark.parametrize("async_mode", ["disabled", "async"])
def test_sidecar_failure(tmp_path: Path, async_mode):
    runner = _CheckpointRunner(tmp_path, config={"ckpt.async_mode": async_mode, "ckpt.interval": 1})
    manager = TorchDistributedCheckpointManager(runner)
    target = "ckpt-e000001-g000000000001-q000001"
    (tmp_path / target / "runner.yaml").mkdir(parents=True)
    runner.train_state.global_step = 1
    with pytest.warns(RuntimeWarning, match="checkpoint failed"):
        manager.save_checkpoint(epochs=0)
        assert manager.close(timeout=10.0)
    _assert_checkpoint_failure(manager, IsADirectoryError, target)
    assert manager.checkpoint_health.last_successful_target is None
    assert not (tmp_path / "latest.pointer").exists()
    assert _checkpoint_targets(tmp_path) == []
    assert not any(path.is_dir() for path in tmp_path.iterdir())


@pytest.mark.parametrize("async_mode", ["disabled", "async"])
def test_latest_publish_failure(tmp_path: Path, async_mode):
    runner = _CheckpointRunner(tmp_path, config={"ckpt.async_mode": async_mode, "ckpt.interval": 1})
    manager = TorchDistributedCheckpointManager(runner)
    (tmp_path / "latest.pointer").mkdir()
    runner.train_state.global_step = 1
    with pytest.warns(RuntimeWarning, match="checkpoint failed"):
        manager.save_checkpoint(epochs=0)
        assert manager.close(timeout=10.0)
    _assert_checkpoint_failure(manager, OSError, "ckpt-e000001-g000000000001-q000001", alias="latest")
    assert manager.checkpoint_health.last_successful_target is None
    assert (tmp_path / "latest.pointer").is_dir()
    assert _checkpoint_targets(tmp_path) == []


def test_partial_alias_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 1})
    runner.is_best = True
    manager = TorchDistributedCheckpointManager(runner)
    (tmp_path / "best.pointer").mkdir()

    runner.train_state.global_step = 1
    with pytest.warns(RuntimeWarning, match="checkpoint failed"):
        manager.save_checkpoint(epochs=0)

    target = "ckpt-e000001-g000000000001-q000001"
    _assert_checkpoint_failure(manager, OSError, target, alias="best")
    assert manager.checkpoint_health.last_successful_target == target
    assert manager.checkpoint_health.last_successful_aliases == ("latest", "ckpt-e000001")
    assert _pointer_target(tmp_path, "latest") == target
    assert _pointer_target(tmp_path, "ckpt-e000001") == target
    assert (tmp_path / target).is_dir()
    assert "checkpoint saved" not in capsys.readouterr().out
    assert manager.close(timeout=1.0) is True


def test_retention_after_alias_failure(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.interval": 1,
            "ckpt.keep_latest_k": 1,
        },
    )
    runner.is_best = True
    manager = TorchDistributedCheckpointManager(runner)
    (tmp_path / "best.pointer").mkdir()

    runner.train_state.global_step = 1
    with pytest.warns(RuntimeWarning, match="checkpoint failed"):
        manager.save_checkpoint(epochs=0)
    first_target = "ckpt-e000001-g000000000001-q000001"
    assert (tmp_path / first_target).is_dir()

    (tmp_path / "best.pointer").rmdir()
    runner.is_best = False
    runner.train_state.global_step = 2
    manager.save_checkpoint(epochs=1)

    second_target = "ckpt-e000002-g000000000002-q000002"
    assert manager.close(timeout=1.0) is True
    assert not (tmp_path / first_target).exists()
    assert not (tmp_path / "ckpt-e000001.pointer").exists()
    assert (tmp_path / second_target).is_dir()
    assert _pointer_target(tmp_path, "latest") == second_target


def test_async_group_failure(tmp_path: Path) -> None:
    require_gloo()
    run_distributed(
        _dcp_async_dedicated_process_group_failure_worker,
        world_size=2,
        worker_args=(str(tmp_path),),
    )


@pytest.mark.skipif(DCP_PINNED_MEMORY_STAGING_AVAILABLE, reason="requires PyTorch without DCP pinned-memory staging")
def test_pinned_async_publication(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.async_mode": "async_with_pinned_mem",
            "ckpt.interval": 1,
        },
    )
    runner.train_state.global_step = 1

    with pytest.warns(RuntimeWarning, match="pinned-memory async staging"):
        manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    assert manager.close(timeout=10.0) is True
    target = _target_path(tmp_path, "latest")
    assert (target / ".metadata").exists()
    assert (target / "runner.yaml").exists()


def test_force_checkpoint_bypasses_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0, force=True)

    assert (tmp_path / "latest.pointer").exists()
    assert len(_checkpoint_targets(tmp_path)) == 1
    assert manager.close(timeout=1.0) is True


def test_resolve_pointer_aliases(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 1})
    runner.is_step_mode = True
    runner.train_state.global_step = 1
    manager = TorchDistributedCheckpointManager(runner)
    try:
        manager.save_checkpoint(force=True)
        for alias in ("latest", "ckpt-s000000000001"):
            assert manager.load_checkpoint(alias)["state"]["global_step"] == 1
    finally:
        manager.close(timeout=1.0)


def test_latest_rotation_preserves_history(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 100})
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(last_step=True)
    first_checkpoint = _target_path(tmp_path, "latest")
    stale_history_pointer = tmp_path / "ckpt-e000000.pointer"
    stale_history_pointer.write_text(f"{first_checkpoint.name}\n", encoding="utf-8")

    runner.train_state.global_step = 2
    manager.save_checkpoint(last_step=True)

    assert first_checkpoint.exists()
    assert stale_history_pointer.exists()
    assert manager.close(timeout=1.0) is True


def test_latest_only_retention(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.interval": 100,
            "ckpt.keep_latest_k": 1,
        },
    )
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(last_step=True)
    first_checkpoint = _target_path(tmp_path, "latest")
    stale_history_pointer = tmp_path / "ckpt-e000000.pointer"
    stale_history_pointer.write_text(f"{first_checkpoint.name}\n", encoding="utf-8")

    runner.train_state.global_step = 2
    manager.save_checkpoint(last_step=True)
    second_checkpoint = _target_path(tmp_path, "latest")

    assert manager.close(timeout=1.0) is True
    assert not first_checkpoint.exists()
    assert not stale_history_pointer.exists()
    assert second_checkpoint.exists()
    assert _pointer_target(tmp_path, "latest") == second_checkpoint.name


def test_history_retention(tmp_path: Path) -> None:
    runner = _CheckpointRunner(
        tmp_path,
        config={
            "ckpt.interval": 1,
            "ckpt.keep_latest_k": 2,
        },
    )
    manager = TorchDistributedCheckpointManager(runner)

    first_checkpoint: Path | None = None
    for epoch in range(3):
        runner.train_state.epoch = epoch
        runner.train_state.global_step = epoch + 1
        manager.save_checkpoint(epochs=epoch)
        if epoch == 0:
            first_checkpoint = _target_path(tmp_path, "ckpt-e000001")

    assert first_checkpoint is not None
    assert manager.close(timeout=1.0) is True
    assert not first_checkpoint.exists()
    assert _target_path(tmp_path, "ckpt-e000002").exists()
    assert _target_path(tmp_path, "ckpt-e000003").exists()
    assert not (tmp_path / "ckpt-e000001.pointer").exists()
    assert (tmp_path / "ckpt-e000002.pointer").exists()
    assert (tmp_path / "ckpt-e000003.pointer").exists()


def test_waits_for_interval(tmp_path: Path) -> None:
    runner, manager = _retaining_dcp_manager(tmp_path)
    runner.train_state.global_step = 10

    manager.save_checkpoint(epochs=0)

    assert manager.close(timeout=1.0) is True
    assert not (tmp_path / "latest.pointer").exists()


def test_published_aliases(tmp_path: Path) -> None:
    manager = _save_best_dcp_checkpoint(tmp_path)
    try:
        targets = {_pointer_target(tmp_path, name) for name in ("latest", "best", "ckpt-e000002")}
        assert len(targets) == 1
        assert (tmp_path / targets.pop()).is_dir()
    finally:
        manager.close(timeout=1.0)


def test_forced_latest_preserves_best(tmp_path: Path) -> None:
    runner, manager = _retaining_dcp_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    best_target = _pointer_target(tmp_path, "best")

    runner.is_best = False
    runner.train_state.global_step = 30
    manager.save_checkpoint(epochs=2, force=True)
    forced_latest_target = _pointer_target(tmp_path, "latest")

    assert forced_latest_target.startswith("latest-g000000000030")
    assert _pointer_target(tmp_path, "best") == best_target
    assert manager.close(timeout=1.0) is True


def test_retention_prunes_forced_latest_target(tmp_path: Path) -> None:
    runner, manager = _retaining_dcp_manager(tmp_path)
    runner.is_best = True
    runner.train_state.global_step = 20
    manager.save_checkpoint(epochs=1)
    best_target = _pointer_target(tmp_path, "best")
    runner.is_best = False
    runner.train_state.global_step = 30
    manager.save_checkpoint(epochs=2, force=True)
    forced_latest_target = _pointer_target(tmp_path, "latest")
    runner.train_state.global_step = 40
    manager.save_checkpoint(epochs=3)
    final_latest_target = _pointer_target(tmp_path, "latest")

    assert manager.close(timeout=1.0) is True
    assert final_latest_target.startswith("ckpt-e000004-g000000000040")
    assert _pointer_target(tmp_path, "best") == best_target
    assert (tmp_path / best_target).is_dir()
    assert not (tmp_path / forced_latest_target).exists()
    assert (tmp_path / final_latest_target).is_dir()


def test_save_checkpoint_respects_save_disabled(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 1, "ckpt.enabled": False})
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint()
    manager.save_checkpoint(last_step=True)

    assert not (tmp_path / "latest.pointer").exists()
    assert _checkpoint_targets(tmp_path) == []
    assert manager.close(timeout=1.0) is True


def test_skips_periodic_writes_without_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path)
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)

    assert not (tmp_path / "latest.pointer").exists()
    assert _checkpoint_targets(tmp_path) == []
    assert manager.close(timeout=1.0) is True


def test_save_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 5})
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    manager.save_checkpoint(epochs=1)
    manager.save_checkpoint(epochs=4)

    assert len(_checkpoint_targets(tmp_path)) == 1
    assert (tmp_path / "ckpt-e000005.pointer").exists()
    assert manager.close(timeout=1.0) is True


def test_last_step_forces_save(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 100})
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    manager.save_checkpoint(epochs=0, last_step=True)

    assert (tmp_path / "latest.pointer").exists()
    assert len(_checkpoint_targets(tmp_path)) == 1
    assert manager.close(timeout=1.0) is True


def test_model_export_preserves_latest(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.export_dtype": "fp16"})
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(last_step=True)
    latest_target = _pointer_target(tmp_path, "latest")

    runner.train_state.global_step = 2
    manager.save_model_checkpoint()

    model_target = _pointer_target(tmp_path, "model")
    model_path = tmp_path / model_target

    assert _pointer_target(tmp_path, "latest") == latest_target
    assert model_path.exists()
    assert not (model_path / "runner.yaml").exists()
    model_payload = manager.load_model_checkpoint("model")
    assert list(model_payload) == ["model"]
    assert model_payload["model"]["weight"].dtype == torch.float16
    assert manager.close(timeout=1.0) is True


def test_step_interval(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 2})
    runner.is_step_mode = True
    manager = TorchDistributedCheckpointManager(runner)

    runner.train_state.global_step = 1
    manager.save_checkpoint(epochs=0)
    assert _checkpoint_targets(tmp_path) == []

    runner.train_state.global_step = 2
    manager.save_checkpoint(epochs=0)
    checkpoint = _target_path(tmp_path, "ckpt-s000000000002")

    assert checkpoint.name.startswith("ckpt-s000000000002-g000000000002-")
    assert manager.close(timeout=1.0) is True


def test_checkpoint_includes_loader_state(tmp_path: Path) -> None:
    runner = _CheckpointRunner(tmp_path, config={"ckpt.interval": 1}, dataloaders=True)
    manager = TorchDistributedCheckpointManager(runner)

    manager.save_checkpoint(epochs=0)
    runner.dataloaders["train"] = _make_stateful_dataloader(0)
    checkpoint = manager.load_checkpoint("latest")

    assert _dataloader_state_position(checkpoint["dataloaders"]["train"]) == 7
    assert _stateful_dataloader_position(runner.dataloaders["train"]) == 0
    assert manager.close(timeout=1.0) is True
