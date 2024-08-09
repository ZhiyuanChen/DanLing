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

from danling.runners import RunnerConfig
from danling.runners.config import DataloaderConfig


def _checkpoint_ft_config() -> RunnerConfig:
    return RunnerConfig(
        {
            "checkpoint": {
                "backend": "DCP",
                "async_mode": "async_with_pinned_mem",
                "dedicated_async_process_group": False,
                "async_process_group_backend": "gloo",
                "enable_ft_dataloader_checkpoints": True,
                "ft_replica_id": "replica-a",
                "ft_dataloader_checkpoint_prefix": "ft-loader",
                "last_save_model_only": True,
                "export_dtype": "bf16",
            },
            "ft": {
                "enabled": True,
                "process_group": "gloo",
                "process_group_timeout_ms": 1234,
                "replica_id": 1,
                "group_size": 2,
                "min_replica_size": 1,
            },
        }
    )


def _fsdp_parallel_config() -> RunnerConfig:
    return RunnerConfig(
        {
            "parallel": {
                "axes": {
                    "replicate": 2,
                    "shard": 4,
                    "context": 1,
                    "pipeline": 4,
                    "tensor": 8,
                    "expert": 1,
                    "expert_tensor": 1,
                },
                "use_device_mesh": False,
                "mesh_device_type": "cuda",
                "pipeline_microbatch_size": 2,
                "pipeline_n_microbatches": 4,
            },
            "fsdp": {
                "enabled": True,
                "reshard_after_forward": False,
                "mp_policy": {"param_dtype": "bf16"},
                "offload_policy": {"pin_memory": True},
                "ignored_params": ["weight"],
            },
        }
    )


def test_runner_config_canonical_ignores_unused_parallelism_blocks_for_ddp() -> None:
    config_a = RunnerConfig(
        {
            "log": False,
            "stack": "ddp",
            "parallel.axes.tensor": 2,
            "parallel.axes.pipeline": 2,
            "parallel.axes.shard": 2,
            "fsdp.enabled": True,
        }
    )
    config_b = RunnerConfig(
        {
            "log": False,
            "stack": "torch",
            "parallel.axes.tensor": 8,
            "parallel.axes.pipeline": 1,
            "parallel.axes.shard": 1,
            "fsdp.enabled": False,
        }
    )

    assert config_a.canonical() == config_b.canonical()


def test_runner_config_canonical_ignores_ft_runtime_policy() -> None:
    config_a = RunnerConfig(
        {
            "log": False,
            "stack": "parallel",
            "parallel.axes.replicate": 2,
            "parallel.axes.shard": 2,
            "fsdp.enabled": True,
            "ft": {
                "enabled": True,
                "process_group": "gloo",
                "replica_id": 0,
                "group_size": 2,
                "min_replica_size": 1,
            },
        }
    )
    config_b = RunnerConfig(
        {
            "log": False,
            "stack": "parallel",
            "parallel.axes.replicate": 2,
            "parallel.axes.shard": 2,
            "fsdp.enabled": True,
            "ft": {
                "enabled": False,
                "process_group": "nccl",
                "replica_id": 1,
                "group_size": 4,
                "min_replica_size": 2,
            },
        }
    )

    assert config_a.canonical() == config_b.canonical()
    assert hash(config_a) == hash(config_b)


def test_runner_config_compile_surface_matches_runtime_options() -> None:
    config = RunnerConfig(
        {
            "compile": {
                "enable": True,
                "backend": "inductor",
                "mode": "max-autotune",
                "options": {"trace.enabled": True},
            }
        }
    )

    assert config.compile.enable is True
    assert config.compile.backend == "inductor"
    assert config.compile.mode == "max-autotune"
    assert dict(config.compile.options) == {"trace": {"enabled": True}}


def test_runner_config_dataloader_surface_matches_stateful_loader_options() -> None:
    config = RunnerConfig(
        {
            "dataloader": {
                "batch_size": 8,
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4,
                "in_order": False,
                "snapshot_every_n_steps": 16,
                "train": {"shuffle": False},
                "val": {"drop_last": False},
            }
        }
    )

    assert isinstance(config.dataloader, DataloaderConfig)
    assert config.dataloader.batch_size == 8
    assert config.dataloader.num_workers == 4
    assert config.dataloader.pin_memory is True
    assert config.dataloader.persistent_workers is True
    assert config.dataloader.prefetch_factor == 4
    assert config.dataloader.in_order is False
    assert config.dataloader.snapshot_every_n_steps == 16
    assert dict(config.dataloader.train) == {"shuffle": False}
    assert dict(config.dataloader.val) == {"drop_last": False}


def test_runner_config_dataloader_keeps_explicit_surface() -> None:
    config = RunnerConfig({"dataloader": {"batch_size": 8}})

    assert isinstance(config.dataloader, DataloaderConfig)
    assert config.dataloader.dict() == {"batch_size": 8}
    assert config.canonical()["dataloader"] == {"batch_size": 8}


def test_runner_config_checkpoint_async_surface_matches_runtime_keys() -> None:
    config = _checkpoint_ft_config()

    assert config.checkpoint.async_mode == "async_with_pinned_mem"
    assert config.checkpoint.dedicated_async_process_group is False
    assert config.checkpoint.async_process_group_backend == "gloo"


def test_runner_config_checkpoint_ft_dataloader_surface_matches_runtime_keys() -> None:
    config = _checkpoint_ft_config()

    assert config.checkpoint.enable_ft_dataloader_checkpoints is True
    assert config.checkpoint.ft_replica_id == "replica-a"
    assert config.checkpoint.ft_dataloader_checkpoint_prefix == "ft-loader"


def test_runner_config_checkpoint_export_surface_matches_runtime_keys() -> None:
    config = _checkpoint_ft_config()

    assert config.checkpoint.last_save_model_only is True
    assert config.checkpoint.export_dtype == "bf16"
    assert config.canonical()["checkpoint"] == {"backend": "dcp"}


def test_runner_config_ft_surface_matches_runtime_keys() -> None:
    config = _checkpoint_ft_config()

    assert config.ft.process_group_timeout_ms == 1234
    assert config.ft.replica_id == 1
    assert config.ft.group_size == 2


def test_runner_config_parallel_axes_match_runtime_keys() -> None:
    config = _fsdp_parallel_config()

    assert config.parallel.axes.replicate == 2
    assert config.parallel.axes.shard == 4
    assert config.parallel.axes.context == 1
    assert config.parallel.axes.pipeline == 4
    assert config.parallel.axes.tensor == 8
    assert config.parallel.axes.expert == 1
    assert config.parallel.axes.expert_tensor == 1


def test_runner_config_parallel_options_match_runtime_keys() -> None:
    config = _fsdp_parallel_config()

    assert config.parallel.use_device_mesh is False
    assert config.parallel.mesh_device_type == "cuda"
    assert config.parallel.pipeline_microbatch_size == 2
    assert config.parallel.pipeline_n_microbatches == 4


def test_runner_config_fsdp_surface_matches_runtime_keys() -> None:
    config = _fsdp_parallel_config()

    assert config.fsdp.reshard_after_forward is False
    assert config.fsdp.enabled is True
    assert dict(config.fsdp.mp_policy) == {"param_dtype": "bf16"}
    assert dict(config.fsdp.offload_policy) == {"pin_memory": True}
    assert list(config.fsdp.ignored_params) == ["weight"]
