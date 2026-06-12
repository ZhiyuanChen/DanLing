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
from danling.runners.config import (
    CheckpointConfig,
    CompileConfig,
    DataloaderCheckpointConfig,
    DataloaderConfig,
    DistributedConfig,
    FaultToleranceConfig,
    Fp8Config,
    FsdpConfig,
    GcConfig,
    HeartbeatConfig,
    LoggingConfig,
    ParallelAxesConfig,
    ParallelConfig,
    ProfilingConfig,
    ScoreConfig,
    TensorboardConfig,
    WandbConfig,
    WorkspaceConfig,
)


def _checkpoint_fault_tolerance_config() -> RunnerConfig:
    return RunnerConfig(
        {
            "ckpt": {
                "backend": "DCP",
                "async_mode": "async_with_pinned_mem",
                "dedicated_async_process_group": False,
                "async_process_group_backend": "gloo",
                "dataloader_checkpoint": {
                    "enabled": True,
                    "replica_id": "replica-a",
                    "prefix": "ft-loader",
                },
                "export_dtype": "bf16",
            },
            "ft": {
                "enabled": True,
                "process_group": "gloo",
                "process_group_timeout_seconds": 1.234,
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
                "pipeline_microbatches": 4,
            },
            "fsdp": {
                "enabled": True,
                "reshard_after_forward": False,
                "mixed_precision_policy": {"param_dtype": "bf16"},
                "offload_policy": {"pin_memory": True},
                "ignored_params": ["weight"],
            },
        }
    )


def test_runner_config_canonical_ignores_unused_parallelism_blocks_for_ddp() -> None:
    config_a = RunnerConfig(
        {
            "logging.enabled": False,
            "stack": "ddp",
            "parallel.axes.tensor": 2,
            "parallel.axes.pipeline": 2,
            "parallel.axes.shard": 2,
            "fsdp.enabled": True,
        }
    )
    config_b = RunnerConfig(
        {
            "logging.enabled": False,
            "stack": "torch",
            "parallel.axes.tensor": 8,
            "parallel.axes.pipeline": 1,
            "parallel.axes.shard": 1,
            "fsdp.enabled": False,
        }
    )

    assert config_a.canonical() == config_b.canonical()


def test_runner_config_canonical_ignores_fault_tolerance_runtime_policy() -> None:
    config_a = RunnerConfig(
        {
            "logging.enabled": False,
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
            "logging.enabled": False,
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


def test_runner_config_materializes_runtime_sections() -> None:
    config = RunnerConfig()

    expected_sections = {
        "compile": CompileConfig,
        "ckpt": CheckpointConfig,
        "workspace": WorkspaceConfig,
        "logging": LoggingConfig,
        "tensorboard": TensorboardConfig,
        "wandb": WandbConfig,
        "score": ScoreConfig,
        "fp8": Fp8Config,
        "ft": FaultToleranceConfig,
        "dist": DistributedConfig,
        "gc": GcConfig,
        "profiling": ProfilingConfig,
        "heartbeat": HeartbeatConfig,
        "dataloader": DataloaderConfig,
        "fsdp": FsdpConfig,
        "parallel": ParallelConfig,
    }
    for name, section_cls in expected_sections.items():
        assert isinstance(config.get(name), section_cls)
        assert isinstance(getattr(config, name), section_cls)

    assert config.get("ckpt.backend") == "auto"
    assert isinstance(config.get("ckpt.dataloader_checkpoint"), DataloaderCheckpointConfig)
    assert config.get("ckpt.dataloader_checkpoint.enabled") is False
    assert config.get("compile.enabled") is False
    assert isinstance(config.get("parallel.axes"), ParallelAxesConfig)
    assert config.get("parallel.axes.shard") == 1
    assert config.get("workspace.root") == "experiments"

    config.workspace.experiment = "changed"
    config.parallel.axes.shard = 2
    assert RunnerConfig().workspace.experiment == "exp"
    assert RunnerConfig().parallel.axes.shard == 1


def test_runner_config_keeps_training_sections_optional() -> None:
    config = RunnerConfig()

    assert config.get("optim") is None
    assert config.get("sched") is None
    assert "optim" not in config
    assert "sched" not in config

    explicit_none = RunnerConfig({"optim": None, "sched": None})
    assert explicit_none.get("optim") is None
    assert explicit_none.get("sched") is None
    assert "optim" not in explicit_none.canonical()
    assert "sched" not in explicit_none.canonical()


def test_runner_config_canonical_ignores_materialized_default_sections() -> None:
    canonical = RunnerConfig().canonical()

    assert "compile" not in canonical
    assert "ckpt" not in canonical
    assert "dataloader" not in canonical
    assert "fp8" not in canonical


def test_runner_config_canonical_drops_defaults_and_preserves_explicit_none_values() -> None:
    config = RunnerConfig(
        {
            "optim": {
                "type": None,
                "lr": 1e-3,
            },
            "sched": {
                "interval": None,
                "monitor": "val.loss",
            },
            "compile": {
                "enabled": False,
                "backend": None,
                "optimize_ddp": None,
            },
            "dataloader": {
                "batch_size": 8,
                "train": {
                    "shuffle": None,
                    "drop_last": False,
                },
            },
        }
    )

    canonical = config.canonical()

    assert canonical["optim"] == {"type": None, "lr": 1e-3}
    assert canonical["sched"] == {"monitor": "val.loss"}
    assert canonical["compile"] == {"optimize_ddp": None}
    assert canonical["dataloader"] == {"batch_size": 8, "train": {"shuffle": None, "drop_last": False}}


def test_runner_config_compile_surface_matches_runtime_options() -> None:
    config = RunnerConfig(
        {
            "compile": {
                "enabled": True,
                "backend": "inductor",
                "mode": "max-autotune",
                "options": {"trace.enabled": True},
                "precompile_artifact_dir": "/tmp/danling-compile",
                "memory_policy": "cpu_offload_all",
            }
        }
    )

    assert config.compile.enabled is True
    assert config.compile.backend == "inductor"
    assert config.compile.mode == "max-autotune"
    assert dict(config.compile.options) == {"trace": {"enabled": True}}
    assert config.compile.precompile_artifact_dir == "/tmp/danling-compile"
    assert config.compile.memory_policy == "cpu_offload_all"


def test_runner_config_common_runtime_surfaces_are_typed() -> None:
    config = RunnerConfig(
        {
            "optim": {
                "type": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "fused": True,
            },
            "sched": {
                "type": "cosine",
                "total_steps": 100,
                "warmup_steps": 5,
                "cooldown_steps": 10,
                "final_lr_ratio": 1e-2,
                "T_max": 100,
                "eta_min": 1e-6,
            },
            "tensorboard": {
                "enabled": True,
                "comment": "debug",
                "max_queue": 4,
                "flush_secs": 30,
                "filename_suffix": ".tb",
            },
            "wandb": {
                "enabled": True,
                "id": "run-a",
                "notes": "smoke",
                "resume": True,
                "save_code": True,
                "sync_tensorboard": True,
            },
            "profiling": {
                "enabled": True,
                "activities": ["cpu"],
                "with_modules": True,
                "acc_events": True,
                "post_processing_timeout_seconds": 12.5,
            },
            "score": {
                "patience": 3,
            },
            "logging": {
                "file": "/tmp/danling.log",
            },
            "dataloader": {
                "sampler": "sampler-a",
                "batch_sampler": "batch-sampler-a",
                "collate_fn": "collate-a",
            },
        }
    )

    assert config.optim.type == "adamw"
    assert config.optim.lr == 1e-3
    assert list(config.optim.betas) == [0.9, 0.95]
    assert config.optim.fused is True
    assert config.sched.type == "cosine"
    assert config.sched.total_steps == 100
    assert config.sched.T_max == 100
    assert config.tensorboard.enabled is True
    assert config.tensorboard.comment == "debug"
    assert config.tensorboard.max_queue == 4
    assert config.wandb.id == "run-a"
    assert config.wandb.resume is True
    assert config.wandb.sync_tensorboard is True
    assert list(config.profiling.activities) == ["cpu"]
    assert config.profiling.with_modules is True
    assert config.profiling.post_processing_timeout_seconds == 12.5
    assert config.score.patience == 3
    assert config.logging.file == "/tmp/danling.log"
    assert config.dataloader.sampler == "sampler-a"
    assert config.dataloader.batch_sampler == "batch-sampler-a"
    assert config.dataloader.collate_fn == "collate-a"


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
    config = _checkpoint_fault_tolerance_config()

    assert config["ckpt"].async_mode == "async_with_pinned_mem"
    assert config["ckpt"].dedicated_async_process_group is False
    assert config["ckpt"].async_process_group_backend == "gloo"


def test_runner_config_checkpoint_dataloader_checkpoint_surface_matches_runtime_keys() -> None:
    config = _checkpoint_fault_tolerance_config()

    assert config["ckpt"].dataloader_checkpoint.enabled is True
    assert config["ckpt"].dataloader_checkpoint.replica_id == "replica-a"
    assert config["ckpt"].dataloader_checkpoint.prefix == "ft-loader"


def test_runner_config_checkpoint_export_surface_matches_runtime_keys() -> None:
    config = _checkpoint_fault_tolerance_config()

    assert config["ckpt"].export_dtype == "bf16"
    assert config.canonical()["ckpt"] == {"backend": "dcp"}


def test_runner_config_fault_tolerance_surface_matches_runtime_keys() -> None:
    config = _checkpoint_fault_tolerance_config()

    assert config.ft.process_group_timeout_seconds == 1.234
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
    assert config.parallel.pipeline_microbatches == 4


def test_runner_config_fsdp_surface_matches_runtime_keys() -> None:
    config = _fsdp_parallel_config()

    assert config.fsdp.reshard_after_forward is False
    assert config.fsdp.enabled is True
    assert dict(config.fsdp.mixed_precision_policy) == {"param_dtype": "bf16"}
    assert dict(config.fsdp.offload_policy) == {"pin_memory": True}
    assert list(config.fsdp.ignored_params) == ["weight"]
