# DanLing
# Copyright (C) 2022-Present  DanLing
#
# This file is part of DanLing.
#
# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0
#
# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Tuple
from warnings import warn

import torch
from lazy_imports import try_import
from torch import distributed as dist
from torch import nn, utils
from tqdm import tqdm

from danling.data import StepProxyLoader
from danling.optim import OPTIMIZERS

from .base_runner import BaseRunner
from .checkpoints import TorchDistributedCheckpointManager
from .compile import maybe_compile_model
from .torch_runner import TorchRunner, get_precision
from .utils import RunnerMode

with try_import() as dcp:
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.device_mesh import init_device_mesh

with try_import() as pipeline:
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import PipelineScheduleMulti, get_schedule_class


@dataclass(init=False)
class TpppTopology:
    tp_degree: int
    pp_degree: int
    dp_degree: int
    tp_rank: int
    pp_rank: int
    dp_rank: int

    def __init__(self, *, world_size: int, rank: int, tp_degree: int, pp_degree: int) -> None:
        if tp_degree < 1 or pp_degree < 1:
            raise ValueError(
                "invalid TPPP topology: tp_degree and pp_degree must be positive integers, "
                f"got tp_degree={tp_degree}, pp_degree={pp_degree}"
            )

        model_parallel_degree = tp_degree * pp_degree
        if world_size % model_parallel_degree != 0:
            raise ValueError(
                "invalid TPPP topology: "
                f"WORLD_SIZE({world_size}) is not divisible by "
                f"tp_degree({tp_degree}) * pp_degree({pp_degree}) = {model_parallel_degree}"
            )

        dp_degree = world_size // model_parallel_degree
        model_parallel_rank = rank % model_parallel_degree
        self.tp_degree = tp_degree
        self.pp_degree = pp_degree
        self.dp_degree = dp_degree
        self.tp_rank = model_parallel_rank % tp_degree
        self.pp_rank = (model_parallel_rank // tp_degree) % pp_degree
        self.dp_rank = rank // model_parallel_degree


class TpppRunner(TorchRunner):
    """Torch runner for TP+PP stacks with backend checks owned by the subclass.

    Checkpoint invariants:
    - Distributed TP/PP runs use `checkpoint.backend="dcp"` only.
    - Model/optimizer state uses torch.distributed.checkpoint state-dict APIs when available.
    - Restore order is model first, then optimizer, then scheduler.
    """

    topology: TpppTopology
    pipeline_schedule: Any | None = None
    pp_has_first_stage: bool = True
    pp_has_last_stage: bool = True

    tp_group = None
    pp_group = None
    dp_group = None
    device_mesh = None
    _tppp_groups_initialized: bool = False

    model_parts: list[nn.Module]

    checkpoint_manager: TorchDistributedCheckpointManager

    def __init__(self, config: Mapping[str, Any]) -> None:
        dcp.check()
        super().__init__(config)

    def init_distributed(self) -> None:
        super().init_distributed()
        if self.world_size <= 1:
            raise RuntimeError("TpppRunner requires distributed mode (WORLD_SIZE > 1)")
        self.topology = self.build_topology()
        if not self._tppp_groups_initialized:
            self.reset_model_parallel_groups()
            self.init_model_parallel_groups()
            self._tppp_groups_initialized = True
        self.configure_checkpoint_backend()

    def build_topology(self) -> TpppTopology:
        return TpppTopology(
            world_size=self.world_size,
            rank=self.rank,
            tp_degree=self.config.tppp.tp_degree,
            pp_degree=self.config.tppp.pp_degree,
        )

    def reset_model_parallel_groups(self) -> None:
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
        self.device_mesh = None

    def init_model_parallel_groups(self) -> None:
        use_device_mesh = self.config.tppp.use_device_mesh
        if not use_device_mesh:
            raise RuntimeError("cannot initialize TPPP process groups: set `tppp.use_device_mesh=True`.")

        mesh_device_type = self.config.tppp.mesh_device_type
        if mesh_device_type is None:
            mesh_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_mesh = init_device_mesh(
            mesh_device_type,
            mesh_shape=(self.dp_degree, self.pp_degree, self.tp_degree),
            mesh_dim_names=("dp", "pp", "tp"),
        )
        self.dp_group = self.device_mesh.get_group("dp")
        self.pp_group = self.device_mesh.get_group("pp")
        self.tp_group = self.device_mesh.get_group("tp")

    def configure_checkpoint_backend(self) -> None:
        backend = self.config.checkpoint.backend.lower()
        if backend == "dcp":
            return
        warn(
            f"{self.__class__.__name__} overrides checkpoint.backend to 'dcp'",
            RuntimeWarning,
            stacklevel=2,
        )
        self.config.checkpoint.backend = "dcp"
        self.checkpoint_manager = TorchDistributedCheckpointManager(self)

    def __post_init__(self):
        if not self.model_parts:
            if self.model is None:
                raise ValueError("cannot initialize model_parts: model is not initialized")
            self.model_parts = [self.model]
        super().__post_init__()

    def materialize_model(self) -> None:
        if self.pipeline_schedule is None and self.pp_degree > 1 and self.pp_group is not None:
            stage_model = self.model_parts[0] if self.model_parts else self.model
            if stage_model is None:
                raise ValueError("cannot materialize TPPP pipeline: model is not initialized")
            self.pipeline_schedule = self.build_pipeline_schedule(stage_model)
            self.model_parts = [stage_model]
            self.model = stage_model
            self.pp_has_first_stage = self.pp_rank == 0
            self.pp_has_last_stage = self.pp_rank == self.pp_degree - 1

        if self.pipeline_schedule is None:
            if self.model is None:
                if self.model_parts:
                    self.model = self.model_parts[0]
                else:
                    raise ValueError("cannot materialize TPPP model: model is not initialized")
            model = self.model.to(self.device)
            model = maybe_compile_model(model, self.config)
            self.model = model
            self.model_parts = [model]
        else:
            if not self.model_parts:
                if self.model is None:
                    raise ValueError("cannot materialize TPPP pipeline: model_parts are not initialized")
                self.model_parts = [self.model]
            self.model_parts = [maybe_compile_model(model.to(self.device), self.config) for model in self.model_parts]
            self.model = self.model_parts[0]
            self.bind_pipeline_modules(self.model_parts)

        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def bind_pipeline_modules(self, modules: Sequence[nn.Module]) -> None:
        if self.pipeline_schedule is None:
            return

        stages = getattr(self.pipeline_schedule, "stages", None)
        if stages is None:
            stage = getattr(self.pipeline_schedule, "stage", None)
            if stage is not None and modules:
                stage.module = modules[0]
                return
            if hasattr(self.pipeline_schedule, "module") and modules:
                self.pipeline_schedule.module = modules[0]
            return

        for stage, module in zip(stages, modules):
            if hasattr(stage, "module"):
                stage.module = module

    def build_optimizer(self) -> None:
        if self.optimizer is not None:
            return
        optim_cfg = self.config.get("optim")
        if optim_cfg is None:
            optim_cfg = self.config.get("optimizer")
        if not isinstance(optim_cfg, Mapping) or not optim_cfg:
            return

        parameter_list: list[nn.Parameter] = []
        seen: set[int] = set()
        parts: list[nn.Module] = list(self.model_parts or [])
        if not parts and self.model is not None:
            parts = [self.model]
        if not parts:
            return
        for model in parts:
            for parameter in model.parameters():
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                parameter_list.append(parameter)

        self.optimizer = OPTIMIZERS.build(params=parameter_list, **dict(optim_cfg))

    def _resolve_pipeline_n_microbatches(self) -> int:
        configured = self.config.tppp.get("pipeline_n_microbatches")
        if configured is not None:
            n_microbatches = int(configured)
            if n_microbatches <= 0:
                raise ValueError(f"invalid tppp.pipeline_n_microbatches: expected a positive integer, got {configured}")
            return n_microbatches

        microbatch_size = int(self.config.tppp.get("pipeline_microbatch_size", 1))
        if microbatch_size <= 0:
            raise ValueError(
                f"invalid tppp.pipeline_microbatch_size: expected a positive integer, got {microbatch_size}"
            )

        try:
            batch_size = int(self.batch_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "cannot infer pipeline microbatch count: set `tppp.pipeline_n_microbatches` "
                "or provide `dataloader.batch_size`."
            ) from exc

        if batch_size <= 0:
            raise ValueError(f"invalid batch size: expected a positive integer, got {batch_size}")
        if batch_size % microbatch_size != 0:
            raise ValueError(
                f"batch size ({batch_size}) must be divisible by tppp.pipeline_microbatch_size ({microbatch_size})"
            )

        n_microbatches = batch_size // microbatch_size
        if n_microbatches < self.pp_degree:
            warn(
                f"n_microbatches ({n_microbatches}) is less than pp_degree ({self.pp_degree}); "
                "pipeline utilization may be suboptimal.",
                RuntimeWarning,
                stacklevel=2,
            )
        return n_microbatches

    def build_pipeline_schedule(self, stage_model: nn.Module) -> Any:
        pipeline.check()
        schedule_name = str(self.config.tppp.get("pipeline_schedule", "1F1B")).strip() or "1F1B"
        n_microbatches = self._resolve_pipeline_n_microbatches()
        schedule_class = get_schedule_class(schedule_name)
        stage = PipelineStage(
            stage_model,
            stage_index=self.pp_rank,
            num_stages=self.pp_degree,
            device=self.device,
            group=self.pp_group,
        )

        # Default to non-interleaved 1F1B for PP schedules in DTPP/TPPP until
        # pytorch/pytorch#164756 is addressed upstream, then we can migrate the
        # default to Interleaved1F1B.
        if issubclass(schedule_class, PipelineScheduleMulti):
            return schedule_class(
                [stage],
                n_microbatches=n_microbatches,
                loss_fn=self.criterion,
                scale_grads=False,
            )
        return schedule_class(
            stage,
            n_microbatches=n_microbatches,
            loss_fn=self.criterion,
            scale_grads=False,
        )

    @contextmanager
    def _step_only_loader_context(self, split: str):
        use_step_only_loader = (
            self.pipeline_schedule is not None and not self.pp_has_first_stage and not self.pp_has_last_stage
        )
        if not use_step_only_loader:
            yield
            return

        loader = self.dataloaders[split]
        self.dataloaders[split] = StepProxyLoader(loader)
        try:
            yield
        finally:
            self.dataloaders[split] = loader

    def _prepare_pipeline_batch(self, data: Any) -> tuple[Any | None, Any | None]:
        if self.pp_has_first_stage:
            if data is None:
                raise ValueError("cannot run pipeline stage: first stage requires dataloader inputs")
            data = self.to_device(data)
            if isinstance(data, Mapping):
                inputs = data["input"]
                target = data.get("target")
            elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
                inputs = data[0]
                target = data[1] if len(data) > 1 else None
            else:
                inputs = data
                target = None
            if not self.pp_has_last_stage:
                target = None
            return inputs, target

        if not self.pp_has_last_stage or data is None:
            return None, None
        if isinstance(data, Mapping):
            if "target" not in data:
                return None, None
            return None, self.to_device(data["target"])
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)) and len(data) > 1:
            return None, self.to_device(data[1])
        target = None
        return None, target

    def train_epoch(self, split: str = "train"):
        with self._step_only_loader_context(split):
            return super().train_epoch(split=split)

    def train_steps(
        self,
        train_splits: list[str] | None = None,
        evaluate_splits: list[str] | None = None,
    ):
        if train_splits is None:
            train_splits = self.train_splits
        use_step_only_loader = (
            self.pipeline_schedule is not None and not self.pp_has_first_stage and not self.pp_has_last_stage
        )
        if not use_step_only_loader:
            return super().train_steps(train_splits=train_splits, evaluate_splits=evaluate_splits)
        with ExitStack() as stack:
            for split in train_splits:
                stack.enter_context(self._step_only_loader_context(split))
            return super().train_steps(train_splits=train_splits, evaluate_splits=evaluate_splits)

    @torch.inference_mode()
    def evaluate_epoch(self, split: str = "val"):
        with self._step_only_loader_context(split):
            return super().evaluate_epoch(split=split)

    @torch.inference_mode()
    def evaluate_steps(self, split: str = "val", steps: int | None = None):
        with self._step_only_loader_context(split):
            return super().evaluate_steps(split=split, steps=steps)

    @contextmanager
    def train_context(self):
        if self.pipeline_schedule is None:
            with super().train_context():
                yield
            return

        if self.fp8_enabled:
            with self.fp8_autocast():
                yield
            return

        precision = self.precision
        if precision is None:
            with nullcontext():
                yield
            return

        with torch.autocast(self.device.type, dtype=get_precision(precision)):
            yield

    def train_step(self, data) -> Tuple[Any, torch.Tensor | None]:
        if self.pipeline_schedule is None:
            return super().train_step(data)

        with self.train_context():
            inputs, target = self._prepare_pipeline_batch(data)
            losses: list[torch.Tensor] = []
            targets = target if self.pp_has_last_stage else None

            if self.pp_has_first_stage:
                self.pipeline_schedule.step(
                    inputs,
                    target=targets,
                    losses=losses,
                )
            else:
                self.pipeline_schedule.step(
                    target=targets,
                    losses=losses,
                )

            if self.pp_has_last_stage and losses:
                loss = torch.mean(torch.stack(losses))
            else:
                loss = None

            pred = None
            self.step()
        return pred, loss

    def evaluate_step(self, data) -> Tuple[Any, torch.Tensor | None]:
        if self.pipeline_schedule is None:
            return super().evaluate_step(data)

        inputs, target = self._prepare_pipeline_batch(data)
        losses: list[torch.Tensor] = []
        targets = target if self.pp_has_last_stage else None

        if self.pp_has_first_stage:
            self.pipeline_schedule.eval(
                inputs,
                target=targets,
                losses=losses,
            )
        else:
            self.pipeline_schedule.eval(
                target=targets,
                losses=losses,
            )

        if self.pp_has_last_stage and losses:
            loss = torch.mean(torch.stack(losses))
        else:
            loss = None

        return None, loss

    def infer(
        self,
        split: str = "infer",
        *,
        steps: int | None = None,
        stream: bool | None = None,
    ) -> list[float] | Iterator[list[float]]:
        if self.pipeline_schedule is None:
            return super().infer(split=split, steps=steps, stream=stream)

        self.mode = RunnerMode.infer
        self.split = split
        loader = self.dataloaders[split]

        if steps is not None and steps < 0:
            raise ValueError(f"invalid steps: expected a non-negative value, got {steps}")

        loader_length = self._loader_length(loader)
        if stream is None:
            stream = steps is None and loader_length is None

        if self.pp_has_first_stage:
            if not stream and loader_length is None and steps is None:
                raise ValueError("infer with stream=False requires `steps` for unsized loaders")
            if steps is not None:
                iterator = (self.infer_step(data) for iteration, data in enumerate(loader) if iteration < steps)
            else:
                iterator = (self.infer_step(data) for data in loader)
            total = steps if steps is not None else loader_length
        else:
            if steps is None:
                if loader_length is None:
                    raise ValueError("infer for non-first pipeline stages requires `steps` for unsized loaders")
                steps = loader_length
            iterator = (self.infer_step(None) for _ in range(steps))
            total = steps

        if stream:
            return iterator

        output: list[float] = []
        for values in tqdm(iterator, total=total, disable=self.distributed and not self.is_main_process):
            output.extend(values)
        return output

    @staticmethod
    def _normalize_infer_output(pred: Any) -> list[float]:
        if pred is None:
            return []
        if torch.is_tensor(pred):
            values = pred.detach().reshape(-1).cpu().tolist()
            if isinstance(values, list):
                return [float(value) for value in values]
            return [float(values)]
        if isinstance(pred, Mapping):
            mapped_values: list[float] = []
            for value in pred.values():
                mapped_values.extend(TpppRunner._normalize_infer_output(value))
            return mapped_values
        if isinstance(pred, Sequence) and not isinstance(pred, (str, bytes)):
            seq_values: list[float] = []
            for value in pred:
                seq_values.extend(TpppRunner._normalize_infer_output(value))
            return seq_values
        if isinstance(pred, (bool, int, float)):
            return [float(pred)]
        raise ValueError(
            "cannot normalize pipeline infer output: unsupported type "
            f"{type(pred).__name__}; override TpppRunner.infer_step for custom formats"
        )

    @torch.inference_mode()
    def infer_step(self, data: Any) -> list[float]:
        if self.pipeline_schedule is None:
            return super().infer_step(data)

        inputs, _ = self._prepare_pipeline_batch(data)
        if self.pp_has_first_stage:
            pred = self.pipeline_schedule.eval(inputs)
        else:
            pred = self.pipeline_schedule.eval()
        return self._normalize_infer_output(pred)

    def close(self, timeout: float | None = None) -> bool:
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
        self.device_mesh = None
        self._tppp_groups_initialized = False
        return super().close(timeout=timeout)

    def state_dict(self, cls: type = dict) -> Mapping:
        state = cls(BaseRunner.state_dict(self, cls))
        state["tppp"] = cls(
            {
                "tp_degree": self.tp_degree,
                "pp_degree": self.pp_degree,
                "dp_degree": self.dp_degree,
            }
        )
        state["ema"] = self.ema.state_dict() if self.ema else None
        state["scheduler"] = self.scheduler.state_dict() if self.scheduler else None

        if len(self.model_parts) != 1:
            state["optimizer"] = self.optimizer.state_dict() if self.optimizer else None
            state["model_parts"] = [self.unwrap(model).state_dict() for model in self.model_parts]
            return state

        model_state_dict, optim_state_dict = self.checkpoint_manager.export_model_optimizer_state(
            model=self.model_parts[0],
            optimizer=self.optimizer,
            options_cls=StateDictOptions,
            strict=True,
        )
        state["model"] = model_state_dict
        if self.optimizer is not None:
            state["optimizer"] = optim_state_dict
        return state

    def load_model(self, state_dict: Mapping[str, Any] | list[Mapping[str, Any]], *args, **kwargs) -> None:
        if isinstance(state_dict, list):
            state_dicts = state_dict
            if len(state_dicts) != len(self.model_parts):
                raise ValueError(
                    "cannot load TPPP checkpoint: model_parts count mismatch: "
                    f"expected {len(self.model_parts)}, got {len(state_dicts)}"
                )
            for model, model_state_dict in zip(self.model_parts, state_dicts):
                self.unwrap(model).load_state_dict(model_state_dict, *args, **kwargs)
            return

        if len(self.model_parts) == 1:
            self.checkpoint_manager.load_model_state(
                model=self.model_parts[0],
                model_state_dict=state_dict,
                options_cls=StateDictOptions,
                strict=True,
            )
            return

        super().load_model(state_dict, *args, **kwargs)

    def adapt_checkpoint_payload_for_load(self, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        checkpoint = super().adapt_checkpoint_payload_for_load(checkpoint)
        saved_topology = self._validate_checkpoint_topology(checkpoint)
        current_topology = (self.tp_degree, self.pp_degree)
        if saved_topology == current_topology:
            return checkpoint

        if len(self.model_parts) != 1:
            raise ValueError(
                "cannot restore TP/PT degree change: degree change restore requires DCP state-dict API "
                "with a single local model part. "
                "Either keep tp/pp degrees unchanged, or restore with a single local model part."
            )

        transformed = dict(checkpoint)
        transformed["tppp"] = {
            "tp_degree": current_topology[0],
            "pp_degree": current_topology[1],
            "dp_degree": self.dp_degree,
        }
        return transformed

    def load_optimizer(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        if self.optimizer is None:
            return super().load_optimizer(state_dict, *args, **kwargs)

        if state_dict is None:
            raise ValueError(
                "cannot restore optimizer: checkpoint has no optimizer state\n"
                "Use `load_pretrained` for model-only checkpoints instead of `load_checkpoint`"
            )

        if len(self.model_parts) != 1:
            self.optimizer.load_state_dict(state_dict, *args, **kwargs)
            return

        self.checkpoint_manager.load_optimizer_state(
            model=self.model_parts[0],
            optimizer=self.optimizer,
            optimizer_state_dict=state_dict,
            options_cls=StateDictOptions,
            strict=True,
        )

    def _validate_checkpoint_topology(self, checkpoint: Mapping[str, Any]) -> tuple[int, int]:
        ckpt_topology = checkpoint.get("tppp")
        current = (self.tp_degree, self.pp_degree)
        if not isinstance(ckpt_topology, Mapping):
            return current

        saved = (int(ckpt_topology.get("tp_degree", current[0])), int(ckpt_topology.get("pp_degree", current[1])))
        if saved == current:
            return saved

        allow_degree_change = self.config.tppp.allow_degree_change
        if allow_degree_change:
            warn(
                "TP/PT degree changed across restart "
                f"(saved tp/pp={saved}, current tp/pp={current}). "
                "Attempting to restore with current runtime mapping.",
                RuntimeWarning,
                stacklevel=2,
            )
            return saved

        raise ValueError(
            "cannot restore checkpoint: TP/PT degree changed across restart "
            f"(saved tp/pp={saved}, current tp/pp={current}). "
            "Set `config.tppp.allow_degree_change=True` to proceed explicitly."
        )

    def build_datasampler(self, dataset, *, split: str, shuffle: bool):
        return utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.dp_degree, rank=self.dp_rank, shuffle=shuffle
        )

    def set_seed(self, seed: int | None = None, bias: int | bool | None = None) -> int:
        if bias is None:
            bias = self.dp_rank
        return super().set_seed(seed=seed, bias=bias)

    def reduce(self, tensor):
        dist.all_reduce(tensor, group=self.dp_group)
        tensor = tensor / max(self.dp_degree, 1)
        return tensor

    def reduce_loss_for_logging(self, loss: torch.Tensor | None) -> torch.Tensor | None:
        if self.pipeline_schedule is None:
            return super().reduce_loss_for_logging(loss)
        payload = torch.zeros((2,), dtype=torch.float32, device=self.device)
        is_reporter = self.pp_has_last_stage and self.tp_rank == 0
        if is_reporter:
            if loss is None:
                payload[0] = 0.0
                payload[1] = 0.0
            else:
                loss_value = loss.detach().to(dtype=torch.float32)
                if loss_value.ndim > 0:
                    loss_value = loss_value.mean()
                payload[0] = loss_value
                payload[1] = 1.0
            if self.dp_group is not None and self.dp_degree > 1:
                dist.all_reduce(payload, group=self.dp_group)

        source_rank = (self.pp_degree - 1) * self.tp_degree
        dist.broadcast(payload, src=source_rank)
        if payload[1].item() <= 0:
            return None
        return payload[0] / payload[1]

    @property
    def tp_degree(self) -> int:
        return self.topology.tp_degree

    @property
    def pp_degree(self) -> int:
        return self.topology.pp_degree

    @property
    def dp_degree(self) -> int:
        return self.topology.dp_degree

    @property
    def tp_rank(self) -> int:
        return self.topology.tp_rank

    @property
    def pp_rank(self) -> int:
        return self.topology.pp_rank

    @property
    def dp_rank(self) -> int:
        return self.topology.dp_rank
