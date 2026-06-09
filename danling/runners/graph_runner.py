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

import json
import os
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Optional
from warnings import warn

import torch
from torch import nn

from .torch_runner import TorchRunner

GraphTrainStepFn = Callable[[Any, Any, torch.Tensor], tuple[Any, torch.Tensor, tuple[Optional[torch.Tensor], ...]]]


class GraphRunner(TorchRunner):
    """
    Experimental runner that captures the default training micro-step as one graph unit.

    `GraphRunner` keeps `TorchRunner`'s outer loop, checkpointing, metrics,
    optimizer stepping, and override surface, but replaces the default
    `train_step` body with a forward/loss/backward graph. The graph returns
    gradients explicitly via `torch.autograd.grad`; the runner then accumulates
    those gradients on parameters and delegates optimizer flushing to
    `TorchRunner.step()`.

    This is intentionally narrower than `ParallelRunner`: it is the place for
    graph-level compile/cache experiments without changing the eager runner
    contract for existing users.

    Precision note: `GraphRunner` inherits `TorchRunner.train_context()`, so
    autocast wraps the captured graph step. Standard torch autocast is expected
    to work; FP8/Transformer Engine autocast with `torch.compile` remains
    experimental and model-dependent.
    """

    _graph_cache_loaded: bool
    _graph_cache_saved: bool

    def __post_init__(self) -> None:
        self._graph_cache_loaded = False
        self._graph_cache_saved = False
        self._validate_graph_runtime_config()
        super().__post_init__()

    def _validate_graph_runtime_config(self) -> None:
        if self.distributed:
            raise NotImplementedError("GraphRunner does not yet support distributed execution")
        memory_policy = self.config.compile.get("memory_policy")
        if memory_policy is not None and str(memory_policy).strip().lower() != "default":
            raise NotImplementedError(
                "GraphRunner currently supports only `compile.memory_policy=None` or `'default'`; "
                "graph-level activation remat/offload policies need an FX pass pipeline."
            )
        if self.config.compile.get("precompile_artifact_dir") and not self.config.compile.get("enabled", False):
            raise ValueError("`compile.precompile_artifact_dir` requires `compile.enabled=True` for GraphRunner")

    def materialize_model(self) -> None:
        """
        Move the model to device without compiling it as a standalone module.

        GraphRunner compiles the forward/loss/backward step, so compiling the
        model here would double-wrap the hot path and obscure the graph cache
        boundary.
        """
        if self.model is None:
            raise ValueError("cannot materialize model: model is not initialized")
        self.model = self.model.to(self.device)
        if self.fp8_enabled:
            self.apply_fp8_module_policy_to_model_parts()
        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def _split_train_batch(self, data: Any) -> tuple[Any, Any]:
        if isinstance(data, Mapping):
            return data["input"], data.get("target")
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return data[0], data[1] if len(data) > 1 else None
        return data, None

    def _model_artifact_signature(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("cannot fingerprint graph artifact: model is not initialized")
        model = self.unwrap(self.model)
        return {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameters": [
                (name, tuple(parameter.shape), str(parameter.dtype), bool(parameter.requires_grad))
                for name, parameter in model.named_parameters()
            ],
            "buffers": [(name, tuple(buffer.shape), str(buffer.dtype)) for name, buffer in model.named_buffers()],
        }

    def _graph_runtime_signature(self) -> dict[str, Any]:
        device: dict[str, Any] = {"type": self.device.type}
        if self.device.type == "cuda" and torch.cuda.is_available():
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            device["capability"] = tuple(torch.cuda.get_device_capability(device_index))
            device["name"] = torch.cuda.get_device_name(device_index)
        return {
            "device": device,
            "world_size": self.world_size,
        }

    def _compile_artifact_signature(self) -> dict[str, Any]:
        compile_config = dict(self.config.compile)
        compile_config.pop("precompile_artifact_dir", None)
        return compile_config

    def graph_artifact_fingerprint(self, extra: Mapping[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "stack": "graph",
            "model": self._model_artifact_signature(),
            "runtime": self._graph_runtime_signature(),
        }
        if extra is not None:
            payload["extra"] = dict(extra)
        return self.compiler.artifact_fingerprint(payload)

    def graph_cache_artifact_path(self) -> Path | None:
        artifact_dir = self.compiler.precompile_artifact_dir
        if artifact_dir is None:
            return None
        return Path(artifact_dir) / f"graph-train-step-{self.graph_artifact_fingerprint()}.pt2cache"

    def graph_cache_metadata_path(self, artifact_path: Path | None = None) -> Path | None:
        if artifact_path is None:
            artifact_path = self.graph_cache_artifact_path()
        if artifact_path is None:
            return None
        return artifact_path.with_suffix(f"{artifact_path.suffix}.json")

    def graph_cache_metadata(self) -> dict[str, Any]:
        return {
            "format": "danling.graph_cache.v1",
            "fingerprint": self.graph_artifact_fingerprint(),
            "torch": torch.__version__,
            "compile": self._compile_artifact_signature(),
            "runtime": self._graph_runtime_signature(),
            "model": self._model_artifact_signature(),
        }

    def _validate_graph_cache_metadata(self, artifact_path: Path) -> bool:
        metadata_path = self.graph_cache_metadata_path(artifact_path)
        if metadata_path is None or not metadata_path.is_file():
            return True
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid GraphRunner compile cache metadata: {metadata_path}") from exc
        expected = self.graph_artifact_fingerprint()
        found = metadata.get("fingerprint")
        if found == expected:
            return True
        warn(
            "skipping GraphRunner compile cache artifact "
            f"{artifact_path}: metadata fingerprint {found!r} does not match expected {expected!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    def _load_graph_cache_artifacts(self) -> None:
        if self._graph_cache_loaded:
            return
        self._graph_cache_loaded = True
        path = self.graph_cache_artifact_path()
        if path is None or not path.is_file():
            return
        if not self._validate_graph_cache_metadata(path):
            return
        load_cache_artifacts = getattr(torch.compiler, "load_cache_artifacts", None)
        if load_cache_artifacts is None:
            raise RuntimeError(
                "cannot load GraphRunner compile cache artifact "
                f"from {path}: this PyTorch build does not expose "
                "`torch.compiler.load_cache_artifacts`; delete the cache file "
                "or upgrade PyTorch"
            )
        load_cache_artifacts(path.read_bytes())

    def _save_graph_cache_artifacts(self) -> None:
        if self._graph_cache_saved:
            return
        self._graph_cache_saved = True
        path = self.graph_cache_artifact_path()
        if path is None:
            return
        save_cache_artifacts = getattr(torch.compiler, "save_cache_artifacts", None)
        if save_cache_artifacts is None:
            raise RuntimeError("this PyTorch build does not expose `torch.compiler.save_cache_artifacts`")
        artifacts = save_cache_artifacts()
        if artifacts is None:
            return
        data, _cache_info = artifacts
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
        metadata_path = self.graph_cache_metadata_path(path)
        if metadata_path is not None:
            metadata_tmp_path = metadata_path.with_name(f"{metadata_path.name}.tmp-{os.getpid()}")
            metadata_tmp_path.write_text(
                json.dumps(self.graph_cache_metadata(), sort_keys=True, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(metadata_tmp_path, metadata_path)

    def _graph_train_step_parameters(self) -> tuple[nn.Parameter, ...]:
        params = tuple(parameter for parameter in self.iter_optimizer_parameters() if parameter.requires_grad)
        if not params:
            raise ValueError("cannot build graph train step: model has no trainable parameters")
        return params

    def build_graph_train_step(self, params: Sequence[nn.Parameter]) -> GraphTrainStepFn:
        """
        Build the callable captured by `GraphRunner.train_step`.

        Override this when the default graph body is too narrow but the outer
        runner behavior should stay intact. The returned callable receives
        `(inputs, target, loss_scale)` and must return `(pred, loss, grads)`,
        where `grads` lines up with `params`.
        """
        if self.model is None:
            raise ValueError("cannot build graph train step: model is not initialized")
        if self.criterion is None:
            raise ValueError("cannot build graph train step: criterion is not initialized")

        model = self.model
        criterion = self.criterion
        params = tuple(params)

        def forward_loss_grad(inputs: Any, target: Any, loss_scale: torch.Tensor):
            pred = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
            loss = criterion(pred, target)
            if loss is None:
                raise ValueError("cannot run train_step: criterion did not produce a loss")
            scaled_loss = loss * loss_scale.to(device=loss.device, dtype=loss.dtype)
            grads = torch.autograd.grad(
                scaled_loss,
                params,
                allow_unused=True,
                materialize_grads=False,
            )
            return pred, loss, grads

        return forward_loss_grad

    def _compile_graph_train_step(self, fn: GraphTrainStepFn) -> GraphTrainStepFn:
        if not self.compiler.enabled:
            return fn
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        self._load_graph_cache_artifacts()
        return torch.compile(fn, **self.compiler.kwargs)

    @cached_property
    def graph_train_step_params(self) -> tuple[nn.Parameter, ...]:
        """Trainable parameters captured by `graph_train_step`."""
        return self._graph_train_step_parameters()

    @cached_property
    def graph_train_step(self) -> GraphTrainStepFn:
        """Compiled or eager graph callable used by `train_step`."""
        return self._compile_graph_train_step(self.build_graph_train_step(self.graph_train_step_params))

    def _accumulate_graph_gradients(
        self,
        params: Sequence[nn.Parameter],
        grads: tuple[torch.Tensor | None, ...],
    ) -> None:
        for parameter, grad in zip(params, grads):
            if grad is None:
                continue
            if parameter.grad is None:
                parameter.grad = grad
            else:
                parameter.grad.add_(grad)

    def train_step(self, data: Any) -> tuple[Any, torch.Tensor | None]:
        """
        Run one graph-backed training micro-step.

        The default batch contract is the same as `TorchRunner.train_step`, but
        forward, criterion, and backward-gradient construction are captured as a
        single callable. Metrics, accumulation counters, optimizer stepping,
        logging, and checkpoint cadence stay in the Python runner loop.
        """
        data = self.to_device(data)
        with self.train_context():
            inputs, target = self._split_train_batch(data)
            loss_scale = torch.tensor(float(self._loss_scale_for_backward()), device=self.device)
            pred, loss, grads = self.graph_train_step(inputs, target, loss_scale)
            self._accumulate_graph_gradients(self.graph_train_step_params, grads)
            self._save_graph_cache_artifacts()
            if self.metrics is not None and pred is not None and target is not None:
                self.metrics.update(pred, target)
            self.step()
        return pred, loss
