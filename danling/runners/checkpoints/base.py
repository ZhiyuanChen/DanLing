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

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from danling.utils import load


class CheckpointManager(ABC):
    """Backend-agnostic checkpoint management contract."""

    _VALID_ASYNC_MODES = frozenset({"disabled", "async", "async_with_pinned_mem"})

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    @abstractmethod
    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
    ) -> None:
        """Persist one checkpoint update."""

    def checkpoint_async_mode(self) -> str:
        mode = self.runner.config.get("checkpoint.async_mode")
        if mode is None:
            mode = self.runner.config.get("checkpoint.async_enabled", True)

        if isinstance(mode, bool):
            return "async" if mode else "disabled"

        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized in {"true", "on", "enable", "enabled"}:
            return "async"
        if normalized in {"false", "off", "disable"}:
            return "disabled"
        if normalized not in self._VALID_ASYNC_MODES:
            valid = ", ".join(sorted(self._VALID_ASYNC_MODES))
            raise ValueError(f"Unknown checkpoint.async_mode: {mode!r}. Valid options are: {valid}")
        return normalized

    def checkpoint_async_enabled(self) -> bool:
        return self.checkpoint_async_mode() != "disabled"

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until pending checkpoint work is drained."""

        return True

    def maybe_wait_for_staging(self, timeout: float | None = None) -> bool:
        """Wait for async staging completion when supported by backend."""

        del timeout
        return True

    def close(self, timeout: float | None = None) -> bool:
        """Finalize checkpoint I/O before runner shutdown."""

        return self.wait(timeout=timeout)

    def load_model_state(
        self,
        *,
        model: Any,
        model_state_dict: Mapping[str, Any],
        options_cls: Any = None,
        strict: bool = True,
    ) -> None:
        """Load model state for the active checkpoint backend."""
        raise NotImplementedError

    def load_optimizer_state(
        self,
        *,
        model: Any,
        optimizer: Any,
        optimizer_state_dict: Mapping[str, Any],
        options_cls: Any = None,
        strict: bool = True,
    ) -> None:
        """Load optimizer state for the active checkpoint backend."""
        raise NotImplementedError

    def load_checkpoint(self, checkpoint: bytes | str | os.PathLike) -> dict[str, Any]:
        """Load a checkpoint payload from backend storage."""

        checkpoint_path = self.resolve_checkpoint_path(checkpoint)
        checkpoint_state = self.runner.load(checkpoint_path)
        if not isinstance(checkpoint_state, Mapping):
            raise ValueError(
                "invalid checkpoint payload: expected a mapping, "
                f"got {type(checkpoint_state).__name__}: {checkpoint_state!r}"
            )
        return dict(checkpoint_state)

    @classmethod
    def resolve_checkpoint_path(cls, checkpoint: bytes | str | os.PathLike) -> str:
        """Resolve checkpoint input path for this backend."""

        del cls
        return os.fsdecode(checkpoint)

    @classmethod
    def is_checkpoint_path(cls, checkpoint: bytes | str | os.PathLike) -> bool:
        """Return whether path points to a backend checkpoint payload."""

        checkpoint_path = cls.resolve_checkpoint_path(checkpoint)
        return os.path.isfile(checkpoint_path)

    @classmethod
    def read_config(cls, checkpoint: bytes | str | os.PathLike) -> Mapping[str, Any]:
        """Read runner config from checkpoint payload path."""

        checkpoint_path = cls.resolve_checkpoint_path(checkpoint)
        checkpoint_state = load(checkpoint_path)
        if not isinstance(checkpoint_state, Mapping):
            raise ValueError(
                "cannot read runner config: checkpoint payload must be a mapping, "
                f"got {type(checkpoint_state).__name__}: {checkpoint_state!r}"
            )
        runner_config = checkpoint_state.get("runner")
        if runner_config is None:
            raise ValueError(
                "cannot read runner config: checkpoint is missing key 'runner'; "
                "use from_pretrained(...) for model-only checkpoints"
            )
        return runner_config

    def export_model_optimizer_state(
        self,
        *,
        model: Any,
        optimizer: Any,
        options_cls: Any = None,
        strict: bool = True,
    ) -> tuple[Any, Any]:
        """Export model/optimizer state for checkpoint payload composition."""

        del options_cls, strict
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict() if optimizer is not None else {}
        return model_state, optimizer_state

    def should_persist_checkpoint(self, *, epochs: int, last_step: bool = False) -> bool:
        if self.runner.config.get("checkpoint.load_only", False):
            return False

        if last_step:
            return True

        interval = self.runner.checkpoint_interval
        if self.runner.is_step_mode:
            progress_index = self.runner.train_state.global_step
        else:
            progress_index = epochs + 1

        if interval <= 0:
            return False

        return progress_index > 0 and progress_index % interval == 0

    @staticmethod
    def _to_model_only_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        model_only: dict[str, Any] = {}
        for key in ("model", "model_parts", "module", "tppp", "dtpp"):
            if key in payload:
                model_only[key] = payload[key]
        return model_only or dict(payload)

    @staticmethod
    def _resolve_export_dtype(dtype_name: str):
        import torch  # pylint: disable=C0415

        normalized = str(dtype_name).strip().lower().replace("-", "_")
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float64": torch.float64,
            "fp64": torch.float64,
            "double": torch.float64,
        }
        if normalized not in mapping:
            valid = ", ".join(sorted(mapping))
            raise ValueError(f"Unknown checkpoint.export_dtype: {dtype_name!r}. Valid options are: {valid}")
        return mapping[normalized]

    def _cast_payload_tensors_dtype(self, payload: Any, dtype: Any) -> Any:
        import torch  # pylint: disable=C0415

        if torch.is_tensor(payload):
            try:
                return payload.to(dtype=dtype)
            except Exception:  # pragma: no cover - dtype cast support depends on tensor/storage implementations.
                return payload

        if isinstance(payload, Mapping):
            return {key: self._cast_payload_tensors_dtype(value, dtype) for key, value in payload.items()}

        if isinstance(payload, list):
            return [self._cast_payload_tensors_dtype(value, dtype) for value in payload]
        if isinstance(payload, tuple):
            return tuple(self._cast_payload_tensors_dtype(value, dtype) for value in payload)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return [self._cast_payload_tensors_dtype(value, dtype) for value in payload]
        return payload

    def build_checkpoint_payload(self, *, last_step: bool = False) -> Mapping[str, Any]:
        payload = self.runner.state_dict()
        if not isinstance(payload, Mapping):
            raise ValueError(f"runner.state_dict() must return a mapping, got {type(payload).__name__}")

        if not last_step or not self.runner.config.get("checkpoint.last_save_model_only", False):
            return self.runner.adapt_checkpoint_payload_for_save(payload)

        model_only_payload = self._to_model_only_payload(payload)

        export_dtype_name = self.runner.config.get("checkpoint.export_dtype")
        if export_dtype_name is None:
            export_dtype_name = self.runner.config.get("export_dtype")
        if export_dtype_name is None:
            return self.runner.adapt_checkpoint_payload_for_save(model_only_payload)

        dtype = self._resolve_export_dtype(str(export_dtype_name))
        return self.runner.adapt_checkpoint_payload_for_save(
            self._cast_payload_tensors_dtype(model_only_payload, dtype)
        )

    def resolve_archive_name(self, epochs: int, *, suffix: str = "") -> str | None:
        """Return periodic archive checkpoint name for the active train mode."""

        is_step_mode = self.runner.is_step_mode
        checkpoint_interval = self.runner.checkpoint_interval
        if is_step_mode:
            archive_index = self.runner.train_state.global_step
            should_archive = checkpoint_interval > 0 and archive_index > 0 and archive_index % checkpoint_interval == 0
        else:
            archive_index = epochs
            should_archive = checkpoint_interval > 0 and (archive_index + 1) % checkpoint_interval == 0
        if not should_archive:
            return None

        if is_step_mode:
            return f"ckpt-s{archive_index:012d}{suffix}"
        return f"ckpt-e{archive_index:06d}{suffix}"
