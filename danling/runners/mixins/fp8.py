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

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any
from warnings import warn

from torch import nn

from ..config import RunnerConfig


def normalize_precision_name(precision: str | None) -> str:
    if precision is None:
        return ""
    return str(precision).strip().lower().replace("-", "_")


def is_fp8_precision(precision: str | None) -> bool:
    normalized = normalize_precision_name(precision)
    return normalized in {
        "fp8",
        "float8",
        "fp8_e4m3",
        "fp8_e4m3fn",
        "fp8_e4m3fnuz",
        "fp8_e5m2",
        "float8_e4m3",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
    }


class Fp8Mixin:
    """Mixin providing FP8 recipe and module-policy hooks."""

    config: RunnerConfig
    model: nn.Module | None = None
    model_parts: list[nn.Module] | None = None

    _fp8_enabled: bool = False
    _fp8_recipe: Any | None = None

    def setup_fp8(self) -> None:
        enabled = self.should_enable_fp8()
        self._fp8_enabled = enabled
        self._fp8_recipe = None
        if not enabled:
            return

        precision = self.config.get("precision")
        if not is_fp8_precision(precision):
            self.config.precision = "fp8"

        self._fp8_recipe = self.build_fp8_recipe()
        self.apply_fp8_module_policy_to_model_parts()

    def should_enable_fp8(self) -> bool:
        fp8_cfg = self.config.get("fp8")
        if isinstance(fp8_cfg, Mapping) and "enabled" in fp8_cfg:
            return bool(fp8_cfg.get("enabled"))
        return is_fp8_precision(self.config.get("precision"))

    def build_fp8_recipe(self) -> Any | None:
        fp8_cfg = self.config.get("fp8")
        if not isinstance(fp8_cfg, Mapping):
            return None

        recipe = fp8_cfg.get("recipe")
        if recipe is not None:
            return recipe

        recipe_cls = fp8_cfg.get("recipe_cls")
        if recipe_cls is None:
            return None

        recipe_kwargs = fp8_cfg.get("recipe_kwargs", {})
        if not isinstance(recipe_kwargs, Mapping):
            raise ValueError("config.fp8.recipe_kwargs must be a mapping")

        if isinstance(recipe_cls, str):
            try:
                import transformer_engine.common.recipe as te_recipe  # pylint: disable=C0415
            except Exception as exc:
                warn(
                    "FP8 recipe class is configured but Transformer Engine is unavailable; "
                    f"falling back to default recipe ({exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

            recipe_ctor = getattr(te_recipe, recipe_cls, None)
            if recipe_ctor is None:
                raise ValueError(f"Unknown FP8 recipe class: {recipe_cls!r}")
            return recipe_ctor(**dict(recipe_kwargs))

        if callable(recipe_cls):
            return recipe_cls(**dict(recipe_kwargs))

        raise ValueError("config.fp8.recipe_cls must be a callable or recipe class name")

    def apply_fp8_module_policy_to_model_parts(self) -> None:
        model_parts = list(self.model_parts or [])
        if not model_parts:
            if self.model is None:
                return
            transformed = self.apply_fp8_module_policy(self.model, recipe=self._fp8_recipe)
            self.model = self.model if transformed is None else transformed
            return

        transformed_parts: list[nn.Module] = []
        for module in model_parts:
            transformed = self.apply_fp8_module_policy(module, recipe=self._fp8_recipe)
            transformed_parts.append(module if transformed is None else transformed)

        self.model_parts = transformed_parts
        self.model = self.model_parts[0]

    def apply_fp8_module_policy(self, module: nn.Module, *, recipe: Any | None = None) -> nn.Module:
        return module

    @contextmanager
    def fp8_autocast(self):
        fp8_context = self.create_fp8_autocast_context()
        if fp8_context is None:
            raise RuntimeError(
                "FP8 precision requested but no FP8 autocast backend is available. "
                "Install Transformer Engine or override `create_fp8_autocast_context`."
            )
        with fp8_context:
            yield

    def create_fp8_autocast_context(self):
        try:
            import transformer_engine.pytorch as te  # pylint: disable=C0415
        except Exception:
            return None

        kwargs: dict[str, Any] = {"enabled": True}
        if self._fp8_recipe is not None:
            kwargs["fp8_recipe"] = self._fp8_recipe

        fp8_cfg = self.config.get("fp8")
        if isinstance(fp8_cfg, Mapping):
            fp8_group = fp8_cfg.get("group")
            if fp8_group is not None:
                kwargs["fp8_group"] = fp8_group
        return te.fp8_autocast(**kwargs)

    @property
    def fp8_enabled(self) -> bool:
        return bool(self._fp8_enabled)

    @property
    def fp8_recipe(self) -> Any | None:
        return self._fp8_recipe
