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
import shutil
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

from chanfig import FlatDict, NestedDict

from danling.metrics import AverageMeter
from danling.utils import RoundDict, catch

from ..config import RunnerConfig
from ..utils import RunnerMode, format_result, on_main_process


class ResultMixin:
    """Cold-path result formatting/persistence helpers extracted from BaseRunner."""

    config: RunnerConfig
    results: RoundDict
    meters: Any
    metrics: Any | None = None
    train_state: Any
    dataloaders: Mapping[str, Any]
    mode: RunnerMode
    writer: Any | None = None
    name: str
    timestamp: str
    dir: str
    save: Any

    @property
    def evaluate_splits(self) -> list[str]:
        raise NotImplementedError

    @property
    def epochs(self) -> int | None:
        raise NotImplementedError

    @property
    def id(self) -> str:
        raise NotImplementedError

    @cached_property
    def score_split(self) -> str | None:
        """Split used for best-score selection."""
        if "score_split" in self.config:
            return self.config.score_split

        splits = self.evaluate_splits
        if not splits:
            return None
        for split in splits:
            if split.lower().startswith("val"):
                return split
        return splits[0]

    @property
    def scores(self) -> FlatDict | None:
        """Index-to-score mapping extracted from `score_split/score_name`."""
        if not self.results:
            return None

        score_split = self.score_split
        if score_split is None:
            return None

        scores = FlatDict()
        for index, result in self.results.items():
            if score_split not in result:
                continue
            split_result = result[score_split]
            if not isinstance(split_result, Mapping):
                continue
            if self.config.score_name not in split_result:
                continue
            scores[index] = split_result[self.config.score_name]

        return scores or None

    @property
    def best_index(self) -> int:
        """Best result index according to configured score metric."""
        if not self.scores:
            return 0

        scores = self.scores
        indices = list(scores.keys())
        reducer = min if self.config.score_name == "loss" else max
        return reducer(reversed(indices), key=scores.get)

    @property
    def latest_result(self) -> RoundDict | None:
        """Most recent appended result row."""
        if not self.results:
            return None

        latest_index = next(reversed(self.results))
        latest = self.results[latest_index]

        ret = RoundDict(latest)
        ret["index"] = latest_index
        return ret

    @property
    def best_result(self) -> RoundDict | None:
        """Best result row according to configured score metric."""
        if not self.results:
            return None

        best_index = self.best_index
        best = self.results[best_index]

        ret = RoundDict(best)
        ret["index"] = best_index
        return ret

    @property
    def latest_score(self) -> float | None:
        """Latest scalar score."""
        scores = self.scores
        if not scores:
            return None

        latest_index = next(reversed(scores))
        return scores[latest_index]

    @property
    def best_score(self) -> float | None:
        """Best scalar score."""
        if not self.scores:
            return None

        return self.scores[self.best_index]

    @property
    def is_best(self) -> bool:
        """Whether latest score matches current best score."""
        if not self.results:
            return True

        try:
            return abs(self.latest_score - self.best_score) < 1e-7  # type: ignore[operator]
        except TypeError:
            return True

    def _merge_result(self, meter_result: Mapping[str, Any], metric_result: Mapping[str, Any]) -> RoundDict:
        merged = RoundDict(meter_result)
        for key, value in metric_result.items():
            if isinstance(value, Mapping) and len(value) == 1:
                value = next(iter(value.values()))
            merged[key] = value
        return merged

    def get_epoch_result(self) -> RoundDict:
        meter_result = self.meters.average()
        if self.metrics is None:
            return RoundDict(meter_result)
        return self._merge_result(meter_result, self.metrics.average())

    def get_step_result(self) -> RoundDict:
        meter_result = self.meters.value()
        if self.metrics is None:
            return RoundDict(meter_result)
        return self._merge_result(meter_result, self.metrics.value())

    def append_result(self, result: RoundDict | Mapping[str, Any], index: int | None = None) -> None:
        if index is None:
            index = self.train_state.epoch

        if not isinstance(result, RoundDict):
            result = RoundDict(result)

        if index in self.results:
            self.results[index].merge(result)
        else:
            self.results[index] = result

    def step_log(self, split: str, iteration: int, length: int | str | None = None) -> RoundDict:
        if length is None:
            try:
                length = len(self.dataloaders[split]) - 1
            except (TypeError, NotImplementedError):
                length = "∞"

        result = self.get_step_result()
        print(self.format_step_result(result, split, iteration, length))

        if self.mode == RunnerMode.train:
            self.write_result(result, split)

        return result

    def format_epoch_result(
        self,
        result: RoundDict[str, Any],
        epochs: int | None = None,
        total_epochs: int | None = None,
    ) -> str:
        epochs = self.train_state.epoch if epochs is None else epochs
        total_epochs = self.epochs if total_epochs is None else total_epochs

        prefix = ""
        if total_epochs is not None:
            prefix = f"epoch [{epochs}/{total_epochs - 1}]"

        return f"{prefix}{self.format_result(result)}"

    def format_step_result(self, result: RoundDict[str, Any], split: str, steps: int, length: int | str) -> str:
        if self.mode == RunnerMode.train:
            prefix = f"training on {split}"
        elif self.mode == RunnerMode.evaluate:
            prefix = f"evaluating on {split}"
        elif self.mode == RunnerMode.infer:
            prefix = f"inferring on {split}"
        else:
            prefix = f"running in {self.mode} on {split}"

        return f"{prefix} [{steps}/{length}]\t{self.format_result(result)}"

    def format_result(self, result: RoundDict[str, Any], format_spec: str = ".4f") -> str:
        return format_result(result, format_spec=format_spec)

    def write_result(self, result: RoundDict[str, Any], split: str, steps: int | None = None) -> None:
        steps = self.train_state.global_step if steps is None else steps

        iterator = result.all_items() if isinstance(result, NestedDict) else result.items()
        for name, score in iterator:
            tag = str(name).replace(".", "/")

            if isinstance(score, AverageMeter):
                score = score.avg

            if isinstance(score, Mapping):
                for sub_name, sub_score in score.items():
                    self.write_score(f"{tag}/{sub_name}", sub_score, split, steps)
                continue

            if isinstance(score, Sequence) and not isinstance(score, (str, bytes)):
                for idx, sub_score in enumerate(score):
                    self.write_score(f"{tag}/{idx}", sub_score, split, steps)
                continue

            self.write_score(tag, score, split, steps)

    def write_score(self, name: str, score: float, split: str, steps: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(f"{split}/{name}", score, steps)

    @catch
    @on_main_process
    def save_result(self) -> None:
        if not self.latest_result:
            return
        payload = {
            "name": self.name,
            "id": self.id,
            "timestamp": self.timestamp,
            "results": round(self.results, 8),
        }
        self.save(payload, os.path.join(self.dir, "results.json"), indent=4)

        latest = round(self.latest_result, 8)
        latest_payload = {"name": self.name, "id": self.id, "timestamp": self.timestamp}
        latest_payload.update(dict(latest))

        latest_path = os.path.join(self.dir, "latest.json")
        self.save(latest_payload, latest_path, indent=4)

        if self.is_best:
            shutil.copy(latest_path, os.path.join(self.dir, "best.json"))
