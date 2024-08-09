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

import os
import random
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from time import time
from typing import Any, Callable, Iterator, Tuple
from warnings import warn

import torch
import torch.distributed
from chanfig import NestedDict
from torch import distributed as dist
from torch import nn, optim, utils
from torch.backends import cudnn
from tqdm import tqdm

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from danling.data import to_device
from danling.optim import OPTIMIZERS, SCHEDULERS
from danling.optim.optimizer import OptimizerContainer
from danling.utils import RoundDict, catch

from .base_runner import BaseRunner
from .checkpoints import FileCheckpointManager, TorchDistributedCheckpointManager
from .compile import maybe_compile_loss, maybe_compile_model, maybe_enable_ddp_optimizer
from .mixins import Fp8Mixin
from .utils import RunnerMode, get_precision, on_main_process


class TorchRunner(Fp8Mixin, BaseRunner):
    r"""
    PyTorch-native runner for training, evaluation, and inference.

    The runner is designed for basic DDP-style training/evaluation/inference with
    a single local model module.
    """

    model: nn.Module
    ema: nn.Module | None = None
    criterion: Callable | None = None
    optimizer: optim.Optimizer | None = None
    scheduler: Any | None = None
    optimizer_container: OptimizerContainer | None = None

    def __post_init__(self):
        if self.model is None:
            raise ValueError("cannot initialize TorchRunner: model is not initialized")
        if self.datasets and not self.dataloaders:
            self.build_dataloaders()
        self.materialize_model()
        self.setup_fp8()
        if self.criterion is not None:
            self.criterion = maybe_compile_loss(self.criterion, self.config)
        self.build_optimizer()
        self.build_scheduler()
        self._bind_optimizer_container()

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        backend = self.config.get("backend", os.getenv("BACKEND"))
        init_method = self.config.get("init_method", os.getenv("INIT_METHOD"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        dist_ready = dist.is_available() and dist.is_initialized()
        if world_size > 1 and not dist_ready:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend, init_method, world_size=world_size, rank=rank)
            dist_ready = bool(dist.is_available() and dist.is_initialized())

        if dist_ready and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        if dist_ready and self.world_size > 1:
            object_list = [self.timestamp]
            dist.broadcast_object_list(object_list)
            self.timestamp = str(object_list[0])

        restart_count = os.getenv("TORCHELASTIC_RESTART_COUNT")
        if restart_count is not None:
            self.elastic_state.restart_count = int(restart_count)

        checkpoint_backend = self.config.checkpoint.backend.lower()
        if checkpoint_backend == "auto":
            checkpoint_backend = "dcp" if self.world_size > 1 else "file"
            self.config.checkpoint.backend = checkpoint_backend

        if checkpoint_backend == "dcp":
            self.checkpoint_manager = TorchDistributedCheckpointManager(self)
        elif checkpoint_backend == "file":
            self.checkpoint_manager = FileCheckpointManager(self)
        else:
            raise ValueError(
                f"invalid checkpoint backend: {checkpoint_backend!r}. Expected one of: 'auto', 'file', 'dcp'."
            )

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        r"""
        Set up TensorBoard SummaryWriter.
        """

        from torch.utils.tensorboard.writer import SummaryWriter  # pylint: disable=C0415

        if "log_dir" not in kwargs:
            kwargs["log_dir"] = os.path.join(self.dir, "tensorboard", self.id)

        self.writer = SummaryWriter(*args, **kwargs)
        self.writer.add_scalar = catch(OSError, verbose=False)(self.writer.add_scalar)

    def set_seed(self, seed: int | None = None, bias: int | bool | None = None) -> int:
        r"""
        Set up random seed.

        Args:
            seed: Random seed to set.
                Defaults to `self.config.seed` (`config.seed`).

            bias: Make the seed different for each processes.
                This is used to ensure the data augmentation are applied differently on every processes.
                Defaults to `self.rank`.
                Set to `False` to disable this feature.
        Returns:
            Random seed set.
        """

        base_seed = seed if seed is not None else self.config.seed  # type: ignore[assignment]
        if base_seed is None:
            base_seed = random.randint(0, 2**32 - 1)
            if self.distributed and dist.is_initialized():
                object_list = [base_seed]
                dist.broadcast_object_list(object_list)
                base_seed = object_list[0]
        base_seed = int(base_seed)
        # Keep `config.seed` as the global/base seed (before per-rank bias).
        self.config.seed = base_seed

        process_seed = base_seed
        bias = self.rank if bias is None else bias
        if bias:
            process_seed += int(bias)

        torch.manual_seed(process_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(process_seed)
        if np_random is not None:
            np_random.seed(process_seed)
        random.seed(process_seed)
        self.rng_state.python = random.getstate()
        self.rng_state.numpy = np_random.get_state() if np_random is not None else None
        self.rng_state.torch_cpu = torch.get_rng_state()
        if torch.cuda.is_available():
            self.rng_state.torch_cuda = torch.cuda.get_rng_state_all()
        else:
            self.rng_state.torch_cuda = None
        return process_seed

    def set_deterministic(self) -> None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def _loader_length(loader: Any) -> int | None:
        try:
            return len(loader)
        except (TypeError, NotImplementedError):
            return None

    @staticmethod
    def _set_loader_epoch(loader: Any, epoch: int) -> None:
        batch_sampler = getattr(loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)  # type: ignore[union-attr]

    def train(
        self,
        train_splits: list[str] | None = None,
        evaluate_splits: list[str] | None = None,
    ) -> RoundDict:
        """
        Run the full training workflow.

        Args:
            train_splits: Optional training splits. When `None`, use `self.train_splits`.
            evaluate_splits: Optional evaluation splits. When `None`, use `self.evaluate_splits`.

        Returns:
            Aggregated runner results (`self.results`).

        Notes:
            Dispatches to `train_steps` when `self.is_step_mode` is true; otherwise to `train_epochs`.
        """

        if train_splits is not None:
            train_splits = sorted(set(train_splits).intersection(self.train_splits))
        else:
            train_splits = self.train_splits
        if not train_splits:
            raise ValueError("cannot start training: no valid training split was resolved")

        if evaluate_splits is not None:
            evaluate_splits = sorted(set(evaluate_splits).intersection(self.evaluate_splits))
        else:
            evaluate_splits = self.evaluate_splits

        print(f"train: splits={train_splits}")
        print(f"evaluate: splits={evaluate_splits}")
        if self.is_step_mode:
            return self.train_steps(train_splits=train_splits, evaluate_splits=evaluate_splits)
        return self.train_epochs(train_splits=train_splits, evaluate_splits=evaluate_splits)

    def train_epochs(
        self,
        train_splits: list[str] | None = None,
        evaluate_splits: list[str] | None = None,
    ) -> RoundDict:
        """
        Run epoch-mode training until `self.epochs` is reached.

        Args:
            train_splits: Training splits for each epoch.
            evaluate_splits: Evaluation splits after each epoch.

        Returns:
            Aggregated runner results (`self.results`).
        """
        if train_splits is None:
            train_splits = self.train_splits
        if evaluate_splits is None:
            evaluate_splits = self.evaluate_splits

        total_epochs = self.epochs
        if total_epochs is None:
            raise ValueError("cannot run epoch-mode training: config.epochs is not set")
        print(f"train: epoch mode start epoch={self.train_state.epoch} total_epochs={total_epochs}")
        checkpoint_cadence = self.checkpoint_interval
        early_stop_counter = 0
        patience = self.patience
        for epoch in range(self.train_state.epoch, total_epochs):
            self.train_state.epoch = epoch
            result = RoundDict()
            for split in train_splits:
                result[split] = self.train_epoch(split)
            for split in evaluate_splits:
                result[split] = self.evaluate_epoch(split)
            self.append_result(result, index=epoch)
            print(self.format_epoch_result(result, epochs=epoch, total_epochs=total_epochs))
            self.save_result()
            self.train_state.epoch = epoch + 1
            if checkpoint_cadence > 0 and self.train_state.epoch % checkpoint_cadence == 0:
                self.save_checkpoint(epochs=epoch)
            early_stop_counter = 0 if self.is_best else early_stop_counter + 1
            if early_stop_counter > patience:
                print("train: early-stop triggered")
                break
        self.save_checkpoint(last_step=True)
        return self.results

    def train_epoch(self, split: str = "train") -> RoundDict:
        """
        Run one full dataloader pass for a training split.

        Args:
            split: Training split name.

        Returns:
            Epoch-level metric mapping for this split.
        """

        self.mode = RunnerMode.train
        self.split = split
        loader = self.dataloaders[split]
        length = len(loader) - 1
        last_print_iteration = -1
        self.meters.reset()
        if self.train_metrics is not None:
            self.metrics = self.train_metrics
        if self.metrics is not None:
            self.metrics.reset()
        self._set_loader_epoch(loader, self.train_state.epoch)
        batch_time = time()

        # Keep gradients clean before the first micro-step of this epoch.
        if self.optimizer_container is not None:
            self.optimizer_container.zero_grad()
        elif self.optimizer is not None:
            self.optimizer.zero_grad()

        for iteration, data in enumerate(loader):
            _, loss = self.train_step(data)

            if self.log_interval > 0 and (iteration > 0 and iteration % self.log_interval == 0 or iteration == length):
                interval = iteration - last_print_iteration
                if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
                    self.meters.lr.update(self.scheduler.get_last_lr()[0])
                reduced_loss = self.reduce_loss_for_logging(loss)
                if reduced_loss is not None:
                    self.meters.loss.update(reduced_loss.item())
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        self._flush_pending_optimizer_step()
        return self.get_epoch_result()

    def train_steps(
        self,
        train_splits: list[str] | None = None,
        evaluate_splits: list[str] | None = None,
    ) -> RoundDict:
        """
        Run step-mode training for the configured global step budget.

        Args:
            train_splits: Training splits to consume in order.
            evaluate_splits: Evaluation splits to run after training steps finish.

        Returns:
            Aggregated runner results (`self.results`).
        """
        if train_splits is None:
            train_splits = self.train_splits
        if evaluate_splits is None:
            evaluate_splits = self.evaluate_splits

        total_steps = self.steps
        if total_steps is None:
            raise ValueError("cannot run step-mode training: config.steps could not be resolved")
        print("train: step mode start " f"global_step={self.train_state.global_step} steps={total_steps}")
        result = RoundDict()
        checkpoint_cadence = self.checkpoint_interval
        while self.train_state.global_step < total_steps:
            round_start_step = self.train_state.global_step
            for split in train_splits:
                self.mode = RunnerMode.train
                self.split = split
                loader = self.dataloaders[split]
                remaining = total_steps - self.train_state.global_step
                if remaining <= 0:
                    break
                split_steps = remaining

                self.meters.reset()
                if self.train_metrics is not None:
                    self.metrics = self.train_metrics
                if self.metrics is not None:
                    self.metrics.reset()

                self._set_loader_epoch(loader, self.train_state.epoch)
                batch_time = time()
                last_print_iteration = -1
                start_global_step = self.train_state.global_step
                target_global_step = start_global_step + split_steps

                # Keep gradients clean before the first micro-step of this step loop.
                if self.optimizer_container is not None:
                    self.optimizer_container.zero_grad()
                elif self.optimizer is not None:
                    self.optimizer.zero_grad()

                for iteration, data in enumerate(loader):
                    if self.train_state.global_step >= target_global_step:
                        break
                    step_before = self.train_state.global_step
                    _, loss = self.train_step(data)
                    step_after = self.train_state.global_step
                    if checkpoint_cadence > 0 and step_after != step_before and step_after % checkpoint_cadence == 0:
                        self.save_checkpoint()

                    if self.log_interval > 0 and iteration > 0 and iteration % self.log_interval == 0:
                        interval = iteration - last_print_iteration
                        if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
                            self.meters.lr.update(self.scheduler.get_last_lr()[0])
                        reduced_loss = self.reduce_loss_for_logging(loss)
                        if reduced_loss is not None:
                            self.meters.loss.update(reduced_loss.item())
                        self.meters.time.update((time() - batch_time) / interval)
                        batch_time = time()
                        self.step_log(split, self.train_state.global_step - start_global_step, split_steps)
                        last_print_iteration = iteration

                step_before = self.train_state.global_step
                self._flush_pending_optimizer_step()
                step_after = self.train_state.global_step
                if checkpoint_cadence > 0 and step_after != step_before and step_after % checkpoint_cadence == 0:
                    self.save_checkpoint()

                result[split] = self.get_epoch_result()

            if self.train_state.global_step == round_start_step:
                remaining_steps = total_steps - self.train_state.global_step
                warn(
                    f"step-mode training made no progress after one full split pass "
                    f"(target={total_steps}, reached={self.train_state.global_step}, remaining={remaining_steps})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            self.train_state.epoch += 1
        remaining_steps = total_steps - self.train_state.global_step
        if remaining_steps > 0:
            warn(
                f"step-mode training finished with {remaining_steps} step(s) remaining "
                f"(target={total_steps}, reached={self.train_state.global_step})",
                RuntimeWarning,
                stacklevel=2,
            )
        for split in evaluate_splits:
            result[split] = self.evaluate_steps(split=split)
        self.append_result(result, index=self.train_state.global_step)
        print(f"train: step mode result={result}")
        self.save_result()
        self.save_checkpoint(last_step=True)
        return self.results

    def train_step(self, data) -> Tuple[Any, torch.Tensor | None]:
        """Execute one micro-step: forward, loss, metric update, backward, and step logic."""
        data = self.to_device(data)
        with self.train_context():
            inputs = data["input"] if isinstance(data, Mapping) else data[0]
            target = data["target"] if isinstance(data, Mapping) else data[1]
            pred = self.model(**inputs) if isinstance(inputs, Mapping) else self.model(inputs)
            loss = self.criterion(pred, target)  # type: ignore[misc]
            if self.metrics is not None and pred is not None:
                self.metrics.update(pred.squeeze(-1), target)
            self.backward(loss)
            self.step()
        return pred, loss

    def backward(self, loss: torch.Tensor) -> None:
        """
        Run backward pass on one micro-step loss.

        Args:
            loss: The loss tensor for this micro-step.
        """

        (loss / self.accum_steps).backward()

    def step(self) -> None:
        """Advance micro-step state and trigger optimizer update on accumulation boundary."""
        micro_steps = self.train_state.micro_step + 1
        self.train_state.micro_step = micro_steps
        if self.accum_steps <= 1 or micro_steps % self.accum_steps == 0:
            self._optimizer_step()

    def _optimizer_step(self) -> bool:
        """
        Perform one optimizer update, with optional grad clipping and non-finite skip.

        Returns:
            `True` when an optimizer update is applied, otherwise `False`.
        """
        max_grad_value = self.max_grad_value
        max_grad_norm = self.max_grad_norm
        skip_nonfinite_grad = self.skip_nonfinite_grad
        if self.optimizer_container is not None:
            if skip_nonfinite_grad:
                has_nonfinite_grad = self.optimizer_container.has_nan_inf_grad()
                if self.distributed and dist.is_available() and dist.is_initialized():
                    backend = str(dist.get_backend()).lower()
                    sync_device = (
                        self.device if backend == "nccl" and torch.cuda.is_available() else torch.device("cpu")
                    )
                    skip_tensor = torch.tensor(float(has_nonfinite_grad), device=sync_device)
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                    has_nonfinite_grad = skip_tensor.item() > 0
                if has_nonfinite_grad:
                    self.optimizer_container.zero_grad()
                    return False

            stepped = self.optimizer_container.step(
                max_grad_value=max_grad_value,
                max_grad_norm=max_grad_norm,
                zero_grad=True,
                skip_nonfinite_grad=False,
            )
            if not stepped:
                return False
        elif self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.train_state.global_step += 1
        return True

    def _flush_pending_optimizer_step(self) -> bool:
        """
        Flush a partial accumulation window at loop boundaries.

        Returns:
            `True` when a boundary flush produced an optimizer update.
        """
        if self.accum_steps <= 1:
            return False
        remainder = self.train_state.micro_step % self.accum_steps
        if remainder == 0:
            return False
        stepped = self._optimizer_step()
        # Boundary flush clears current accumulation window; realign to the next
        # accumulation boundary so the next loop starts with a fresh full window.
        self.train_state.micro_step += self.accum_steps - remainder
        return stepped

    def evaluate(self, evaluate_splits: list[str] | None = None) -> RoundDict:
        """
        Run evaluation across splits with epoch-mode semantics.

        Args:
            evaluate_splits: Optional evaluation splits. When `None`, use `self.evaluate_splits`.

        Returns:
            Mapping of split -> evaluation result for this call.
        """

        evaluate_splits = sorted(evaluate_splits) if evaluate_splits is not None else self.evaluate_splits
        if not evaluate_splits:
            raise ValueError("cannot start evaluation: no valid evaluation split was resolved")
        print("evaluate: start")
        print(f"evaluate: splits={evaluate_splits}")
        result = RoundDict()
        for split in evaluate_splits:
            result[split] = self.evaluate_epoch(split=split)
        print(self.format_epoch_result(result))
        return result

    @torch.inference_mode()
    def evaluate_epoch(self, split: str = "val") -> RoundDict:
        """
        Run one full dataloader pass for an evaluation split.

        Args:
            split: Evaluation split name.

        Returns:
            Epoch-level metric mapping for this split.
        """

        self.mode = RunnerMode.evaluate
        self.split = split
        loader = self.dataloaders[split]
        length = len(loader) - 1
        last_print_iteration = -1
        self.meters.reset()
        if self.evaluate_metrics is not None:
            self.metrics = self.evaluate_metrics
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()

        for iteration, data in enumerate(loader):
            _, loss = self.evaluate_step(data)

            if self.log_interval > 0 and (iteration > 0 and iteration % self.log_interval == 0 or iteration == length):
                interval = iteration - last_print_iteration
                reduced_loss = self.reduce_loss_for_logging(loss)
                if reduced_loss is not None:
                    self.meters.loss.update(reduced_loss.item())
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        result = self.get_epoch_result()
        self.write_result(result, split, self.train_state.epoch)
        return result

    @torch.inference_mode()
    def evaluate_steps(self, split: str = "val", steps: int | None = None) -> RoundDict:
        """
        Run bounded evaluation steps on one split.

        Args:
            split: Evaluation split name.
            steps: Number of batches to evaluate. When `None`, defaults to `max(self.steps // 20, 1)`.

        Returns:
            Step-bounded evaluation metrics.
        """
        self.mode = RunnerMode.evaluate
        self.split = split
        loader = self.dataloaders[split]

        if steps is None:
            total_steps = self.steps
            if total_steps is None:
                raise ValueError("cannot infer evaluation steps: step budget is unavailable; pass `steps`")
            steps = max(total_steps // 20, 1)
        if steps < 0:
            raise ValueError(f"invalid steps: expected a non-negative value, got {steps}")

        self.meters.reset()
        if self.evaluate_metrics is not None:
            self.metrics = self.evaluate_metrics
        if self.metrics is not None:
            self.metrics.reset()
        if steps == 0:
            result = self.get_epoch_result()
            self.write_result(result, split, self.train_state.global_step)
            return result

        batch_time = time()
        last_print_iteration = -1
        length = steps - 1

        consumed = 0
        for iteration, data in enumerate(loader):
            if iteration >= steps:
                break
            consumed = iteration + 1
            _, loss = self.evaluate_step(data)

            if self.log_interval > 0 and (iteration > 0 and iteration % self.log_interval == 0 or iteration == length):
                interval = iteration - last_print_iteration
                reduced_loss = self.reduce_loss_for_logging(loss)
                if reduced_loss is not None:
                    self.meters.loss.update(reduced_loss.item())
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        if consumed < steps:
            raise ValueError(
                f"evaluate steps exhausted early on split '{split}': requested {steps} step(s), got {consumed}"
            )

        result = self.get_epoch_result()
        self.write_result(result, split, self.train_state.global_step)
        return result

    def evaluate_step(self, data) -> Tuple[Any, torch.Tensor | None]:
        """Execute one evaluation step (forward + optional loss + metric update)."""
        data = self.to_device(data)

        inputs = data["input"] if isinstance(data, Mapping) else data[0]
        target = data["target"] if isinstance(data, Mapping) else data[1]

        if self.model is None:
            raise ValueError("cannot run evaluate_step: model is not initialized")
        model = self.ema or self.model
        if isinstance(inputs, Mapping):
            pred = model(**inputs)
        else:
            pred = model(inputs)
        loss = self.criterion(pred, target) if self.criterion is not None else None

        if self.metrics is not None and pred is not None:
            self.metrics.update(pred.squeeze(-1), target)

        return pred, loss

    def infer(
        self,
        split: str = "infer",
        *,
        steps: int | None = None,
        stream: bool | None = None,
    ) -> list[float] | Iterator[list[float]]:
        """
        Run inference on one split.

        Args:
            split: Inference split name.
            steps: Optional max number of batches to consume.
            stream: `True` returns a generator of per-batch outputs, `False` returns a flattened list.
                When `None`, stream only for unsized loaders without explicit `steps`.

        Returns:
            Flattened predictions or a streaming iterator of batch predictions.
        """

        self.mode = RunnerMode.infer
        self.split = split
        loader = self.dataloaders[split]
        if steps is not None and steps < 0:
            raise ValueError(f"invalid steps: expected a non-negative value, got {steps}")

        loader_length = self._loader_length(loader)
        if stream is None:
            stream = steps is None and loader_length is None

        if not stream and loader_length is None and steps is None:
            raise ValueError("infer with stream=False requires `steps` for unsized loaders")

        if steps is not None:
            iterator = (self.infer_step(data) for iteration, data in enumerate(loader) if iteration < steps)
        else:
            iterator = (self.infer_step(data) for data in loader)
        if stream:
            return iterator

        total = steps if steps is not None else loader_length
        output: list[float] = []
        for values in tqdm(iterator, total=total, disable=self.distributed and not self.is_main_process):
            output.extend(values)
        return output

    @contextmanager
    def train_context(self):
        """Context for one training micro-step (autocast + optional DDP no_sync)."""
        # The step after this context closes.
        micro_steps = self.train_state.micro_step + 1
        if self.fp8_enabled:
            autocast_context = self.fp8_autocast()
        else:
            precision = self.precision
            if precision is None:
                autocast_context = nullcontext()
            else:
                autocast_context = torch.autocast(self.device.type, dtype=get_precision(precision))

        if (
            self.accum_steps > 1
            and micro_steps % self.accum_steps != 0
            and isinstance(self.model, nn.parallel.DistributedDataParallel)
        ):
            with autocast_context, self.model.no_sync():
                yield
            return

        with autocast_context:
            yield

    @staticmethod
    def collate_fn(batch):
        return utils.data.dataloader.default_collate(batch)

    def to_device(self, data: Any):
        """Move one batch to runtime device; override in subclasses for custom fast paths."""
        return to_device(data, self.device)

    @torch.inference_mode()
    def infer_step(self, data: Any) -> list[float]:
        """Execute one inference step and return CPU scalar/list predictions."""
        data = self.to_device(data)
        inputs = data["input"] if isinstance(data, Mapping) else data[0]
        if self.model is None:
            raise ValueError("cannot run infer_step: model is not initialized")
        model = self.ema or self.model
        pred = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
        values = pred.squeeze(-1).detach().cpu().tolist()
        if isinstance(values, list):
            return values
        return [float(values)]

    def materialize_model(self) -> None:
        """Move model to runtime device, optionally compile, and wrap with DDP when distributed."""
        if self.model is None:
            raise ValueError("cannot materialize model: model is not initialized")

        model = self.model.to(self.device)
        should_wrap_ddp = self.distributed and not isinstance(
            model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)
        )
        if should_wrap_ddp:
            # Must be set before torch.compile to take effect.
            maybe_enable_ddp_optimizer(self.config)
        model = maybe_compile_model(model, self.config)
        if should_wrap_ddp:
            model = nn.parallel.DistributedDataParallel(model)
        self.model = model

        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def build_optimizer(self) -> None:
        """Auto-build optimizer from config when `self.optimizer` is absent."""
        if self.optimizer is not None or self.model is None:
            return
        optim_cfg = self.config.get("optim")
        if optim_cfg is None:
            optim_cfg = self.config.get("optimizer")
        if not isinstance(optim_cfg, Mapping) or not optim_cfg:
            return
        self.optimizer = OPTIMIZERS.build(params=self.unwrap(self.model).parameters(), **dict(optim_cfg))

    def build_scheduler(self) -> None:
        """Auto-build scheduler from config when `self.scheduler` is absent."""
        if self.scheduler is not None or self.optimizer is None:
            return
        sched_cfg = self.config.get("sched")
        if sched_cfg is None:
            sched_cfg = self.config.get("scheduler")
        if not isinstance(sched_cfg, Mapping) or not sched_cfg:
            return
        scheduler_kwargs = dict(sched_cfg)
        if "total_steps" not in scheduler_kwargs:
            steps = self.steps
            if steps is not None:
                scheduler_kwargs["total_steps"] = steps
        self.scheduler = SCHEDULERS.build(self.optimizer, **scheduler_kwargs)

    def _bind_optimizer_container(self) -> None:
        if self.optimizer is None:
            self.optimizer_container = None
            return
        self.optimizer_container = OptimizerContainer(self.optimizer, scheduler=self.scheduler)

    def build_dataloaders(self):
        """Build dataloaders for dataset splits not already materialized."""
        datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
        dataloader_config = self.config.get("dataloader", NestedDict())
        default_kwargs = NestedDict({k: v for k, v in dataloader_config.items() if k not in self.datasets})
        split_kwargs = NestedDict({k: v for k, v in dataloader_config.items() if k in self.datasets})
        for k, dataset in datasets.items():
            kwargs = NestedDict(default_kwargs)
            if k in split_kwargs:
                kwargs.merge(split_kwargs[k], overwrite=True)
            is_train_split = k in self.train_splits
            shuffle = kwargs.pop("shuffle", is_train_split)
            kwargs.setdefault("drop_last", is_train_split)
            sampler = self.build_datasampler(dataset, split=k, shuffle=shuffle)
            self.dataloaders[k] = utils.data.DataLoader(dataset, sampler=sampler, collate_fn=self.collate_fn, **kwargs)

    def build_datasampler(self, dataset, *, split: str, shuffle: bool):
        """Build split sampler (distributed or local)."""
        if self.distributed:
            return utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        return utils.data.RandomSampler(dataset) if shuffle else utils.data.SequentialSampler(dataset)

    def reduce_loss_for_logging(self, loss: torch.Tensor | None) -> torch.Tensor | None:
        """Detach and all-reduce loss tensor for logging."""
        if loss is None:
            return None
        return self.reduce(loss.detach())

    def unwrap(self, model: nn.Module) -> nn.Module:
        if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
            return model.module
        return model

    def state_dict(self, cls: type = dict) -> Mapping:
        """Return TorchRunner checkpoint payload (runner + model + optimizer + scheduler + ema)."""
        if self.model is None:
            raise ValueError("cannot build checkpoint state: model is not initialized")
        state = cls(super().state_dict(cls))
        state["ema"] = self.ema.state_dict() if self.ema else None
        state["optimizer"] = self.optimizer.state_dict() if self.optimizer else None
        state["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        state["model"] = self.unwrap(self.model).state_dict()
        return state

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore runner state and RNG state from checkpoint payload."""
        super().load_state_dict(checkpoint)
        state_dict = checkpoint.get("state") or {}
        rng_state = state_dict.get("rng")
        if isinstance(rng_state, Mapping) and "torch_cpu" in rng_state and self.rng_state.torch_cpu is not None:
            torch.set_rng_state(self.rng_state.torch_cpu)
        if (
            torch.cuda.is_available()
            and isinstance(rng_state, Mapping)
            and "torch_cuda" in rng_state
            and self.rng_state.torch_cuda is not None
        ):
            torch.cuda.set_rng_state_all(self.rng_state.torch_cuda)

    def load_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        """Load a full checkpoint and rebind optimizer container afterwards."""
        super().load_checkpoint(checkpoint, *args, **kwargs)
        self._bind_optimizer_container()

    @catch
    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
    ) -> None:
        """Save checkpoint through the active backend manager."""
        if self.config.checkpoint.backend.lower() != "dcp":
            return super().save_checkpoint(name=name, epochs=epochs, save_best=save_best, last_step=last_step)

        epochs = self.train_state.epoch if epochs is None else epochs
        self.checkpoint_manager.save_checkpoint(name=name, epochs=epochs, save_best=save_best, last_step=last_step)

    def read_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> Mapping[str, Any]:
        """Read checkpoint payload from mapping/file/DCP directory input."""
        if isinstance(checkpoint, Mapping):
            return checkpoint

        if self.config.checkpoint.backend.lower() == "dcp":
            return self.checkpoint_manager.load_checkpoint(checkpoint)
        return super().read_checkpoint(checkpoint, *args, **kwargs)

    @classmethod
    def read_config(
        cls,
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args,
        **kwargs,
    ) -> Mapping[str, Any]:
        """Read runner config from checkpoint payload, including DCP directory inputs."""
        if isinstance(checkpoint, Mapping):
            return super().read_config(checkpoint, *args, **kwargs)

        if isinstance(
            checkpoint, (bytes, str, os.PathLike)
        ) and TorchDistributedCheckpointManager.is_checkpoint_path(checkpoint):
            return TorchDistributedCheckpointManager.read_config(checkpoint)

        return super().read_config(checkpoint, *args, **kwargs)

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    @property
    def mode(self) -> RunnerMode:
        return self._mode

    @mode.setter
    def mode(self, mode: str | RunnerMode) -> None:
        if isinstance(mode, str):
            mode = RunnerMode(mode)
        self._mode = mode

        is_train = mode == RunnerMode.train
        if self.model is not None:
            self.model.train(is_train)
        if self.ema is not None:
            self.ema.train(is_train)

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return int(os.getenv("RANK", "0"))

    @property
    def world_size(self) -> int:
        r"""
        Number of Processes.
        """
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return int(os.getenv("WORLD_SIZE", "1"))

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @staticmethod
    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        """Average-reduce tensor across distributed world when initialized."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist.all_reduce(tensor)
            world_size = max(dist.get_world_size(), 1)
            tensor = tensor / world_size
        return tensor

    def close(self, timeout: float | None = None) -> bool:
        """Close runner resources."""
        return super().close(timeout=timeout)
