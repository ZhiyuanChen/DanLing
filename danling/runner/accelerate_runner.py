from __future__ import annotations

import os
import random
from collections.abc import Callable, Mapping
from contextlib import suppress
from time import time
from typing import Any
from warnings import warn

# pylint: disable=redefined-builtin
import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from chanfig import NestedDict
from torch import distributed as dist
from torch import nn, optim, utils
from torch.backends import cudnn
from tqdm import tqdm

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from danling.utils import catch

from .base_runner import BaseRunner
from .utils import RunnerMode, on_main_process


class AccelerateRunner(BaseRunner):  # pylint: disable=too-many-public-methods
    r"""
    Set up everything for running a job.

    `AccelerateRunner` uses [`accelerate`][accelerate] as distributed backend to
    provide seamless distributed training experience.

    `AccelerateRunner` will automatically [`prepare`][accelerate.Accelerator.prepare] everything,
    including `model`, `criterion`, `optimizer`, `scheduler`, and `dataloaders` for distribute training,
    mixed precision, and deepspeed (optional).

    In fact, you don't even need to create `dataloaders`, just define
    `datasets` and `AccelerateRunner` will create `dataloaders` for you.
    `AccelerateRunner` will inspect the `train` flag in corresponding dataset to
    automatically set `shuffle`.

    Attributes:
        accelerator (Accelerator):
        accelerate: Arguments to pass when building accelerator. Defaults to `{}`.
    """

    accelerator: Accelerator
    accelerate: dict

    model: nn.Module
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    def __init__(self, *args, **kwargs) -> None:
        if len(args) != 1 or kwargs:
            message = (
                "Passing multiple args & kwargs to build Runner is deprecated and will be removed in DanLing v0.3.\n"
                "Please only pass a config dict instead."
            )
            warn(message, DeprecationWarning, stacklevel=2)
            config = NestedDict(*args, **kwargs)
        else:
            config = args[0]
        if "accelerate" not in self:  # class attributes
            self.accelerate = {}
        self.accelerate.update(config.get("accelerate", {}))
        super().__init__(config)

    def __post_init__(self) -> None:
        self.model, self.criterion, self.optimizer = self.prepare(self.model, self.criterion, self.optimizer)
        self.scheduler = self.prepare(self.scheduler)
        if self.datasets:
            datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
            default_kwargs = self.state.setdefault("dataloader", NestedDict())
            dataloader_kwargs = NestedDict({k: default_kwargs.pop(k) for k in self.datasets if k in default_kwargs})
            for k, d in datasets.items():
                dataloader_kwargs.setdefault(k, NestedDict())
                dataloader_kwargs[k].merge(default_kwargs, overwrite=False)
                dataloader_kwargs[k].setdefault("shuffle", getattr(d, "train", True))
                dataloader_kwargs[k].setdefault("drop_last", not getattr(d, "train", True))
                self.dataloaders[k] = utils.data.DataLoader(d, **dataloader_kwargs[k])
            default_kwargs.update(dataloader_kwargs)
        for k, d in self.dataloaders.items():
            self.dataloaders[k] = self.prepare(d)
        if self.state.get("log_interval") is None:
            self.state.log_interval = max(len(d) for d in self.dataloaders.values()) // 10

    @property
    def deepspeed(self) -> dict | None:
        if "accelerator" not in self:
            raise ValueError("accelerator is not used")
        if self.accelerator.state.deepspeed_plugin is not None:
            return self.accelerator.state.deepspeed_plugin.deepspeed_config
        return None

    def train(self, train_splits: list[str] | None = None, eval_splits: list[str] | None = None) -> NestedDict:
        r"""
        Perform training on `split`.

        Args:
            train_splits (list[str]): list of split to run train.
                Defaults to `["train"]`.
            eval_splits (list[str]): list of split to run evaluate.
                Defaults to `self.dataloaders` except for those in `train_splits`.

        Return:
            NestedDict: train results
        """

        early_stop_counter = 0
        if train_splits is None:
            train_splits = ["train"]
        if eval_splits is None:
            eval_splits = [s for s in self.dataloaders if s not in train_splits]
        self.state.epoch_begin = self.state.epochs
        print(f"Begin training from {self.state.epoch_begin} to {self.state.epoch_end}")
        print(f"Training splits: {train_splits}")
        print(f"Evaluation splits: {eval_splits}")
        patience = self.state.get("patience", float("inf"))
        for epochs in range(self.state.epoch_begin, self.state.epoch_end):  # type: ignore
            self.state.epochs = epochs
            result = NestedDict()
            result.setattr("convert_mapping", True)
            for split in train_splits:
                result[split] = self.train_epoch(split)
            for split in eval_splits:
                result[split] = self.evaluate_epoch(split)
            self.append_result(result)
            print(self.format_epoch_result(result))
            self.save_result()
            if self.state.save_interval is not None:
                self.save_checkpoint(epochs)
            """@nni.report_intermediate_result(self.latest_score)"""
            early_stop_counter = 0 if self.is_best else early_stop_counter + 1
            if early_stop_counter > patience:
                print("early stop")
                break
        """@nni.report_final_result(self.latest_score)"""
        return self.results

    def train_epoch(self, split: str = "train") -> NestedDict:
        r"""
        Train one epoch on `split`.

        Args:
            split (str): split to run train

        Return:
            NestedDict: train result
        """

        self.mode = "train"  # type: ignore
        self.split = split
        loader = self.dataloaders[split]
        length = len(loader) - 1
        last_print_iteration = -1
        log_interval = self.state.get("log_interval", -1)
        self.meters.reset()
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(self.epochs)
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.epochs)

        for iteration, data in enumerate(loader):
            with self.autocast(), self.accumulate():
                input = data["input"] if isinstance(data, Mapping) else data[0]
                target = data["target"] if isinstance(data, Mapping) else data[1]
                pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
                loss = self.criterion(pred, target)
                if self.metrics is not None:
                    self.metrics.update(pred.squeeze(-1), target)
                self.step(loss)

            if log_interval > 0 and (iteration > 0 and iteration % log_interval == 0 or iteration == length):
                interval = iteration - last_print_iteration
                if self.device == torch.device("cuda"):
                    torch.cuda.synchronize()
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        result = self.meters.average()
        if self.metrics is not None:
            result.merge(self.metrics.average())
        return result

    def evaluate(self, eval_splits: list[str] | None = None) -> NestedDict:
        r"""
        Perform evaluation on `eval_splits`.

        Args:
            eval_splits (list[str]): list of split to run evaluate.
                Defaults to `["eval"]`.

        Return:
            NestedDict: evaluation result
        """

        if eval_splits is None:
            eval_splits = ["eval"]

        print("Begin evaluation")
        print(f"Evaluation splits: {eval_splits}")
        result = NestedDict()
        result.setattr("convert_mapping", True)
        for split in eval_splits:
            result[split] = self.evaluate_epoch(split=split)
        print(self.format_epoch_result(result))
        return result

    @torch.inference_mode()
    def evaluate_epoch(self, split: str = "val") -> NestedDict:
        r"""
        Evaluate one epoch on `split`.

        Args:
            split (str): split to run evaluate

        Return:
            NestedDict: evaluation result
        """

        self.mode = "eval"  # type: ignore
        self.split = split
        loader = self.dataloaders[split]
        length = len(loader) - 1
        last_print_iteration = -1
        log_interval = self.state.get("log_interval", -1)
        self.meters.reset()
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()

        for iteration, data in enumerate(loader):
            input = data["input"] if isinstance(data, Mapping) else data[0]
            target = data["target"] if isinstance(data, Mapping) else data[1]
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
            loss = self.criterion(pred, target)
            if self.metrics is not None:
                self.metrics.update(pred.squeeze(-1), target)

            if log_interval > 0 and (iteration > 0 and iteration % log_interval == 0 or iteration == length):
                interval = iteration - last_print_iteration
                if self.device == torch.device("cuda"):
                    torch.cuda.synchronize()
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        result = self.meters.average()
        if self.metrics is not None:
            result.merge(self.metrics.average())
        self.write_result(result, split, self.state.epochs)
        return result

    @torch.inference_mode()
    def inference(self, split: str = "inf") -> list:
        r"""
        Perform inference on `split`.

        Args:
            split (str): split to run inference

        Return:
            Tensor: inference outputs
        """

        # pylint: disable=E1102, W0622
        self.mode = "inf"  # type: ignore
        loader = self.dataloaders[split]
        self.meters.reset()
        output = []
        for _, data in tqdm(enumerate(loader), total=len(loader)):
            input = data["input"] if isinstance(data, Mapping) else data[0]
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
            output.extend(pred.squeeze(-1).tolist())

        if self.distributed:
            torch.cuda.synchronize()
            output = self.gather_for_metrics(output)
        return output

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true":
            deepspeed_config = self.state.get("deepspeed", os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE"))
            self.accelerate["deepspeed_plugin"] = DeepSpeedPlugin(hf_ds_config=self.init_deepspeed(deepspeed_config))
        self.accelerator = Accelerator(**self.accelerate)
        if self.distributed:
            object_list = [self.id, self.timestamp]
            dist.broadcast_object_list(object_list)
            self.id, self.timestamp = object_list

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        r"""
        Set up Tensoraoard SummaryWriter.
        """
        from torch.utils.tensorboard.writer import SummaryWriter  # pylint: disable=C0415

        if "log_dir" not in kwargs:
            kwargs["log_dir"] = self.dir

        self.writer = SummaryWriter(*args, **kwargs)
        self.writer.add_scalar = catch(OSError, verbose=False)(self.writer.add_scalar)

    def set_seed(self, seed: int | None = None, bias: int | None = None) -> None:
        r"""
        Set up random seed.

        Args:
            seed: Random seed to set.
                Defaults to `self.state.seed` (`config.seed`).

            bias: Make the seed different for each processes.
                This is used to ensure the data augmentation are applied differently on every processes.
                Defaults to `self.rank`.
                Set to `False` to disable this feature.
        """

        seed = seed or self.state.seed
        if self.distributed:
            object_list = [seed]
            dist.broadcast_object_list(object_list)
            seed = object_list[0]
        bias = bias or self.rank
        if bias:
            seed += bias
        self.state.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if np_random is not None:
            np_random.seed(seed)
        random.seed(seed)

    def set_deterministic(self) -> None:
        r"""
        Set up deterministic.
        """

        cudnn.benchmark = False
        cudnn.deterministic = True
        if torch.__version__ >= "1.8.0":
            torch.use_deterministic_algorithms(True)

    def step(self, loss, batch_size: int | None = None, zero_grad: bool = True) -> None:
        r"""
        Backward loss and step optimizer & scheduler.

        This method increment `self.state.steps`.

        This method also increment `self.state.iters` when `batch_size` is specified.

        Args:
            zero_grad: Whether to zero the gradients.
        """

        self.accelerator.backward(loss)
        if self.sync_gradients:
            if self.state.get("max_grad_value") is not None:
                self.clip_grad_value_(self.model.parameters(), self.state.get("max_grad_value"))
            if self.state.get("max_grad_norm") is not None:
                self.clip_grad_norm_(self.model.parameters(), self.state.get("max_grad_norm"))
        if self.optimizer is not None:
            self.optimizer.step()
            if zero_grad:
                self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.steps += 1
        if batch_size is None:
            batch_size = self.batch_size_equivalent
        self.state.iters += batch_size
        # TODO: Support `drop_last = False`
        # self.state.iters += self.batch_size_equivalent

    def state_dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Return dict of all attributes for checkpoint.
        """

        if self.model is None:
            raise ValueError("Model must be defined when calling state_dict")
        model = self.accelerator.unwrap_model(self.model)
        return cls(
            runner=self.state.dict(),
            model=model.state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
        )

    def prepare(self, *args: list[Any], device_placement: list[bool] | None = None) -> list[Any]:
        r"""
        Prepare all objects passed in `args` for distributed training and mixed precision,
        then return them in the same order.
        """

        return self.accelerator.prepare(*args, device_placement=device_placement)

    def accumulate(self, model: nn.Module | None = None):
        r"""
        Context manager that enables gradient accumulate.
        """

        model = model or self.model
        return self.accelerator.accumulate(model)

    def autocast(self):
        r"""
        Context manager that enables auto-casting for the forward pass (and maybe backward pass).
        """

        return self.accelerator.autocast()

    def backward(self, loss) -> None:
        r"""
        Backward loss to compute gradients.
        """

        return self.accelerator.backward(loss)

    def unwrap_model(self, model: nn.Module | None = None) -> nn.Module:
        r"""
        Unwrap DDP model.

        Args:
            model (Optional[nn.Module]):
                Defaults to `self.model`.
        """

        if model is not None:
            model = self.model
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        if self.distributed:
            return model.module
        return model

    @property
    def mode(self) -> RunnerMode:
        return self._mode

    @mode.setter
    def mode(self, mode: str | RunnerMode) -> None:
        if isinstance(mode, str):
            mode = RunnerMode(mode)
        self._mode = mode
        if self.model is not None:
            self.model.train(mode == RunnerMode.train)

    @property
    def batch_size(self) -> int:
        r"""
        Batch size.

        Notes:
            If `train` is in `dataloaders`, then `batch_size` is the batch size of `train`.
            Otherwise, `batch_size` is the batch size of the first dataloader.

        Returns:
            (int):
        """

        batch_size = self.state.get("dataloader.batch_size")
        if batch_size:
            return batch_size
        if self.dataloaders:
            loader = self.dataloaders.get("train", next(iter(self.dataloaders.values())))
            if loader.batch_size:
                return loader.batch_size
            batch_sampler = loader.batch_sampler if loader.batch_sampler is not None else loader.sampler
            return batch_sampler.batch_size
        raise AttributeError("batch_size could not be inferred, since no dataloader found.")

    @property
    def accum_steps(self) -> int:
        r"""
        Gradient accumulation steps.

        Returns:
            (int):
        """

        return self.accelerator.gradient_accumulation_steps

    @property
    def device(self) -> torch.device:
        r"""
        Device of runner.
        """

        return self.accelerator.device

    @property
    def world_size(self) -> int:
        r"""
        Number of Processes.
        """

        return self.accelerator.num_processes

    @property
    def rank(self) -> int:
        r"""
        Process index in all processes.
        """

        return self.accelerator.process_index

    @property
    def local_rank(self) -> int:
        r"""
        Process index in local processes.
        """

        return self.accelerator.local_process_index

    def gather(self, tensor) -> torch.Tensor:
        r"""
        Gather tensor.
        """

        return self.accelerator.gather(tensor)

    def reduce(self, tensor, reduction: str = "sum") -> torch.Tensor:
        r"""
        Reduce tensor.
        """

        return self.accelerator.reduce(tensor, reduction=reduction)

    def __getattr__(self, name: str) -> Any:
        with suppress(AttributeError):
            return super().__getattr__(name)
        if "accelerator" in self.__dict__ and hasattr(self.accelerator, name):
            return getattr(self.accelerator, name)
        raise super().__getattribute__(name)
