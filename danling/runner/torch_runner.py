from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from contextlib import suppress
from time import time
from typing import Any
from warnings import warn

import torch
from accelerate import Accelerator
from chanfig import NestedDict
from torch import distributed as dist
from torch import nn
from torch.backends import cudnn
from tqdm import tqdm

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from danling.utils import catch

from .base_runner import BaseRunner
from .runner_base import RunnerMode
from .utils import on_main_process


class TorchRunner(BaseRunner):
    r"""
    Set up everything for running a job.

    Attributes:
        accelerator (Accelerator):
        accelerate: Arguments to pass when building accelerator. Defaults to `{}`.
    """

    # pylint: disable=R0902

    accelerator: Accelerator = None  # type: ignore
    accelerate: Mapping[str, Any]

    def __init__(self, *args, **kwargs) -> None:
        if len(args) != 1 or kwargs:
            message = (
                "Passing multiple args & kwargs to build Runner is deprecated and will be removed in DanLing v0.3.\n"
                "Please only pass a config dict instead."
            )
            warn(message, DeprecationWarning, stacklevel=2)
        if "accelerate" not in self:
            self.accelerate = {}
        if len(args) == 1 and isinstance(args[0], dict):
            self.accelerate.update(args[0].pop("accelerate", {}))  # type: ignore
        if "accelerate" in kwargs:
            self.accelerate.update(kwargs.pop("accelerate"))  # type: ignore
        super().__init__(*args, **kwargs)

    def train(self):
        early_stop_counter = 0
        print("begin training")
        self.state.epoch_begin = self.state.epochs
        for self.state.epochs in range(self.state.epoch_begin, self.state.epoch_end):  # noqa: B020
            result = NestedDict()
            result.setattr("convert_mapping", True)
            result.train = self.train_epoch()
            if "val" in self.dataloaders:
                result.val = self.evaluate_epoch("val")
            if "test" in self.dataloaders:
                result.test = self.evaluate_epoch("test")
            self.append_result(result)
            print(self.format_epoch_result(result))
            self.save_result()
            self.save_checkpoint()
            early_stop_counter = 0 if self.is_best else early_stop_counter + 1
            if self.patience and early_stop_counter > self.patience:
                print("early stop")
                break
        return self.results

    def train_epoch(self, split: str = "train"):
        r"""
        Train one epoch on `split`.

        Args:
            split (str): split to run train

        Return:
            Dict[str, float]: train result
        """

        # pylint: disable=E1101, E1102, W0622
        self.mode = RunnerMode("train")
        loader = self.dataloaders[split]
        length = len(loader) - 1
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(self.epochs)
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.epochs)
        self.meters.reset()
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()

        for iteration, (input, target) in enumerate(loader):  # pylint: disable=W0622
            with self.accelerator.accumulate(self.model):
                pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)  # type: ignore
                loss = self.criterion(pred, target)  # type: ignore
                if self.metrics is not None:
                    self.metrics.update(pred, target)
                self.accelerator.backward(loss)
                if self.sync_gradients:
                    max_grad_value = self.state.get("max_grad_value")
                    if max_grad_value:
                        self.clip_grad_value_(self.model.parameters(), max_grad_value)  # type: ignore
                    max_grad_norm = self.state.get("max_grad_norm")
                    if max_grad_norm:
                        self.clip_grad_norm_(self.model.parameters(), max_grad_norm)  # type: ignore
                self.step()

            if self.print_interval > 0 and iteration % self.print_interval == 0:
                if self.device == torch.device("cuda"):
                    torch.cuda.synchronize()
                self.meters.time.update((time() - batch_time) / self.print_interval)
                batch_time = time()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.step_log(split, iteration, length)

        result = self.meters.avg
        if self.metrics is not None:
            result.merge(self.metrics.avg)
        return result

    def evaluate(self):
        print("begin evaluation")
        result = self.evaluate_epoch()
        print(self.format_epoch_result({"evaluate": result}))
        return result

    @torch.inference_mode()
    def evaluate_epoch(self, split: str = "val"):
        r"""
        Evaluate one epoch on `split`.

        Args:
            split (str): split to run evaluate

        Return:
            Dict[str, float]: evaluation result
        """

        # pylint: disable=E1101, E1102, W0622
        self.mode = RunnerMode("eval")
        loader = self.dataloaders[split]
        length = len(loader) - 1
        self.meters.reset()
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()

        for iteration, (input, target) in enumerate(loader):
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)  # type: ignore
            loss = self.criterion(pred, target)  # type: ignore
            if self.metrics is not None:
                self.metrics.update(pred, target)

            if self.print_interval > 0 and iteration % self.print_interval == 0:
                if self.device == torch.device("cuda"):
                    torch.cuda.synchronize()
                self.meters.time.update((time() - batch_time) / self.print_interval)
                batch_time = time()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.step_log(split, iteration, length)

        result = self.meters.avg
        if self.metrics is not None:
            result.merge(self.metrics.avg)
        self.write_result(result, split, self.state.epochs)
        return result

    @torch.inference_mode()
    def inference(self, split: str = "inf"):
        r"""
        Perform inference on `split`.

        Args:
            split (str): split to run inference

        Return:
            Tensor: inference outputs
        """

        # pylint: disable=E1102, W0622
        self.mode = RunnerMode("inf")
        loader = self.dataloaders[split]
        self.meters.reset()
        output = []
        for _, (input, _) in tqdm(enumerate(loader), total=len(loader)):
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)  # type: ignore
            output.extend(pred.tolist())

        if self.distributed:
            torch.cuda.synchronize()
            output = self.gather_for_metrics(output)
        return output

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        if self.accelerate is None:
            self.accelerate = {}
        self.accelerator = Accelerator(**self.accelerate)
        if self.distributed:
            object_list = [self.state.id]
            dist.broadcast_object_list(object_list)
            self.state.id = object_list[0]

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        r"""
        Set up Tensoraoard SummaryWriter.
        """
        from torch.utils.tensorboard.writer import SummaryWriter  # pylint: disable=C0415

        if "log_dir" not in kwargs:
            kwargs["log_dir"] = self.dir

        self.writer = SummaryWriter(*args, **kwargs)
        self.writer.add_scalar = catch(OSError, verbose=False)(self.writer.add_scalar)  # type: ignore

    def set_seed(self, seed: int = None, bias: int | None = None) -> None:  # type: ignore
        r"""
        Set up random seed.

        Args:
            seed: Random seed to set.
                Defaults to `self.state.seed` (`config.seed`).

            bias: Make the seed different for each processes.

                This avoids same data augmentation are applied on every processes.

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
            seed += bias  # type: ignore
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

    def prepare(self, *args, device_placement: list[bool] | None = None) -> None:
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
            model = self.model  # type: ignore
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        if self.distributed:
            return model.module  # type: ignore
        return model  # type: ignore

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

        batch_size = self.state.get("batch_size")
        if batch_size:
            return batch_size
        if self.dataloaders:
            loader = self.dataloaders["train"] if "train" in self.dataloaders else next(iter(self.dataloaders.values()))
            if loader.batch_size:
                return loader.batch_size
            batch_sampler = loader.batch_sampler if loader.batch_sampler is not None else loader.sampler
            return batch_sampler.batch_size
        raise AttributeError("batch_size could not be inferred, since no dataloaedr found.")

    @property
    def accum_steps(self) -> int:
        r"""
        Gradient accumulation steps.

        Returns:
            (int):
        """

        return self.accelerator.gradient_accumulation_steps

    @property
    def device(self) -> torch.device:  # pylint: disable=E1101
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
        if self.accelerator is not None and hasattr(self.accelerator, name):
            return getattr(self.accelerator, name)
        raise super().__getattribute__(name)
