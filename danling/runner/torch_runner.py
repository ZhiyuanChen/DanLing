from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from time import time
from typing import Any
from warnings import warn

import torch
from chanfig import FlatDict, NestedDict
from torch import Tensor
from torch import distributed as dist
from torch._C._distributed_c10d import ProcessGroup, ReduceOp
from torch.backends import cudnn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import utils
from tqdm import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

    np_random = None

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from danling.utils import catch

from .base_runner import BaseRunner
from .utils import is_criterion, on_main_process


class TorchRunner(BaseRunner):
    r"""
    Set up everything for running a job.

    `TorchRunner` provides a basic runner for running PyToch models.

    `TorchRunner` will automatically prepare everything,
    including `model`, `criterion`, `optimizer`, `scheduler`, and `dataloaders` for distribute training,
    mixed precision.

    In fact, you don't even need to create `dataloaders`, just define
    `datasets` and `TorchRunner` will create `dataloaders` for you.
    `TorchRunner` will inspect the `train` flag in corresponding dataset to
    automatically set `shuffle`.
    """

    # pylint: disable=R0902

    _models: FlatDict[str, Module]
    _criterions: FlatDict[str, Module]
    _optimizers: FlatDict[str, Optimizer]
    _schedulers: FlatDict[str, LRScheduler]

    def __init__(self, *args, **kwargs):
        self._models = FlatDict()
        self._criterions = FlatDict()
        self._optimizers = FlatDict()
        self._schedulers = FlatDict()
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        self.prepare_attr(self._models)
        self.prepare_attr(self._criterions)
        self.prepare_attr(self._optimizers)
        self.prepare_attr(self._schedulers)
        if self.datasets:
            datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
            dataloader_kwargs = self.state.get("dataloader", {})
            dataloaders_kwargs = {k: dataloader_kwargs.pop(k) for k in self.datasets.keys() if k in dataloader_kwargs}
            for k, d in datasets.items():
                if k in dataloaders_kwargs:
                    kwargs = dataloaders_kwargs[k]
                    kwargs.merge(dataloader_kwargs, overwrite=False)
                else:
                    kwargs = dataloader_kwargs.clone()
                kwargs.setdefault("shuffle", getattr(d, "train", True))
                kwargs.setdefault("drop_last", not getattr(d, "train", True))
                self.dataloaders[k] = self.prepare(DataLoader(d, **kwargs))

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
            self.save_checkpoint()
            """@nni.report_intermediate_result(self.latest_score)"""  # pylint: disable=W0105
            early_stop_counter = 0 if self.is_best else early_stop_counter + 1
            if early_stop_counter > patience:
                print("early stop")
                break
        """@nni.report_final_result(self.latest_score)"""  # pylint: disable=W0105
        return self.results

    def train_epoch(self, split: str = "train") -> NestedDict:
        r"""
        Train one epoch on `split`.

        Args:
            split (str): split to run train

        Return:
            NestedDict: train result
        """

        # pylint: disable=E1101, E1102, W0622
        self.mode = "train"  # type: ignore
        self.split = split
        loader = self.dataloaders[split]
        length = len(loader) - 1
        last_print_iteration = -1
        self.meters.reset()
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(self.epochs)
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.epochs)

        for iteration, data in enumerate(loader):
            with self.autocast():
                input = data["input"] if isinstance(data, Mapping) else data[0]
                target = data["target"] if isinstance(data, Mapping) else data[1]
                input, target = input.to(self.device), target.to(self.device)
                pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
                loss = self.criterion(pred, target)
                if self.metrics is not None:
                    self.metrics.update(pred, target)
                self.step(loss)

            if self.print_interval > 0 and (
                iteration > 0 and iteration % self.print_interval == 0 or iteration == length
            ):
                interval = iteration - last_print_iteration
                if self.device == torch.device("cuda"):
                    torch.cuda.synchronize()
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        result = self.meters.avg
        if self.metrics is not None:
            result.merge(self.metrics.avg)
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

        # pylint: disable=E1101, E1102, W0622
        self.mode = "eval"  # type: ignore
        self.split = split
        loader = self.dataloaders[split]
        length = len(loader) - 1
        last_print_iteration = -1
        self.meters.reset()
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()

        for iteration, data in enumerate(loader):
            input = data["input"] if isinstance(data, Mapping) else data[0]
            target = data["target"] if isinstance(data, Mapping) else data[1]
            input, target = input.to(self.device), target.to(self.device)
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
            loss = self.criterion(pred, target)
            if self.metrics is not None:
                self.metrics.update(pred, target)

            if self.print_interval > 0 and (
                iteration > 0 and iteration % self.print_interval == 0 or iteration == length
            ):
                interval = iteration - last_print_iteration
                if self.device == torch.device("cuda"):
                    torch.cuda.synchronize()
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        result = self.meters.avg
        if self.metrics is not None:
            result.merge(self.metrics.avg)
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
            output.extend(pred.tolist())

        if self.distributed:
            torch.cuda.synchronize()
            output = self.gather_for_metrics(output)
        return output

    def step(self, loss, batch_size: int | None = None, zero_grad: bool = True) -> None:
        r"""
        Backward loss and step optimizer & scheduler.

        This method increment `self.state.steps`.

        This method also increment `self.state.iters` when `batch_size` is specified.

        Args:
            zero_grad: Whether to zero the gradients.
        """

        self.backward(loss)
        if True:  # TODO: do not clip gradients if no sync
            if self.state.get("max_grad_value") is not None:
                self.clip_grad_value_(self.model.parameters(), self.state.get("max_grad_value"))  # type: ignore
            if self.state.get("max_grad_norm") is not None:
                self.clip_grad_norm_(self.model.parameters(), self.state.get("max_grad_norm"))  # type: ignore
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

    def autocast(self):
        r"""
        Context manager that enables auto-casting for the forward pass (and maybe backward pass).
        """

        return torch.autocast(self.device.type)

    def backward(self, loss) -> None:
        r"""
        Backward loss to compute gradients.
        """

        return loss.backward()

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        if True:
            dist.init_process_group("nccl")
            # object_list = [self.state.run_id]
            # dist.broadcast_object_list(object_list)
            # self.state.id = object_list[0]

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
        # if self.distributed:
        #     object_list = [seed]
        #     dist.broadcast_object_list(object_list)
        #     seed = object_list[0]
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

    def prepare(self, *args: list[Any], device: list[torch.device] | torch.device | None = None) -> None:
        r"""
        Prepare all objects passed in `args` for distributed training and mixed precision,
        then return them in the same order.
        """

        if isinstance(device, Sequence):
            if len(args) != len(device):
                raise ValueError("num of devices must match num of objects in prepare")
            result = tuple(self._prepare_one(obj, d) for obj, d in zip(args, device))
        else:
            if device is None:
                device = self.device
            result = tuple(self._prepare_one(obj, device) for obj in args)
        return result if len(result) > 1 else result[0]

    def prepare_attr(self, objs: Mapping[str, Any]) -> Mapping:
        for name, obj in objs.items():
            setattr(self, name, self.prepare(obj))
        return objs

    def _prepare_one(self, obj: Any, device: torch.device | None = None) -> None:
        r"""
        Prepare one object for distributed training and mixed precision.
        """
        if device is None:
            device = self.device

        if isinstance(obj, DataLoader):
            return self.prepare_data_loader(obj, device=device)
        if isinstance(obj, Module):
            return self.prepare_model(obj, device=device)
        if isinstance(obj, Optimizer):
            return self.prepare_optimizer(obj, device=device)
        if isinstance(obj, LRScheduler):
            return self.prepare_scheduler(obj)
        return obj

    def prepare_data_loader(self, obj: Any, device: torch.device | None = None) -> None:
        r"""
        Prepare dataloader for distributed training and mixed precision.
        """
        if device is None:
            device = self.device

        sampler = utils.data.distributed.DistributedSampler(obj.dataset, shuffle=obj.shuffle, drop_last=obj.drop_last)
        return DataLoader(obj.dataset, obj.batch_size, num_workers=obj.num_workers, sampler=sampler)

    def prepare_model(self, obj: Any, device: torch.device | None = None) -> None:
        r"""
        Prepare module for distributed training and mixed precision.
        """
        if device is None:
            device = self.device

        return obj.to(device)

    def prepare_optimizer(self, obj: Any, device: torch.device | None = None) -> None:
        r"""
        Prepare optimizer for distributed training and mixed precision.
        """
        if device is None:
            device = self.device

        return obj

    def prepare_scheduler(self, obj: Any, device: torch.device | None = None) -> None:
        r"""
        Prepare lr scheduler for distributed training and mixed precision.
        """
        if device is None:
            device = self.device

        return obj

    def state_dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Return dict of all attributes for checkpoint.
        """

        if self.model is None:
            raise ValueError("Model must be defined when calling state_dict")
        model = self.unwrap_model(self.model)
        return cls(
            runner=self.state.dict(),
            model=model.state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
        )

    def unwrap_model(self, model: Module | None = None) -> Module:
        r"""
        Unwrap DDP model.

        Args:
            model (Optional[Module]):
                Defaults to `self.model`.
        """

        if model is None:
            model = self.model
        if self.distributed:
            return model.module
        return model

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
    def device(self) -> torch.device:  # pylint: disable=E1101
        r"""
        Device of runner.
        """

        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    @property
    def world_size(self) -> int:
        r"""
        Number of Processes.
        """

        if dist.is_initialized():
            return dist.get_world_size()
        return super().world_size

    @property
    def rank(self) -> int:
        r"""
        Process index in all processes.
        """

        if dist.is_initialized():
            return dist.get_rank()
        return super().rank

    def reduce(
        self, tensor: Tensor, op: ReduceOp = ReduceOp.SUM, group: ProcessGroup = None, async_op: bool = False
    ) -> Tensor:
        r"""
        Perform all reduce across the world.
        """
        if not self.distributed:
            return tensor
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            if is_criterion(value):
                self._criterions[name] = value
            else:
                self._models[name] = value
        elif isinstance(value, Optimizer):
            self._optimizers[name] = value
        elif isinstance(value, LRScheduler):
            self._schedulers[name] = value

        super().__setattr__(name, value)
