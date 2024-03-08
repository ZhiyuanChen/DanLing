from .accelerate_runner import AccelerateRunner as TorchRunner

import random
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from time import time
from typing import Any
from warnings import warn

import deepspeed
import torch
from chanfig import NestedDict
from deepspeed import DeepSpeedEngine, comm
from deepspeed.comm import ReduceOp
from deepspeed.accelerator import get_accelerator
from torch import Tensor
from torch import distributed as dist
from torch import nn, optim
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # type: ignore

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from danling.data import DataLoader
from danling.utils import catch

from .base_runner import BaseRunner
from .utils import Precision, on_main_process


class TorchRunner(BaseRunner):
    r"""
    Set up everything for running a job.

    `TorchRunner` uses [`deepspeed`][deepspeed] as distributed backend to
    provide seamless experience for large-scale training.

    `TorchRunner` will automatically [`prepare`][accelerate.Accelerator.prepare] everything,
    including `model`, `criterion`, `optimizer`, `scheduler`, and `dataloaders` for distribute training,
    mixed precision, and deepspeed (optional).

    In fact, you don't even need to create `dataloaders`, just define
    `datasets` and `TorchRunner` will create `dataloaders` for you.
    `TorchRunner` will inspect the `train` flag in corresponding dataset to
    automatically set `shuffle`.
    """

    # pylint: disable=R0902

    _model: nn.Module
    _criterion: nn.Module
    _optimizer: optim.Optimizer
    _scheduler: optim.lr_scheduler.LRScheduler

    amp: Any = None
    accelerator: Any = None

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
        self.accelerator = get_accelerator()
        self.amp = self.accelerator.amp()

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__()
        self.initialize()

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
        print(f"Training splits: {train_splits}")
        print(f"Evaluation splits: {eval_splits}")
        print(f"Begin training from epoch {self.state.epoch_begin} to epoch {self.state.epoch_end - 1}")
        patience = self.state.get("patience", float("inf"))
        for epochs in range(self.state.epoch_begin, self.state.epoch_end):  # type: ignore
            self.state.epochs = epochs
            print(f"epoch [{epochs}/{self.state.epoch_end - 1}]")
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
            input = data["input"] if isinstance(data, Mapping) else data[0]
            target = data["target"] if isinstance(data, Mapping) else data[1]
            with self.autocast():
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
                reduced_loss = loss.item()
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
            with self.autocast():
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
                reduced_loss = loss.item()
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

        self.model.backward(loss)
        self.model.step()
        # if self.sync_gradients:
        #     if self.state.get("max_grad_value") is not None:
        #         self.clip_grad_value_(self.model.parameters(), self.state.get("max_grad_value"))  # type: ignore
        #     if self.state.get("max_grad_norm") is not None:
        #         self.clip_grad_norm_(self.model.parameters(), self.state.get("max_grad_norm"))  # type: ignore
        # if self.optimizer is not None:
        #     self.optimizer.step()
        #     if zero_grad:
        #         self.optimizer.zero_grad()
        # if self.scheduler is not None:
        #     self.scheduler.step()
        self.state.steps += 1
        if batch_size is None:
            batch_size = self.batch_size_equivalent
        self.state.iters += batch_size
        # TODO: Support `drop_last = False`
        # self.state.iters += self.batch_size_equivalent

    def backward(
        self,
        loss: Tensor,
        allreduce_gradients: bool = True,
        release_loss: bool = False,
        retain_graph=False,
        scale_wrt_gas: bool = True,
    ):
        r"""
        Perform the backward pass to compute the gradients.
        """

        self.model.backward(
            loss,
            allreduce_gradients=allreduce_gradients,
            release_loss=release_loss,
            retain_graph=retain_graph,
            scale_wrt_gas=scale_wrt_gas,
        )

    def reduce(self, tensor, op=ReduceOp.AVG, group=None, async_op=False):
        if not self.distributed:
            return tensor
        comm.all_reduce(tensor, op=op, group=group, async_op=async_op)
        return tensor

    def initialize(self, **kwargs) -> None:
        r"""
        Prepare all objects passed in `args` for distributed training and mixed precision,
        then return them in the same order.
        """

        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            config=self.deepspeed,
            dist_init_required=False,
            **kwargs
        )

        if self.datasets:
            datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
            dataloader_kwargs = self.state.setdefault("dataloader", self.state.empty())
            for k, d in datasets.items():
                dataloader_kwargs.setdefault("shuffle", getattr(d, "train", k == "train"))
                dataloader_kwargs.setdefault("batch_size", self.batch_size)
                self.dataloaders[k] = DataLoader(d, **dataloader_kwargs)
        if self.dataloaders:
            for d in self.dataloaders.values():
                if hasattr(d, "device"):
                    d.device = self.device

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        dist_backend = "gloo"
        if self.device != torch.device("cpu"):
            torch.cuda.set_device(self.rank)
            dist_backend = "nccl"
        deepspeed.init_distributed(dist_backend=dist_backend, distributed_port=self.state.get("port", 29500))
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

    def unwrap_model(self, model: nn.Module | None = None) -> nn.Module:
        r"""
        Unwrap model for saving checkpoints.

        Args:
            model (Optional[nn.Module]): Model to unwrap.
                Defaults to `self.model`.
        """

        model = model or self.model

        while isinstance(model, (DeepSpeedEngine, DistributedDataParallel)):
            model = model.module
        return model

    def autocast(self, enabled: bool = True, dtype: torch.dtype | str | None = None, cache_enabled: bool = True):
        r"""
        Context manager that enables auto-casting for the forward pass (and maybe backward pass).
        """

        if self.deepspeed.get("fp16.auto_cast"):
            return nullcontext()
        if dtype is None:
            dtype = self.dtype
        return self.amp.autocast(enabled, dtype, cache_enabled)

    @cached_property
    def dtype(self) -> torch.dtype:
        precision = self.state.get("precision", "notset")
        if precision == "notset":
            return torch.float32
        if precision == "fp32":
            return torch.float32
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision == "fp8":
            raise ValueError("fp8 is not currently supported")
            # return torch.float8
        if precision == "int8":
            return torch.int8
        raise ValueError(
            f"Precision should be one of 'fp32', 'bf16', 'fp16', 'fp8', 'int8', or 'notset, but got {precision}"
        )

    @cached_property
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
            return torch.device(self.accelerator.current_device_name())
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

    @cached_property
    def deepspeed(self) -> dict:  # pylint: disable=R0912,R0915
        deepspeed_config = self.state.setdefault("deepspeed", self.state.empty())

        if self.state.zero is not None:
            zero = deepspeed_config.setdefault("zero_optimization", {})
            zero["stage"] = self.state.zero
            if zero.get("allgather_partitions") == "auto":
                zero["allgather_partitions"] = True
            if zero.get("overlap_comm") == "auto":
                zero["overlap_comm"] = True
            if zero.get("reduce_scatter") == "auto":
                zero["reduce_scatter"] = True
            if zero.get("contiguous_gradients") == "auto":
                zero["contiguous_gradients"] = True
            if zero.get("allgather_bucket_size") == "auto":
                zero["allgather_bucket_size"] = 1e6
            if zero.get("reduce_bucket_size") == "auto":
                zero["reduce_bucket_size"] = 1e6
            if zero.get("stage3_max_live_parameters") == "auto":
                zero["stage3_max_live_parameters"] = 1e8
            if zero.get("stage3_max_live_gradients") == "auto":
                zero["stage3_max_live_gradients"] = 1e8
            if zero.get("stage3_max_reuse_distance") == "auto":
                zero["stage3_max_reuse_distance"] = 1e8
            if zero.get("stage3_prefetch_bucket_size") == "auto":
                zero["stage3_prefetch_bucket_size"] = 1e6
            if zero.get("stage3_param_persistence_threshold") == "auto":
                zero["stage3_param_persistence_threshold"] = 1e8

        if self.state.amp is not None:
            amp = deepspeed_config.setdefault("amp", {})
            if amp.get("enabled", "auto") == "auto":
                amp["enabled"] = True
            if amp.get("opt_level", "auto") == "auto":
                amp["opt_level"] = self.state.amp

        if self.state.precision == "bf16":
            bf16 = deepspeed_config.setdefault("bf16", {})
            if bf16.get("enabled", "auto") == "auto":
                bf16["enabled"] = True
        elif self.state.precision == "fp16":
            fp16 = deepspeed_config.setdefault("fp16", {})
            if fp16.get("enabled", "auto") == "auto":
                fp16["enabled"] = True

        if deepspeed_config.get("bf16.enabled", False):
            if self.state.precision == "notset":
                self.state.precision = Precision.bf16
            elif self.state.precision != "bf16":
                raise ValueError(
                    f"Precision is {self.state.precision} in state, which differs from bf16 in deepspeed config"
                )
        if deepspeed_config.get("fp16.enabled", False):
            if self.state.precision == "notset":
                self.state.precision = Precision.fp16
            elif self.state.precision != "fp16":
                raise ValueError(
                    f"Precision is {self.state.precision} in state, which differs from fp16 in deepspeed config"
                )

        if self.state.accum_steps > 1:
            deepspeed_config["gradient_accumulation_steps"] = self.state.accum_steps
        elif "gradient_accumulation_steps" in deepspeed_config:
            self.state.accum_steps = deepspeed_config["gradient_accumulation_steps"]

        if "optimizer" in deepspeed_config:
            if "params" not in deepspeed_config["optimizer"]:
                deepspeed_config["optimizer"]["params"] = {}
            optimizer = deepspeed_config["optimizer"]["params"]
            if optimizer.get("lr", "auto") == "auto":
                optimizer["lr"] = self.state.get("optim.lr", 1e-3)
            if optimizer.get("weight_decay", "auto") == "auto":
                optimizer["weight_decay"] = self.state.get("optim.weight_decay", 1e-2)
            if optimizer.get("betas") == "auto":
                optimizer["betas"] = (0.9, 0.999)
            if optimizer.get("eps") == "auto":
                optimizer["eps"] = 1e-8
        if "scheduler" in deepspeed_config:
            if "params" not in deepspeed_config["scheduler"]:
                deepspeed_config["scheduler"]["params"] = {}
            scheduler = deepspeed_config["scheduler"]["params"]
            if scheduler.get("total_num_steps", "auto") == "auto":
                scheduler["total_num_steps"] = self.total_steps
            if scheduler.get("warmup_num_steps", "auto") == "auto":
                scheduler["warmup_num_steps"] = scheduler["total_num_steps"] // 20
            if scheduler.get("warmup_max_lr", "auto") == "auto":
                if self.optimizer:
                    scheduler["warmup_max_lr"] = self.optimizer.param_groups[0]["lr"]
                elif "optimizer" in deepspeed_config:
                    scheduler["warmup_max_lr"] = deepspeed_config["optimizer"]["params"]["lr"]
                else:
                    raise ValueError("warmup_max_lr is not defined and cannot be inferred")
            if scheduler.get("warmup_min_lr", "auto") == "auto":
                scheduler["warmup_min_lr"] = 1e-7
        if deepspeed_config.get("steps_per_print", "auto") == "auto":
            deepspeed_config["steps_per_print"] = self.print_interval
        if deepspeed_config.get("train_micro_batch_size_per_gpu", "auto") == "auto":
            deepspeed_config["train_micro_batch_size_per_gpu"] = self.state.batch_size
        return deepspeed_config

    # def __setattr__(self, name, value):
    #     if name not in ("model", "criterion") and isinstance(value, nn.Module):
    #         if is_criterion(value):
    #             self.criterion = value
    #         else:
    #             self.model = value
    #     elif name != "optimizer" and isinstance(value, optim.Optimizer):
    #         self.optimizer = value
    #     elif name != "scheduler" and isinstance(value, optim.lr_scheduler.LRScheduler):
    #         self.scheduler = value

    #     super().__setattr__(name, value)
