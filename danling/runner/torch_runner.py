import random
from typing import Any, Callable, List, Mapping, Optional

import numpy as np
import torch
from accelerate import Accelerator
from torch import distributed as dist
from torch import nn
from torch.backends import cudnn

from danling.utils import catch

from .base_runner import BaseRunner
from .utils import on_main_process


class TorchRunner(BaseRunner):
    r"""
    Set up everything for running a job.

    Attributes:
        accelerator (Accelerator):
        accelerate: Defaults to `{}`.
            if is `None`, will not use `accelerate`.
    """

    # pylint: disable=R0902

    accelerator: Accelerator
    accelerate: Mapping[str, Any]

    def __init__(self, *args, **kwargs) -> None:
        self.accelerate = {}
        super().__init__(*args, **kwargs)

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        self.accelerator = Accelerator(**self.accelerate)
        if self.distributed:
            object_list = [self.id, self.uuid]
            dist.broadcast_object_list(object_list)
            self.id, self.uuid = object_list[0], object_list[1]

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

    def set_seed(self, seed: Optional[int] = None, bias: Optional[int] = None) -> None:
        r"""
        Set up random seed.

        Args:
            seed: Random seed to set.
                Defaults to `self.seed` (`config.seed`).

            bias: Make the seed different for each processes.

                This avoids same data augmentation are applied on every processes.

                Defaults to `self.rank`.

                Set to `False` to disable this feature.
        """

        if seed is None:
            seed = self.seed
        if self.distributed:
            object_list = [seed]
            dist.broadcast_object_list(object_list)
            seed = object_list[0]
        if bias is None:
            bias = self.rank
        if bias:
            seed += bias
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
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
            runner=self.dict(),
            model=model.state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
        )

    def prepare(self, *args, device_placement: Optional[List[bool]] = None) -> None:
        r"""
        Prepare all objects passed in `args` for distributed training and mixed precision,
        then return them in the same order.
        """

        return self.accelerator.prepare(*args, device_placement=device_placement)

    def autocast(self):
        r"""
        Context manager that enables autocasting for the forward pass (and maybe backward pass).
        """

        return self.accelerator.autocast()

    def backward(self, loss) -> None:
        r"""
        Backward loss to compute gradients.
        """

        return self.accelerator.backward(loss)

    def unwrap_model(self, model: Optional[nn.Module] = None) -> nn.Module:
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
        if self.accelerator is not None and hasattr(self.accelerator, name):
            return getattr(self.accelerator, name)
        return super().__getattr__(name)
