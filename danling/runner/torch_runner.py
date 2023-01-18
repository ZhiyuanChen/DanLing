import os
import random
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch
from accelerate import Accelerator
from torch import distributed as dist
from torch.backends import cudnn

from danling.typing import File
from danling.utils import catch

from .base_runner import BaseRunner
from .utils import on_main_process

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter


class TorchRunner(BaseRunner):
    r"""
    Set up everything for running a job.

    Attributes
    ----------
    accelerator: Accelerator
    accelerate: Dict[str, Any] = {}
    """

    # pylint: disable=R0902

    accelerator: Accelerator
    accelerate: Dict[str, Any] = {}

    def __init__(self, *args, **kwargs) -> None:
        self.accelerator = Accelerator(**self.accelerate)
        super().__init__(*args, **kwargs)

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        r"""
        Set up Tensoraoard SummaryWriter.
        """
        from torch.utils.tensorboard.writer import SummaryWriter  # pylint: disable=C0415

        if "log_dir" not in kwargs:
            kwargs["log_dir"] = self.dir

        self.writer = SummaryWriter(*args, **kwargs)

    def set_seed(self, bias: Optional[int] = None) -> None:
        r"""
        Set up random seed.

        Parameters
        ----------
        bias: Optional[int] = self.rank
            Make the seed different for each processes.
            This avoids same data augmentation are applied on every processes.
            Set to `False` to disable this feature.
        """

        if self.distributed:
            # TODO: use broadcast_object instead.
            self.seed = (
                self.accelerator.gather(torch.tensor(self.seed).cuda())
                .unsqueeze(0)
                .flatten()[0]  # pylint: disable=E1101
            )
        if bias is None:
            bias = self.rank
        seed = self.seed + bias if bias else self.seed
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

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.

        .. deprecated:: 0.1.0
            `init_distributed` is deprecated in favor of `Accelerator()`.
        """

        # pylint: disable=W0201

        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        torch.cuda.set_device(self.local_rank)
        self.is_main_process = self.rank == 0
        self.is_local_main_process = self.local_rank == 0

    @catch
    def save(self, obj: Any, f: File, main_process_only: bool = True) -> File:  # pylint: disable=C0103
        r"""
        Save object to a path or file.
        Returns
        -------
        File
        """

        if main_process_only and self.is_main_process or not on_main_process:
            self.accelerator.save(obj, f)
        return f

    @staticmethod
    def load(f: File, *args, **kwargs) -> Any:  # pylint: disable=C0103
        r"""
        Load object from a path or file.
        Returns
        -------
        Any
        """

        return torch.load(f, *args, **kwargs)  # type: ignore

    @property
    def world_size(self) -> int:
        r"""
        Number of Processes.

        Returns
        -------
        int
        """

        return self.accelerator.num_processes

    @property
    def rank(self) -> int:
        r"""
        Process index in all processes.

        Returns
        -------
        int
        """

        return self.accelerator.process_index

    @property
    def local_rank(self) -> int:
        r"""
        Process index in local processes.

        Returns
        -------
        int
        """

        return self.accelerator.local_process_index
