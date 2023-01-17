import atexit
import json
import logging
import logging.config
import os
import random
import shutil
from collections.abc import Mapping
from typing import Callable, Optional, Union

import numpy as np
import torch
from chanfig import Config, FlatDict
from torch import Tensor
from torch import distributed as dist
from torch.backends import cudnn

from danling.utils import catch

from .base_runner import BaseRunner
from .utils import on_main_process


class Runner(BaseRunner):
    """
    Set up everything for running a job
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.seed is not None:
            self.init_seed()

        if self.deterministic:
            self.init_deterministic()

        if self.log:
            self.init_logger()
            self.init_print()

        if self.tensorboard:
            self.writer = self.init_tensorboard()

        atexit.register(self.print_result)

    def init_distributed(self) -> None:
        """
        Set up distributed training
        """

        # pylint: disable=W0201

        dist.init_process_group(backend="nccl")
        self.process_index = dist.get_rank()
        self.num_processes = dist.get_world_size()
        self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
        torch.cuda.set_device(self.local_process_index)
        self.is_main_process = self.process_index == 0
        self.is_local_main_process = self.local_process_index == 0

    def init_deterministic(self) -> None:
        """
        Set up deterministic
        """

        cudnn.benchmark = False
        cudnn.deterministic = True

    def init_seed(self) -> None:
        """
        Set up random seed
        """

        if self.seed is None:
            self.seed = random.randint(0, 100000)
            if self.distributed:
                self.seed = self.accelerator.gather(Tensor(self.seed).cuda()).unsqueeze(0).flatten()[0]
        torch.manual_seed(self.seed + self.process_index)
        torch.cuda.manual_seed(self.seed + self.process_index)
        np.random.seed(self.seed + self.process_index)
        random.seed(self.seed + self.process_index)

    @on_main_process
    def init_logger(self) -> None:
        """
        Set up logger
        """

        # Why is setting up proper logging so !@?#! ugly?
        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
                },
                "handlers": {
                    "stdout": {
                        "level": "INFO",
                        "formatter": "standard",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                    },
                    "logfile": {
                        "level": "DEBUG",
                        "formatter": "standard",
                        "class": "logging.FileHandler",
                        "filename": self.log_path,
                        "mode": "a",
                    },
                },
                "loggers": {
                    "": {
                        "handlers": ["stdout", "logfile"],
                        "level": "DEBUG",
                        "propagate": True,
                    },
                },
            }
        )
        logging.captureWarnings(True)
        self.logger = logging.getLogger("runner")
        self.logger.flush = lambda: [h.flush() for h in self.logger.handlers]

    def init_print(self, process: Optional[int] = 0, precision: Optional[int] = 10) -> None:
        """
        Set up print.

        Only print on a specific process or when force is indicated.

        Replace default print function with logging.info.
        """

        torch.set_printoptions(precision=precision)

        logger = logging.getLogger("print")
        logger.flush = lambda: [h.flush for h in logger.handlers]
        import builtins as __builtin__  # pylint: disable=C0415

        builtin_print = __builtin__.print

        @catch
        def print(*args, force=False, end="\n", file=None, flush=False, **kwargs):  # pylint: disable=W0622
            if self.process_index == process or force:
                if self.log:
                    logger.info(*args, **kwargs)
                else:
                    builtin_print(*args, end=end, file=file, flush=flush, **kwargs)

        __builtin__.print = print

    @on_main_process
    def init_tensorboard(self, *args, **kwargs):
        from torch.utils.tensorboard.writer import SummaryWriter  # pylint: disable=C0415

        return SummaryWriter(log_dir=self.dir, *args, **kwargs)

    def init_lr(
        self,
        lr_scale_factor: Optional[float] = None,
        batch_size_base: Optional[int] = None,
    ) -> None:
        """
        Set up learning rate according to linear scaling rule
        """

        # pylint: disable=W0201

        if lr_scale_factor is None:
            if batch_size_base is None:
                if batch_size_base := getattr(self, "batch_size_base", None) is None:
                    raise ValueError("batch_size_base must be specified to auto scale lr")
            lr_scale_factor = self.batch_size_actual / batch_size_base
        self.lr_scale_factor = lr_scale_factor
        self.lr = self.lr * self.lr_scale_factor
        self.lr_final = self.lr_final * self.lr_scale_factor

    @catch
    @on_main_process
    def save_checkpoint(self) -> None:
        """
        Save checkpoint to checkpoint_dir
        """
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        self.save(self.state_dict(), latest_path)
        if hasattr(self, "save_freq") and (self.epochs + 1) % self.save_freq == 0:
            save_path = os.path.join(self.checkpoint_dir, f"epoch-{self.epochs}.pth")
            shutil.copy(latest_path, save_path)
        if self.is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            shutil.copy(latest_path, best_path)

    def load_checkpoint(
        self, checkpoint: Union[Mapping, str], override_config: bool = True, *args, **kwargs
    ) -> None:  # pylint: disable=W1113
        """
        Load runner from checkpoint
        """
        if isinstance(checkpoint, str):
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"checkpoint at {checkpoint} is not a file")
            checkpoint = self.load(checkpoint, *args, **kwargs)
        if override_config:
            self.__dict__.update(checkpoint["runner"])
        if self.model is not None and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        if self.optimizer is not None and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    @classmethod
    def from_checkpoint(
        cls, checkpoint, map_location: str = "cpu", convert_mapping: bool = True, *args, **kwargs
    ) -> BaseRunner:
        """
        Load Runner from checkpoint.
        """

        checkpoint = cls.load(checkpoint, *args, map_location=map_location, **kwargs)
        config = Config(**checkpoint["runner"]) if convert_mapping else checkpoint["runner"]
        runner = cls(**config)
        runner.load_checkpoint(checkpoint, override_config=False)
        return runner

    def state_dict(self, cls: Callable = dict) -> Mapping:
        """
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

    @catch
    @on_main_process
    def save_result(self) -> None:
        r"""
        Save result to `runner.dir`.
        """

        ret = {"id": self.id, "name": self.name}
        result = self.latest_result
        if isinstance(result, FlatDict):
            result = result.dict()
        ret.update(result)  # This is slower but ensure id in the first
        latest_path = os.path.join(self.dir, "latest.json")
        with FlatDict.open(latest_path, "w") as f:
            json.dump(ret, f, indent=4)
        if self.is_best:
            best_path = os.path.join(self.dir, "best.json")
            shutil.copy(latest_path, best_path)

    def print_result(self) -> None:
        r"""
        Print latest and best result.
        """

        print(f"latest result: {self.latest_result}")
        print(f"best result: {self.best_result}")

    def append_result(self, result) -> None:
        r"""
        Add latest result and update best result.
        """

        self.results.append(result)

    def step(self, zero_grad: bool = True) -> None:
        r"""
        Step optimizer and scheduler.
        """

        if self.optimizer is not None:
            self.optimizer.step()
            if zero_grad:
                self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.steps += 1
