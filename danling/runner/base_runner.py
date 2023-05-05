from __future__ import annotations

import logging
import logging.config
import os
import random
import shutil
from typing import Callable, Mapping, Optional, Union
from warnings import warn

import numpy as np
from chanfig import FlatDict, NestedDict

from danling.utils import catch

from .runner_base import RunnerBase
from .utils import on_main_process


class BaseRunner(RunnerBase):
    r"""
    Base class for running a neural network.

    `BaseRunner` sets up basic running environment, including `seed`, `deterministic`, and `logging`.

    `BaseRunner` also provides some basic methods, such as, `step`, `state_dict`, `save_checkpoint`, `load_checkpoint`.

    All runners should inherit `BaseRunner`.
    """

    # pylint: disable=R0902

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.init_distributed()
        if self.state.seed is not None:
            self.set_seed()
        if self.state.deterministic:
            self.set_deterministic()
        if self.state.log:
            self.init_logging()
        self.init_print()
        if self.state.tensorboard:
            self.init_tensorboard()

    @on_main_process
    def init_logging(self) -> None:
        r"""
        Set up logging.
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
        self.logger.flush = lambda: [h.flush() for h in self.logger.handlers]  # type: ignore

    def init_print(self, process: int = 0) -> None:
        r"""
        Set up `print`.

        Only print on a specific `process` or when `force = True`.

        Args:
            process: The process to `print` on.

        Notes
        -----
        If `self.state.log = True`, the default `print` function will be override by `logging.info`.
        """

        logger = logging.getLogger("print")
        logger.flush = lambda: [h.flush for h in logger.handlers]  # type: ignore
        import builtins as __builtin__  # pylint: disable=C0415

        builtin_print = __builtin__.print

        @catch
        def print(*args, force=False, end="\n", file=None, flush=False, **kwargs):  # pylint: disable=W0622
            if self.rank == process or force:
                if self.state.log:
                    logger.info(*args, **kwargs)
                else:
                    builtin_print(*args, end=end, file=file, flush=flush, **kwargs)

        __builtin__.print = print

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        r"""
        Set up Tensoraoard SummaryWriter.
        """
        raise NotImplementedError

    def set_seed(self, seed: Optional[int] = None, bias: Optional[int] = None) -> None:
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

        if seed is None:
            seed = self.state.seed
        if bias is None:
            bias = self.rank
        if bias:
            seed += bias
        np.random.seed(seed)
        random.seed(seed)

    def set_deterministic(self) -> None:
        r"""
        Set up deterministic.
        """

        raise NotImplementedError

    def scale_lr(
        self,
        lr: float,  # pylint: disable=C0103
        lr_scale_factor: Optional[float] = None,
        batch_size_base: Optional[int] = None,
    ) -> float:
        r"""
        Scale learning rate according to [linear scaling rule](https://arxiv.org/abs/1706.02677).
        """

        # pylint: disable=W0201

        if lr_scale_factor is None:
            if batch_size_base is None:
                batch_size_base = getattr(self, "batch_size_base", None)
                if batch_size_base is None:
                    raise ValueError("batch_size_base must be specified to auto scale lr")
            lr_scale_factor = self.batch_size_equivalent / batch_size_base
        elif batch_size_base is not None:
            warn(
                "batch_size_base will be ignored if lr_scale_factor is specified",
                RuntimeWarning,
            )
        lr = lr * lr_scale_factor  # pylint: disable=C0103, E1101
        self.lr_scale_factor = lr_scale_factor
        return lr

    def step(self, zero_grad: bool = True, batch_size: Optional[int] = None) -> None:
        r"""
        Step optimizer and scheduler.

        This method increment `self.state.steps`.

        This method also increment `self.state.iters` when `batch_size` is specified.

        Args:
            zero_grad: Whether to zero the gradients.
        """

        if self.optimizer is not None:
            self.optimizer.step()
            if zero_grad:
                self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.steps += 1
        if batch_size is not None:
            self.state.iters += batch_size
        # TODO: Support `drop_last = False`
        # self.state.iters += self.batch_size_equivalent

    def state_dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Return dict of all attributes for checkpoint.
        """

        raise NotImplementedError

    @catch
    @on_main_process
    def save_checkpoint(self) -> None:
        r"""
        Save checkpoint to `self.checkpoint_dir`.

        The checkpoint will be saved to `self.checkpoint_dir/latest.pth`.

        If `self.state.save_interval` is positive and `self.state.epochs + 1` is a multiple of `save_interval`,
        the checkpoint will also be copied to `self.checkpoint_dir/epoch-{self.state.epochs}.pth`.

        If `self.state.is_best` is `True`, the checkpoint will also be copied to `self.checkpoint_dir/best.pth`.
        """

        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        self.save(self.state_dict(), latest_path)
        if (
            hasattr(self, "save_interval")
            and self.save_interval > 0
            and (self.state.epochs + 1) % self.save_interval == 0
        ):
            save_path = os.path.join(self.checkpoint_dir, f"epoch-{self.state.epochs}.pth")
            shutil.copy(latest_path, save_path)
        if self.is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            shutil.copy(latest_path, best_path)

    def load_checkpoint(  # pylint: disable=W1113
        self, checkpoint: Optional[Union[Mapping, str]] = None, override_state: bool = False, *args, **kwargs
    ) -> None:
        """
        Load info from checkpoint.

        Args:
            checkpoint: Checkpoint (or its path) to load.
                Defaults to `self.checkpoint_dir/latest.pth`.
            override_state: If True, override runner state with checkpoint state.
                Defaults to `False`.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Raises:
            FileNotFoundError: If `checkpoint` does not exists.

        See Also:
            [`from_checkpoint`][danling.BaseRunner.from_checkpoint]: Build runner from checkpoint.
            [`load_pretrained`][danling.BaseRunner.load_pretrained]: Load parameters from pretrained checkpoint.
        """

        if checkpoint is None:
            checkpoint = os.path.join(self.checkpoint_dir, "latest.pth")  # type: ignore
        # TODO: Support loading checkpoints in other format
        if isinstance(checkpoint, str):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"checkpoint is set to {checkpoint} but does not exist.")
            self.checkpoint = checkpoint  # pylint: disable=W0201
            checkpoint: Mapping = self.load(checkpoint, *args, **kwargs)  # type: ignore
        # TODO: Wrap state_dict in a dataclass
        if override_state:
            self.__dict__.update(NestedDict(**checkpoint["runner"]))  # type: ignore
        if self.model is not None and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])  # type: ignore
        if self.optimizer is not None and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])  # type: ignore

    def load_pretrained(self, checkpoint: Union[Mapping, str], *args, **kwargs) -> None:
        """
        Load parameters from pretrained checkpoint.

        Args:
            checkpoint: Pretrained checkpoint (or its path) to load.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Raises:
            FileNotFoundError: If `checkpoint` does not exists.

        See Also:
            [`load_checkpoint`][danling.BaseRunner.load_checkpoint]: Load info from checkpoint.
        """

        # TODO: Support loading checkpoints in other format
        if isinstance(checkpoint, str):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"pretrained is set to {checkpoint} but does not exist.")
            checkpoint: Mapping = self.load(checkpoint, *args, **kwargs)  # type: ignore
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]  # type: ignore
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]  # type: ignore
        self.model.load_state_dict(checkpoint)  # type: ignore

    @classmethod
    def from_checkpoint(cls, checkpoint: Union[Mapping, str], *args, **kwargs) -> BaseRunner:
        r"""
        Build BaseRunner from checkpoint.

        Args:
            checkpoint: Checkpoint (or its path) to load.
                Defaults to `self.checkpoint_dir/latest.pth`.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Returns:
            (BaseRunner):
        """

        if isinstance(checkpoint, str):
            checkpoint = cls.load(checkpoint, *args, **kwargs)
        runner = cls(**checkpoint["runner"])  # type: ignore
        runner.load_checkpoint(checkpoint, override_state=False)
        return runner

    def append_result(self, result) -> None:
        r"""
        Append result to `self.state.results`.

        Warnings:
            `self.state.results` is heavily relied upon for computing metrics.

            Failed to use this method may lead to unexpected behavior.
        """

        self.state.results.append(result)

    def print_result(self) -> None:
        r"""
        Print latest and best result.
        """

        print(f"results: {self.state.results}")
        print(f"latest result: {self.latest_result}")
        print(f"best result: {self.best_result}")

    @catch
    @on_main_process
    def save_result(self) -> None:
        r"""
        Save result to `self.dir`.

        This method will save latest and best result to
        `self.dir/latest.json` and `self.dir/best.json` respectively.
        """

        results_path = os.path.join(self.dir, "results.json")
        self.save({"id": self.state.id, "name": self.state.name, "results": self.state.results}, results_path, indent=4)
        ret = {"id": self.state.id, "name": self.state.name}
        result = self.latest_result  # type: ignore
        if isinstance(result, FlatDict):
            result = result.dict()  # type: ignore
        # This is slower but ensure id is the first key
        if result is not None:
            ret.update(result)  # type: ignore
        latest_path = os.path.join(self.dir, "latest.json")
        self.save(ret, latest_path, indent=4)
        if self.is_best:
            best_path = os.path.join(self.dir, "best.json")
            shutil.copy(latest_path, best_path)