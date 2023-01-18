from __future__ import annotations

import atexit
import json
import logging
import logging.config
import os
import random
import shutil
from typing import Callable, Mapping, Optional, Union

import numpy as np
from chanfig import FlatDict

from danling.utils import catch

from .bases import RunnerBase
from .utils import on_main_process


class BaseRunner(RunnerBase):
    r"""
    Set up everything for running a job.
    """

    # pylint: disable=R0902

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.seed is not None:
            self.set_seed()
        if self.deterministic:
            self.set_deterministic()
        if self.log:
            self.init_logging()
        self.init_print()
        if self.tensorboard:
            self.init_tensorboard()

        atexit.register(self.print_result)

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
        Set up print.

        Only print on a specific process or when force is indicated.

        Parameters
        ----------
        process: int, optional
            The process to print on.

        Notes
        -----
        If `self.log = True`, the default `print` function will be override by `logging.info`.
        """

        logger = logging.getLogger("print")
        logger.flush = lambda: [h.flush for h in logger.handlers]  # type: ignore
        import builtins as __builtin__  # pylint: disable=C0415

        builtin_print = __builtin__.print

        @catch
        def print(*args, force=False, end="\n", file=None, flush=False, **kwargs):  # pylint: disable=W0622
            if self.rank == process or force:
                if self.log:
                    logger.info(*args, **kwargs)
                else:
                    builtin_print(*args, end=end, file=file, flush=flush, **kwargs)

        __builtin__.print = print

    def set_deterministic(self) -> None:
        r"""
        Set up deterministic.
        """

        raise NotImplementedError

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

        if bias is None:
            bias = self.rank
        seed = self.seed + bias if bias else self.seed
        np.random.seed(seed)
        random.seed(seed)

    def scale_lr(
        self,
        lr_scale_factor: Optional[float] = None,
        batch_size_base: Optional[int] = None,
    ) -> None:
        r"""
        Scale learning rate according to [linear scaling rule](https://arxiv.org/abs/1706.02677).
        """

        # pylint: disable=W0201

        if lr_scale_factor is None:
            if batch_size_base is None:
                if batch_size_base := getattr(self, "batch_size_base", None) is None:
                    raise ValueError("batch_size_base must be specified to auto scale lr")
            lr_scale_factor = self.batch_size_equivalent / batch_size_base
        self.lr_scale_factor = lr_scale_factor
        self.lr = self.lr * self.lr_scale_factor  # type: float  # pylint: disable=C0103
        self.lr_final = self.lr_final * self.lr_scale_factor  # type: float

    def step(self, zero_grad: bool = True) -> None:
        r"""
        Step optimizer and scheduler.

        This method also increment the `self.steps` attribute.

        Parameters
        ----------
        zero_grad: bool, optional
            Whether to zero the gradients.
        """

        if self.optimizer is not None:
            self.optimizer.step()
            if zero_grad:
                self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.steps += 1
        # TODO: Support `drop_last = False`
        self.iters += self.batch_size_equivalent

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

    @catch
    @on_main_process
    def save_checkpoint(self) -> None:
        r"""
        Save checkpoint to `runner.checkpoint_dir`.

        The checkpoint will be saved to `runner.checkpoint_dir/latest.pth`.

        If `save_freq` is specified and `self.epochs + 1` is a multiple of `save_freq`,
        the checkpoint will also be copied to `runner.checkpoint_dir/epoch-{self.epochs}.pth`.

        If `self.is_best` is `True`, the checkpoint will also be copied to `runner.checkpoint_dir/best.pth`.
        """

        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        self.save(self.state_dict(), latest_path)
        if hasattr(self, "save_freq") and (self.epochs + 1) % self.save_freq == 0:
            save_path = os.path.join(self.checkpoint_dir, f"epoch-{self.epochs}.pth")
            shutil.copy(latest_path, save_path)
        if self.is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            shutil.copy(latest_path, best_path)

    def load_checkpoint(  # pylint: disable=W1113
        self, checkpoint: Optional[Union[Mapping, str]] = None, override_config: bool = True, *args, **kwargs
    ) -> None:
        """
        Load info from checkpoint.

        Parameters
        ----------
        checkpoint: Optional[Union[Mapping, str]] = latest_checkpoint
            Checkpoint (or its path) to load.
        override_config: bool = True
            If True, override runner config with checkpoint config.
        *args, **kwargs
            Additional arguments to pass to `runner.load`.

        Raises
        ------
        FileNotFoundError
            If `checkpoint` does not exists.

        See also
        --------
        from_checkpoint: Build runner from checkpoint.
        """

        if checkpoint is None:
            checkpoint = os.path.join(self.checkpoint_dir, "latest.pth")
        # TODO: Support loading checkpoints in other format
        if isinstance(checkpoint, str):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"checkpoint is set to {checkpoint} but does not exist.")
            state_dict = self.load(checkpoint, *args, **kwargs)
        # TODO: Wrap state_dict in a dataclass
        if override_config:
            self.__dict__.update(state_dict["runner"])  # type: ignore
        if self.model is not None and "model" in state_dict:  # type: ignore
            self.model.load_state_dict(state_dict["model"])  # type: ignore
        if self.optimizer is not None and "optimizer" in state_dict:  # type: ignore
            self.optimizer.load_state_dict(state_dict["optimizer"])  # type: ignore
        if self.scheduler is not None and "scheduler" in state_dict:  # type: ignore
            self.scheduler.load_state_dict(state_dict["scheduler"])  # type: ignore
        self.checkpoint = checkpoint  # pylint: disable=W0201

    @classmethod
    def from_checkpoint(cls, checkpoint: Union[Mapping, str], *args, **kwargs) -> BaseRunner:
        r"""
        Build BaseRunner from checkpoint.

        Parameters
        ----------
        checkpoint: Optional[Union[Mapping, str]] = latest_checkpoint
            Checkpoint (or its path) to load.
        *args, **kwargs
            Additional arguments to pass to `runner.load`.

        Returns
        -------
        BaseRunner
        """

        if isinstance(checkpoint, str):
            checkpoint = cls.load(checkpoint, *args, **kwargs)
        runner = cls(**checkpoint["runner"])  # type: ignore
        runner.load_checkpoint(checkpoint, override_config=False)
        return runner

    def append_result(self, result) -> None:
        r"""
        Append result to `self.results`.

        Warnings
        --------
        `self.results` is heavily relied upon for computing metrics.

        Failed to use this method may lead to unexpected behavior.
        """

        self.results.append(result)

    def print_result(self) -> None:
        r"""
        Print latest and best result.
        """

        print(f"latest result: {self.latest_result}")
        print(f"best result: {self.best_result}")

    @catch
    @on_main_process
    def save_result(self) -> None:
        r"""
        Save result to `runner.dir`.

        This method will save latest and best result to
        `runner.dir/latest.json` and `runner.dir/best.json` respectively.
        """

        ret = {"id": self.id, "name": self.name}
        result = self.latest_result  # type: ignore
        if isinstance(result, FlatDict):
            result = result.dict()  # type: ignore
        # This is slower but ensure id is the first key
        ret.update(result)  # type: ignore
        latest_path = os.path.join(self.dir, "latest.json")
        with FlatDict.open(latest_path, "w") as f:  # pylint: disable=C0103
            json.dump(ret, f, indent=4)
        if self.is_best:
            best_path = os.path.join(self.dir, "best.json")
            shutil.copy(latest_path, best_path)
