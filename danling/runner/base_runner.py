from __future__ import annotations

import atexit
import json
import logging
import logging.config
import os
import random
import shutil
from collections.abc import Mapping
from json import dumps as json_dumps
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from chanfig import Config, NestedDict, OrderedDict
from torch import nn, optim
from torch.backends import cudnn
from yaml import dump as yaml_dump

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from danling.utils import catch, is_json_serializable

from .utils import ensure_dir, on_main_process

PathStr = Union[PathLike, str, bytes]
File = Union[PathStr, IO]


class BaseRunner:
    """
    Set up everything for running a job
    """

    config: Config
    id: str = None
    name: str = "danling"

    seed: int = 1031
    deterministic: bool = False

    experiment_dir: str = "experiments"
    checkpoint_dir_name: str = "checkpoints"

    steps: int = 0
    epochs: int = 0

    accelerator: Accelerator
    accelerate: Dict = {}

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    datasets: NestedDict = NestedDict()
    datasamplers: NestedDict = NestedDict()
    dataloaders: NestedDict = NestedDict()

    batch_size: int = 0

    criterion: Tuple[nn.Module] = tuple()

    metric: str = "loss"
    results: List[NestedDict] = []
    result_best: NestedDict = NestedDict()
    result_latest: NestedDict = NestedDict()
    score_best: float = 0
    score_latest: float = 0
    is_best: bool = False

    log: bool = True
    logger: Optional[logging.Logger] = None
    tensorboard: bool = False
    writer: Optional[SummaryWriter] = None

    def __init__(self, config: Optional[Config] = None, *args, **kwargs) -> None:
        super().__init__()
        if config.id is None:
            config.id = f"{self.name}-{self.seed}"
       self.config = config or Config()

        self.accelerator = Accelerator(**self.accelerate)

        if self.seed is not None:
            self.init_seed()

        if self.deterministic:
            self.init_deterministic()

        if self.log:
            self.init_logger()
            self.init_print()

        if self.tensorboard:
            self.writer = self.init_tensorboard()

        if self.is_main_process:
            self.yaml(os.path.join(self.dir, "runner.yaml"))

        atexit.register(self.print_result)

    def init_distributed(self) -> None:
        """
        Set up distributed training
        """

        dist.init_process_group(backend="nccl")
        self.process_index = dist.get_rank()
        self.num_processes = dist.get_world_size()
        self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
        torch.cuda.set_device(self.local_process_index)
        self.is_main_process = self.process_index == 0
        self.is_local_main_process = self.local_process_index == 0

    @staticmethod
    def init_deterministic() -> None:
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
                self.seed = self.accelerator.gather(torch.tensor(self.seed).cuda()).unsqueeze(0).flatten()[0]
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
                        "filename": os.path.join(self.dir, "run.log"),
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
        Set up print
        Only print from a specific process or force indicated
        Replace default print function with logging.info
        """

        torch.set_printoptions(precision=precision)

        logger = logging.getLogger("print")
        logger.flush = lambda: [h.flush for h in logger.handlers]
        import builtins as __builtin__

        builtin_print = __builtin__.print

        @catch
        def print(*args, force=False, end="\n", file=None, flush=False, **kwargs):
            if self.process_index == process or force:
                if self.log:
                    logger.info(*args, **kwargs)
                else:
                    builtin_print(*args, end=end, file=file, flush=flush, **kwargs)

        __builtin__.print = print

    @on_main_process
    def init_tensorboard(self, *args, **kwargs):
        from torch.utils.tensorboard.writer import SummaryWriter

        return SummaryWriter(log_dir=self.dir, *args, **kwargs)

    def init_lr(
        self,
        lr_scale_factor: Optional[float] = None,
        batch_size_base: Optional[int] = None,
    ) -> None:
        """
        Set up learning rate according to linear scaling rule
        """

        if lr_scale_factor is None:
            if batch_size_base is None:
                if batch_size_base := getattr(self, "batch_size_base", None) is None:
                    raise ValueError("batch_size_base must be specified to auto scale lr")
            lr_scale_factor = self.batch_size_actual / batch_size_base
        self.lr_scale_factor = lr_scale_factor
        self.lr = self.lr * self.lr_scale_factor
        self.lr_final = self.lr_final * self.lr_scale_factor

    @property
    def batch_size_actual(self) -> int:
        """
        Actual batch size
        """

        return self.batch_size * self.num_processes * getattr(self, "accum_steps", 1)

    @catch
    @on_main_process
    def save(self, obj: Any, f: File) -> None:
        """
        Save object to a path or file
        """

        self.accelerator.save(obj, f)

    def state_dict(self, cls: Callable = dict) -> Mapping:
        """
        Return dict of all attributes for checkpoint
        """

        if self.model is None:
            raise ValueError("Model must be defined when calling state_dict")
        model = self.accelerator.unwrap_model(self.model)
        return cls(
            runner=self.to(dict),
            model=model.state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
        )

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

    def load_checkpoint(self, path, *args, **kwargs) -> None:
        """
        Load runner from checkpoint
        """

        print(f'=> loading checkpoint "{path}"')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint at {path} is not a file")
        checkpoint = torch.load(path, *args, **kwargs)
        self.config.update(checkpoint["runner"])
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        if self.optimizer is not None and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        print(f'=> loaded checkpoint "{path}"')

    @catch
    @on_main_process
    def save_result(self) -> None:
        """
        Save result to dir
        """

        ret = {"id": self.id, "name": self.name}
        result = self.result_latest
        if isinstance(result, OrderedDict):
            result = result.dict()
        ret.update(result)  # This is slower but ensure id in the first
        latest_path = os.path.join(self.dir, "latest.json")
        with open(latest_path, "w") as f:
            json.dump(ret, f, indent=4)
        if self.is_best:
            best_path = os.path.join(self.dir, "best.json")
            shutil.copy(latest_path, best_path)

    def print_result(self) -> None:
        """
        Print latest and best result
        """

        print(f"latest result: {self.result_latest}")
        print(f"best result: {self.result_best}")

    def append_result(self, result) -> None:
        """
        Add latest result and update best result
        """

        self.results.append(result)
        self.result_latest = result

    def update_score(self, score) -> None:
        """
        Add latest result and update best result
        """

        self.is_best = False
        self.score_latest = score
        if self.score_latest > self.score_best:
            self.is_best = True
            self.score_best = self.score_latest
            self.result_best = self.result_latest

    @property
    def distributed(self):
        """
        If runner is in distributed mode
        """

        return self.num_processes > 1

    @property
    @ensure_dir
    def dir(self) -> str:
        return os.path.join(self.experiment_dir, self.id)

    @property
    @ensure_dir
    def checkpoint_dir(self) -> str:
        return os.path.join(self.dir, self.checkpoint_dir_name)

    def step(self) -> None:
        """
        step optimizer and scheduler
        """

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.steps += 1

    @property
    def iters(self) -> int:
        """
        Number of iterations
        """

        return self.steps * self.batch_size_actual

    @property
    def progress(self) -> float:
        """
        Training Progress
        """

        if hasattr(self, "iter_end"):
            return self.iters / self.iter_end
        elif hasattr(self, "step_end"):
            return self.steps / self.step_end
        elif hasattr(self, "epoch_end"):
            return self.epochs / self.epoch_end
        return 0

    def to(self, cls: Callable = dict) -> Mapping:
        ret = cls()
        for k, v in self.__dict__.items():
            if isinstance(v, OrderedDict):
                v = v.convert(cls)
            if is_json_serializable(v):
                ret[k] = v
        return ret

    convert = to

    def json(self, file: File, *args, **kwargs) -> None:
        """
        Dump Runner to json file
        """

        with NestedDict.open(file, mode="w") as fp:
            fp.write(self.jsons(*args, **kwargs))

    def jsons(self, *args, **kwargs) -> str:
        """
        Dump Runner to json string
        """

        return json_dumps(self.to(dict), *args, **kwargs)

    def yaml(self, file: File, *args, **kwargs) -> None:
        """
        Dump Runner to yaml file
        """

        with NestedDict.open(file, mode="w") as fp:
            self.yamls(fp, *args, **kwargs)

    def yamls(self, *args, **kwargs) -> str:
        """
        Dump Runner to yaml string
        """

        return yaml_dump(self.to(dict), *args, **kwargs)  # type: ignore

    def __getattr__(self, name) -> Any:
        if name in self.config:
            return self.config[name]
        if hasattr(self.accelerator, name):
            return getattr(self.accelerator, name)
        raise AttributeError(f'"Runner" object has no attribute "{name}"')

    def __contains__(self, name) -> bool:
        return name in self.__dict__ or name in self.config

    def __repr__(self):
        r"""
        Representation of OrderedDict.

        Example:
        ```python
        >>> d = OrderedDict(a=1, b=2, c=3)
        >>> repr(d)
        'OrderedDict(\n  (a): 1\n  (b): 2\n  (c): 3\n)'

        ```
        """

        lines = []
        for key, value in self.__dict__.items():
            value_str = repr(value)
            value_str = self._add_indent(value_str)
            lines.append("(" + key + "): " + value_str)

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def _add_indent(self, s):
        st = s.split("\n")
        # don't do anything for single-line stuff
        if len(st) == 1:
            return s
        first = st.pop(0)
        st = [(2 * " ") + line for line in st]  # hardcode indent to 2
        st = "\n".join(st)
        st = first + "\n" + st
        return st
