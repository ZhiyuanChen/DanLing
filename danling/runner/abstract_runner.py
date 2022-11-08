from __future__ import annotations

import logging
import logging.config
import os
from collections import Mapping
from json import dumps as json_dumps
from typing import IO, Any, Callable, List, Optional, Tuple, Union

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from accelerate import Accelerator
from chanfig import Config, NestedDict, OrderedDict
from yaml import dump as yaml_dump

from danling.utils import catch, is_json_serializable

from .utils import ensure_dir, on_main_process

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]


class AbstractRunner:

    config: Config

    id: str
    name: str

    seed: int
    deterministic: bool

    experiment_dir: str
    checkpoint_dir_name: str

    steps: int
    epochs: int

    accelerator: Accelerator
    accelerate: NestedDict[str, Any]

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    datasets: OrderedDict[str, data.Dataset]
    datasamplers: OrderedDict[str, data.Sampler]
    dataloaders: OrderedDict[str, data.DataLoader]

    batch_size: int

    criterion: Tuple[nn.Module]

    metric: str
    results: List[NestedDict[str, any]]
    result_best: NestedDict[str, any]
    result_latest: NestedDict[str, any]
    score_best: float
    score_latest: float
    is_best: bool

    log: bool
    logger: logging.Logger
    tensorboard: bool
    writer: Callable

    def __init__(self, config: Optional[Config] = None, *args, **kwargs):
        super().__init__()
        config = config or Config()
        super().__setattr__("config", config)
        self.id = None
        self.name = "danling"
        self.seed = 1031
        self.deterministic = False

        self.experiment_dir = "experiments"
        self.checkpoint_dir_name = "checkpoints"

        self.steps = 0
        self.epochs = 0
        self.accelerate = {}

        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None

        self.datasets = OrderedDict()
        self.datasamplers = OrderedDict()
        self.dataloaders = OrderedDict()

        self.batch_size = 1

        self.criterion = None

        self.metric = "loss"
        self.results = []
        self.result_best = NestedDict()
        self.result_latest = NestedDict()
        self.score_best = 0
        self.score_latest = 0
        self.is_best = False

        self.log = True
        self.tensorboard = False
        self.writer = None
        self.config.update(args)
        self.config.update(kwargs)

    @property
    def distributed(self):
        """
        If runner is in distributed mode
        """

        return self.num_processes > 1

    @property
    def batch_size_actual(self) -> int:
        """
        Actual batch size
        """

        return self.batch_size * self.num_processes * getattr(self, "accum_steps", 1)

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

    @property
    @ensure_dir
    def dir(self) -> str:
        return os.path.join(self.experiment_dir, self.id)

    @property
    @ensure_dir
    def checkpoint_dir(self) -> str:
        return os.path.join(self.dir, self.checkpoint_dir_name)

    @catch
    @on_main_process
    def save(self, obj: Any, f: File) -> None:
        """
        Save object to a path or file
        """

        self.accelerator.save(obj, f)

    def __getattr__(self, name) -> Any:
        if name in self.config:
            return self.config[name]
        if hasattr(self.accelerator, name):
            return getattr(self.accelerator, name)
        raise AttributeError(f'"Runner" object has no attribute "{name}"')

    def __contains__(self, name) -> bool:
        return name in self.config

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
