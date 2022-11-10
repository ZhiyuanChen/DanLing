from __future__ import annotations

import logging
import logging.config
import os
from collections import Mapping
from json import dumps as json_dumps
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from accelerate import Accelerator
from chanfig import NestedDict, OrderedDict
from yaml import dump as yaml_dump

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from danling.utils import catch, is_json_serializable

from .utils import ensure_dir, on_main_process

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]


class AbstractRunner:

    id: str = None
    name: str = "danling"

    seed: int = 1031
    deterministic: bool = False

    experiment_dir: str = "experiments"
    checkpoint_dir_name: str = "checkpoints"

    steps: int = 0
    epochs: int = 0

    accelerator: Accelerator
    accelerate: Dict[str, Any] = {}

    model: Optional[nn.Module] = None
    optimizer: Optional[optim.Optimizer] = None
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

    datasets: OrderedDict[str, data.Dataset] = OrderedDict()
    datasamplers: OrderedDict[str, data.Sampler] = OrderedDict()
    dataloaders: OrderedDict[str, data.DataLoader] = OrderedDict()

    batch_size: int = 1

    criterion: Tuple[nn.Module] = None

    results: List[NestedDict[str, any]] = []
    metric_set: Optional[str] = None
    metric: str = "loss"

    log: bool = True
    logger: Optional[logging.Logger] = None
    tensorboard: bool = False
    writer: Optional[SummaryWriter] = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__dict__ = NestedDict()
        self.__dict__.update(args)
        self.__dict__.update(kwargs)

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
    def best_result(self) -> NestedDict:
        return self.results[reversed(self.scores).index(self.best_score)] if self.results else None

    @property
    def latest_result(self) -> NestedDict:
        return self.results[-1] if self.results else None

    @property
    def scores(self) -> List[float]:
        if not self.results:
            return []
        metric_set = self.metric_set or next(reversed(self.results[-1]))
        return [r[metric_set][self.metric] for r in self.results]

    @property
    def best_score(self) -> float:
        return max(self.scores) if self.results else None

    @property
    def latest_score(self) -> float:
        return self.scores[-1] if self.results else None

    @property
    def is_best(self) -> bool:
        return self.latest_score == self.best_score

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
    def save(self, obj: Any, f: File, on_main_process: bool = True) -> None:
        """
        Save object to a path or file
        """

        if on_main_process and self.is_main_process or not on_main_process:
            self.accelerator.save(obj, f)

    def __getattr__(self, name) -> Any:
        if hasattr(self.accelerator, name):
            return getattr(self.accelerator, name)
        raise AttributeError(f'"Runner" object has no attribute "{name}"')

    def __contains__(self, name) -> bool:
        return name in self.__dict__

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
                v = v.to(cls)
            if is_json_serializable(v):
                ret[k] = v
        return ret

    convert = to

    @catch
    def json(self, file: File, on_main_process: bool = True, *args, **kwargs) -> None:
        """
        Dump Runner to json file
        """

        if on_main_process and self.is_main_process or not on_main_process:
            with NestedDict.open(file, mode="w") as fp:
                fp.write(self.jsons(*args, **kwargs))

    def jsons(self, *args, **kwargs) -> str:
        """
        Dump Runner to json string
        """

        return json_dumps(self.to(dict), *args, **kwargs)

    @catch
    def yaml(self, file: File, on_main_process: bool = True, *args, **kwargs) -> None:
        """
        Dump Runner to yaml file
        """

        if on_main_process and self.is_main_process or not on_main_process:
            with NestedDict.open(file, mode="w") as fp:
                self.yamls(fp, *args, **kwargs)

    def yamls(self, *args, **kwargs) -> str:
        """
        Dump Runner to yaml string
        """

        return yaml_dump(self.to(dict), *args, **kwargs)  # type: ignore
