from __future__ import annotations

import logging
import logging.config
import os
from collections.abc import Mapping
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from chanfig import Config, NestedDict, OrderedDict
from chanfig.config import JsonEncoder, YamlDumper, json_dumps, yaml_dump
from torch import nn, optim
from torch.utils import data

from danling.utils import catch, is_json_serializable

from .utils import ensure_dir

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]


class Runner:

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
    index_set: Optional[str] = None
    index: str = "loss"

    log: bool = True
    logger: Optional[logging.Logger] = None
    tensorboard: bool = False
    writer: Optional[SummaryWriter] = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict) and not kwargs:
            args, kwargs = (), args[0]
        self.__dict__ = NestedDict()
        self.__dict__.update(args)
        self.__dict__.update(kwargs)
        self.accelerator = Accelerator(**self.accelerate)

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
        """
        Best result
        """

        return self.results[-1 - self.scores[::-1].index(self.best_score)] if self.results else None

    @property
    def latest_result(self) -> NestedDict:
        """
        Latest Results
        """

        return self.results[-1] if self.results else None

    @property
    def scores(self) -> List[float]:
        """
        Scores
        """

        if not self.results:
            return []
        index_set = self.index_set or next(reversed(self.results[-1]))
        return [r[index_set][self.index] for r in self.results]

    @property
    def best_score(self) -> float:
        """
        Actual batch size
        """

        return self.best_fn(self.scores) if self.results else None

    @property
    def latest_score(self) -> float:
        """
        Actual batch size
        """

        return self.scores[-1] if self.results else None

    @staticmethod
    def best_fn(scores: Sequence[float]) -> float:
        """
        Function to determine best score from a list of scores
        """

        return max(scores)

    @property
    def is_best(self) -> bool:
        """
        If current epoch is the best epoch
        """

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
        if hasattr(self, "step_end"):
            return self.steps / self.step_end
        if hasattr(self, "epoch_end"):
            return self.epochs / self.epoch_end
        return 0

    @property
    @ensure_dir
    def dir(self) -> str:
        """
        Directory of experiment
        """

        return os.path.join(self.experiment_dir, self.id)

    @property
    @ensure_dir
    def checkpoint_dir(self) -> str:
        """
        Directory of checkpoints
        """

        return os.path.join(self.dir, self.checkpoint_dir_name)

    @catch
    def save(self, obj: Any, f: File, on_main_process: bool = True) -> None:
        """
        Save object to a path or file
        """

        if on_main_process and self.is_main_process or not on_main_process:
            self.accelerator.save(obj, f)
        return f

    @staticmethod
    def load(f: File, *args, **kwargs) -> Any:
        """
        Load object from a path or file
        """

        return torch.load(f, *args, **kwargs)

    def __getattr__(self, name) -> Any:
        if "accelerator" not in self:
            raise RuntimeError(f"{self.__class__.__name__} is not properly initialised")
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

    def to(self, cls: Callable = dict) -> Mapping:  # pylint: disable=C0103
        """
        Convert config to `cls`
        """

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

    @classmethod
    def from_json(cls, file: File, *args, **kwargs) -> Runner:
        r"""
        Construct Runner from json file.

        This function calls `self.from_jsons()` to construct object from json string.
        You may overwrite `from_jsons` in case something is not json serializable.

        """

        with open(file) as fp:
            return cls.from_jsons(fp.read(), *args, **kwargs)

    def jsons(self, *args, **kwargs) -> str:
        """
        Dump Runner to json string
        """

        if "cls" not in kwargs:
            kwargs["cls"] = JsonEncoder
        return json_dumps(self.to(dict), *args, **kwargs)

    @classmethod
    def from_jsons(cls, string: str, *args, **kwargs) -> Runner:
        r"""
        Construct Runner from json string.

        """

        return cls(**Config.from_jsons(string, *args, **kwargs))

    @catch
    def yaml(self, file: File, on_main_process: bool = True, *args, **kwargs) -> None:
        """
        Dump Runner to yaml file
        """

        if on_main_process and self.is_main_process or not on_main_process:
            with NestedDict.open(file, mode="w") as fp:
                self.yamls(fp, *args, **kwargs)

    @classmethod
    def from_yaml(cls, file: File, *args, **kwargs) -> Runner:
        r"""
        Construct Runner from yaml file.

        This function calls `self.from_yamls()` to construct object from yaml string.
        You may overwrite `from_yamls` in case something is not yaml serializable.

        """

        with open(file) as fp:
            return cls.from_yamls(fp.read(), *args, **kwargs)

    def yamls(self, *args, **kwargs) -> str:
        """
        Dump Runner to yaml string
        """

        if "Dumper" not in kwargs:
            kwargs["Dumper"] = YamlDumper
        return yaml_dump(self.to(dict), *args, **kwargs)  # type: ignore

    @classmethod
    def from_yamls(cls, string: str, *args, **kwargs) -> Runner:
        r"""
        Construct Runner from yaml string.

        """

        return cls(**Config.from_yamls(string, *args, **kwargs))
