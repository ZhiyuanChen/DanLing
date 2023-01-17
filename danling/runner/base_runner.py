from __future__ import annotations

import logging
import logging.config
import os
from collections.abc import Mapping
from json import dumps as json_dumps
from random import randint
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from chanfig import Config, FlatDict, NestedDict
from chanfig.utils import JsonEncoder, YamlDumper
from torch import nn, optim
from yaml import dump as yaml_dump

from danling.utils import catch, is_json_serializable

from .utils import ensure_dir

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]


class BaseRunner:

    r"""
    Base class for all runners.

    Attributes
    ----------
    id: str = f"{self.name}-{self.seed}"
    name: str = "danling"

    seed: int = randint(0, 2 ** 32 - 1)
    deterministic: bool = False

    accelerator: Accelerator
    accelerate: Dict[str, Any] = {}
        keyword arguments for :class:`accelerate`.

    steps: int = 0
        Current running steps.
    epochs: int = 0
        Current running epochs.
    step_begin: int = 0
        End running steps.
    epoch_begin: int = 0
        End running epochs.
    step_end: int = 0
        End running steps.
    epoch_end: int = 0
        End running epochs.

    model: Optional[nn.Module] = None
    criterion: Optional[Tuple[nn.Module]] = None
    optimizer: Optional[optim.Optimizer] = None
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

    datasets: FlatDict
        All datasets, should be in the form of ``{subset: dataset}``.
    datasamplers: FlatDict
        All datasamplers, should be in the form of ``{subset: datasampler}``.
    dataloaders: FlatDict
        All dataloaders, should be in the form of ``{subset: dataloader}``.

    batch_size: int = 1

    results: List[NestedDict] = []
        All results, should be in the form of ``[{subset: {index: score}}]``.
    index_set: Optional[str] = 'val'
        The subset to calculate the core score.
    index: str = 'loss'
        The index to calculate the core score.

    experiments_root: str = "experiments"
        The root directory to save all experiments.
    checkpoint_dir_name: str = "checkpoints"
        The name of the directory under run_dir to save checkpoints.

    log: bool = True
        Whether to log the results.
    logger: Optional[logging.Logger] = None
    tensorboard: bool = False
        Whether to use tensorboard.
    writer: Optional[SummaryWriter] = None
    """

    id: str = ""
    name: str = "DanLing"

    seed: int
    deterministic: bool = False

    accelerator: Accelerator
    accelerate: Dict[str, Any] = {}

    steps: int = 0
    epochs: int = 0
    step_begin: int
    epoch_begin: int
    step_end: int
    epoch_end: int
    """step_begin, epoch_begin, step_end, epoch_end may not present in some runners, so they are not initialised."""

    model: Optional[nn.Module] = None
    criterion: Optional[Tuple[nn.Module]] = None
    optimizer: Optional[optim.Optimizer] = None
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

    datasets: FlatDict
    datasamplers: FlatDict
    dataloaders: FlatDict

    batch_size: int = 1

    results: List[NestedDict] = []
    index_set: Optional[str] = None
    index: str = "loss"

    experiments_root: str = "experiments"
    checkpoint_dir_name: str = "checkpoints"
    log: bool = True
    logger: Optional[logging.Logger] = None
    tensorboard: bool = False
    writer: Optional[SummaryWriter] = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], FlatDict) and not kwargs:
            args, kwargs = (), args[0]
        self.__dict__ = NestedDict(*args, **kwargs)
        if "seed" not in self:
            self.seed = randint(0, 2**32 - 1)
        if not self.id:
            self.id = f"{self.name}-{self.seed}"
        self.accelerator = Accelerator(**self.accelerate)
        self.datasets = FlatDict()
        self.datasamplers = FlatDict()
        self.dataloaders = FlatDict()

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
    def best_result(self) -> Optional[NestedDict]:
        """
        Best result
        """

        return self.results[-1 - self.scores[::-1].index(self.best_score)] if self.results else None

    @property
    def latest_result(self) -> Optional[NestedDict]:
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
    def best_score(self) -> Optional[float]:
        """
        Actual batch size
        """

        return self.best_fn(self.scores) if self.results else None

    @property
    def latest_score(self) -> Optional[float]:
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

        return os.path.join(self.experiments_root, self.id)

    @property
    def log_path(self) -> str:
        """
        Path of log file
        """

        return os.path.join(self.dir, "run.log")

    @property
    @ensure_dir
    def checkpoint_dir(self) -> str:
        """
        Directory of checkpoints
        """

        return os.path.join(self.dir, self.checkpoint_dir_name)

    @catch
    def save(self, obj: Any, f: File, on_main_process: bool = True) -> File:
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
        raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")

    def __contains__(self, name) -> bool:
        return name in self.__dict__

    def dict(self, cls: Callable = dict) -> Mapping:
        """
        Convert config to `cls`
        """

        # pylint: disable=C0103

        ret = cls()
        for k, v in self.__dict__.items():
            if isinstance(v, FlatDict):
                v = v.dict(cls)
            if is_json_serializable(v):
                ret[k] = v
        return ret

    @catch
    def json(self, file: File, on_main_process: bool = True, *args, **kwargs) -> None:
        """
        Dump Runner to json file
        """

        if on_main_process and self.is_main_process or not on_main_process:
            with FlatDict.open(file, mode="w") as fp:
                fp.write(self.jsons(*args, **kwargs))

    @classmethod
    def from_json(cls, file: File, *args, **kwargs) -> BaseRunner:
        r"""
        Construct Runner from json file.

        This function calls `self.from_jsons()` to construct object from json string.
        You may overwrite `from_jsons` in case something is not json serializable.

        """

        with FlatDict.open(file) as fp:
            return cls.from_jsons(fp.read(), *args, **kwargs)

    def jsons(self, *args, **kwargs) -> str:
        """
        Dump Runner to json string
        """

        if "cls" not in kwargs:
            kwargs["cls"] = JsonEncoder
        return json_dumps(self.dict(), *args, **kwargs)

    @classmethod
    def from_jsons(cls, string: str, *args, **kwargs) -> BaseRunner:
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
            with FlatDict.open(file, mode="w") as fp:
                self.yamls(fp, *args, **kwargs)

    @classmethod
    def from_yaml(cls, file: File, *args, **kwargs) -> BaseRunner:
        r"""
        Construct Runner from yaml file.

        This function calls `self.from_yamls()` to construct object from yaml string.
        You may overwrite `from_yamls` in case something is not yaml serializable.
        """

        with FlatDict.open(file) as fp:
            return cls.from_yamls(fp.read(), *args, **kwargs)

    def yamls(self, *args, **kwargs) -> str:
        """
        Dump Runner to yaml string
        """

        if "Dumper" not in kwargs:
            kwargs["Dumper"] = YamlDumper
        return yaml_dump(self.dict(), *args, **kwargs)  # type: ignore

    @classmethod
    def from_yamls(cls, string: str, *args, **kwargs) -> BaseRunner:
        r"""
        Construct Runner from yaml string.
        """

        return cls(**Config.from_yamls(string, *args, **kwargs))

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

    def __repr__(self):
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
