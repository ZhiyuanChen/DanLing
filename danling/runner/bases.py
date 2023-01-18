from __future__ import annotations

import logging
import logging.config
import os
from json import dumps as json_dumps
from random import randint
from typing import IO, Any, Callable, List, Mapping, Optional, Sequence, Union

from chanfig import Config, FlatDict, NestedDict
from chanfig.utils import JsonEncoder, YamlDumper
from yaml import dump as yaml_dump

from danling.utils import catch, ensure_dir, is_json_serializable

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]


class RunnerBase:
    r"""
    Base class for all runners.

    `RunnerBase` is designed as a "dataclass".

    It defines all basic attributes and relevant properties such as `scores`, `progress`, etc.

    `RunnerBase` also defines basic IO operations such as `save`, `load`, `json`, `yaml`, etc.

    Attributes
    ----------
    id: str = f"{self.name}-{self.seed}"
    name: str = "danling"
    seed: int = randint(0, 2**32 - 1)
    deterministic: bool = False
        Ensure [deterministic](https://pytorch.org/docs/stable/notes/randomness.html) operations.
    iters: int = 0
        Current running iters.
        Iters refers to the number of data samples processed.
        Iters equals to steps when batch size is 1.
    steps: int = 0
        Current running steps.
        Steps refers to the number of `step` calls.
    epochs: int = 0
        Current running epochs.
        Epochs refers to the number of complete passes over the datasets.
    iter_end: int
        End running iters.
        Note that `step_end` not initialised since this variable may not apply to some Runners.
    step_end: int
        End running steps.
        Note that `step_end` not initialised since this variable may not apply to some Runners.
    epoch_end: int
        End running epochs.
        Note that `epoch_end` not initialised since this variable may not apply to some Runners.
    model: Optional = None
    criterion: Optional = None
    optimizer: Optional = None
    scheduler: Optional = None
    datasets: FlatDict
        All datasets, should be in the form of ``{subset: dataset}``.
    datasamplers: FlatDict
        All datasamplers, should be in the form of ``{subset: datasampler}``.
    dataloaders: FlatDict
        All dataloaders, should be in the form of ``{subset: dataloader}``.
    batch_size: int = 1
    results: List[NestedDict] = []
        All results, should be in the form of ``[{subset: {index: score}}]``.
    index_set: str = 'val'
        The subset to calculate the core score.
    index: str = 'loss'
        The index to calculate the core score.
    experiments_root: str = "experiments"
        The root directory for all experiments.
    checkpoint_dir_name: str = "checkpoints"
        The name of the directory under `runner.dir` to save checkpoints.
    log: bool = True
        Whether to log the results.
    logger: Optional[logging.Logger] = None
    tensorboard: bool = False
        Whether to use tensorboard.
    writer: Optional[SummaryWriter] = None

    Notes
    -----
    The `RunnerBase` class is not intended to be used directly, nor to be directly inherit from.

    This is because `RunnerBase` is designed as a "dataclass",
    and is meant for demonstrating all attributes and properties only.

    See Also
    --------
    [`BaseRunner`][danling.base_runner.BaseRunner]: The base runner class.
    """

    # pylint: disable=R0902, R0904

    id: str = ""
    name: str = "DanLing"

    seed: int
    deterministic: bool

    iters: int
    steps: int
    epochs: int
    # iter_begin: int  # Deprecated
    # step_begin: int  # Deprecated
    # epoch_begin: int  # Deprecated
    iter_end: int
    step_end: int
    epoch_end: int

    model: Optional = None
    criterion: Optional = None
    optimizer: Optional = None
    scheduler: Optional = None

    datasets: FlatDict
    datasamplers: FlatDict
    dataloaders: FlatDict

    batch_size: int

    results: List[NestedDict] = []
    index_set: Optional[str]
    index: str

    experiments_root: str = "experiments"
    checkpoint_dir_name: str = "checkpoints"
    log: bool
    logger: Optional[logging.Logger] = None
    tensorboard: bool
    writer: Optional = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Init attributes that should be kept in checkpoint inside `__init__`.
        # Note that attributes should be init before redefine `self.__dict__`.
        self.deterministic = False
        self.iters = 0
        self.steps = 0
        self.epochs = 0
        self.batch_size = 1
        self.seed = randint(0, 2**32 - 1)
        self.datasets = FlatDict()
        self.datasamplers = FlatDict()
        self.dataloaders = FlatDict()
        self.index_set = None
        self.index = "loss"
        if len(args) == 1 and isinstance(args[0], FlatDict) and not kwargs:
            args, kwargs = (), args[0]
        self.__dict__ = NestedDict(**self.__dict__)
        self.__dict__.update(args)
        self.__dict__.update(kwargs)
        if not self.id:
            self.id = f"{self.name}-{self.seed}"  # pylint: disable=C0103

    @property
    def progress(self) -> float:
        r"""
        Training Progress.

        Returns
        -------
        float

        Raises
        ------
        RuntimeError
            If no terminal is defined.
        """

        if hasattr(self, "iter_end"):
            return self.iters / self.iter_end
        if hasattr(self, "step_end"):
            return self.steps / self.step_end
        if hasattr(self, "epoch_end"):
            return self.epochs / self.epoch_end
        raise RuntimeError("DanLing cannot determine progress since no terminal is defined.")

    @property
    def batch_size_equivalent(self) -> int:
        r"""
        Actual batch size.

        `batch_size` * `world_size` * `accum_steps`
        """

        return self.batch_size * self.world_size * getattr(self, "accum_steps", 1)

    @property
    def latest_result(self) -> Optional[NestedDict]:
        r"""
        Latest result.

        Returns
        -------
        Optional[NestedDict]
        """

        return self.results[-1] if self.results else None

    @property
    def best_result(self) -> Optional[NestedDict]:
        r"""
        Best result.

        Returns
        -------
        Optional[NestedDict]
        """

        return self.results[-1 - self.scores[::-1].index(self.best_score)] if self.results else None  # type: ignore

    @property
    def scores(self) -> List[float]:
        r"""
        All scores.

        Scores are extracted from results by `index_set` and `runner.index`,
        following `[r[index_set][self.index] for r in self.results]`.

        By default, `index_set` points to `self.index_set` and is set to `val`,
        if `self.index_set` is not set, it will be the last key of the last result.

        Scores are considered as the index of the performance of the model.
        It is useful to determine the best model and the best hyper-parameters.

        Returns
        -------
        List[float]
        """

        if not self.results:
            return []
        index_set = self.index_set or next(reversed(self.results[-1]))
        return [r[index_set][self.index] for r in self.results]

    @property
    def latest_score(self) -> Optional[float]:
        r"""
        Latest score.

        Returns
        -------
        Optional[float]
        """

        return self.scores[-1] if self.results else None

    @property
    def best_score(self) -> Optional[float]:
        r"""
        Best score.

        Returns
        -------
        Optional[float]
        """

        return self.best_fn(self.scores) if self.results else None

    @staticmethod
    def best_fn(scores: Sequence[float], fn: Callable = max) -> float:  # pylint: disable=C0103
        r"""
        Function to determine the best score from a list of scores.

        Subclass can override this method to accommodate needs, such as `min(scores)`.

        Parameters
        ----------
        scores: Sequence[float]
            List of scores.
        fn: Callable = max
            Function to determine the best score from a list of scores.

        Returns
        -------
        best_score: float
            The best score from a list of scores.
        """

        return fn(scores)

    @property
    def is_best(self) -> bool:
        r"""
        If current epoch is the best epoch.

        Returns
        -------
        bool
        """

        return abs(self.latest_score - self.best_score) < 1e-7
        # return self.latest_score == self.best_score

    @property
    def world_size(self) -> int:
        r"""
        Number of Processes.

        Returns
        -------
        int
        """

        return 1

    @property
    def rank(self) -> int:
        r"""
        Process index in all processes.

        Returns
        -------
        int
        """

        return 0

    @property
    def local_rank(self) -> int:
        r"""
        Process index in local processes.

        Returns
        -------
        int
        """

        return 0

    @property
    def distributed(self) -> bool:
        r"""
        If runner is running in distributed mode.
        """

        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        r"""
        If current process is the main process.

        Returns
        -------
        bool
        """

        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        r"""
        If current process is the main process in local.

        Returns
        -------
        bool
        """

        return self.local_rank == 0

    @property
    @ensure_dir
    def dir(self) -> str:
        r"""
        Directory of the experiment.

        Returns
        -------
        str
        """

        return os.path.join(self.experiments_root, self.id)

    @property
    def log_path(self) -> str:
        r"""
        Path of log file.

        Returns
        -------
        str
        """

        return os.path.join(self.dir, "run.log")

    @property
    @ensure_dir
    def checkpoint_dir(self) -> str:
        r"""
        Directory of checkpoints.

        Returns
        -------
        str
        """

        return os.path.join(self.dir, self.checkpoint_dir_name)

    @catch
    def save(self, obj: Any, f: File, main_process_only: bool = True) -> File:  # pylint: disable=C0103
        r"""
        Save object to a path or file.

        Returns
        -------
        File
        """

        raise NotImplementedError

    @staticmethod
    def load(f: File, *args, **kwargs) -> Any:  # pylint: disable=C0103
        r"""
        Load object from a path or file.

        Returns
        -------
        Any
        """

        raise NotImplementedError

    def dict(self, cls: Callable = dict, only_json_serializable: bool = True) -> Mapping:
        r"""
        Convert config to Mapping.

        Note that all non-json-serializable objects will be removed.

        Parameters
        ----------
        cls : Callable = dict
            Class to convert to.
        only_json_serializable : bool = True
            If only json serializable objects should be kept.

        Returns
        -------
        Mapping
        """

        # pylint: disable=C0103

        ret = cls()
        for k, v in self.__dict__.items():
            if isinstance(v, FlatDict):
                v = v.dict(cls)
            if not only_json_serializable or is_json_serializable(v):
                ret[k] = v
        return ret

    @catch
    def json(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=W1113
        r"""
        Dump Runner to json file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            with FlatDict.open(file, mode="w") as fp:  # pylint: disable=C0103
                fp.write(self.jsons(*args, **kwargs))

    @classmethod
    def from_json(cls, file: File, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from json file.

        This function calls `self.from_jsons()` to construct object from json string.
        You may overwrite `from_jsons` in case something is not json serializable.

        Returns
        -------
        RunnerBase
        """

        with FlatDict.open(file) as fp:  # pylint: disable=C0103
            return cls.from_jsons(fp.read(), *args, **kwargs)

    def jsons(self, *args, **kwargs) -> str:
        r"""
        Dump Runner to json string.

        Returns
        -------
        json: str
        """

        if "cls" not in kwargs:
            kwargs["cls"] = JsonEncoder
        return json_dumps(self.dict(), *args, **kwargs)

    @classmethod
    def from_jsons(cls, string: str, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from json string.

        Returns
        -------
        RunnerBase
        """

        return cls(**Config.from_jsons(string, *args, **kwargs))

    @catch
    def yaml(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=W1113
        r"""
        Dump Runner to yaml file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            with FlatDict.open(file, mode="w") as fp:  # pylint: disable=C0103
                self.yamls(fp, *args, **kwargs)

    @classmethod
    def from_yaml(cls, file: File, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from yaml file.

        This function calls `self.from_yamls()` to construct object from yaml string.
        You may overwrite `from_yamls` in case something is not yaml serializable.

        Returns
        -------
        RunnerBase
        """

        with FlatDict.open(file) as fp:  # pylint: disable=C0103
            return cls.from_yamls(fp.read(), *args, **kwargs)

    def yamls(self, *args, **kwargs) -> str:
        r"""
        Dump Runner to yaml string.

        Returns
        -------
        yaml: str
        """

        if "Dumper" not in kwargs:
            kwargs["Dumper"] = YamlDumper
        return yaml_dump(self.dict(), *args, **kwargs)  # type: ignore

    @classmethod
    def from_yamls(cls, string: str, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from yaml string.

        Returns
        -------
        RunnerBase
        """

        return cls(**Config.from_yamls(string, *args, **kwargs))

    def __getattr__(self, name) -> Any:
        if "accelerator" not in self:
            raise RuntimeError(f"{self.__class__.__name__} is not properly initialised")
        if hasattr(self.accelerator, name):
            return getattr(self.accelerator, name)
        raise AttributeError(f"{self.__class__.__name__} does not contain {name}")

    def __contains__(self, name) -> bool:
        return name in self.__dict__

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

    def _add_indent(self, text):
        lines = text.split("\n")
        # don't do anything for single-line stuff
        if len(lines) == 1:
            return text
        first = lines.pop(0)
        # add 2 spaces to each line but the first
        lines = [(2 * " ") + line for line in lines]
        lines = "\n".join(lines)
        lines = first + "\n" + lines
        return lines
