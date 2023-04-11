from __future__ import annotations

import logging
import logging.config
import os
from typing import IO, Any, Callable, Mapping, Optional, Union

from chanfig import Config, FlatDict, Variable

from danling.utils import catch, load, save

from .runner_state import RunnerState

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]

DEFAULT_EXPERIMENT_NAME = "DanLing"
DEFAULT_EXPERIMENT_ID = "xxxxxxxxxxxxxxxx"


class RunnerBase:
    r"""
    Base class for all runners.

    `RunnerBase` is designed as a "dataclass".

    It defines all basic attributes and relevant properties such as `scores`, `progress`, etc.

    `RunnerBase` also defines basic IO operations such as `save`, `load`, `json`, `yaml`, etc.

    Attributes: Model:
        model (Callable):
        criterion (Callable):
        optimizer:
        scheduler:

    Attributes: Data:
        datasets (FlatDict): All datasets, should be in the form of ``{subset: dataset}``.
        datasamplers (FlatDict): All datasamplers, should be in the form of ``{subset: datasampler}``.
        dataloaders (FlatDict): All dataloaders, should be in the form of ``{subset: dataloader}``.
        batch_size (int, property): Number of samples per batch in train dataloader or the first dataloader.
        batch_size_equivalent (int, property): Total batch_size (`batch_size * world_size * accum_steps`).

    `datasets`, `datasamplers`, `dataloaders` should be a dict with the same keys.
    Their keys should be `split` (e.g. `train`, `val`, `test`).

    Attributes: Parallel Training:
        world_size (int, property): Number of processes.
        rank (int, property): Process index of all processes.
        local_rank (int, property): Process index of local processes.
        distributed (bool, property): If runner is running in distributed mode.
        is_main_process (bool, property): If current process is the main process of all processes.
        is_local_main_process (bool, property): If current process is the main process of local processes.

    Attributes: logging:
        logger:
        writer:

    Notes:
        The `RunnerBase` class is not intended to be used directly, nor to be directly inherit from.

        This is because `RunnerBase` is designed as a "dataclass",
        and is meant for demonstrating all attributes and properties only.

    See Also:
        [`RunnerState`][danling.runner.runner_state.RunnerState]: The runeer base that stores runtime information.
        [`BaseRunner`][danling.runner.BaseRunner]: The base runner class.
    """

    # pylint: disable=R0902, R0904
    # DO NOT set default value in class, as they won't be stored in `__dict__`.

    state: RunnerState

    model: Optional[Callable] = None
    criterion: Optional[Callable] = None
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None

    datasets: FlatDict
    datasamplers: FlatDict
    dataloaders: FlatDict

    logger: Optional[logging.Logger] = None
    writer: Optional[Any] = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.state = RunnerState(*args, **kwargs)
        self.datasets = FlatDict()
        self.datasamplers = FlatDict()
        self.dataloaders = FlatDict()

    @property
    def batch_size(self) -> int:
        r"""
        Batch size.

        Notes:
            If `train` is in `dataloaders`, then `batch_size` is the batch size of `train`.
            Otherwise, `batch_size` is the batch size of the first dataloader.

        Returns:
            (int):
        """

        loader = self.dataloaders["train"] if "train" in self.dataloaders else next(iter(self.dataloaders.values()))
        return loader.batch_size

    @property
    def batch_size_equivalent(self) -> int:
        r"""
        Actual batch size.

        Returns:
            (int): `batch_size` * `world_size` * `accum_steps`
        """

        return self.batch_size * self.world_size * getattr(self, "accum_steps", 1)

    @property
    def accum_steps(self) -> int:
        r"""
        Accumulated steps.

        Returns:
            (int):
        """

        raise AttributeError("accum_steps is not defined.")

    @property
    def device(self) -> Any:
        r"""
        Device of runner.
        """

        raise NotImplementedError

    @property
    def world_size(self) -> int:
        r"""
        Number of processes.
        """

        return 1

    @property
    def rank(self) -> int:
        r"""
        Process index of all processes.
        """

        return 0

    @property
    def local_rank(self) -> int:
        r"""
        Process index of local processes.
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
        If current process is the main process of all processes.
        """

        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        r"""
        If current process is the main process of local processes.
        """

        return self.local_rank == 0

    @catch
    def save(  # pylint: disable=W1113
        self, obj: Any, file: PathStr, main_process_only: bool = True, *args, **kwargs
    ) -> File:
        r"""
        Save any file with supported extensions.

        `Runner.save` internally calls `dl.save`,
        but with additional arguments to allow it save only on the main process.
        Moreover, any error raised by `Runner.save` will be caught and logged.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return save(obj, file, *args, **kwargs)
        return file

    @staticmethod
    def load(file: PathStr, *args, **kwargs) -> Any:  # pylint: disable=C0103
        r"""
        Load any file with supported extensions.

        `Runner.load` is identical to `dl.save`.
        """

        return load(file, *args, **kwargs)

    def dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Convert state to Mapping.

        Args:
            cls: Target `clc to convert to.
        """

        # pylint: disable=C0103

        return self.state.dict(cls)

    @catch
    def json(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=W1113
        r"""
        Dump Runner State to json file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return self.state.json(file, *args, **kwargs)

    @classmethod
    def from_json(cls, file: File, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from json file.

        This function calls `self.from_jsons()` to construct object from json string.
        You may overwrite `from_jsons` in case something is not json serializable.
        """

        with FlatDict.open(file) as fp:  # pylint: disable=C0103
            return cls.from_jsons(fp.read(), *args, **kwargs)

    def jsons(self, *args, **kwargs) -> str:
        r"""
        Dump Runner State to json string.
        """

        return self.state.jsons(*args, **kwargs)

    @classmethod
    def from_jsons(cls, string: str, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from json string.
        """

        return cls(Config.from_jsons(string, *args, **kwargs))

    @catch
    def yaml(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=W1113
        r"""
        Dump Runner State to yaml file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return self.state.yaml(file, *args, **kwargs)

    @classmethod
    def from_yaml(cls, file: File, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from yaml file.

        This function calls `self.from_yamls()` to construct object from yaml string.
        You may overwrite `from_yamls` in case something is not yaml serializable.
        """

        with FlatDict.open(file) as fp:  # pylint: disable=C0103
            return cls.from_yamls(fp.read(), *args, **kwargs)

    def yamls(self, *args, **kwargs) -> str:
        r"""
        Dump Runner State to yaml string.
        """

        return self.state.yamls(*args, **kwargs)

    @classmethod
    def from_yamls(cls, string: str, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from yaml string.
        """

        return cls(Config.from_yamls(string, *args, **kwargs))

    def __getattr__(self, name) -> Any:
        if name in self.state:
            return self.state[name]
        if name in dir(self.state):
            return getattr(self.state, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value) -> None:
        if name in self.__dict__ and isinstance(self.__dict__[name], Variable):
            self.__dict__[name].set(value)
        else:
            self.__dict__[name] = value

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
