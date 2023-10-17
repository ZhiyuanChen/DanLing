from __future__ import annotations

import logging
import logging.config
import os
import random
import shutil
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from sys import version_info
from typing import Any
from warnings import warn

from chanfig import Config, FlatDict, NestedDict, Variable

from danling.metrics import AverageMeter, AverageMeters, Metrics
from danling.typing import File, PathStr
from danling.utils import catch, ensure_dir, load, save

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # type: ignore

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from .runner_state import RunnerState
from .utils import RunnerMeta, RunnerMode, on_main_process

PY38_PLUS = version_info >= (3, 8)
IGNORED_SET_NAMES = ("index", "epoch", "step", "iter")
__APPEND_RESULT_COUNTER__ = 0


class BaseRunner(metaclass=RunnerMeta):
    r"""
    Base class for all runners.

    `BaseRunner` sets up basic running environment, including `seed`, `deterministic`, and `logging`.

    `BaseRunner` also provides some basic methods, such as, `step`, `state_dict`, `save_checkpoint`, `load_checkpoint`.

    `BaseRunner` defines all basic attributes and relevant properties such as `scores`, `progress`, etc.

    Attributes: Core:
        mode (RunnerMode, property): Running mode.
        state (RunnerState): Running state. See `RunnerState` for details.

    Attributes: Model:
        model (Callable):
        criterion (Callable):
        optimizer:
        scheduler:

    Attributes: Data:
        datasets (FlatDict): All datasets, should be in the form of ``{subset: dataset}``.
            Initialised to `FlatDict` by default.
        datasamplers (FlatDict): All datasamplers, should be in the form of ``{subset: datasampler}``.
            Initialised to `FlatDict` by default.
        dataloaders (FlatDict): All dataloaders, should be in the form of ``{subset: dataloader}``.
            Initialised to `FlatDict` by default.
        batch_size (int, property): Number of samples per batch in train dataloader or the first dataloader.
        batch_size_equivalent (int, property): Total batch_size (`batch_size * world_size * accum_steps`).

    `datasets`, `datasamplers`, `dataloaders` should be a dict with the same keys.
    Their keys should be `split` (e.g. `train`, `val`, `test`).

    Attributes: Progress:
        progress (float, property): Running Progress, in `range(0, 1)`.

    Attributes: Results:
        results (NestedDict): Results include all metric information of the model.
            Results should be in the form of `{epoch: {subset: {metric: score}}}`.
        latest_result (NestedDict, property): Most recent result, should be in the form of `{subset: {metric: score}}`.
        best_result (NestedDict, property): Best result, should be in the form of `{subset: {metric: score}}`.
        scores (List[float], property): Score is the core metric that is used to evaluate the performance of the model.
            Scores should be in the form of `{epoch: score}`.
        latest_score (float, property): Most recent score, should be in the form of `score`.
        best_score (float, property): Best score, should be in the form of `score`.
        score_set (Optional[str]): The subset to calculate the score.
            If is `None`, will use the last set of the result.
        score_name (str): The metric name of score.
            Defaults to `"loss"`.
        is_best (bool, property): If `latest_score == best_score`.

    Attributes: IO:
        dir (str, property): Directory of the run.
            Defaults to `os.path.join(self.project_root, f"{self.name}-{self.id}")`.
        checkpoint_dir (str, property): Directory of checkpoints.
        log_path (str, property):  Path of log file.
        checkpoint_dir_name (str): The name of the directory under `runner.dir` to save checkpoints.
            Defaults to `"checkpoints"`.

    Attributes: Parallel Training:
        world_size (int, property): Number of processes.
        rank (int, property): Process index of all processes.
        local_rank (int, property): Process index of local processes.
        distributed (bool, property): If runner is running in distributed mode.
        is_main_process (bool, property): If current process is the main process of all processes.
        is_local_main_process (bool, property): If current process is the main process of local processes.

    Attributes: logging:
        meters (AverageMeters): Average meters.
            Initialised to `AverageMeters` by default.
        metrics (Metrics): Metrics for evaluating.
        logger:
        writer:

    See Also:
        [`RunnerState`][danling.runner.runner_state.RunnerState]: The runeer base that stores runtime information.
        [`BaseRunner`][danling.runner.BaseRunner]: The base runner class.
    """

    # pylint: disable=R0902, R0904
    # DO NOT set default value in class, as they won't be stored in `__dict__`.

    _mode: RunnerMode
    state: RunnerState

    model: Callable | None = None
    criterion: Callable | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None

    datasets: FlatDict
    datasamplers: FlatDict
    dataloaders: FlatDict

    meters: AverageMeters
    metrics: Metrics | None = None
    logger: logging.Logger | None = None
    writer: Any | None = None

    def __init__(self, config: NestedDict) -> None:
        if "datasets" not in self.__dict__:
            self.datasets = FlatDict()
        if "datasamplers" not in self.__dict__:
            self.datasamplers = FlatDict()
        if "dataloaders" not in self.__dict__:
            self.dataloaders = FlatDict()
        self._mode = RunnerMode.train
        self.meters = AverageMeters()
        self.metrics = None
        # must init state at last to avoid conflict names
        self.state = RunnerState(config)
        self.init_distributed()
        if self.state.seed is not None:
            self.set_seed()
        if self.state.deterministic:
            self.set_deterministic()
        if os.listdir(self.dir):
            warn(
                f"Directory `{self.dir}` is not empty.",
                category=RuntimeWarning,
                stacklevel=2,
            )
        if self.state.log:
            self.init_logging()
        self.init_print()
        if self.state.tensorboard:
            self.init_tensorboard()

    def __post_init__(self, *args, **kwargs) -> None:
        pass

    @property
    def mode(self) -> RunnerMode:
        return self._mode

    @mode.setter
    def mode(self, mode: str | RunnerMode) -> None:
        if isinstance(mode, str):
            mode = RunnerMode(mode)
        self._mode = mode
        if self.model is not None:
            self.model.train(mode == RunnerMode.train)  # type: ignore

    @cached_property
    def batch_size(self) -> int:
        r"""
        Batch size.

        Notes:
            If `train` is in `dataloaders`, then `batch_size` is the batch size of `train`.
            Otherwise, `batch_size` is the batch size of the first dataloader.

        Returns:
            (int):
        """

        batch_size = self.state.get("dataloader.batch_size")
        if batch_size:
            return batch_size
        if self.dataloaders:
            loader = self.dataloaders["train"] if "train" in self.dataloaders else next(iter(self.dataloaders.values()))
            return loader.batch_size
        raise AttributeError("batch_size could not be inferred, since no dataloader found.")

    @property
    def batch_size_equivalent(self) -> int:
        r"""
        Actual batch size.

        Returns:
            (int): `batch_size` * `world_size` * `accum_steps`
        """

        return self.batch_size * self.world_size * getattr(self, "accum_steps", 1)

    @cached_property
    def total_epochs(self) -> int:
        if self.state.epoch_end:
            return self.state.epoch_end - self.state.epoch_begin
        raise ValueError("epoch_end is not specified")

    @cached_property
    def total_steps(self) -> int:
        if self.state.step_end:
            return self.state.step_end - self.state.step_begin
        dataset = self.datasets.get("train", next(iter(self.datasets.values())))
        return self.total_epochs * ceil(len(dataset) / self.batch_size)

    @cached_property
    def accum_steps(self) -> int:
        r"""
        Accumulated steps.

        Returns:
            (int):
        """

        return self.state.get("accum_steps", 1)

    def init_distributed(self) -> None:
        r"""
        Initialise distributed running environment.
        """

        raise NotImplementedError

    @property
    def device(self) -> Any:
        r"""
        Device of runner.
        """

        return "cpu"

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

        `Runner.load` is identical to `dl.load`.
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
    def from_json(cls, file: File, *args, **kwargs) -> BaseRunner:
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
    def from_jsons(cls, string: str, *args, **kwargs) -> BaseRunner:
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
    def from_yaml(cls, file: File, *args, **kwargs) -> BaseRunner:
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
    def from_yamls(cls, string: str, *args, **kwargs) -> BaseRunner:
        r"""
        Construct Runner from yaml string.
        """

        return cls(Config.from_yamls(string, *args, **kwargs))

    @property
    def progress(self) -> float:
        r"""
        Training Progress.

        Returns:
            (float):

        Raises:
            RuntimeError: If no terminal is defined.
        """

        return self.steps / self.total_steps

    @property
    def best_fn(self) -> Callable:
        r"""
        Function to determine the best score from a list of scores.

        By default, the `best_fn` returns `min` if `self.state.score_name` is `loss`,
        otherwise, returns `max`.

        Subclass can override this method to accommodate needs, such as `min`.

        Returns:
            (callable):
        """

        return max if self.state.score_name != "loss" else min

    @property
    def best_index(self) -> int:
        r"""
        Find the best index from all scores.

        Returns:
            (int):
        """

        if not self.scores:
            return 0
        values = list(self.scores.values())
        return self.best_fn(range(len(values)), key=values.__getitem__)

    @property
    def latest_result(self) -> NestedDict | None:
        r"""
        Latest result.
        """

        if not self.state.results:
            return None
        latest_index = next(reversed(self.state.results if PY38_PLUS else list(self.state.results)))  # type: ignore
        ret = self.state.results[latest_index].clone()
        ret["index"] = latest_index
        return ret

    @property
    def best_result(self) -> NestedDict | None:
        r"""
        Best result.
        """

        if not self.state.results:
            return None
        best_index = self.best_index
        ret = self.state.results[best_index].clone()
        ret["index"] = best_index
        return ret

    @property
    def scores(self) -> FlatDict | None:
        r"""
        All scores.

        Scores are extracted from results by `score_set` and `runner.state.score_name`,
        following `[r[score_set][self.state.score_name] for r in self.state.results]`.

        Scores are considered as the index of the performance of the model.
        It is useful to determine the best model and the best hyper-parameters.

        `score_set` is defined in `self.state.score_set`.
        If it is not set, `DanLing` will use `val` or `validate` if they appear in the `latest_result`.
        If `DanLing` still could not find, it will fall back to the second key in the `latest_result`
        if it contains more that one element, or the first key.

        Note that certain keys are ignored when falling back, they are defined in {IGNORED_SET_NAMES}.
        """

        if not self.state.results:
            return None
        subsets = [i for i in self.latest_result.keys() if i not in IGNORED_SET_NAMES]  # type: ignore
        score_set = self.state.get("score_set")
        if score_set is None and "val" in subsets:
            score_set = "val"
        if score_set is None and "validate" in subsets:
            score_set = "validate"
        if score_set is None:
            score_set = subsets[1] if len(subsets) > 1 else subsets[0]
        return FlatDict({k: v[score_set][self.state.score_name] for k, v in self.state.results.items()})

    @property
    def latest_score(self) -> float | None:
        r"""
        Latest score.
        """

        if not self.state.results:
            return None
        if not PY38_PLUS:
            return next(reversed(list(self.scores.values())))  # type: ignore
        return next(reversed(self.scores.values()))  # type: ignore

    @property
    def best_score(self) -> float | None:
        r"""
        Best score.
        """

        if not self.state.results:
            return None
        return self.scores[self.best_index]  # type: ignore

    @property
    def is_best(self) -> bool:
        r"""
        If current epoch is the best epoch.
        """

        if not self.state.results:
            return True
        try:
            return abs(self.latest_score - self.best_score) < 1e-7  # type: ignore
        except TypeError:
            return True

    @cached_property
    @ensure_dir
    def dir(self) -> str:
        r"""
        Directory of the run.
        """

        if "dir" in self.state:
            return self.state.dir
        return os.path.join(self.project_root, f"{self.name}-{self.id}")

    @cached_property
    def log_path(self) -> str:
        r"""
        Path of log file.
        """

        if "log_path" in self.state:
            return self.state.log_path
        return os.path.join(self.dir, "run.log")

    @cached_property
    @ensure_dir
    def checkpoint_dir(self) -> str:
        r"""
        Directory of checkpoints.
        """

        if "checkpoint_dir" in self.state:
            return self.state.checkpoint_dir
        return os.path.join(self.dir, self.checkpoint_dir_name)

    def __getattribute__(self, name) -> Any:
        if name in ("__class__", "__dict__"):
            return super().__getattribute__(name)
        if name in dir(self):
            return super().__getattribute__(name)
        if name in self.__dict__:
            return self.__dict__[name]
        if "state" in self and name in self.state:
            return self.state[name]
        return super().__getattribute__(name)

    def __getattr__(self, name) -> Any:
        if "state" not in self:
            raise RuntimeError("Runner is not initialised yet.")
        if name in dir(self.state):
            return getattr(self.state, name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value) -> None:
        if name in self.__dict__:
            if isinstance(self.__dict__[name], Variable):
                self.__dict__[name].set(value)
            else:
                self.__dict__[name] = value
        elif "state" in self and name in self.state:
            if isinstance(self.state[name], Variable):
                self.state[name].set(value)
            else:
                self.state[name] = value
        else:
            object.__setattr__(self, name, value)

    def __contains__(self, name) -> bool:
        return name in dir(self) or ("state" in self.__dict__ and name in dir(self.state))

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

    def init_deepspeed(self, config: dict | None = None) -> dict:  # type: ignore # pylint: disable=R0912,R0915
        r"""
        Preprocess DeepSpeed config.
        """

        if config is None:
            config = self.state.get("deepspeed")
        if config is None:
            return {}
        if isinstance(config, str):
            config = NestedDict.load(config)
        if config.get("steps_per_print", "auto") == "auto":
            config["steps_per_print"] = self.print_interval
        if config.get("train_micro_batch_size_per_gpu", "auto") == "auto":
            config["train_micro_batch_size_per_gpu"] = self.batch_size
        if "amp" in config:
            amp = config["amp"]
            if amp.get("enabled", "auto") == "auto":
                amp["enabled"] = "true"
            if amp.get("opt_level", "auto") == "auto":
                amp["opt_level"] = "O1"
        if "zero_optimization" in config:
            zero = config["zero_optimization"]
            if zero.get("allgather_bucket_size") == "auto":
                zero["allgather_bucket_size"] = 1e6
            if zero.get("reduce_bucket_size") == "auto":
                zero["reduce_bucket_size"] = 1e6
            if zero.get("stage3_max_live_parameters") == "auto":
                zero["stage3_max_live_parameters"] = 1e8
            if zero.get("stage3_max_live_gradients") == "auto":
                zero["stage3_max_live_gradients"] = 1e8
            if zero.get("stage3_max_reuse_distance") == "auto":
                zero["stage3_max_reuse_distance"] = 1e8
            if zero.get("stage3_prefetch_bucket_size") == "auto":
                zero["stage3_prefetch_bucket_size"] = 1e6
            if zero.get("stage3_param_persistence_threshold") == "auto":
                zero["stage3_param_persistence_threshold"] = 1e8
            if "amp" in config:
                if "fp16" not in config:
                    config["fp16"] = {}
                if config["fp16"].get("enabled", "auto"):
                    config["fp16"]["enabled"] = config["amp"]["enabled"]
                warn(
                    f"AMP is not compatible with ZeRO. Automatically set 'fp16' to {config['amp']['enabled']}",
                    stacklevel=2,
                )
                del config["amp"]
        if "optimizer" in config:
            if "params" not in config["optimizer"]:
                config["optimizer"]["params"] = {}
            optimizer = config["optimizer"]["params"]
            if optimizer.get("lr", "auto") == "auto":
                optimizer["lr"] = self.state.get("optim.lr", 1e-3)
            if optimizer.get("weight_decay", "auto") == "auto":
                optimizer["weight_decay"] = self.state.get("optim.weight_decay", 1e-2)
            if optimizer.get("betas") == "auto":
                optimizer["betas"] = (0.9, 0.999)
            if optimizer.get("eps") == "auto":
                optimizer["eps"] = 1e-8
        if "scheduler" in config:
            if "params" not in config["scheduler"]:
                config["scheduler"]["params"] = {}
            scheduler = config["scheduler"]["params"]
            if scheduler.get("total_num_steps", "auto") == "auto":
                scheduler["total_num_steps"] = self.total_steps
            if scheduler.get("warmup_num_steps", "auto") == "auto":
                scheduler["warmup_num_steps"] = scheduler["total_num_steps"] // 20
            if scheduler.get("warmup_max_lr", "auto") == "auto":
                if self.optimizer:
                    scheduler["warmup_max_lr"] = self.optimizer.param_groups[0]["lr"]
                elif "optimizer" in config:
                    scheduler["warmup_max_lr"] = config["optimizer"]["params"]["lr"]
                else:
                    raise ValueError("warmup_max_lr is not defined and cannot be inferred")
            if scheduler.get("warmup_min_lr", "auto") == "auto":
                scheduler["warmup_min_lr"] = 1e-7
        return config

    @on_main_process
    def init_logging(self) -> None:
        r"""
        Set up logging.
        """

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
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

    def set_seed(self, seed: int | None = None, bias: int | None = None) -> None:
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

        seed = seed or self.state.seed
        bias = bias or self.rank
        if bias:
            seed += bias
        if np_random is not None:
            np_random.seed(seed)
        random.seed(seed)

    def set_deterministic(self) -> None:
        r"""
        Set up deterministic.
        """

        raise NotImplementedError

    def scale_lr(
        self,
        lr: float,  # pylint: disable=C0103
        lr_scale_factor: float | None = None,
        batch_size_base: int | None = None,
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
                "batch_size_base will be ignored if lr_scale_factor is specified", category=RuntimeWarning, stacklevel=2
            )
        lr = lr * lr_scale_factor  # pylint: disable=C0103, E1101
        self.lr_scale_factor = lr_scale_factor
        return lr

    def step(self, loss, batch_size: int | None = None, zero_grad: bool = True) -> None:
        r"""
        Backward loss and step optimizer & scheduler.

        This method increment `self.state.steps`.

        This method also increment `self.state.iters` when `batch_size` is specified.

        Args:
            zero_grad: Whether to zero the gradients.
        """

        raise NotImplementedError

    def state_dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Return dict of all attributes for checkpoint.
        """

        return cls(self.state)

    @catch
    @on_main_process
    def save_checkpoint(self) -> None:
        r"""
        Save checkpoint to `self.checkpoint_dir`.

        The checkpoint will be saved to `self.checkpoint_dir/latest.pth`.

        If `self.state.save_interval` is positive and `self.state.epochs + 1` is a multiple of `save_interval`,
        the checkpoint will also be copied to `self.checkpoint_dir/epoch-{self.state.epochs}.pth`.

        If `self.is_best` is `True`, the checkpoint will also be copied to `self.checkpoint_dir/best.pth`.
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
        self,
        checkpoint: Mapping | bytes | str | os.PathLike | None = None,
        auto_resume: bool | None = None,
        override_state: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Load info from checkpoint.

        Args:
            checkpoint: Checkpoint (or its path) to load.
                Defaults to `self.state.checkpoint`.
            auto_resume: Automatically resume from latest checkpoint if exists.
                Defaults to `False`.
                If is `True` and `checkpoint` is None, will set it to `self.checkpoint_dir/latest.pth`.
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

        checkpoint = checkpoint if checkpoint is not None else self.state.get("checkpoint")
        auto_resume = auto_resume if auto_resume is not None else self.state.get("auto_resume", False)

        # TODO: Support loading checkpoints in other format
        if checkpoint is not None:
            if auto_resume:
                warn(
                    "latest checkpoint is preempted by value specified in checkpoint",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if isinstance(checkpoint, (bytes, str, os.PathLike)):
                if not os.path.exists(checkpoint):
                    raise FileNotFoundError(f"checkpoint is set to {checkpoint!r} but does not exist.")
                self.state.checkpoint = checkpoint
                ckpt = self.load(checkpoint, *args, **kwargs)
            elif isinstance(checkpoint, Mapping):
                ckpt = checkpoint
            else:
                raise ValueError(f"pretrained is set to {checkpoint!r} but is not a valid checkpoint.")
        elif auto_resume:
            checkpoint = os.path.join(self.checkpoint_dir, "latest.pth")
            if os.path.exists(checkpoint):
                self.state.checkpoint = checkpoint
                ckpt = self.load(checkpoint, *args, **kwargs)
            else:
                warn("latest checkpoint does not exits", category=RuntimeWarning, stacklevel=2)
                return
        else:
            raise ValueError("checkpoint is not specified and auto_resume is not set to True")

        # TODO: Wrap state_dict in a dataclass
        self.state.merge(ckpt["runner"], overwrite=override_state)
        if self.model is not None and "model" in ckpt:
            model = self.unwrap_model(self.model)
            model.load_state_dict(ckpt["model"])
        if self.optimizer is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.state.iter_begin = self.state.iters
        self.state.step_begin = self.state.steps
        self.state.epoch_begin = self.state.epochs

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> BaseRunner:
        r"""
        Build BaseRunner from checkpoint.

        Args:
            checkpoint: Checkpoint (or its path) to load.
            *args: Additional arguments to pass to `cls.load`.
            **kwargs: Additional keyword arguments to pass to `cls.load`.

        Returns:
            (BaseRunner):
        """

        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            ckpt = cls.load(checkpoint, *args, **kwargs)
        elif isinstance(checkpoint, Mapping):
            ckpt = checkpoint
        else:
            raise ValueError(f"checkpoint is set to {checkpoint} but is not a valid checkpoint.")
        runner = cls(**ckpt["runner"])
        runner.load_checkpoint(ckpt, override_state=False)
        return runner

    def load_pretrained(  # pylint: disable=W1113
        self, checkpoint: Mapping | bytes | str | os.PathLike | None = None, *args, **kwargs
    ) -> None:
        """
        Load parameters from pretrained checkpoint.

        This method only loads the model weights.

        Args:
            checkpoint: Pretrained checkpoint (or its path) to load.
                Defaults to `self.state.pretrained`.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Raises:
            FileNotFoundError: If `checkpoint` does not exists.

        See Also:
            [`load_checkpoint`][danling.BaseRunner.load_checkpoint]: Load info from checkpoint.
        """

        # TODO: Support loading checkpoints in other format
        checkpoint = checkpoint if checkpoint is not None else self.state.get("pretrained")
        if checkpoint is None:
            raise ValueError("pretrained is not specified")
        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"pretrained is set to {checkpoint!r} but does not exist.")
            ckpt = self.load(checkpoint, *args, **kwargs)
        elif isinstance(checkpoint, Mapping):
            ckpt = checkpoint
        else:
            raise ValueError(f"pretrained is set to {checkpoint!r} but is not a valid checkpoint.")
        ckpt = ckpt.get("model", ckpt)
        ckpt = ckpt.get("state_dict", ckpt)
        model = self.unwrap_model(self.model)
        model.load_state_dict(ckpt)

    def append_result(self, result: NestedDict, index: int | None = None) -> None:
        r"""
        Append result to `self.state.results`.

        Warnings:
            `self.state.results` is heavily relied upon for computing metrics.

            Failed to use this method may lead to unexpected behavior.
        """

        if index is None:
            index = self.state.epochs
            global __APPEND_RESULT_COUNTER__
            __APPEND_RESULT_COUNTER__ += 1
            if index == 0 and __APPEND_RESULT_COUNTER__ > 1:
                warn(
                    """
                    Automatically set index to `self.state.epochs`.
                    Please ensure `self.state.epochs` updates before calling `append_result`
                    """,
                    category=RuntimeWarning,
                    stacklevel=2,
                )
        if index in self.state.results:
            self.state.results[index].merge(result)
        else:
            self.state.results[index] = result

    def print_result(self) -> None:
        r"""
        Print latest and best result.
        """

        print(f"results: {self.state.results}")
        print(f"latest result: {self.latest_result}")
        print(f"best result: {self.best_result}")

    def step_log(self, split: str, iteration: int, length: int | None = None):
        if length is None:
            length = len(self.dataloaders[split]) - 1
        result = self.meters.val
        if self.metrics is not None:
            result.merge(self.metrics.val)
        print(self.format_step_result(result, split, iteration, length))
        if self.mode == "train":
            self.write_result(result, split, iteration)
        return result

    def format_step_result(self, result: NestedDict, split: str, steps: int, length: int) -> str:
        result = NestedDict(result).clone()
        repr_str = ""
        if split is not None:
            if self.mode == "train":
                repr_str = f"training on {split} "
            elif self.mode == "eval":
                repr_str = f"evaluating on {split} "
            else:
                repr_str = f"running in {self.mode} mode on {split} "
        repr_str += f"[{steps}/{length}]\t"
        return repr_str + self.format_result(result)

    def format_epoch_result(self, result: NestedDict, epochs: int | None = None, epoch_end: int | None = None) -> str:
        result = NestedDict(result).clone()
        epochs = epochs or self.state.epochs
        epoch_end = epoch_end or self.state.epoch_end
        repr_str = f"epoch [{epochs}/{epoch_end - 1}]\n" if epochs is not None and epoch_end else ""
        repr_str += "\n".join([f"{k}:\t{self.format_result(v)}" for k, v in result.items()])
        return repr_str

    def format_result(self, result):
        return "\t".join([f"{k}: {v}" for k, v in result.items()])

    def write_result(self, result: NestedDict, split: str, steps: int):
        for name, score in result.all_items():
            name = name.replace(".", "/")
            if name == "loss" and isinstance(score, AverageMeter):
                score = score.avg
            if isinstance(score, Sequence):
                for i, s in enumerate(score):
                    self.write_score(f"{name}/{i}", s, split, steps)
            elif isinstance(score, Mapping):
                for k, s in score.items():
                    self.write_score(f"{name}/{k}", s, split, steps)
            else:
                self.write_score(name, score, split, steps)

    def write_score(self, name: str, score: float, split: str, steps: int):
        if self.writer:
            self.writer.add_scalar(f"{split}/{name}", score, steps)  # type: ignore

    @catch
    @on_main_process
    def save_result(self) -> None:
        r"""
        Save result to `self.dir`.

        This method will save latest and best result to
        `self.dir/latest.json` and `self.dir/best.json` respectively.
        """

        results_path = os.path.join(self.dir, "results.json")
        self.save(
            {
                "id": self.state.id,
                "name": self.state.name,
                "results": self.state.results,
            },
            results_path,
            indent=4,
        )
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
