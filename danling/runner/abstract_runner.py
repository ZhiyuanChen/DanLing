from __future__ import annotations

import logging
import logging.config
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple

import accelerate
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from chanfig import NestedDict

class AbstractRunner(NestedDict):

    id: str
    name: str
    seed: int
    deterministic: bool

    experiment_dir: str
    checkpoint_dir_name: str

    steps: int
    epochs: int

    accelerator: accelerate.Accelerator
    accelerate_kwargs: list

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    datasets: NestedDict
    datasamplers: NestedDict
    dataloaders: NestedDict

    batch_size: int

    criterion: Tuple[nn.Module]

    results: List[Dict[str, Any]]
    result_best: Dict[str, Any]
    result_last: Dict[str, Any]
    score_best: float
    score_last: float

    log: bool
    logger: logging.Logger
    tensorboard: bool
    writer: Callable

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.id = None
        self.name = "danling"
        self.seed = 1031
        self.deterministic = False

        self.experiment_dir = "experiments"
        self.checkpoint_dir_name = "checkpoints"

        self.steps = 0
        self.epochs = 0
        self.accelerate_kwargs = []

        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None

        self.datasets = NestedDict()
        self.datasamplers = NestedDict()
        self.dataloaders = NestedDict()

        self.batch_size = 1

        self.criterion = None

        self.results = []
        self.result_best = {}
        self.result_last = {}
        self.score_best = 0
        self.score_last = 0

        self.log = False
        self.tensorboard = False
        for key, value in args:
            self.set(key, value, convert_mapping=True)
        for key, value in kwargs.items():
            self.set(key, value, convert_mapping=True)
