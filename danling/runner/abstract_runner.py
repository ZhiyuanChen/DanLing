from __future__ import annotations

import logging
import logging.config
from typing import Any, Callable, List, Tuple

import accelerate
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from chanfig import Dict, NestedDict

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

    datasets: Dict[str, data.Dataset]
    datasamplers: Dict[str, data.Sampler]
    dataloaders: Dict[str, data.DataLoader]

    batch_size: int

    criterion: Tuple[nn.Module]

    metric: str
    results: List[Dict[str, any]]
    result_best: Dict[str, any]
    result_latest: Dict[str, any]
    score_best: float
    score_latest: float
    is_best: bool

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

        self.datasets = Dict()
        self.datasamplers = Dict()
        self.dataloaders = Dict()

        self.batch_size = 1

        self.criterion = None

        self.metric = 'loss'
        self.results = []
        self.result_best = Dict()
        self.result_latest = Dict()
        self.score_best = 0
        self.score_latest = 0
        self.is_best = False

        self.log = True
        self.tensorboard = False
        for key, value in args:
            self.set(key, value, convert_mapping=True)
        for key, value in kwargs.items():
            self.set(key, value, convert_mapping=True)
