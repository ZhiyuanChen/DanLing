import atexit
import json
import logging
import os
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple

import accelerate
# import click
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from danling.utils import catch
from torch.utils.tensorboard import SummaryWriter


class BaseRunner(object):

    config: Any

    id: str
    name: str
    seed: int = 616
    directory: str
    device: torch.device

    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    is_main_process: bool = True
    is_local_main_process: bool = True

    dataloaders: Dict[str, data.DataLoader] = {}

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    criterions: Tuple[nn.Module]

    accelerator: accelerate.Accelerator

    epochs: int
    epoch: int = 0
    start_epoch: int = 0
    is_best_epoch: bool = False

    results: List[dict]
    result_best: dict = None
    result_last: dict = None
    score_best: float = 0
    score_last: float = 0

    log: bool = True
    logger: logging.Logger
    tensorboard: bool = True
    writer: SummaryWriter

    def __init__(self, config):
        """
        Set up everything for running
        """
        self.config = config
        self.id = config.id
        self.name = config.name
        self.deterministic = config.deterministic
        self.epochs = config.epochs
        self.log = config.log
        self.tensorboard = config.tensorboard
        self.directory = os.path.join(config.experiments_root, self.id)
        self.checkpoint_dir = os.path.join(self.directory, config.checkpoint_dir)

        # self.init_distributed()
        self.accelerator = accelerate.Accelerator()
        self.device = self.accelerator.device
        self.seed = self.accelerator.gather(torch.tensor(config.seed)).item()
        self.is_main_process = self.accelerator.is_main_process
        self.is_local_main_process = self.accelerator.is_local_main_process

        if self.seed is not None:
            self.init_seed()

        if self.deterministic:
            self.init_deterministic()

        if self.is_main_process:
            os.makedirs(self.directory, exist_ok=True)
            if config.train:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.log:
                self.init_logger()
            if self.tensorboard:
                self.writer = SummaryWriter(self.directory)
        elif config.nni:
            config.nni = False

        if config.log:
            self.init_print()

        print(config)
        atexit.register(self.print_result)

    def init_distributed(self):
        """
        Set up distributed training
        """
        dist.init_process_group(backend='nccl')
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(self.local_rank)

    def init_deterministic(self):
        """
        Set up deterministic
        """
        cudnn.benchmark = False
        cudnn.deterministic = True

    def init_seed(self):
        """
        Set up random seed
        """
        torch.manual_seed(self.rank + self.seed)
        torch.cuda.manual_seed(self.rank + self.seed)
        np.random.seed(self.rank + self.seed)
        random.seed(self.rank + self.seed)

    def init_logger(self):
        """
        Set up logger
        """
        # TODO consider move stream output out of stderr
        # Why is setting up proper logging so !@?#! ugly?
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
            },
            'handlers': {
                'stderr': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stderr',
                },
                'logfile': {
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'class': 'logging.FileHandler',
                    'filename': os.path.join(self.directory, 'run.log'),
                    'mode': 'a',
                }
            },
            'loggers': {
                '': {
                    'handlers': ['stderr', 'logfile'],
                    'level': 'DEBUG',
                    'propagate': True
                },
            }
        })
        logging.captureWarnings(True)
        self.logger = logging.getLogger('runner')
        self.logger.flush = lambda: [h.flush() for h in self.logger.handlers]

    def init_print(self, process: Optional[int] = 0, precision: Optional[int] = 10):
        """
        Set up print
        Only print from a specific process or force indicated
        Replace default print function with logging.info
        """
        torch.set_printoptions(precision=precision)

        logger = logging.getLogger('print')
        logger.flush = lambda: [h.flush() for h in logger.handlers]
        import builtins as __builtin__
        builtin_print = __builtin__.print

        @catch()
        def print(*config, end='\n', file=None, flush=False, force=False, **kwconfig):
            if self.rank == process or force:
                if self.log:
                    logger.info(*config, **kwconfig)
                else:
                    builtin_print(*config, end='\n', file=None, flush=False, force=False, **kwconfig)

        __builtin__.print = print

    def scale_lr(self, scale_factor: Optional[float] = None, batch_size_base: Optional[int] = None):
        if batch_size_base is None:
            batch_size_base = self.config.batch_size_base
        if scale_factor is None:
            batch_size_actual = self.config.batch_size * self.world_size * self.config.accum_steps
            scale_factor = batch_size_actual / batch_size_base
        self.config.scale_factor = scale_factor
        self.config.lr = self.config.lr * self.config.scale_factor
        self.config.lr_final = self.config.lr_final * self.config.scale_factor

    @catch()
    def save_checkpoint(self):
        """
        Save checkpoint to checkpoint_dir
        """
        model = self.model
        if self.distributed:
            self.accelerator.wait_for_everyone()
            model = self.accelerator.unwrap_model(self.model)

        state_dict = {
            'epoch': self.epoch,
            'config': self.config,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'result': self.result_last
        }
        last_path = os.path.join(self.checkpoint_dir, 'last.pth')
        accelerate.save(state_dict, last_path)
        if (self.epoch + 1) % self.config.save_freq == 0:
            save_path = os.path.join(self.checkpoint_dir, f'epoch-{self.epoch}.pth')
            shutil.copy(last_path, save_path)
        if self.is_best_epoch:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            shutil.copy(last_path, best_path)

    @catch()
    def save_result(self):
        """
        Save result
        """
        ret = {'id': self.id, 'name': self.name}
        ret.update(self.result_last)  # This is slower but ensure id in the first
        last_path = os.path.join(self.directory, 'last.json')
        with open(last_path, 'w') as f:
            json.dump(ret, f, indent=4)
        if self.is_best_epoch:
            best_path = os.path.join(self.directory, 'best.json')
            shutil.copy(last_path, best_path)

    def load_checkpoint(self, checkpoint: str):
        """
        Load checkpoint from checkpoint
        """
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError('checkpoint does not exist')
        print(f'=> loading checkpoint "{checkpoint}"')
        checkpoint = torch.load(checkpoint)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        if self.optimizer is not None and 'optimizer' in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint.keys():
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'config' in checkpoint.keys():
            self.config = checkpoint['config']
        if 'epoch' in checkpoint.keys():
            self.start_epoch = checkpoint['epoch']
        if 'result' in checkpoint.keys():
            self.result_best = checkpoint['result']
        print(f'=> loaded  checkpoint "{checkpoint}"')

    def print_result(self):
        print(f'best result: {self.result_best}')
        print(f'last result: {self.result_last}')

    def __repr__(self):
        return self.id

    def __str__(self):
        return self.name
