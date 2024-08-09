# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

import torchvision
from torch import nn

import danling as dl


class MNISTConfig(dl.RunnerConfig):
    epochs: int = 2
    log: bool = False
    tensorboard: bool = False
    score_split: str = "val"
    score_name: str = "loss"
    debug: bool = False
    patience: int = 1

    def __init__(self):
        super().__init__()
        self.network.type = "resnet18"
        self.dataset.download = True
        self.dataset.root = "data"
        self.dataloader.batch_size = 8
        self.dataloader.num_workers = 0
        self.optim.type = "adamw"
        self.optim.lr = 1e-3
        self.optim.weight_decay = 1e-4
        self.sched.type = "cosine"

    def post(self):
        super().post()
        self.experiment = f"{self.network.type}_{self.optim.type}@{self.optim.lr}"


class MNISTRunner(dl.TorchRunner):
    def __init__(self, config: dl.RunnerConfig):
        super().__init__(config)

        self.config.dataset.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.datasets.train = torchvision.datasets.MNIST(train=True, **self.config.dataset)
        self.datasets.val = torchvision.datasets.MNIST(train=False, **self.config.dataset)
        # only run on a few samples to speed up testing process
        self.datasets.train.data = self.datasets.train.data[:100]
        self.datasets.val.data = self.datasets.val.data[:100]

        self.model = getattr(torchvision.models, self.config.network.type)(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, 1, bias=False)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = dl.metrics.multiclass_metrics(num_classes=10)
        self.meters.loss.reset()
        self.meters.time.reset()


if __name__ == "__main__":
    config = MNISTConfig()
    config.parse()
    with dl.debug(config.get("debug", False)):
        runner = MNISTRunner(config)
        runner.train()
        runner.evaluate(["val"])
