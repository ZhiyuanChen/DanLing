import torch
from chanfig import Config
from torch import nn

import danling as dl


class MNISTConfig(Config):
    def __init__(self):
        self.network = "ResNet18"


class Runner(dl.Runner):
    def __init__(self, config: MNISTConfig):
        super().__init__(**config)
        print(self)


if __name__ == "__main__":
    config = MNISTConfig()
    config.parse()
    runner = Runner(config)
