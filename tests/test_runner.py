import time

import torch
import torchvision
from chanfig import Config, FlatDict, NestedDict
from torch import nn, optim
from torch.utils import data

import danling as dl

OPTIMIZERS = dl.Registry()
OPTIMIZERS.register(optim.AdamW, "adamw")
OPTIMIZERS.register(optim.SGD, "sgd")


class MNISTConfig(Config):
    __test__ = False

    def __init__(self):
        self.network.name = "resnet18"
        self.dataset.download = True
        self.dataset.root = "data"
        self.dataloader.batch_size = 8
        self.epoch_end = 2
        self.optim.name = "adamw"
        self.optim.lr = 1e-3
        self.optim.weight_decay = 1e-4
        self.log = False
        self.tensorboard = False
        self.gradient_clip = False
        self.print_freq = 10
        self.train_iterations_per_epoch = 64
        self.val_iterations_per_epoch = 16
        self.index_set = "val"
        self.index = "loss"

    def post(self):
        self.experiment_name = f"{self.network.name}_{self.optim.name}@{self.optim.lr}"


class MNISTRunner(dl.TorchRunner):
    __test__ = False

    def __init__(self, config: Config):
        super().__init__(**config)

        self.dataset.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.datasets.train = torchvision.datasets.MNIST(train=True, **self.dataset)
        self.datasets.val = torchvision.datasets.MNIST(train=False, **self.dataset)
        self.dataloaders.train = self.prepare(data.DataLoader(self.datasets.train, shuffle=True, **self.dataloader))
        self.dataloaders.val = self.prepare(data.DataLoader(self.datasets.val, shuffle=True, **self.dataloader))

        self.model = getattr(torchvision.models, self.network.name)(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, 1, bias=False)
        self.optimizer = OPTIMIZERS.build(params=self.model.parameters(), **self.optim)
        self.model, self.optimizer = self.prepare(self.model, self.optimizer)

        self.criterion = nn.CrossEntropyLoss()
        self.meters = FlatDict(default_factory=dl.metrics.AverageMeter)

    def run(self):
        for self.epochs in range(self.epochs, self.epoch_end):
            result = NestedDict(epoch=self.epochs)
            result.train = self.train_epoch()
            result.val = self.evaluate_epoch()
            self.append_result(result)
            self.print_result()

    def train_epoch(self, split: str = "train"):
        self.model.train()
        loader = self.dataloaders[split]
        self.meters.loss.reset()
        self.meters.time.reset()
        batch_time = time.time()
        for iteration, (input, target) in enumerate(loader):
            predict = self.model(input)
            loss = self.criterion(predict, target)
            self.backward(loss)
            if self.gradient_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.step()
            if iteration % self.print_freq == 0:
                if self.device == torch.device("cuda"):  # pylint: disable=E1101
                    torch.cuda.synchronize()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.meters.time.update((time.time() - batch_time) / self.print_freq)
                batch_time = time.time()
            # break early to speed up tests
            if iteration > self.train_iterations_per_epoch:
                break
        return FlatDict(loss=self.meters.loss.avg)

    @torch.inference_mode()
    def evaluate_epoch(self, split: str = "val"):
        self.model.eval()
        self.meters.loss.reset()
        loader = self.dataloaders[split]
        for iteration, (input, target) in enumerate(loader):
            predict = self.model(input)
            loss = self.criterion(predict, target)
            self.meters.loss.update(loss.item())
            # break early to speed up tests
            if iteration > self.val_iterations_per_epoch:
                break
        return FlatDict(loss=self.meters.loss.avg)

    @property
    def best_fn(self):
        return min


class Test:
    config = MNISTConfig()
    runner = MNISTRunner(config)

    def test_runner(self):
        self.runner.run()


if __name__ == "__main__":
    config = MNISTConfig()
    config.parse()
    runner = MNISTRunner(config)
    runner.run()
