import torchvision
from chanfig import Config, Registry
from torch import nn, optim

import danling as dl

OPTIMIZERS = Registry()
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
        self.log_interval = None
        self.save_interval = None
        self.score_split = "val"
        self.score_name = "loss"

    def post(self):
        self.experiment_name = f"{self.network.name}_{self.optim.name}@{self.optim.lr}"


class MNISTRunner(dl.TorchRunner):
    __test__ = False

    def __init__(self, config: Config):
        super().__init__(config)

        self.dataset.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.datasets.train = torchvision.datasets.MNIST(train=True, **self.dataset)
        self.datasets.val = torchvision.datasets.MNIST(train=False, **self.dataset)
        # only run on a few samples to speed up testing process
        self.datasets.train.data = self.datasets.train.data[:64]
        self.datasets.val.data = self.datasets.val.data[:64]

        self.model = getattr(torchvision.models, self.network.name)(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, 1, bias=False)
        self.optimizer = OPTIMIZERS.build(params=self.model.parameters(), **self.optim)
        self.criterion = nn.CrossEntropyLoss()

        self.meters.loss.reset()
        self.meters.time.reset()


class Test:
    config = MNISTConfig()
    runner = MNISTRunner(config)

    def test_train(self):
        self.runner.train()

    def test_evaluate(self):
        self.runner.evaluate(["val"])


if __name__ == "__main__":
    config = MNISTConfig()
    config.parse()
    runner = MNISTRunner(config)
    runner.train()
