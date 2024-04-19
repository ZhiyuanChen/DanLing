import torchvision
from chanfig import Config, Registry
from torch import nn, optim

import danling as dl

OPTIMIZERS = Registry()
OPTIMIZERS.register(optim.AdamW, "adamw")
OPTIMIZERS.register(optim.SGD, "sgd")


class MNISTConfig(Config):
    epoch_end = 2
    log = False
    tensorboard = False
    log_interval = 1000
    score_split = "val"
    score_name = "loss"
    debug = False
    patience = 1

    def __init__(self):
        self.network.name = "resnet18"
        self.dataset.download = True
        self.dataset.root = "data"
        self.dataloader.batch_size = 8
        self.optim.name = "adamw"
        self.optim.lr = 1e-3
        self.optim.weight_decay = 1e-4

    def post(self):
        self.experiment_name = f"{self.network.name}_{self.optim.name}@{self.optim.lr}"


class MNISTRunner(dl.TorchRunner):
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

        self.model = getattr(torchvision.models, self.network.name)(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, 1, bias=False)
        self.optimizer = OPTIMIZERS.build(params=self.model.parameters(), **self.optim)
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
