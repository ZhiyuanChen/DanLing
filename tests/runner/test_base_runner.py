from chanfig import Config as Config_
from chanfig import NestedDict

import danling as dl


class Runner(dl.BaseRunner):
    def init_distributed(self) -> None:
        pass


class Config(Config_):
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
        self.print_interval = 10
        self.train_iterations_per_epoch = 64
        self.val_iterations_per_epoch = 16
        self.index_set = "val"
        self.index = "loss"


class Test:
    config = Config()
    runner = Runner(config)

    def test_results(self):
        runner = self.runner
        state = runner.state
        state.results = NestedDict(
            {
                0: {
                    "val": {
                        "loss": 1.0,
                        "acc": 0.0,
                    },
                },
                1: {
                    "val": {
                        "loss": 0.5,
                        "acc": 0.5,
                    },
                },
                2: {
                    "val": {
                        "loss": 0.2,
                        "acc": 0.8,
                    },
                },
                3: {
                    "val": {
                        "loss": 0.6,
                        "acc": 0.4,
                    },
                },
            }
        )
        assert runner.best_result.dict() == {
            "index": 2,
            "val": {
                "loss": 0.2,
                "acc": 0.8,
            },
        }
        assert runner.latest_result.dict() == {
            "index": 3,
            "val": {
                "loss": 0.6,
                "acc": 0.4,
            },
        }
        assert runner.best_score == 0.2
        assert runner.latest_score == 0.6
