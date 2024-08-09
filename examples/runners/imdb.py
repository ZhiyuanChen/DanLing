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

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

import danling as dl


class IMDBConfig(dl.RunnerConfig):
    epochs: int = 2
    log: bool = False
    tensorboard: bool = False
    log_interval: int = 1000
    score_split: str = "val"
    score_name: str = "loss"
    debug: bool = False
    patience: int = 1

    def __init__(self):
        super().__init__()
        self.pretrained = "prajjwal1/bert-tiny"
        self.dataset.path = "stanfordnlp/imdb"
        self.dataloader.batch_size = 8
        self.dataloader.num_workers = 4
        self.optim.type = "adamw"
        self.optim.lr = 1e-3
        self.optim.weight_decay = 1e-4
        self.sched.type = "cosine"

    def post(self):
        super().post()
        self.transformers = AutoConfig.from_pretrained(self.pretrained)
        self.experiment = f"{self.pretrained}_{self.optim.type}@{self.optim.lr}"


def transform(data):
    text = dl.NestedTensor(data.pop("text"))
    data["input_ids"] = text.tensor
    data["attention_mask"] = text.mask
    data["labels"] = torch.tensor(data.pop("label"))
    return data


def preprocess_data(dataset, tokenizer):
    def tokenization(example):
        example["text"] = tokenizer(example["text"], truncation=True, max_length=510)["input_ids"]
        return example

    dataset = dataset.map(tokenization, batched=True)
    dataset.set_transform(transform)
    dataset.__getitems__ = dataset.__getitem__
    return dataset


class IMDBRunner(dl.TorchRunner):
    def __init__(self, config: dl.RunnerConfig):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained, use_fast=False)
        self.datasets.train = load_dataset(split="train", **self.config.dataset)
        self.datasets.val = load_dataset(split="test", **self.config.dataset)
        # only run on a few samples to speed up testing process
        self.datasets.train._data = self.datasets.train._data[:64]
        self.datasets.val._data = self.datasets.val._data[:64]
        self.datasets.train = preprocess_data(self.datasets.train, self.tokenizer)
        self.datasets.val = preprocess_data(self.datasets.val, self.tokenizer)

        self.model = AutoModelForSequenceClassification.from_config(self.config.transformers)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = dl.metrics.binary_metrics()
        self.meters.loss.reset()
        self.meters.time.reset()

    def build_optimizer(self) -> None:
        if self.optimizer is not None or self.model is None:
            return
        self.optimizer = dl.optim.OPTIMIZERS.build(params=self.model.parameters(), **self.config.optim)

    def build_scheduler(self) -> None:
        if self.scheduler is not None or self.optimizer is None:
            return
        sched_kwargs = dict(self.config.sched)
        if "total_steps" not in sched_kwargs and self.steps is not None:
            sched_kwargs["total_steps"] = int(self.steps)
        self.scheduler = dl.optim.SCHEDULERS.build(self.optimizer, **sched_kwargs)

    def train_step(self, data):
        data = dl.to_device(data, self.device)
        with self.train_context():
            pred = self.model(**data)
            loss = pred["loss"]
            self.backward(loss)
            self.step()
            self.metrics.update(pred["logits"][:, 0], data["labels"])
        return pred, loss

    def evaluate_step(self, data):
        data = dl.to_device(data, self.device)
        pred = self.model(**data)
        loss = pred["loss"]
        self.metrics.update(pred["logits"][:, 0], data["labels"])
        return pred, loss


if __name__ == "__main__":
    config = IMDBConfig()
    config.parse()
    with dl.debug(config.get("debug", False)):
        runner = IMDBRunner(config)
        runner.train()
        runner.evaluate(["val"])
