# DanLing
# Copyright (C) 2022-Present  DanLing

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from danling.runners import TorchRunner


class TinyFaultToleranceTorchRunner(TorchRunner):
    @property
    def device(self):
        return torch.device("cpu")

    def __init__(self, config):
        super().__init__(config)
        total_examples = int(self.config.steps) * int(self.world_size)
        self.datasets["train"] = [
            {"input": torch.full((4,), float(index + 1)), "target": torch.zeros(2)} for index in range(total_examples)
        ]
        self.model = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replica-id", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--auto-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    runner = TinyFaultToleranceTorchRunner(
        {
            "log": False,
            "dir": str(run_dir),
            "stack": "ddp",
            "backend": "gloo",
            "steps": args.steps,
            "train_splits": ["train"],
            "evaluate_splits": [],
            "auto_resume": args.auto_resume,
            "dataloader": {"batch_size": 1, "shuffle": False},
            "optim": {"type": "sgd", "lr": 0.1},
            "checkpoint": {
                "backend": "dcp",
                "interval": 1,
                "async_mode": "disabled",
            },
            "ft": {
                "enabled": True,
                "process_group": "gloo",
                "replica_id": args.replica_id,
                "group_size": args.group_size,
                "min_replica_size": 1,
            },
        }
    )
    try:
        runner.train_steps(train_splits=["train"], evaluate_splits=[])
        if runner.rank == 0:
            status = {
                "global_step": int(runner.train_state.global_step),
                "cursor": int(runner.dataloaders["train"].state_dict()["_num_yielded"]),
            }
            status_path = run_dir / f"ft-status-replica-{args.replica_id}.json"
            with status_path.open("w", encoding="utf-8") as fp:
                json.dump(status, fp)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
