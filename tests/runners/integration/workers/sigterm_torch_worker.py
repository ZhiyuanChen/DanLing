# DanLing
# Copyright (C) 2022-Present  DanLing

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

import torch
from torch import nn

from danling.runners import TorchRunner


class SigtermTorchRunner(TorchRunner):
    @property
    def device(self):
        return torch.device("cpu")

    def __init__(self, config):
        super().__init__(config)
        total_examples = int(self.config.steps) * int(os.getenv("WORLD_SIZE", "1"))
        self.datasets["train"] = [
            {"input": torch.full((4,), float(index + 1)), "target": torch.zeros(2)} for index in range(total_examples)
        ]
        self.model = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self._sigterm_sent = False

    def train_step(self, data):
        result = super().train_step(data)
        signal_after_steps = int(os.getenv("DANLING_SIGNAL_AFTER_STEPS", "0") or 0)
        if signal_after_steps > 0 and not self._sigterm_sent and self.train_state.global_step >= signal_after_steps:
            self._sigterm_sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--signal-after-steps", type=int, default=0)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--status-name", default="status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    exit_code = 0
    runner = SigtermTorchRunner(
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
                "interval": 100,
                "async_mode": "disabled",
            },
        }
    )
    try:
        try:
            runner.train_steps(train_splits=["train"], evaluate_splits=[])
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
            raise
    finally:
        if runner.rank == 0:
            status = {
                "exit_code": exit_code,
                "global_step": int(runner.train_state.global_step),
                "signal_after_steps": int(args.signal_after_steps),
            }
            status_path = run_dir / f"sigterm-{args.status_name}.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            with status_path.open("w", encoding="utf-8") as fp:
                json.dump(status, fp)
        runner.close()


if __name__ == "__main__":
    main()
