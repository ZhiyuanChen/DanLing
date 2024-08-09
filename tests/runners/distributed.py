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

from __future__ import annotations

import os
import random
import socket
from contextlib import contextmanager

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn


def can_bind_localhost() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except OSError:
        return False
    return True


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def configure_distributed_env(rank: int, world_size: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def run_distributed(worker, world_size: int = 2, *, seed: int = 1016, worker_args: tuple = ()) -> None:
    if not can_bind_localhost():
        pytest.skip("Local TCP sockets are unavailable in this environment.")

    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_port())
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        spawn(worker, args=(world_size, *worker_args), nprocs=world_size, join=True)
    finally:
        if master_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = master_addr
        if master_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = master_port


@contextmanager
def process_group(backend: str, rank: int, world_size: int):
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        yield
    finally:
        destroy_process_group()


def destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def require_gloo(world_size: int = 2) -> None:
    del world_size
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable in this PyTorch build.")


def require_nccl_cuda(world_size: int = 2) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable in this PyTorch build.")
    if not dist.is_nccl_available():
        pytest.skip("NCCL is unavailable in this PyTorch build.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this environment.")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Expected at least {world_size} CUDA devices, found {torch.cuda.device_count()}.")
