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
r"""CUDA stream pooling for parallel per-element kernel launches."""

from __future__ import annotations

import atexit
import os

import torch

_stream_pools: dict = {}
_stream_pools_owner_pid = os.getpid()


def _reset_stream_pools_after_fork():
    r"""Drop inherited CUDA stream handles in forked child processes."""
    global _stream_pools_owner_pid
    _stream_pools.clear()
    _stream_pools_owner_pid = os.getpid()


def _get_streams(device: torch.device, n: int, max_streams: int = 8) -> list:
    r"""Return a cached pool of CUDA streams for the given device."""
    global _stream_pools_owner_pid
    current_pid = os.getpid()
    if current_pid != _stream_pools_owner_pid:
        _reset_stream_pools_after_fork()
    key = device.index if device.index is not None else 0
    pool = _stream_pools.get(key)
    if pool is None:
        pool = [torch.cuda.Stream(device=device) for _ in range(max_streams)]
        _stream_pools[key] = pool
    return pool[: min(n, max_streams)]


def _run_on_streams(items, fn, max_streams: int = 8) -> list:
    r"""
    Run fn on each item using a pool of CUDA streams for parallel kernel launches.

    Returns the list of results; callers are responsible for reconstruction.
    """
    n = len(items)
    current = torch.cuda.current_stream()
    streams = _get_streams(current.device, n, max_streams)

    results = [None] * n
    for i, item in enumerate(items):
        s = streams[i % len(streams)]
        s.wait_stream(current)
        with torch.cuda.stream(s):
            results[i] = fn(item)

    for s in streams[: min(n, len(streams))]:
        current.wait_stream(s)

    return results


def cleanup_stream_pools():
    r"""Clear the cached CUDA stream pools. Call this in torchelastic worker cleanup."""
    _stream_pools.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_stream_pools_after_fork)

atexit.register(cleanup_stream_pools)
