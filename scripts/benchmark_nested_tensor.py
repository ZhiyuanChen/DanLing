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
"""Comprehensive benchmark: DanLing NestedTensor vs padded Tensor vs torch.nested.

Compares performance (inference + fwd/bwd) across two categories:

Part 1 — Models: TransformerEncoder, TransformerDecoder, Transformer, ResNet-50
Part 2 — Operators: F.linear, F.layer_norm, F.softmax, F.embedding,
         F.relu, F.gelu, torch.matmul, torch.add

Outputs a markdown table suitable for README.

Usage:
    python scripts/benchmark_nested_tensor.py
    python scripts/benchmark_nested_tensor.py --device cuda --markdown true
    python scripts/benchmark_nested_tensor.py --part ops --hidden_size 512
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chanfig  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from danling.tensors import NestedTensor  # noqa: E402

NT = NestedTensor


class Config(chanfig.Config):

    hidden_size: int = 1024
    num_hidden_layers: int = 2
    num_attention_heads: int = 16
    intermediate_size: int = 4096

    batch_size: int = 32
    max_seq_len: int = 2048
    max_img_size: int = 1024
    seed: int = 1016

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    warmup: int = 5
    repeats: int = 20
    occupancies: list[float] = [0.2, 0.4, 0.8]
    part: str = "all"
    markdown: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seq(config, device, dtype, *, seed_offset=0, occupancy=None):
    """Variable-length sequences → (list[Tensor], NestedTensor, padded, mask, lengths)."""
    batch = config.batch_size
    max_len = config.max_seq_len
    dim = config.hidden_size
    rng = torch.Generator()
    rng.manual_seed(config.seed + seed_offset)
    if occupancy is not None:
        # Generate lengths so actual occupancy matches target.
        # One element is forced to max_len; the remaining (batch-1) are
        # drawn from Uniform[low, high] with mean = target_avg_rest.
        target_sum = int(batch * max_len * occupancy)
        target_sum_rest = max(batch - 1, target_sum - max_len)
        target_avg_rest = target_sum_rest / max(1, batch - 1)
        half_range = min(target_avg_rest - 1, max_len - target_avg_rest)
        half_range = max(1, int(half_range))
        low = max(1, int(target_avg_rest - half_range))
        high = min(max_len, int(target_avg_rest + half_range))
        lengths = torch.randint(low, high + 1, (batch,), generator=rng).tolist()
        lengths[-1] = max_len  # ensure at least one full-length
    else:
        lengths = torch.randint(max(1, max_len // 4), max_len + 1, (batch,), generator=rng).tolist()
        lengths[-1] = max_len
    tensors = [torch.randn(seq_len, dim, device=device, dtype=dtype) for seq_len in lengths]
    nt = NT(tensors)
    padded = nt.tensor
    mask = ~nt.mask  # True = ignore
    torch_nt = nt.to_torch_nested()
    return tensors, nt, padded, mask, torch_nt, lengths


def _bench(fn, warmup, repeats, device):
    """Return ms per call, or None on error."""
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / repeats
    except Exception:
        return None


def _bench_train(fn, warmup, repeats, device):
    """Return ms per fwd+bwd call, or None on error."""
    try:
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / repeats
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Part 1: Model benchmarks
# ---------------------------------------------------------------------------


def _run_model(name, model_factory, make_inputs_fn, config, device, dtype, occupancy=None):
    """Run a single model benchmark. Returns result dict."""
    model = model_factory().to(device, dtype).eval()

    inputs = make_inputs_fn(config, device, dtype, occupancy=occupancy)
    torch_nt_in = inputs["torch_nt"]
    lengths = inputs["lengths"]

    # --- Timing helpers ---
    skip_compile = "resnet" in name.lower() or "vit" in name.lower()

    def _infer(kw_key):
        return _bench(lambda: model(**inputs[kw_key]), config.warmup, config.repeats, device)

    def _infer_compiled(kw_key):
        if skip_compile:
            return None
        try:
            torch._dynamo.reset()
            cm = torch.compile(copy.deepcopy(model).eval())
            return _bench(lambda: cm(**inputs[kw_key]), config.warmup, config.repeats, device)
        except Exception:
            return None

    target = inputs.get("target")

    def _make_train_fn(m, kw_key):
        kw = inputs[kw_key]

        def fn():
            m.zero_grad()
            out = m(**kw)
            out_t = out.tensor if isinstance(out, NT) else out
            if target is not None:
                F.mse_loss(out_t, target).backward()
            else:
                out_t.sum().backward()

        return fn

    def _train(kw_key):
        m = copy.deepcopy(model).train()
        return _bench_train(_make_train_fn(m, kw_key), config.warmup, config.repeats, device)

    def _train_compiled(kw_key):
        if skip_compile:
            return None
        try:
            torch._dynamo.reset()
            m = torch.compile(copy.deepcopy(model).train())
            return _bench_train(_make_train_fn(m, kw_key), config.warmup, config.repeats, device)
        except Exception:
            return None

    # --- Collect all 2 (eager/compiled) × 2 (infer/train) × 3 (pad/dl/tn) ---
    model.eval()
    r = {
        "name": name,
        "occ": inputs.get("_occ_override") or (sum(lengths) / (len(lengths) * max(lengths))),
        # Inference eager
        "infer_pad": _infer("pad_kwargs"),
        "infer_dl": _infer("dl_kwargs"),
        "infer_tn": _infer("tn_kwargs") if torch_nt_in is not None else None,
        # Inference compiled
        "compile_infer_pad": _infer_compiled("pad_kwargs"),
        "compile_infer_dl": _infer_compiled("dl_kwargs"),
        "compile_infer_tn": None,  # torch.nested can't compile through nn.Transformer
        # Train eager
        "train_pad": _train("pad_kwargs"),
        "train_dl": _train("dl_kwargs"),
        "train_tn": None,  # torch.nested can't train through nn.Transformer
        # Train compiled
        "compile_train_pad": _train_compiled("pad_kwargs"),
        "compile_train_dl": _train_compiled("dl_kwargs"),
        "compile_train_tn": None,
    }
    return r


def _encoder_inputs(config, device, dtype, occupancy=None):
    tensors, nt, padded, mask, torch_nt, lengths = _make_seq(config, device, dtype, occupancy=occupancy)
    target = torch.randn_like(padded)
    single_kwargs_list = [{"src": t.unsqueeze(0)} for t in tensors]
    return {
        "nt": nt,
        "padded": padded,
        "mask": mask,
        "torch_nt": torch_nt,
        "lengths": lengths,
        "target": target,
        "single_kwargs_list": single_kwargs_list,
        "dl_kwargs": {"src": nt},
        "pad_kwargs": {"src": padded, "src_key_padding_mask": mask},
        "pad_nomask_kwargs": {"src": padded},
        "tn_kwargs": {"src": torch_nt} if torch_nt is not None else {},
    }


def _decoder_inputs(config, device, dtype, occupancy=None):
    tensors_tgt, nt_tgt, pad_tgt, mask_tgt, tn_tgt, len_tgt = _make_seq(config, device, dtype, occupancy=occupancy)
    tensors_mem, nt_mem, pad_mem, mask_mem, tn_mem, _ = _make_seq(
        config, device, dtype, seed_offset=1, occupancy=occupancy
    )
    target = torch.randn_like(pad_tgt)
    single_kwargs_list = [{"tgt": t.unsqueeze(0), "memory": m.unsqueeze(0)} for t, m in zip(tensors_tgt, tensors_mem)]
    return {
        "nt": nt_tgt,
        "padded": pad_tgt,
        "mask": mask_tgt,
        "torch_nt": tn_tgt,
        "lengths": len_tgt,
        "target": target,
        "single_kwargs_list": single_kwargs_list,
        "dl_kwargs": {"tgt": nt_tgt, "memory": nt_mem},
        "pad_kwargs": {
            "tgt": pad_tgt,
            "memory": pad_mem,
            "tgt_key_padding_mask": mask_tgt,
            "memory_key_padding_mask": mask_mem,
        },
        "pad_nomask_kwargs": {"tgt": pad_tgt, "memory": pad_mem},
        "tn_kwargs": {"tgt": tn_tgt, "memory": tn_mem} if tn_tgt is not None else {},
    }


def _transformer_inputs(config, device, dtype, occupancy=None):
    tensors_src, nt_src, pad_src, mask_src, tn_src, len_src = _make_seq(config, device, dtype, occupancy=occupancy)
    tensors_tgt, nt_tgt, pad_tgt, mask_tgt, tn_tgt, len_tgt = _make_seq(
        config, device, dtype, seed_offset=1, occupancy=occupancy
    )
    target = torch.randn_like(pad_tgt)
    single_kwargs_list = [{"src": s.unsqueeze(0), "tgt": t.unsqueeze(0)} for s, t in zip(tensors_src, tensors_tgt)]
    return {
        "nt": nt_src,
        "padded": pad_src,
        "mask": mask_src,
        "torch_nt": tn_src,
        "lengths": len_tgt,
        "target": target,
        "single_kwargs_list": single_kwargs_list,
        "dl_kwargs": {"src": nt_src, "tgt": nt_tgt},
        "pad_kwargs": {
            "src": pad_src,
            "tgt": pad_tgt,
            "src_key_padding_mask": mask_src,
            "tgt_key_padding_mask": mask_tgt,
        },
        "pad_nomask_kwargs": {"src": pad_src, "tgt": pad_tgt},
        "tn_kwargs": {"src": tn_src, "tgt": tn_tgt} if tn_src is not None else {},
    }


def _resnet_inputs(config, device, dtype, occupancy=None):
    max_size = config.max_img_size
    rng = torch.Generator()
    rng.manual_seed(config.seed)
    # Variable image sizes; per-element dispatch means each image runs through
    # the model individually.
    # Caveat: BatchNorm with per-element dispatch is only correct in eval mode
    # (fixed running stats).  In train mode, BN with batch_size=1 produces
    # degenerate statistics (zero variance), so train-mode results for
    # BN-heavy models are timing-only — numerics are NOT correct.
    if occupancy is not None:
        # Occupancy is on the side length: target avg side = occupancy * max_size.
        # Area occupancy will be ~occupancy² (e.g., 80% side → ~64% area).
        min_size = max(32, int(max_size * occupancy))
    else:
        min_size = max_size // 2
    img_sizes = torch.randint(min_size, max_size + 1, (config.batch_size,), generator=rng).tolist()
    img_sizes[-1] = max_size
    sizes = [(s, s) for s in img_sizes]
    tensors = [torch.randn(3, h, w, device=device, dtype=dtype) for h, w in sizes]
    nt = NT(tensors)
    padded = nt.tensor
    lengths = list(range(len(tensors)))
    single_kwargs_list = [{"x": t.unsqueeze(0)} for t in tensors]
    target = torch.randn(config.batch_size, 1000, device=device, dtype=dtype)
    occ = sum(h * w for h, w in sizes) / (config.batch_size * max_size * max_size)
    return {
        "nt": nt,
        "padded": padded,
        "mask": None,
        "torch_nt": None,
        "lengths": lengths,
        "target": target,
        "_occ_override": occ,
        "single_kwargs_list": single_kwargs_list,
        "dl_kwargs": {"x": nt + 0},  # non-leaf for inplace ops
        "pad_kwargs": {"x": padded},
        "pad_nomask_kwargs": {"x": padded},
        "tn_kwargs": {},
    }


def run_models(config, device, dtype):
    d, h, ff, layers = (
        config.hidden_size,
        config.num_attention_heads,
        config.intermediate_size,
        config.num_hidden_layers,
    )
    occupancies = config.occupancies

    benchmarks = [
        (
            "TransformerEncoder",
            lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, ff, batch_first=True, dropout=0.0), layers),
            _encoder_inputs,
        ),
        (
            "TransformerDecoder",
            lambda: nn.TransformerDecoder(nn.TransformerDecoderLayer(d, h, ff, batch_first=True, dropout=0.0), layers),
            _decoder_inputs,
        ),
        (
            "Transformer",
            lambda: nn.Transformer(d, h, layers, layers, ff, batch_first=True, dropout=0.0),
            _transformer_inputs,
        ),
    ]

    try:
        import torchvision.models as models

        benchmarks.append(("ResNet-50", lambda: models.resnet50(weights=None), _resnet_inputs))
    except ImportError:
        pass

    rows = []
    for name, factory, input_fn in benchmarks:
        for occ in occupancies:
            torch.manual_seed(config.seed)
            r = _run_model(name, factory, input_fn, config, device, dtype, occupancy=occ)
            occ_val = r["occ"]
            for mode, pk, cpk, dk, cdk in [
                ("Infer", "infer_pad", "compile_infer_pad", "infer_dl", "compile_infer_dl"),
                ("Train", "train_pad", "compile_train_pad", "train_dl", "compile_train_dl"),
            ]:
                rows.append(
                    {
                        "Model": name,
                        "Mode": mode,
                        "Occ.": f"{occ_val:.0%}",
                        "Padded (eager)": r[pk],
                        "Padded (compiled)": r[cpk],
                        "DanLing (eager)": r[dk],
                        "DanLing (compiled)": r.get(cdk),
                    }
                )
    df = pd.DataFrame(rows)
    df["DL vs Padded"] = df["Padded (eager)"] / df["DanLing (eager)"]
    df["DL vs Compiled"] = df["Padded (compiled)"] / df["DanLing (eager)"]
    return df


# ---------------------------------------------------------------------------
# Part 2: Operator benchmarks
# ---------------------------------------------------------------------------


def _run_op(name, op_fn_dl, op_fn_pad, op_fn_tn, occ, config, device):
    """Benchmark a single operator."""
    infer_pad = _bench(op_fn_pad, config.warmup, config.repeats, device)
    infer_dl = _bench(op_fn_dl, config.warmup, config.repeats, device)
    infer_tn = _bench(op_fn_tn, config.warmup, config.repeats, device) if op_fn_tn is not None else None

    # Compiled
    try:
        torch._dynamo.reset()
        c_pad = torch.compile(op_fn_pad)
        compile_pad = _bench(c_pad, config.warmup, config.repeats, device)
    except Exception:
        compile_pad = None
    try:
        torch._dynamo.reset()
        c_dl = torch.compile(op_fn_dl)
        compile_dl = _bench(c_dl, config.warmup, config.repeats, device)
    except Exception:
        compile_dl = None
    compile_tn = None
    if op_fn_tn is not None:
        try:
            torch._dynamo.reset()
            c_tn = torch.compile(op_fn_tn)
            compile_tn = _bench(c_tn, config.warmup, config.repeats, device)
        except Exception:
            compile_tn = None

    return {
        "name": name,
        "occ": occ,
        "infer_pad": infer_pad,
        "infer_dl": infer_dl,
        "infer_tn": infer_tn,
        "compile_pad": compile_pad,
        "compile_dl": compile_dl,
        "compile_tn": compile_tn,
    }


def _make_op_list(nt, padded, torch_nt, lengths, d, device, dtype):
    """Build the list of (name, dl_fn, pad_fn, tn_fn) for a given data setup."""
    w_linear = torch.randn(d // 2, d, device=device, dtype=dtype)
    b_linear = torch.randn(d // 2, device=device, dtype=dtype)
    ln_w = torch.ones(d, device=device, dtype=dtype)
    ln_b = torch.zeros(d, device=device, dtype=dtype)
    embed_w = torch.randn(1000, d, device=device, dtype=dtype)
    idx_tensors = [torch.randint(0, 1000, (seq_len,), device=device) for seq_len in lengths]
    idx_nt = NT(idx_tensors)
    idx_padded = idx_nt.tensor
    try:
        idx_torch_nt = torch.nested.nested_tensor(idx_tensors, layout=torch.jagged)
    except Exception:
        idx_torch_nt = None

    def _tn_or_none(fn):
        if torch_nt is None:
            return None
        return fn

    return [
        (
            "F.linear",
            lambda: F.linear(nt, w_linear, b_linear),
            lambda: F.linear(padded, w_linear, b_linear),
            _tn_or_none(lambda: F.linear(torch_nt, w_linear, b_linear)),
        ),
        (
            "F.layer_norm",
            lambda: F.layer_norm(nt, (d,), ln_w, ln_b),
            lambda: F.layer_norm(padded, (d,), ln_w, ln_b),
            _tn_or_none(lambda: F.layer_norm(torch_nt, (d,), ln_w, ln_b)),
        ),
        ("F.relu", lambda: F.relu(nt), lambda: F.relu(padded), _tn_or_none(lambda: F.relu(torch_nt))),
        ("F.gelu", lambda: F.gelu(nt), lambda: F.gelu(padded), _tn_or_none(lambda: F.gelu(torch_nt))),
        (
            "F.softmax",
            lambda: F.softmax(nt, dim=-1),
            lambda: F.softmax(padded, dim=-1),
            _tn_or_none(lambda: F.softmax(torch_nt, dim=-1)),
        ),
        (
            "F.embedding",
            lambda: F.embedding(idx_nt, embed_w),
            lambda: F.embedding(idx_padded, embed_w),
            _tn_or_none(lambda: F.embedding(idx_torch_nt, embed_w)) if idx_torch_nt is not None else None,
        ),
        (
            "torch.matmul",
            lambda: torch.matmul(nt, w_linear.T),
            lambda: torch.matmul(padded, w_linear.T),
            _tn_or_none(lambda: torch.matmul(torch_nt, w_linear.T)),
        ),
        ("torch.add", lambda: nt + nt, lambda: padded + padded, _tn_or_none(lambda: torch_nt + torch_nt)),
    ]


def run_ops(config, device, dtype):
    d = config.hidden_size
    occ = float(config.occupancies[len(config.occupancies) // 2])
    _, nt, padded, _, torch_nt, lengths = _make_seq(config, device, dtype, occupancy=occ)
    actual_occ = sum(lengths) / (len(lengths) * max(lengths))
    ops = _make_op_list(nt, padded, torch_nt, lengths, d, device, dtype)
    rows = []
    for name, fn_dl, fn_pad, fn_tn in ops:
        r = _run_op(name, fn_dl, fn_pad, fn_tn, actual_occ, config, device)
        rows.append(
            {
                "Operator": r["name"],
                "Occ.": f"{r['occ']:.0%}",
                "Padded (eager)": r["infer_pad"],
                "Padded (compiled)": r["compile_pad"],
                "DanLing (eager)": r["infer_dl"],
                "DanLing (compiled)": r["compile_dl"],
                "torch.nested (eager)": r["infer_tn"],
                "torch.nested (compiled)": r.get("compile_tn"),
            }
        )
    df = pd.DataFrame(rows)
    df["DL vs Padded"] = df["Padded (compiled)"] / df["DanLing (compiled)"]
    df["DL vs torch.nested"] = df["torch.nested (compiled)"] / df["DanLing (compiled)"]
    return df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _format_df(df, float_format="{:.2f} ms", ratio_format="{:.2f}x"):
    """Format a DataFrame for display: floats as 'X.XX ms', ratios as 'X.XXx'."""
    out = df.copy()
    for col in out.columns:
        if col in ("Model", "Operator", "Mode", "Occ."):
            continue
        if "vs" in col:
            # Ratio columns
            out[col] = out[col].map(lambda v: ratio_format.format(v) if pd.notna(v) else "N/A")
        else:
            # Timing columns
            out[col] = out[col].map(lambda v: float_format.format(v) if pd.notna(v) else "ERR ms")
    return out


def _print_df(df, title, markdown=False):
    """Print a DataFrame as markdown table or plain text."""
    formatted = _format_df(df)
    if markdown:
        print(f"\n### {title}\n")
        print(formatted.to_markdown(index=False))
    else:
        print(f"\n=== {title} ===\n")
        print(formatted.to_string(index=False))


def main():

    config = Config().parse()

    device = torch.device(config.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(
        f"Config: d_model={config.hidden_size} nhead={config.num_attention_heads}"
        f" d_ff={config.intermediate_size} layers={config.num_hidden_layers}"
    )
    print(f"        batch={config.batch_size} max_seq_len={config.max_seq_len}" f" device={device} dtype={dtype}")

    if config.part in ("all", "models"):
        model_df = run_models(config, device, dtype)
        _print_df(model_df, "Models", config.markdown)

    if config.part in ("all", "ops"):
        ops_df = run_ops(config, device, dtype)
        _print_df(ops_df, "Operators", config.markdown)


if __name__ == "__main__":
    main()
