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
"""
Train a BERT-large-shaped Transformer classifier on IMDB.

NestedTensor vs Padded side-by-side.

- Uses real IMDB from HuggingFace `datasets`.
- Uses `torch.nn.TransformerEncoder` with standard BERT-large dimensions.
- Uses `PNTensor` default DataLoader collation.

Two identical models train side-by-side on IMDB.
"""

import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from danling.tensors import NestedTensor, PNTensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

MODEL_NAME = "bert-large-uncased"
TOTAL_EPOCHS = 2
BATCH_SIZE = 32
TRAIN_SPLIT = "train"
VAL_SPLIT = "test"
VOCAB_SIZE = 30522
MAX_LEN = 8192
D_MODEL = 1024
NHEAD = 16
NUM_LAYERS = 24


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        num_classes=2,
        max_len=MAX_LEN,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.embedding_norm = nn.LayerNorm(d_model, eps=1e-12)
        # dropout=0 so NT dispatch and padded paths are deterministically comparable
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
            layer_norm_eps=1e-12,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, input_ids, padding_mask=None):
        is_nested = isinstance(input_ids, NestedTensor)

        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.embedding_norm(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        # Mean-pool over non-padding positions
        if is_nested:
            return self.classifier(x.mean(dim=1))
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


class IMDBDataset(Dataset):
    def __init__(self, split=TRAIN_SPLIT):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_LEN)
        self.tokenizer.model_max_length = MAX_LEN
        self.dataset = load_dataset("stanfordnlp/imdb", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        ids = PNTensor(
            self.tokenizer(
                row["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_LEN,
            )["input_ids"]
        ).to(torch.long)
        return {"input_ids": ids, "label": row["label"]}


def sync_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def cuda_memory_allocated():
    if DEVICE.type != "cuda":
        return 0
    return torch.cuda.memory_allocated()


def reset_cuda_peak():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_delta(baseline):
    if DEVICE.type != "cuda":
        return 0
    return max(torch.cuda.max_memory_allocated() - baseline, 0)


def format_gib(num_bytes):
    return f"{num_bytes / 1024**3:.2f} GiB"


def train():
    print(
        f"Config: model={MODEL_NAME} epochs={TOTAL_EPOCHS} batch_size={BATCH_SIZE} max_len={MAX_LEN} "
        f"d_model={D_MODEL} nhead={NHEAD} num_layers={NUM_LAYERS}"
    )

    train_ds = IMDBDataset(TRAIN_SPLIT)
    val_ds = IMDBDataset(VAL_SPLIT)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model_nt = TransformerClassifier(d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, max_len=MAX_LEN).to(
        device=DEVICE, dtype=DTYPE
    )
    model_pad = copy.deepcopy(model_nt)
    opt_nt = torch.optim.AdamW(model_nt.parameters(), lr=1e-3)
    opt_pad = torch.optim.AdamW(model_pad.parameters(), lr=1e-3)

    time_nt_total, time_pad_total = 0.0, 0.0
    peak_nt_memory, peak_pad_memory = 0, 0

    for epoch in range(TOTAL_EPOCHS):
        model_nt.train()
        model_pad.train()
        loss_nt_sum, loss_pad_sum = 0.0, 0.0
        correct_nt, correct_pad, total = 0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids_nt = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            ids_pad = input_ids_nt.tensor
            mask_pad = ~input_ids_nt.mask

            opt_nt.zero_grad(set_to_none=True)
            sync_device()
            reset_cuda_peak()
            memory_nt_baseline = cuda_memory_allocated()
            t0 = time.perf_counter()
            logits_nt = model_nt(input_ids_nt)
            loss_nt = F.cross_entropy(logits_nt, labels)
            loss_nt.backward()
            sync_device()
            step_time_nt = time.perf_counter() - t0
            time_nt_total += step_time_nt
            peak_nt_memory = max(peak_nt_memory, cuda_peak_delta(memory_nt_baseline))
            opt_nt.step()

            opt_pad.zero_grad(set_to_none=True)
            sync_device()
            reset_cuda_peak()
            memory_pad_baseline = cuda_memory_allocated()
            t0 = time.perf_counter()
            logits_pad = model_pad(ids_pad, padding_mask=mask_pad)
            loss_pad = F.cross_entropy(logits_pad, labels)
            loss_pad.backward()
            sync_device()
            step_time_pad = time.perf_counter() - t0
            time_pad_total += step_time_pad
            peak_pad_memory = max(peak_pad_memory, cuda_peak_delta(memory_pad_baseline))
            opt_pad.step()

            if epoch == 0 and step == 0:
                max_diff = (logits_nt - logits_pad).abs().max().item()
                print(f"  Step 0 max logit diff: {max_diff:.6f}")
                assert max_diff < 0.05, f"Step 0: max diff {max_diff} too large!\nNT: {logits_nt}\nPad: {logits_pad}"

            loss_nt_sum += loss_nt.item()
            loss_pad_sum += loss_pad.item()
            correct_nt += (logits_nt.argmax(1) == labels).sum().item()
            correct_pad += (logits_pad.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model_nt.eval()
        model_pad.eval()
        val_correct_nt, val_correct_pad, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids_nt = batch["input_ids"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                ids_pad = input_ids_nt.tensor
                mask_pad = ~input_ids_nt.mask

                vl_nt = model_nt(input_ids_nt)
                vl_pad = model_pad(ids_pad, padding_mask=mask_pad)

                val_correct_nt += (vl_nt.argmax(1) == labels).sum().item()
                val_correct_pad += (vl_pad.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_loss_nt = loss_nt_sum / (step + 1)
        train_loss_pad = loss_pad_sum / (step + 1)
        train_acc_nt = correct_nt / total
        train_acc_pad = correct_pad / total
        val_acc_nt = val_correct_nt / val_total
        val_acc_pad = val_correct_pad / val_total

        print(
            f"Epoch {epoch + 1}/{TOTAL_EPOCHS}  "
            f"loss_nt={train_loss_nt:.4f}  loss_pad={train_loss_pad:.4f}  "
            f"acc_nt={train_acc_nt:.4f}  acc_pad={train_acc_pad:.4f}  "
            f"val_nt={val_acc_nt:.4f}  val_pad={val_acc_pad:.4f}"
        )

    print("\nTiming (forward+backward, all epochs):")
    print(f"  NestedTensor: {time_nt_total * 1000:.1f} ms")
    print(f"  Padded:       {time_pad_total * 1000:.1f} ms")
    ratio = time_pad_total / time_nt_total if time_nt_total > 0 else float("inf")
    faster = "NestedTensor" if time_nt_total < time_pad_total else "Padded"
    speedup = max(ratio, 1 / ratio)
    print(f"  {faster} is {speedup:.2f}x faster")
    if DEVICE.type == "cuda":
        print("\nPeak extra CUDA memory per training step:")
        print(f"  NestedTensor: {format_gib(peak_nt_memory)}")
        print(f"  Padded:       {format_gib(peak_pad_memory)}")


if __name__ == "__main__":
    train()
