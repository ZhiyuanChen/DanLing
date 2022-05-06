r"""
x: x center
y: y center
l: x left
r: x right
t: y top
b: y buttom
w: x width
h: y height
"""
import torch
from torch import Tensor


def ltrb2ltwh(bbox: Tensor) -> Tensor:
    l, t, r, b = bbox.T if bbox.ndim == 2 else bbox
    w, h = r - l, b - t
    return torch.stack((l, t, w, h)).T


def ltwh2ltrb(bbox: Tensor) -> Tensor:
    l, t, w, h = bbox.T if bbox.ndim == 2 else bbox
    r, b = l + w, t + h
    return torch.stack((l, t, r, b)).T


def ltrb2xywh(bbox: Tensor) -> Tensor:
    l, t, r, b = bbox.T if bbox.ndim == 2 else bbox
    w, h = r - l, b - t
    x, y = l + w / 2, t + h / 2
    return torch.stack((x, y, w, h)).T


def xywh2ltrb(bbox: Tensor) -> Tensor:
    x, y, w, h = bbox.T if bbox.ndim == 2 else bbox
    l, t, r, b = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    return torch.stack((l, t, r, b)).T


def xywh2ltwh(bbox: Tensor) -> Tensor:
    x, y, w, h = bbox.T if bbox.ndim == 2 else bbox
    l, t = x - w / 2, y - h / 2
    return torch.stack((l, t, w, h)).T
