from __future__ import annotations

import torch

from danling.runners.utils import format_result


def test_format_result_formats_nested_scalar_like_values() -> None:
    result = {
        "train": {
            "loss": 0.7303,
            "auroc": torch.tensor(0.9849324226379395),
            "auprc": torch.tensor(0.9360466003417969),
            "acc": torch.tensor(0.7621212601661682),
            "f1": torch.tensor(0.7479169368743896),
            "mcc": torch.tensor(0.7965899109840393),
        },
        "val": {
            "loss": 1.7002,
            "auroc": torch.tensor(0.8909330368041992),
            "auprc": torch.tensor(0.6458794474601746),
            "acc": torch.tensor(0.39649349451065063),
            "f1": torch.tensor(0.33815914392471313),
            "mcc": torch.tensor(0.42628031969070435),
        },
    }

    formatted = format_result(result)

    assert "0.9849" in formatted
    assert "0.9360" in formatted
    assert "0.7621" in formatted
    assert "0.8909" in formatted
    assert "0.6459" in formatted
    assert "0.9849324226379395" not in formatted
