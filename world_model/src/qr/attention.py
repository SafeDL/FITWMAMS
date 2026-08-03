"""Small attention helpers shared by QR-WM modules."""

from __future__ import annotations

import torch


def safe_key_padding_mask(valid: torch.Tensor) -> torch.Tensor:
    """Mask invalid keys without creating an all-masked attention row."""
    padding = ~valid.bool()
    empty = padding.all(dim=1)
    if empty.any():
        padding = padding.clone()
        padding[empty, 0] = False
    return padding
