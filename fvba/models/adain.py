"""Class-conditioned adaptive instance normalization."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConditionalAdaIN(nn.Module):
    """Apply learned affine modulation derived from a condition vector."""

    def __init__(self, channels: int, condition_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = nn.Linear(condition_dim, channels * 2)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("features must have shape [N, channels, H, W]")
        if condition.ndim != 2 or condition.shape[0] != features.shape[0]:
            raise ValueError("condition must have shape [N, condition_dim]")
        mean = features.mean(dim=(2, 3), keepdim=True)
        variance = features.var(dim=(2, 3), keepdim=True, unbiased=False)
        normalized = (features - mean) / torch.sqrt(variance + self.eps)
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        return (1 + gamma[:, :, None, None]) * normalized + beta[:, :, None, None]

