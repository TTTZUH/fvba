"""Feature-alignment and perceptual objectives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor


def cfal_loss(
    features: Tensor,
    centroids: Tensor,
    targets: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Cosine feature alignment loss."""
    if features.ndim != 2 or centroids.ndim != 2:
        raise ValueError("features and centroids must be rank-two tensors")
    if features.shape[0] != targets.numel() or features.shape[1] != centroids.shape[1]:
        raise ValueError("features, centroids, and targets have incompatible shapes")
    if targets.numel() and (targets.min() < 0 or targets.max() >= centroids.shape[0]):
        raise ValueError("target index is outside the centroid table")
    selected = centroids[targets]
    similarity = functional.cosine_similarity(features, selected, dim=1, eps=eps)
    return (1 - similarity).mean()


def _gaussian_window(size: int, sigma: float, channels: int, like: Tensor) -> Tensor:
    coordinates = torch.arange(size, device=like.device, dtype=like.dtype) - (size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(
    x: Tensor,
    y: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Mean differentiable structural similarity for NCHW images."""
    if x.shape != y.shape:
        raise ValueError("SSIM inputs must have the same shape")
    if x.ndim != 4:
        raise ValueError("SSIM inputs must have shape [N, C, H, W]")
    if min(x.shape[-2:]) < window_size:
        window_size = min(x.shape[-2:])
        if window_size % 2 == 0:
            window_size -= 1
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x)
    padding = window_size // 2
    mu_x = functional.conv2d(x, window, padding=padding, groups=channels)
    mu_y = functional.conv2d(y, window, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = functional.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y = functional.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = functional.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(torch.finfo(x.dtype).tiny)).mean()


def visual_loss(generated: Tensor, clean: Tensor) -> Tensor:
    """MSE plus structural dissimilarity."""
    return functional.mse_loss(generated, clean) + (1 - ssim(generated, clean))
