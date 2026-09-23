"""Differentiable block-DCT utilities used by FPBA."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _basis(size: int, like: Tensor) -> Tensor:
    positions = torch.arange(size, device=like.device, dtype=like.dtype)
    frequencies = positions[:, None]
    scale = torch.full((size,), math.sqrt(2.0 / size), device=like.device, dtype=like.dtype)
    scale[0] = math.sqrt(1.0 / size)
    return scale[:, None] * torch.cos(
        math.pi * (2 * positions[None, :] + 1) * frequencies / (2 * size)
    )


def _as_blocks(x: Tensor, block_size: int) -> Tensor:
    if x.ndim != 4:
        raise ValueError("Expected image tensor with shape [N, C, H, W]")
    n, c, height, width = x.shape
    if height % block_size or width % block_size:
        raise ValueError(f"Image height and width must be divisible by {block_size}")
    return (
        x.reshape(n, c, height // block_size, block_size, width // block_size, block_size)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )


def block_dct2(x: Tensor, block_size: int = 8) -> Tensor:
    """Return coefficients shaped `[N,C,H/B,W/B,B,B]`."""
    blocks = _as_blocks(x, block_size)
    basis = _basis(block_size, x)
    return torch.matmul(torch.matmul(basis, blocks), basis.transpose(0, 1))


def block_idct2(coefficients: Tensor, block_size: int = 8) -> Tensor:
    """Invert coefficients produced by :func:`block_dct2`."""
    if coefficients.ndim != 6 or coefficients.shape[-2:] != (block_size, block_size):
        raise ValueError("Expected coefficients with shape [N, C, H/B, W/B, B, B]")
    basis = _basis(block_size, coefficients)
    blocks = torch.matmul(
        torch.matmul(basis.transpose(0, 1), coefficients), basis
    )
    n, c, height_blocks, width_blocks, _, _ = blocks.shape
    return (
        blocks.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(n, c, height_blocks * block_size, width_blocks * block_size)
    )


def mid_high_mask(
    block_size: int = 8,
    low_freq_size: int = 3,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a frequency mask with an excluded upper-left square."""
    if not 0 <= low_freq_size <= block_size:
        raise ValueError("low_freq_size must lie in [0, block_size]")
    mask = torch.ones((block_size, block_size), device=device, dtype=dtype)
    mask[:low_freq_size, :low_freq_size] = 0
    return mask


def frequency_perturb(
    carrier: Tensor,
    auxiliary: Tensor,
    *,
    strength: float = 1.5,
    low_freq_size: int = 3,
    block_size: int = 8,
    clamp: bool = True,
) -> Tensor:
    """Apply a frequency-domain perturbation using another class's coefficients."""
    if carrier.shape != auxiliary.shape:
        raise ValueError("carrier and auxiliary must have the same shape")
    carrier_coeff = block_dct2(carrier, block_size)
    auxiliary_coeff = block_dct2(auxiliary, block_size)
    mask = mid_high_mask(
        block_size,
        low_freq_size,
        device=carrier.device,
        dtype=carrier.dtype,
    )
    result = block_idct2(carrier_coeff + strength * mask * auxiliary_coeff, block_size)
    return result.clamp(0, 1) if clamp else result
