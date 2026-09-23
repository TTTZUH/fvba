"""Runtime determinism and device selection."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all supported random generators and optionally request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(
    requested: str,
    *,
    full_run: bool,
    allow_cpu_full_run: bool,
) -> torch.device:
    """Resolve a requested device and guard accidental full CPU launches."""
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if full_run and device.type == "cpu" and not allow_cpu_full_run:
        raise RuntimeError(
            "Full runs require CUDA; set runtime.allow_cpu_full_run=true to override"
        )
    return device
