"""Atomic, resumable PyTorch checkpoints including random-number state."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint_atomic(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "metadata": metadata or {},
        "rng": _rng_state(),
    }
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        random.setstate(payload["rng"]["python"])
        torch.set_rng_state(payload["rng"]["torch"])
        if torch.cuda.is_available() and "cuda" in payload["rng"]:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return {"epoch": payload["epoch"], "metadata": payload["metadata"]}

