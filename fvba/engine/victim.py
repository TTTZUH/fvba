"""Standard classifier training used for clean and poisoned victims."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def train_classifier_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device, dtype=torch.long)
        logits = model(images)
        loss = functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite victim classification loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = images.shape[0]
        total_loss += float(loss.detach()) * count
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += count
    if seen == 0:
        raise ValueError("Victim training loader is empty")
    return {"loss": total_loss / seen, "accuracy": correct / seen}


def build_victim_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

