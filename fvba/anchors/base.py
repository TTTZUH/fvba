"""Shared anchor adapter and clean feature-centroid computation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenAnchor(nn.Module):
    """Freeze model parameters without disabling gradients to image inputs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not hasattr(model, "forward_features"):
            raise TypeError("Anchor model must expose forward_features(images)")
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def logits_and_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.model.forward_features(images)
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "fc"):
            logits = self.model.backbone.fc(features)
        elif hasattr(self.model, "classifier"):
            logits = self.model.classifier(features)
        else:
            logits = self.model(images)
        return logits, features


@torch.no_grad()
def compute_centroids(
    anchor: FrozenAnchor,
    loader,
    *,
    num_classes: int,
    device: torch.device,
) -> Tensor:
    """Compute class centroids with explicit missing and invalid-class checks."""
    sums: Tensor | None = None
    counts = torch.zeros(num_classes, device=device, dtype=torch.long)
    anchor.to(device).eval()
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device, dtype=torch.long)
        if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
            raise ValueError("Reference labels contain a class outside [0, num_classes)")
        _, features = anchor.logits_and_features(images)
        if sums is None:
            sums = torch.zeros(
                num_classes, features.shape[1], device=device, dtype=features.dtype
            )
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))
    missing = torch.where(counts == 0)[0]
    if missing.numel():
        raise ValueError(f"Reference data has no samples for classes: {missing.tolist()}")
    if sums is None:
        raise ValueError("Reference loader is empty")
    return sums / counts[:, None].to(sums.dtype)

