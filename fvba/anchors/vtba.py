"""Frozen pretrained ViT semantic reference used by VTBA."""

from __future__ import annotations

from pathlib import Path
import hashlib

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class ViTSemanticAnchor(nn.Module):
    """Frozen ViT backbone plus a task-specific linear classification head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        checkpoint_path: Path | str | None = None,
        allow_download: bool = False,
        *,
        backbone: nn.Module | None = None,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        if backbone is None:
            checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
            if checkpoint is None or not checkpoint.is_file():
                if not allow_download:
                    raise RuntimeError(
                        "Pretrained VTBA weights are unavailable: provide checkpoint_path "
                        "or set allow_download=true"
                    )
                import timm

                backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            else:
                import timm

                backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
                payload = torch.load(checkpoint, map_location="cpu")
                state = payload.get("state_dict", payload.get("model", payload))
                state = {
                    key.removeprefix("module.").removeprefix("model."): value
                    for key, value in state.items()
                }
                incompatible = backbone.load_state_dict(state, strict=False)
                unexpected = [
                    key
                    for key in incompatible.unexpected_keys
                    if not key.startswith(("head.", "head_dist."))
                ]
                if incompatible.missing_keys or unexpected:
                    raise RuntimeError(
                        "ViT checkpoint coverage validation failed; "
                        f"missing={incompatible.missing_keys[:8]}, unexpected={unexpected[:8]}"
                    )
                digest = hashlib.sha256()
                with checkpoint.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                self.weight_fingerprint = digest.hexdigest()
        if not hasattr(backbone, "forward_features") or not hasattr(backbone, "num_features"):
            raise TypeError("ViT backbone must expose forward_features and num_features")
        self.backbone = backbone
        self.backbone.requires_grad_(False).eval()
        self.head = nn.Linear(int(backbone.num_features), num_classes)
        pretrained = getattr(backbone, "pretrained_cfg", {}) or {}
        configured_size = pretrained.get("input_size", (3, 224, 224))[-1]
        self.image_size = int(image_size if image_size is not None else configured_size)
        self.interpolation = str(pretrained.get("interpolation", "bilinear"))
        mean = pretrained.get("mean", (0.485, 0.456, 0.406))
        std = pretrained.get("std", (0.229, 0.224, 0.225))
        self.register_buffer(
            "normalization_mean", torch.tensor(mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "normalization_std", torch.tensor(std).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, images: Tensor) -> Tensor:
        resize_kwargs = {
            "size": (self.image_size, self.image_size),
            "mode": self.interpolation,
        }
        if self.interpolation in {"bilinear", "bicubic"}:
            resize_kwargs.update(align_corners=False, antialias=True)
        resized = functional.interpolate(images, **resize_kwargs)
        normalized = (resized - self.normalization_mean) / self.normalization_std
        features = self.backbone.forward_features(normalized)
        if features.ndim == 3:
            features = features[:, 0]
        if features.ndim != 2:
            raise ValueError("ViT backbone must return [N,D] or token features [N,T,D]")
        return features

    def logits_and_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.extract_features(images)
        return self.head(features), features

    def forward(self, images: Tensor) -> Tensor:
        return self.logits_and_features(images)[0]

    def freeze_head(self) -> None:
        self.head.requires_grad_(False).eval()


def fit_linear_head(
    anchor: ViTSemanticAnchor,
    loader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """Fit the downstream classifier on frozen features, then freeze it."""
    anchor.to(device)
    anchor.backbone.requires_grad_(False).eval()
    anchor.head.requires_grad_(True).train()
    optimizer = torch.optim.SGD(anchor.head.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device, dtype=torch.long)
            with torch.no_grad():
                features = anchor.extract_features(images)
            loss = functional.cross_entropy(anchor.head(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    anchor.freeze_head()
