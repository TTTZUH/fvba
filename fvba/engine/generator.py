"""LCNG training phases for FPBA and VTBA."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from fvba.attacks.dct import frequency_perturb
from fvba.attacks.losses import cfal_loss, visual_loss
from fvba.data.splits import different_class_permutation


def _accumulate(totals: dict[str, float], values: Mapping[str, Tensor], count: int) -> None:
    for name, value in values.items():
        totals[name] += float(value.detach()) * count


def train_fpba_epoch(
    generator: nn.Module,
    anchor: nn.Module,
    centroids: Tensor,
    loader,
    optimizer: torch.optim.Optimizer,
    config: Mapping,
    device: torch.device,
) -> dict[str, float]:
    """Train one FPBA epoch using Eqs. 8 and 10-14."""
    generator.train()
    anchor.eval()
    centroids = centroids.to(device)
    totals = {name: 0.0 for name in ("loss", "classification", "cfal", "visual")}
    seen = 0
    pairing_generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    dct = config["dct"]

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device, dtype=torch.long)
        targets = batch[3].to(device, dtype=torch.long) if len(batch) > 3 else labels
        if not torch.equal(labels, targets):
            raise ValueError("FPBA carrier conditions must preserve the clean-label target")
        auxiliary_indices = different_class_permutation(labels.cpu(), pairing_generator).to(device)
        dct_input = frequency_perturb(
            images,
            images[auxiliary_indices],
            strength=float(dct["strength"]),
            low_freq_size=int(dct["low_freq_size"]),
            block_size=int(dct["block_size"]),
        )
        generated = generator(dct_input, targets, reference_image=images)
        logits, features = anchor.logits_and_features(generated)
        classification = functional.cross_entropy(logits, targets)
        alignment = cfal_loss(features, centroids, targets)
        visual = visual_loss(generated, images)
        loss = (
            float(config["alpha"]) * classification
            + float(config["beta"]) * alignment
            + float(config["gamma"]) * visual
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite FPBA generator loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        count = images.shape[0]
        _accumulate(
            totals,
            {
                "loss": loss,
                "classification": classification,
                "cfal": alignment,
                "visual": visual,
            },
            count,
        )
        seen += count
    if seen == 0:
        raise ValueError("FPBA carrier loader is empty")
    return {name: total / seen for name, total in totals.items()}


def sample_cross_class_targets(
    labels: Tensor,
    num_classes: int,
    generator: torch.Generator | None,
) -> Tensor:
    """Uniformly choose a target from all classes except the source class."""
    if num_classes < 2:
        raise ValueError("Cross-class targets require at least two classes")
    if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("source label is outside [0, num_classes)")
    offsets = torch.randint(1, num_classes, labels.shape, generator=generator)
    return ((labels.cpu() + offsets) % num_classes).to(labels.device)


def _train_vtba_epoch(
    stage: int,
    generator: nn.Module,
    anchor: nn.Module,
    centroids: Tensor,
    loader,
    optimizer: torch.optim.Optimizer,
    config: Mapping,
    device: torch.device,
) -> dict[str, float]:
    generator.train()
    anchor.eval()
    centroids = centroids.to(device)
    totals = {name: 0.0 for name in ("loss", "classification", "cfal")}
    seen = 0
    target_generator = config.get("_target_generator")
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device, dtype=torch.long)
        if stage == 1:
            targets = sample_cross_class_targets(
                labels, int(config["num_classes"]), target_generator
            )
        else:
            targets = batch[3].to(device, dtype=torch.long) if len(batch) > 3 else labels
            if not torch.equal(targets, labels):
                raise ValueError("VTBA Stage 2 conditions must equal the in-class carrier label")
        generated = generator(images, targets)
        logits, features = anchor.logits_and_features(generated)
        classification = functional.cross_entropy(logits, targets)
        alignment = cfal_loss(features, centroids, targets)
        loss = float(config["alpha"]) * classification + float(config["beta"]) * alignment
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite VTBA Stage {stage} generator loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = images.shape[0]
        _accumulate(
            totals,
            {"loss": loss, "classification": classification, "cfal": alignment},
            count,
        )
        seen += count
    if seen == 0:
        raise ValueError(f"VTBA Stage {stage} loader is empty")
    return {name: total / seen for name, total in totals.items()}


def train_vtba_stage1_epoch(
    generator,
    anchor,
    centroids,
    loader,
    optimizer,
    config,
    device,
) -> dict[str, float]:
    return _train_vtba_epoch(
        1, generator, anchor, centroids, loader, optimizer, config, device
    )


def train_vtba_stage2_epoch(
    generator,
    anchor,
    centroids,
    loader,
    optimizer,
    config,
    device,
) -> dict[str, float]:
    return _train_vtba_epoch(
        2, generator, anchor, centroids, loader, optimizer, config, device
    )
