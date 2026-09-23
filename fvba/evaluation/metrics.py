"""Metrics for attack success and image quality."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from fvba.attacks.losses import ssim


@torch.no_grad()
def benign_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.to(device).eval()
    correct = 0
    seen = 0
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device, dtype=torch.long)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels).sum())
        seen += labels.numel()
    if seen == 0:
        raise ValueError("Benign evaluation loader is empty")
    return correct / seen


@torch.no_grad()
def full_target_asr(
    model: nn.Module,
    generator: nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
) -> dict[str, object]:
    """Compute Eqs. 25-26, excluding source samples already in each target."""
    model.to(device).eval()
    generator.to(device).eval()
    successes: list[int] = []
    counts: list[int] = []
    for target_class in range(num_classes):
        target_successes = 0
        target_count = 0
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device, dtype=torch.long)
            if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
                raise ValueError("Evaluation label is outside [0, num_classes)")
            keep = labels != target_class
            if not torch.any(keep):
                continue
            source = images[keep]
            targets = torch.full(
                (source.shape[0],), target_class, device=device, dtype=torch.long
            )
            triggered = generator(source, targets)
            predictions = model(triggered).argmax(dim=1)
            target_successes += int((predictions == targets).sum())
            target_count += targets.numel()
        if target_count == 0:
            raise ValueError(f"No non-target samples exist for target class {target_class}")
        successes.append(target_successes)
        counts.append(target_count)
    class_asr = [success / count for success, count in zip(successes, counts)]
    return {
        "asr_per_target": class_asr,
        "counts_per_target": counts,
        "successes_per_target": successes,
        "mean_asr": sum(class_asr) / num_classes,
    }


@torch.no_grad()
def visual_metrics(
    clean: Tensor,
    poisoned: Tensor,
    *,
    require_lpips: bool,
    epsilon_budget: float = 80 / 255,
    psnr_threshold: float = 30.0,
) -> dict[str, object]:
    if clean.shape != poisoned.shape or clean.ndim != 4:
        raise ValueError("clean and poisoned images must share shape [N,C,H,W]")
    if clean.shape[0] == 0:
        raise ValueError("visual metric input must not be empty")
    difference = poisoned - clean
    mse = difference.square().flatten(1).mean(dim=1)
    psnr_values = torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        10 * torch.log10(torch.ones_like(mse) / mse),
    )
    linf_values = difference.abs().flatten(1).amax(dim=1)
    lpips_value: float | None = None
    lpips_status = "not_requested"
    if require_lpips:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "LPIPS evaluation requested but package 'lpips' is not installed"
            ) from error
        metric = lpips.LPIPS(net="alex").to(clean.device).eval()
        lpips_value = float(metric(clean * 2 - 1, poisoned * 2 - 1).mean())
        lpips_status = "complete"
    return {
        "psnr_db": float(psnr_values.mean()),
        "ssim": float(ssim(clean, poisoned)),
        "linf_max": float(linf_values.max()),
        "lpips": lpips_value,
        "lpips_status": lpips_status,
        "psnr_below_30_count": int((psnr_values < psnr_threshold).sum()),
        "linf_above_budget_count": int((linf_values > epsilon_budget).sum()),
        "sample_count": clean.shape[0],
    }
