"""End-to-end training phases shared by smoke and CIFAR-10 runs."""

from __future__ import annotations

import json
import itertools
import hashlib
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image

from fvba.anchors.base import FrozenAnchor, compute_centroids
from fvba.anchors.vtba import ViTSemanticAnchor, fit_linear_head
from fvba.data.cifar import SyntheticClassificationDataset, build_cifar10
from fvba.data.imagefolder import build_imagefolder
from fvba.data.poison import AppendedPoisonDataset, PoisonBundle, materialize_poison
from fvba.data.splits import stratified_indices
from fvba.engine.checkpoint import load_checkpoint, save_checkpoint_atomic
from fvba.engine.generator import (
    train_fpba_epoch,
    train_vtba_stage1_epoch,
    train_vtba_stage2_epoch,
)
from fvba.engine.victim import build_victim_scheduler, train_classifier_epoch
from fvba.evaluation.metrics import benign_accuracy, full_target_asr, visual_metrics
from fvba.models.classifiers import build_classifier
from fvba.models.lcng import LCNG
from fvba.runtime import resolve_device, seed_everything


class _TinySemanticBackbone(nn.Module):
    num_features = 16

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, self.num_features, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward_features(self, images: Tensor) -> Tensor:
        return torch.flatten(self.network(images), 1)


class _LimitedLoader:
    def __init__(self, loader, max_batches: int) -> None:
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.max_batches)

    def __len__(self) -> int:
        return min(len(self.loader), self.max_batches)


def limit_batches(loader, max_batches: int | None):
    """Return a reusable loader view capped for real-data preflight runs."""
    if max_batches is None:
        return loader
    if int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when configured")
    return _LimitedLoader(loader, int(max_batches))


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_save_atomic(value: Any, path: Path) -> None:
    handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_hash(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _Pipeline:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        runtime = self.config["runtime"]
        self.device = resolve_device(
            runtime["device"],
            full_run=bool(self.config["experiment"]["full_run"]),
            allow_cpu_full_run=bool(runtime["allow_cpu_full_run"]),
        )
        self.seed = int(runtime["seed"])
        seed_everything(self.seed)

    @property
    def output(self) -> Path:
        return Path(self.config["output"]["dir"])

    @property
    def num_classes(self) -> int:
        return int(self.config["data"]["num_classes"])

    def datasets(self):
        data = self.config["data"]
        if data["dataset"] == "synthetic":
            size = int(data["synthetic_size"])
            train = SyntheticClassificationDataset(
                size=size,
                num_classes=self.num_classes,
                image_size=int(data.get("image_size", 32)),
                seed=self.seed,
            )
            test = SyntheticClassificationDataset(
                size=size,
                num_classes=self.num_classes,
                image_size=int(data.get("image_size", 32)),
                seed=self.seed + 10_000,
            )
            return train, train, test
        if data["dataset"] == "cifar10":
            builder = lambda train, augment: build_cifar10(
                data["root"], train=train, augment=augment, download=bool(data["download"])
            )
        elif data["dataset"] in {"animals90", "imagenet100"}:
            builder = lambda train, augment: build_imagefolder(
                data["root"], train=train, augment=augment, image_size=int(data["image_size"])
            )
        else:
            raise ValueError(f"Unsupported dataset: {data['dataset']}")
        deterministic = builder(True, False)
        augmented = builder(True, True)
        test = builder(False, False)
        return deterministic, augmented, test

    def loader(self, dataset, *, shuffle: bool, batch_size: int | None = None):
        data = self.config["data"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size or int(data["batch_size"]),
            shuffle=shuffle,
            num_workers=int(data.get("num_workers", 0)),
            pin_memory=bool(data.get("pin_memory", False)),
            persistent_workers=bool(data.get("persistent_workers", False)) and int(data.get("num_workers", 0)) > 0,
        )
        return limit_batches(loader, data.get("max_batches"))

    def carrier_indices(self, dataset) -> list[int]:
        return stratified_indices(
            dataset.targets,
            per_class=int(self.config["attack"]["carriers_per_class"]),
            num_classes=self.num_classes,
            seed=self.seed,
        )

    def classifier(self, role: str) -> nn.Module:
        model = self.config.get("model", {})
        name = model.get(role, model.get("victim", "resnet18"))
        return build_classifier(name, self.num_classes)

    def semantic_anchor(self) -> nn.Module:
        model = self.config["model"]
        if self.config["data"]["dataset"] == "synthetic":
            return ViTSemanticAnchor(
                "tiny", self.num_classes, backbone=_TinySemanticBackbone(), image_size=32
            )
        checkpoint = model.get("vit_checkpoint")
        return ViTSemanticAnchor(
            model["vit_name"],
            self.num_classes,
            checkpoint_path=checkpoint,
            allow_download=bool(model.get("allow_weight_download", False)),
        )

    def anchor(self) -> nn.Module:
        if self.config["experiment"]["method"] == "vtba":
            anchor = self.semantic_anchor()
            load_checkpoint(self.output / "anchor.pt", model=anchor, restore_rng=False)
        else:
            model = self.classifier("anchor")
            load_checkpoint(self.output / "anchor.pt", model=model, restore_rng=False)
            anchor = FrozenAnchor(model)
        return anchor.to(self.device).eval()

    def generator(self) -> LCNG:
        model = self.config["model"]
        generator = LCNG(
            self.num_classes,
            float(self.config["attack"]["epsilon"]),
            embedding_dim=int(model["embedding_dim"]),
            channels=tuple(model["lcng_channels"]),
        )
        load_checkpoint(self.output / "lcng.pt", model=generator, restore_rng=False)
        return generator.to(self.device)

    def train_classifier(
        self,
        model: nn.Module,
        dataset,
        epochs: int,
        path: Path,
        *,
        initial_state_hash: str | None = None,
        force_rebuild: bool = False,
    ) -> None:
        training = self.config["training"]
        model.to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(training["victim_lr"]),
            momentum=float(training["momentum"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = build_victim_scheduler(optimizer)
        start_epoch = 0
        metadata = {
            "research_artifact": "classifier",
            "initial_state_hash": initial_state_hash or _state_hash(model.state_dict()),
        }
        if path.is_file() and not force_rebuild:
            restored = load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                restore_rng=True,
            )
            start_epoch = int(restored["epoch"])
            metadata.update(restored["metadata"])
        loader = self.loader(dataset, shuffle=True)
        for epoch in range(start_epoch, epochs):
            train_classifier_epoch(model, loader, optimizer, self.device)
            scheduler.step()
            save_checkpoint_atomic(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metadata=metadata,
            )

    def phase_anchor(self, context):
        deterministic, augmented, _ = self.datasets()
        training = self.config["training"]
        if self.config["experiment"]["method"] == "fpba":
            model = self.classifier("anchor")
            self.train_classifier(
                model,
                augmented,
                int(training["anchor_epochs"]),
                self.output / "anchor.pt",
                force_rebuild=bool(context.get("force_rebuild")),
            )
        else:
            anchor = self.semantic_anchor().to(self.device)
            anchor.head.requires_grad_(True).train()
            optimizer = torch.optim.SGD(
                anchor.head.parameters(), lr=float(training.get("linear_head_lr", 0.1))
            )
            start_epoch = 0
            anchor_path = self.output / "anchor.pt"
            if anchor_path.is_file() and not context.get("force_rebuild"):
                restored = load_checkpoint(
                    anchor_path, model=anchor, optimizer=optimizer, restore_rng=True
                )
                start_epoch = int(restored["epoch"])
                anchor.head.requires_grad_(True).train()
            reference_loader = self.loader(deterministic, shuffle=True)
            total_epochs = int(training["linear_head_epochs"])
            for epoch in range(start_epoch, total_epochs):
                for batch in reference_loader:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device, dtype=torch.long)
                    with torch.no_grad():
                        features = anchor.extract_features(images)
                    loss = functional.cross_entropy(anchor.head(features), labels)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                save_checkpoint_atomic(
                    anchor_path,
                    model=anchor,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch + 1,
                    metadata={"research_artifact": "vtba_semantic_reference"},
                )
            anchor.freeze_head()
        return {"checkpoint": "anchor.pt", "artifacts": ["anchor.pt"]}

    def phase_centroids(self, _context):
        deterministic, _, _ = self.datasets()
        centroids = compute_centroids(
            self.anchor(),
            self.loader(deterministic, shuffle=False),
            num_classes=self.num_classes,
            device=self.device,
        )
        _torch_save_atomic(centroids.cpu(), self.output / "centroids.pt")
        return {
            "checkpoint": "centroids.pt",
            "shape": list(centroids.shape),
            "artifacts": ["centroids.pt"],
        }

    def phase_generator(self, context):
        deterministic, _, _ = self.datasets()
        indices = self.carrier_indices(deterministic)
        _write_json(self.output / "carrier_indices.json", indices)
        carriers = Subset(deterministic, indices)
        generator = LCNG(
            self.num_classes,
            float(self.config["attack"]["epsilon"]),
            embedding_dim=int(self.config["model"]["embedding_dim"]),
            channels=tuple(self.config["model"]["lcng_channels"]),
        ).to(self.device)
        anchor = self.anchor()
        centroids = torch.load(self.output / "centroids.pt", map_location=self.device)
        optimizer = torch.optim.Adam(
            generator.parameters(), lr=float(self.config["training"]["generator_lr"])
        )
        generator_path = self.output / "lcng.pt"
        start_epoch = 0
        if generator_path.is_file() and not context.get("force_rebuild"):
            restored = load_checkpoint(
                generator_path,
                model=generator,
                optimizer=optimizer,
                restore_rng=True,
            )
            start_epoch = int(restored["epoch"])
        attack_config = dict(self.config["attack"])
        attack_config.update(seed=self.seed, num_classes=self.num_classes)
        training = self.config["training"]
        if self.config["experiment"]["method"] == "fpba":
            carrier_loader = self.loader(carriers, shuffle=True, batch_size=len(carriers))
            epochs = int(training["generator_epochs"])
            for epoch in range(start_epoch, epochs):
                train_fpba_epoch(
                    generator, anchor, centroids, carrier_loader, optimizer, attack_config, self.device
                )
                save_checkpoint_atomic(
                    generator_path,
                    model=generator,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch + 1,
                    metadata={"research_artifact": "backdoor_generator", "method": "fpba"},
                )
        else:
            stage1_epochs = int(training["stage1_epochs"])
            stage2_epochs = int(training["stage2_epochs"])
            epochs = stage1_epochs + stage2_epochs
            stage1_loader = self.loader(deterministic, shuffle=True)
            carrier_loader = self.loader(carriers, shuffle=True, batch_size=len(carriers))
            for epoch in range(start_epoch, epochs):
                if epoch < stage1_epochs:
                    train_vtba_stage1_epoch(
                        generator,
                        anchor,
                        centroids,
                        stage1_loader,
                        optimizer,
                        attack_config,
                        self.device,
                    )
                    stage = 1
                else:
                    train_vtba_stage2_epoch(
                        generator,
                        anchor,
                        centroids,
                        carrier_loader,
                        optimizer,
                        attack_config,
                        self.device,
                    )
                    stage = 2
                save_checkpoint_atomic(
                    generator_path,
                    model=generator,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch + 1,
                    metadata={
                        "research_artifact": "backdoor_generator",
                        "method": "vtba",
                        "stage": stage,
                    },
                )
        return {
            "checkpoint": "lcng.pt",
            "carrier_count": len(indices),
            "artifacts": ["lcng.pt", "carrier_indices.json"],
        }

    def phase_poison(self, _context):
        deterministic, _, _ = self.datasets()
        indices = json.loads((self.output / "carrier_indices.json").read_text(encoding="utf-8"))
        bundle = materialize_poison(
            self.generator(),
            deterministic,
            indices,
            self.device,
            batch_size=int(self.config["data"]["batch_size"]),
        )
        _torch_save_atomic(bundle, self.output / "poison.pt")
        return {
            "checkpoint": "poison.pt",
            "poison_rate": bundle.poison_rate(len(deterministic)),
            "artifacts": ["poison.pt"],
        }

    def phase_victims(self, context):
        _, augmented, _ = self.datasets()
        bundle: PoisonBundle = torch.load(self.output / "poison.pt", map_location="cpu")
        poison_transform = None
        if self.config["data"]["dataset"] == "cifar10":
            poison_transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            )
        poisoned_dataset = AppendedPoisonDataset(augmented, bundle, poison_transform)
        epochs = int(self.config["training"]["victim_epochs"])
        seed_everything(self.seed + 2_000)
        base_model = self.classifier("victim")
        initial_state = base_model.state_dict()
        initial_hash = _state_hash(initial_state)
        clean_model = self.classifier("victim")
        clean_model.load_state_dict(initial_state)
        backdoored_model = self.classifier("victim")
        backdoored_model.load_state_dict(initial_state)
        seed_everything(self.seed + 3_000)
        self.train_classifier(
            clean_model,
            augmented,
            epochs,
            self.output / "clean_victim.pt",
            initial_state_hash=initial_hash,
            force_rebuild=bool(context.get("force_rebuild")),
        )
        seed_everything(self.seed + 3_000)
        self.train_classifier(
            backdoored_model,
            poisoned_dataset,
            epochs,
            self.output / "backdoored_victim.pt",
            initial_state_hash=initial_hash,
            force_rebuild=bool(context.get("force_rebuild")),
        )
        return {
            "clean": "clean_victim.pt",
            "backdoored": "backdoored_victim.pt",
            "artifacts": ["clean_victim.pt", "backdoored_victim.pt"],
        }

    def _load_victim(self, filename: str) -> nn.Module:
        model = self.classifier("victim")
        load_checkpoint(self.output / filename, model=model, restore_rng=False)
        return model.to(self.device).eval()

    def phase_evaluation(self, _context):
        deterministic, _, test = self.datasets()
        clean_victim = self._load_victim("clean_victim.pt")
        backdoored = self._load_victim("backdoored_victim.pt")
        generator = self.generator().eval()
        test_loader = self.loader(test, shuffle=False)
        clean_ba = benign_accuracy(clean_victim, test_loader, self.device)
        backdoored_ba = benign_accuracy(backdoored, test_loader, self.device)
        attack = full_target_asr(
            backdoored, generator, test_loader, self.num_classes, self.device
        )
        metrics = {
            "clean_ba": clean_ba,
            "backdoored_ba": backdoored_ba,
            "accuracy_degradation": clean_ba - backdoored_ba,
            **attack,
        }
        _write_json(self.output / "metrics.json", metrics)
        bundle: PoisonBundle = torch.load(self.output / "poison.pt", map_location="cpu")
        clean_carriers = torch.stack(
            [deterministic[index][0] for index in bundle.carrier_indices]
        )
        visual = visual_metrics(
            clean_carriers.to(self.device),
            bundle.images.to(self.device),
            require_lpips=bool(self.config["evaluation"]["require_lpips"]),
            epsilon_budget=float(self.config["attack"]["epsilon"]),
        )
        _write_json(self.output / "visual_metrics.json", visual)
        count = min(8, bundle.images.shape[0])
        grid = torch.cat(
            [
                clean_carriers[:count],
                bundle.images[:count],
                (bundle.images[:count] - clean_carriers[:count]).abs().mul(5).clamp(0, 1),
            ]
        )
        save_image(grid, self.output / "poison_examples.png", nrow=count)
        return {
            "metrics": "metrics.json",
            "visual_metrics": "visual_metrics.json",
            "artifacts": ["metrics.json", "visual_metrics.json", "poison_examples.png"],
        }


def build_pipeline_phases(config: Mapping[str, Any]):
    """Build the resumable phase graph for one experiment cell."""
    pipeline = _Pipeline(config)
    return {
        "anchor": pipeline.phase_anchor,
        "centroids": pipeline.phase_centroids,
        "generator": pipeline.phase_generator,
        "poison": pipeline.phase_poison,
        "victims": pipeline.phase_victims,
        "evaluation": pipeline.phase_evaluation,
    }
