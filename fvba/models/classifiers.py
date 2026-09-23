"""CIFAR-adapted victim and proxy classifiers with a shared feature API."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import models


class CifarResNet18(nn.Module):
    feature_dim = 512

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=None, num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

    def forward_features(self, images: Tensor) -> Tensor:
        model = self.backbone
        hidden = model.relu(model.bn1(model.conv1(images)))
        hidden = model.maxpool(hidden)
        hidden = model.layer1(hidden)
        hidden = model.layer2(hidden)
        hidden = model.layer3(hidden)
        hidden = model.layer4(hidden)
        return torch.flatten(model.avgpool(hidden), 1)

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone.fc(self.forward_features(images))


class CifarVGG19(nn.Module):
    feature_dim = 512

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        original = models.vgg19(weights=None)
        self.features = original.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, images: Tensor) -> Tensor:
        return torch.flatten(self.avgpool(self.features(images)), 1)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.forward_features(images))


class TinyClassifier(nn.Module):
    """Compact classifier used only by the offline pipeline smoke test."""

    feature_dim = 16

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, self.feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, images: Tensor) -> Tensor:
        return torch.flatten(self.features(images), 1)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.forward_features(images))


def build_classifier(name: str, num_classes: int) -> nn.Module:
    normalized = name.lower().replace("-", "")
    if normalized == "resnet18":
        return CifarResNet18(num_classes)
    if normalized == "vgg19":
        return CifarVGG19(num_classes)
    if normalized == "tiny":
        return TinyClassifier(num_classes)
    raise ValueError(f"Unsupported classifier: {name}")
