"""Label-Conditioned Noise Generator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .adain import ConditionalAdaIN


class _EncoderBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class LCNG(nn.Module):
    """Four-stage conditional residual encoder-decoder."""

    def __init__(
        self,
        num_classes: int,
        epsilon: float,
        embedding_dim: int = 512,
        channels: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("LCNG requires exactly four encoder channel values")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0, 1]")
        self.num_classes = num_classes
        self.epsilon = float(epsilon)
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        encoder_channels = (3, *channels)
        self.encoder = nn.Sequential(
            *[
                _EncoderBlock(encoder_channels[index], encoder_channels[index + 1])
                for index in range(4)
            ]
        )
        decoder_channels = (channels[3], channels[2], channels[1], channels[0], 3)
        self.decoder = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    decoder_channels[index],
                    decoder_channels[index + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
                for index in range(4)
            ]
        )
        self.modulation = nn.ModuleList(
            [ConditionalAdaIN(output_channels, embedding_dim) for output_channels in decoder_channels[1:]]
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        generator_input: Tensor,
        target: Tensor,
        reference_image: Tensor | None = None,
    ) -> Tensor:
        if generator_input.ndim != 4 or generator_input.shape[1] != 3:
            raise ValueError("generator_input must have shape [N, 3, H, W]")
        if target.ndim != 1 or target.shape[0] != generator_input.shape[0]:
            raise ValueError("target must have shape [N]")
        if target.numel() and (target.min() < 0 or target.max() >= self.num_classes):
            raise ValueError("target index is outside the configured classes")
        if reference_image is None:
            reference_image = generator_input
        if reference_image.shape != generator_input.shape:
            raise ValueError("reference_image must have the same shape as generator_input")

        condition = self.embedding(target)
        hidden = self.encoder(generator_input)
        for index, (decoder, modulation) in enumerate(zip(self.decoder, self.modulation)):
            hidden = modulation(decoder(hidden), condition)
            if index < len(self.decoder) - 1:
                hidden = self.activation(hidden)
        residual = self.epsilon * torch.tanh(hidden)
        return (reference_image + residual).clamp(0, 1)
