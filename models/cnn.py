"""
CNN — Convolutional Neural Network for image classification.

Configurable conv layers with batch normalization, max pooling,
and a fully connected classifier head.
"""
import torch
import torch.nn as nn
from . import register


@register("cnn", description="Convolutional Neural Network for image classification")
class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28,
        conv_channels: list[int] | None = None,
        fc_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]

        layers = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            ch_in = ch_out

        self.features = nn.Sequential(*layers)

        # Calculate flattened size after conv layers
        feat_size = input_size
        for _ in conv_channels:
            feat_size = feat_size // 2  # each MaxPool2d(2) halves spatial dims

        flat_dim = conv_channels[-1] * feat_size * feat_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
