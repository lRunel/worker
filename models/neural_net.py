"""
NeuralNet — Basic feedforward neural network (vanilla fully-connected).

The simplest architecture: input → hidden → output.
No batch normalization or dropout — intentionally minimal for baselines.
"""
import torch
import torch.nn as nn
from . import register


@register("neural_net", description="Basic feedforward neural network (vanilla FC)")
class NeuralNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
