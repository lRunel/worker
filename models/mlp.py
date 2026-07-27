"""
MLP — Multi-Layer Perceptron for classification.

Configurable number of hidden layers with batch normalization,
ReLU activations, and optional dropout.
"""
import torch
import torch.nn as nn
from . import register


@register("mlp", description="Multi-Layer Perceptron with configurable hidden layers")
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | None = None,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        dim_in = input_dim
        for dim_out in hidden_dims:
            layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out

        layers.append(nn.Linear(dim_in, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input (e.g., [B, 1, 28, 28] -> [B, 784])
        x = x.view(x.size(0), -1)
        return self.net(x)
