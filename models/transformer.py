"""
Transformer — Encoder-based Transformer for classification.

Splits input images into patches, projects them to an embedding space,
adds learned positional encoding, and uses a Transformer encoder stack.
A [CLS] token is prepended and its final representation is used for classification.
"""
import math
import torch
import torch.nn as nn
from . import register


@register("transformer", description="Transformer encoder classifier with patch embedding")
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 28,
        patch_size: int = 7,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_size % patch_size == 0, \
            f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"

        self.patch_size = patch_size
        num_patches = (input_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding: flatten each patch and project to d_model
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image tensor [B, C, H, W] to patch sequence [B, num_patches, patch_dim]."""
        B, C, H, W = x.shape
        p = self.patch_size
        # Unfold into patches
        x = x.unfold(2, p, p).unfold(3, p, p)   # [B, C, H//p, W//p, p, p]
        x = x.contiguous().view(B, C, -1, p, p)  # [B, C, num_patches, p, p]
        x = x.permute(0, 2, 1, 3, 4)             # [B, num_patches, C, p, p]
        x = x.contiguous().view(B, -1, C * p * p) # [B, num_patches, patch_dim]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Patchify and embed
        patches = self._patchify(x)               # [B, num_patches, patch_dim]
        x = self.patch_embed(patches)              # [B, num_patches, d_model]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)     # [B, num_patches+1, d_model]

        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x)                        # [B, num_patches+1, d_model]

        # Classify from [CLS] token
        cls_output = x[:, 0]                       # [B, d_model]
        return self.classifier(cls_output)
