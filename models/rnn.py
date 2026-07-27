"""
RNN — LSTM-based Recurrent Neural Network for sequence classification.

For image inputs (e.g., MNIST 28×28), treats each row as a timestep
with `input_size` features, producing a sequence of length `seq_len`.
The final hidden state is passed through a classifier head.
"""
import torch
import torch.nn as nn
from . import register


@register("rnn", description="LSTM-based RNN classifier (treats image rows as sequences)")
class RNN(nn.Module):
    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        classifier_input = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> reshape to [B, H, W] (squeeze channel for grayscale)
        if x.dim() == 4:
            x = x.squeeze(1)  # [B, H, W] — each row is a timestep
        # x: [B, seq_len, input_size]

        # LSTM forward
        lstm_out, (h_n, _) = self.lstm(x)

        # Use final hidden state from all layers' last direction
        if self.bidirectional:
            # Concatenate final forward and backward hidden states
            h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_final = h_n[-1]  # [B, hidden_size]

        return self.classifier(h_final)
