from __future__ import annotations

import torch
from torch import nn


class FeatureSequenceLSTM(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.input_features = input_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence = features.unsqueeze(-1)
        _, (hidden, _) = self.lstm(sequence)
        final_hidden = self.norm(hidden[-1])
        return self.head(final_hidden)

