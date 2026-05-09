from __future__ import annotations

import torch
from torch import nn


def _mlp_layers(
    input_features: int,
    hidden_sizes: list[int] | tuple[int, ...],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_features)
    for hidden_size in hidden_sizes:
        layers.extend(
            [
                nn.Linear(previous, int(hidden_size)),
                nn.ReLU(),
                nn.BatchNorm1d(int(hidden_size)),
                nn.Dropout(float(dropout)),
            ]
        )
        previous = int(hidden_size)
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


class DenseMLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = _mlp_layers(input_features, hidden_sizes, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        categorical_cardinalities: list[int] | tuple[int, ...],
        num_continuous: int,
        embedding_dim: int = 16,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dims = [
            max(2, min(int(embedding_dim), int(round(cardinality**0.5)) + 1))
            for cardinality in categorical_cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(int(cardinality), int(dim))
                for cardinality, dim in zip(categorical_cardinalities, self.embedding_dims)
            ]
        )
        input_features = int(num_continuous) + int(sum(self.embedding_dims))
        self.net = _mlp_layers(input_features, hidden_sizes, dropout)

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            embedded = [
                embedding(x_categ[:, index].clamp(min=0, max=embedding.num_embeddings - 1))
                for index, embedding in enumerate(self.embeddings)
            ]
            features = torch.cat([*embedded, x_cont], dim=1)
        else:
            features = x_cont
        return self.net(features)
