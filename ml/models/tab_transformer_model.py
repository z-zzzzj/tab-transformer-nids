from __future__ import annotations

import numpy as np
import torch
from torch import nn
from tab_transformer_pytorch import TabTransformer


def build_tab_transformer(
    categorical_cardinalities: list[int],
    num_continuous: int,
    model_config: dict,
    continuous_mean_std: np.ndarray,
) -> TabTransformer:
    cont_tensor = torch.as_tensor(continuous_mean_std, dtype=torch.float32)
    return TabTransformer(
        categories=tuple(categorical_cardinalities),
        num_continuous=num_continuous,
        dim=model_config["dim"],
        dim_out=1,
        depth=model_config["depth"],
        heads=model_config["heads"],
        attn_dropout=model_config["attn_dropout"],
        ff_dropout=model_config["ff_dropout"],
        mlp_hidden_mults=tuple(model_config.get("mlp_hidden_mults", [4, 2])),
        # Recent tab-transformer-pytorch forwards `mlp_act=None` into x-mlps,
        # which builds `Sequential(linear, None)` and crashes at runtime.
        mlp_act=nn.ReLU(),
        continuous_mean_std=cont_tensor,
    )


def load_tab_transformer_checkpoint(
    *,
    checkpoint: dict[str, object],
    categorical_cardinalities: list[int],
    num_continuous: int,
    device: torch.device,
) -> tuple[nn.Module, float]:
    state_dict = checkpoint["state_dict"]
    continuous_mean_std = np.array(checkpoint["continuous_mean_std"], dtype=np.float32)
    model = build_tab_transformer(
        categorical_cardinalities=categorical_cardinalities,
        num_continuous=num_continuous,
        model_config=checkpoint["model_config"],
        continuous_mean_std=continuous_mean_std,
    )
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, float(checkpoint.get("temperature", 1.0))
