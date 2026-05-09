from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ml.models.tab_transformer_model import load_tab_transformer_checkpoint
from ml.scripts.train import run_tab_transformer, run_tab_transformer_no_engineered
from tab_transformer_nids.evaluation import choose_threshold_with_plateau
from tab_transformer_nids.preprocessing import TabularPreprocessor


def build_toy_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    attack_families = ["Bot", "Web Attack - XSS", "Web Attack - Brute Force"]
    for index in range(64):
        is_attack = index % 3 == 0
        rows.append(
            {
                "protocol": "TCP" if index % 2 == 0 else "UDP",
                "destination_port_bucket": "80" if index % 4 < 2 else "443",
                "flow_duration": float(index + 1),
                "flow_bytes_per_s": float((index + 1) * 10),
                "total_packets_bin": "small" if index % 2 == 0 else "large",
                "total_packets": float(index + 2),
                "target_binary": 1 if is_attack else 0,
                "label_original": (
                    attack_families[(index // 3) % len(attack_families)]
                    if is_attack
                    else "BENIGN"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_training_config() -> dict[str, object]:
    return {
        "dataset": {"benign_label": "BENIGN"},
        "training": {
            "batch_size": 8,
            "eval_batch_size": 16,
            "epochs": 2,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "patience": 1,
            "positive_class_weight": 1.0,
            "threshold_selection_policy": "f1_plateau",
            "selection_split_name": "inner_val",
            "auto_batch_tune": {"enabled": False},
        },
        "tab_transformer": {
            "dim": 16,
            "depth": 2,
            "heads": 4,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
            "mlp_hidden_mults": [2, 1],
        },
    }


def fit_preprocessor(frame: pd.DataFrame) -> TabularPreprocessor:
    train_df = frame.iloc[:40].reset_index(drop=True)
    preprocessor = TabularPreprocessor(
        categorical_columns=["protocol", "destination_port_bucket", "total_packets_bin"],
        continuous_columns=["flow_duration", "flow_bytes_per_s", "total_packets"],
    )
    preprocessor.fit(train_df)
    return preprocessor


def feature_schema_from_preprocessor(preprocessor: TabularPreprocessor) -> dict[str, object]:
    return {
        "categorical_columns": list(preprocessor.categorical_columns),
        "continuous_columns": list(preprocessor.continuous_columns),
        "categorical_cardinalities": list(preprocessor.categorical_cardinalities),
        "category_maps": preprocessor.category_maps,
        "continuous_mean_std": preprocessor.continuous_mean_std_tensor().tolist(),
    }


def test_choose_threshold_with_plateau_prefers_higher_recall_and_lower_threshold() -> None:
    y_true = np.asarray([1, 1, 0, 0], dtype=np.int64)
    scores = np.asarray([0.60, 0.59, 0.58, 0.10], dtype=np.float32)

    selection = choose_threshold_with_plateau(y_true, scores, plateau_tolerance=0.21)

    assert selection["threshold"] == pytest.approx(0.58, abs=1e-6)
    assert selection["recall"] == pytest.approx(1.0)
    assert selection["selection_rule"].startswith("f1_plateau_tol_")


def test_run_tab_transformer_writes_current_public_artifacts(tmp_path: Path) -> None:
    frame = build_toy_frame()
    train_df = frame.iloc[:40].reset_index(drop=True)
    val_df = frame.iloc[40:52].reset_index(drop=True)
    test_df = frame.iloc[52:].reset_index(drop=True)
    preprocessor = fit_preprocessor(frame)
    feature_schema = feature_schema_from_preprocessor(preprocessor)

    result = run_tab_transformer(
        config=make_training_config(),
        preprocessor=preprocessor,
        feature_schema=feature_schema,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        experiment_dir=tmp_path / "run",
    )

    assert result.epochs_completed >= 1
    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "threshold.json").exists()
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_load_tab_transformer_checkpoint_uses_current_public_format(tmp_path: Path) -> None:
    frame = build_toy_frame()
    train_df = frame.iloc[:40].reset_index(drop=True)
    val_df = frame.iloc[40:52].reset_index(drop=True)
    test_df = frame.iloc[52:].reset_index(drop=True)
    preprocessor = fit_preprocessor(frame)
    feature_schema = feature_schema_from_preprocessor(preprocessor)
    run_tab_transformer(
        config=make_training_config(),
        preprocessor=preprocessor,
        feature_schema=feature_schema,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        experiment_dir=tmp_path / "run",
    )
    checkpoint = torch.load(tmp_path / "run" / "model.pt", map_location="cpu")

    model, temperature = load_tab_transformer_checkpoint(
        checkpoint=checkpoint,
        categorical_cardinalities=feature_schema["categorical_cardinalities"],
        num_continuous=len(preprocessor.continuous_columns),
        device=torch.device("cpu"),
    )

    assert checkpoint["model_type"] == "tab_transformer"
    assert isinstance(temperature, float)
    assert model.training is False


def test_no_engineered_ablation_entry_writes_artifacts(tmp_path: Path) -> None:
    frame = build_toy_frame()
    train_df = frame.iloc[:40].reset_index(drop=True)
    val_df = frame.iloc[40:52].reset_index(drop=True)
    test_df = frame.iloc[52:].reset_index(drop=True)
    preprocessor = fit_preprocessor(frame)
    feature_schema = feature_schema_from_preprocessor(preprocessor)

    ablated_preprocessor, ablated_schema = run_tab_transformer_no_engineered(
        make_training_config(),
        preprocessor,
        feature_schema,
        train_df,
        val_df,
        test_df,
        tmp_path / "ablation",
    )

    assert "total_packets_bin" not in ablated_preprocessor.categorical_columns
    assert "total_packets" not in ablated_preprocessor.continuous_columns
    assert "total_packets_bin" not in ablated_schema["categorical_columns"]
    assert (tmp_path / "ablation" / "metrics.json").exists()
