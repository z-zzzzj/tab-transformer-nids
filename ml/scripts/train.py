from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.models.isolation_forest_baseline import build_isolation_forest
from ml.models.lstm_baseline import FeatureSequenceLSTM
from ml.models.mlp_baseline import DenseMLP, EmbeddingMLP
from ml.models.tab_transformer_model import build_tab_transformer, load_tab_transformer_checkpoint
from tab_transformer_nids.evaluation import (
    DEFAULT_SELECTION_BETA,
    build_attack_family_metrics,
    build_selection_metrics,
    choose_best_threshold,
    choose_threshold_with_plateau,
    classification_metrics,
    save_confusion_matrix,
    save_roc_curve,
)
from tab_transformer_nids.preprocessing import (
    ENGINEERED_CATEGORICAL_COLUMNS,
    ENGINEERED_CONTINUOUS_COLUMNS,
    TabularPreprocessor,
    load_dataframe,
)
from tab_transformer_nids.settings import ensure_dir, load_config


@dataclass
class TrainingResult:
    best_epoch: int
    best_f1: float
    epochs_completed: int


@dataclass(frozen=True)
class ArrayTriplet:
    x_categ: np.ndarray
    x_cont: np.ndarray
    y: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.y.shape[0])

    @property
    def total_bytes(self) -> int:
        return int(self.x_categ.nbytes + self.x_cont.nbytes + self.y.nbytes)


class TabDataset(Dataset):
    def __init__(self, x_categ: np.ndarray | torch.Tensor, x_cont: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, device: torch.device | None = None):
        self.x_categ = torch.as_tensor(x_categ, dtype=torch.long, device=device)
        self.x_cont = torch.as_tensor(x_cont, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_categ[index], self.x_cont[index], self.y[index]


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, y: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.y[index]


def flatten_tabular_arrays(x_categ: np.ndarray, x_cont: np.ndarray) -> np.ndarray:
    return np.concatenate([x_categ.astype(np.float32), x_cont.astype(np.float32)], axis=1)


def read_processed_assets(processed_dir: Path) -> tuple[TabularPreprocessor, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessor = TabularPreprocessor.load(processed_dir / "preprocessor.joblib")
    feature_schema = json.loads((processed_dir / "feature_schema.json").read_text(encoding="utf-8"))
    train_df = load_dataframe(processed_dir / "train.pkl.gz")
    val_df = load_dataframe(processed_dir / "val.pkl.gz")
    test_df = load_dataframe(processed_dir / "test.pkl.gz")
    return preprocessor, feature_schema, train_df, val_df, test_df


def prepare_arrays(preprocessor: TabularPreprocessor, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_categ, x_cont = preprocessor.transform(df)
    y = df["target_binary"].astype(np.float32).to_numpy()
    return x_categ, x_cont, y


def clone_preprocessor_with_columns(
    preprocessor: TabularPreprocessor,
    categorical_columns: Sequence[str],
    continuous_columns: Sequence[str],
) -> TabularPreprocessor:
    payload = preprocessor.to_dict()
    categorical_index = {
        column: index for index, column in enumerate(payload["categorical_columns"])
    }
    continuous_index = {
        column: index for index, column in enumerate(payload["continuous_columns"])
    }
    selected_categorical = [column for column in categorical_columns if column in categorical_index]
    selected_continuous = [column for column in continuous_columns if column in continuous_index]
    return TabularPreprocessor.from_dict(
        {
            "categorical_columns": selected_categorical,
            "continuous_columns": selected_continuous,
            "category_maps": {
                column: payload["category_maps"][column] for column in selected_categorical
            },
            "categorical_cardinalities": [
                payload["categorical_cardinalities"][categorical_index[column]]
                for column in selected_categorical
            ],
            "numeric_fill_values": {
                column: payload["numeric_fill_values"][column] for column in selected_continuous
            },
            "scaler_mean": [
                payload["scaler_mean"][continuous_index[column]]
                for column in selected_continuous
            ],
            "scaler_var": [
                payload["scaler_var"][continuous_index[column]]
                for column in selected_continuous
            ],
            "scaler_scale": [
                payload["scaler_scale"][continuous_index[column]]
                for column in selected_continuous
            ],
        }
    )


def subset_feature_schema(
    feature_schema: dict[str, Any],
    categorical_columns: Sequence[str],
    continuous_columns: Sequence[str],
) -> dict[str, Any]:
    source_categorical = list(feature_schema["categorical_columns"])
    categorical_index = {column: index for index, column in enumerate(source_categorical)}
    selected_categorical = [column for column in categorical_columns if column in categorical_index]
    selected_continuous = [
        column for column in continuous_columns if column in feature_schema["continuous_columns"]
    ]
    output = dict(feature_schema)
    output["categorical_columns"] = selected_categorical
    output["continuous_columns"] = selected_continuous
    output["categorical_cardinalities"] = [
        feature_schema["categorical_cardinalities"][categorical_index[column]]
        for column in selected_categorical
    ]
    if "category_maps" in output:
        output["category_maps"] = {
            column: feature_schema["category_maps"][column]
            for column in selected_categorical
            if column in feature_schema["category_maps"]
        }
    output["ablation"] = "no_engineered_behavior_features"
    return output


def subset_triplet_columns(
    preprocessor: TabularPreprocessor,
    triplet: ArrayTriplet,
    categorical_columns: Sequence[str],
    continuous_columns: Sequence[str],
) -> ArrayTriplet:
    categorical_index = {
        column: index for index, column in enumerate(preprocessor.categorical_columns)
    }
    continuous_index = {
        column: index for index, column in enumerate(preprocessor.continuous_columns)
    }
    categorical_indices = [
        categorical_index[column] for column in categorical_columns if column in categorical_index
    ]
    continuous_indices = [
        continuous_index[column] for column in continuous_columns if column in continuous_index
    ]
    return ArrayTriplet(
        x_categ=triplet.x_categ[:, categorical_indices].copy(),
        x_cont=triplet.x_cont[:, continuous_indices].copy(),
        y=triplet.y,
    )


def arrays_to_triplet(
    arrays: ArrayTriplet | tuple[np.ndarray, np.ndarray, np.ndarray]
) -> ArrayTriplet:
    if isinstance(arrays, ArrayTriplet):
        return arrays
    return ArrayTriplet(x_categ=arrays[0], x_cont=arrays[1], y=arrays[2])


def tensor_from_array(array: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=dtype)
    if device.type == "cuda":
        return tensor.to(device, non_blocking=True)
    return tensor


def get_available_physical_memory_bytes() -> int | None:
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None

    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return int(available_pages * page_size)


def _filesystem_path_str(path: str | Path) -> str:
    raw_path = os.path.abspath(os.fspath(path))
    if os.name != "nt":
        return raw_path
    if raw_path.startswith("\\\\?\\") or raw_path.startswith("\\\\.\\"):
        return raw_path
    if raw_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw_path[2:]
    return "\\\\?\\" + raw_path


def _ensure_parent_dir(path: str | Path) -> None:
    parent = Path(path).parent
    os.makedirs(_filesystem_path_str(parent), exist_ok=True)


def _write_text_file(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    _ensure_parent_dir(path)
    with open(_filesystem_path_str(path), "w", encoding=encoding) as handle:
        handle.write(text)


def _read_text_file(path: str | Path, *, encoding: str = "utf-8") -> str:
    with open(_filesystem_path_str(path), "r", encoding=encoding) as handle:
        return handle.read()


def resolve_epoch_selection_metric(config: dict[str, Any]) -> str:
    metric = str(config.get("training", {}).get("epoch_selection_metric", "f_beta")).strip().lower()
    if metric not in {"f1", "f_beta"}:
        raise ValueError(f"Unsupported epoch_selection_metric: {metric}")
    return metric


def epoch_selection_metric_label(metric: str, *, beta: float = DEFAULT_SELECTION_BETA) -> str:
    return "val_f1" if metric == "f1" else f"val_f_beta_beta_{beta}"


def epoch_selection_score(selection: dict[str, Any], metric: str) -> float:
    if metric == "f1":
        return float(selection["f1"])
    return float(selection["f_beta"])


def is_better_epoch_selection(
    current: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    metric: str,
    atol: float = 1e-12,
) -> bool:
    if best is None:
        return True

    current_values = (
        epoch_selection_score(current, metric),
        float(current["recall"]),
        float(current["f_beta"]),
        -float(current["threshold"]),
    )
    best_values = (
        epoch_selection_score(best, metric),
        float(best["recall"]),
        float(best["f_beta"]),
        -float(best["threshold"]),
    )
    for current_value, best_value in zip(current_values, best_values):
        if current_value > best_value + atol:
            return True
        if current_value < best_value - atol:
            return False
    return False


def resolve_auto_batch_tune_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    auto_batch_tune = training.get("auto_batch_tune") or {}
    return {
        "enabled": bool(auto_batch_tune.get("enabled", False)),
        "gpu_reserve_mb": int(auto_batch_tune.get("gpu_reserve_mb", 1024)),
        "ram_reserve_mb": int(auto_batch_tune.get("ram_reserve_mb", 4096)),
        "max_train_batch_multiplier": max(1, int(auto_batch_tune.get("max_train_batch_multiplier", 4))),
        "max_eval_batch_multiplier": max(1, int(auto_batch_tune.get("max_eval_batch_multiplier", 4))),
    }


def batch_size_candidates(base_batch_size: int, max_multiplier: int, max_rows: int) -> list[int]:
    base = max(1, int(base_batch_size))
    max_allowed = max(base, min(int(max_rows), base * max(1, int(max_multiplier))))
    candidates = {base}
    multiplier = 2
    while base * multiplier <= max_allowed:
        candidates.add(base * multiplier)
        multiplier *= 2
    if max_allowed < base:
        candidates.add(max_allowed)
    return sorted(candidate for candidate in candidates if candidate > 0)


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _cleanup_cuda_after_probe() -> None:
    if torch.cuda.is_available():
        gc.collect()
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        gc.collect()


def resolve_max_batches_per_epoch(config: dict[str, Any]) -> int | None:
    raw_value = config.get("training", {}).get("max_batches_per_epoch")
    if raw_value in (None, "", 0, "0"):
        return None
    return max(1, int(raw_value))


def _probe_single_batch_size(
    *,
    label: str,
    base_batch_size: int,
    max_multiplier: int,
    max_rows: int,
    reserve_bytes: int,
    probe_fn: Callable[[int], dict[str, Any] | None],
    minimum_batch_size: int = 1,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    max_rows = max(1, int(max_rows))
    minimum_batch_size = max(1, min(int(minimum_batch_size), max_rows))
    configured_batch_size = min(max(int(base_batch_size), minimum_batch_size), max_rows)
    safe_batch_size = configured_batch_size

    def run_probe(
        candidate: int,
        *,
        direction: str,
        allow_reserve_override: bool,
        allow_pre_probe_reserve_block: bool,
    ) -> tuple[dict[str, Any], bool, bool]:
        _cleanup_cuda_after_probe()
        free_bytes_before, total_bytes = torch.cuda.mem_get_info()
        attempt = {
            "batch_size": int(candidate),
            "direction": direction,
            "free_cuda_bytes_before": int(free_bytes_before),
            "total_cuda_bytes": int(total_bytes),
            "reserve_bytes": int(reserve_bytes),
            "reserve_headroom_before_probe": bool(free_bytes_before > reserve_bytes),
        }
        if free_bytes_before <= reserve_bytes and not allow_pre_probe_reserve_block:
            attempt["status"] = "reserve_blocked_before_probe"
            return attempt, False, False
        try:
            probe_result = probe_fn(int(candidate)) or {}
            free_bytes_after_probe = probe_result.get("free_cuda_bytes_after_probe")
            if free_bytes_after_probe is not None:
                estimated_free_after_peak = int(max(0, int(free_bytes_after_probe)))
                reserve_headroom_after_probe = estimated_free_after_peak >= reserve_bytes
                attempt["free_cuda_bytes_after_probe"] = estimated_free_after_peak
                if "extra_peak_bytes" in probe_result:
                    attempt["extra_peak_bytes"] = int(probe_result["extra_peak_bytes"])
            else:
                extra_peak_bytes = int(probe_result.get("extra_peak_bytes", 0))
                estimated_free_after_peak = int(max(0, free_bytes_before - extra_peak_bytes))
                reserve_headroom_after_probe = estimated_free_after_peak >= reserve_bytes
                attempt["extra_peak_bytes"] = extra_peak_bytes
            attempt["estimated_free_cuda_bytes_after_peak"] = estimated_free_after_peak
            attempt["reserve_headroom_after_probe"] = reserve_headroom_after_probe
            if reserve_headroom_after_probe:
                attempt["status"] = "accepted"
                return attempt, True, False
            if allow_reserve_override:
                attempt["status"] = "accepted_without_reserve_headroom"
                return attempt, True, False
            attempt["status"] = "reserve_blocked_after_probe"
            return attempt, False, False
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            attempt["status"] = "oom"
            attempt["error"] = str(exc)
            _cleanup_cuda_after_probe()
            return attempt, False, True

    upward_candidates = batch_size_candidates(
        configured_batch_size,
        max_multiplier=max_multiplier,
        max_rows=max_rows,
    )

    baseline_attempt, baseline_non_oom, baseline_oom = run_probe(
        configured_batch_size,
        direction="baseline",
        allow_reserve_override=True,
        allow_pre_probe_reserve_block=True,
    )
    attempts.append(baseline_attempt)

    if baseline_oom:
        safe_batch_size = minimum_batch_size
        last_non_oom_batch_without_reserve: int | None = None
        current = configured_batch_size
        seen_candidates: set[int] = {configured_batch_size}
        while current > minimum_batch_size:
            current = max(minimum_batch_size, current // 2)
            if current in seen_candidates:
                if current == minimum_batch_size:
                    break
                continue
            seen_candidates.add(current)
            attempt, non_oom, is_oom = run_probe(
                current,
                direction="fallback",
                allow_reserve_override=False,
                allow_pre_probe_reserve_block=True,
            )
            attempts.append(attempt)
            if is_oom:
                if current == minimum_batch_size:
                    break
                continue
            if non_oom:
                safe_batch_size = int(current)
                break
            last_non_oom_batch_without_reserve = int(current)
            if current == minimum_batch_size:
                break
        if (
            safe_batch_size == minimum_batch_size
            and last_non_oom_batch_without_reserve is not None
        ):
            safe_batch_size = last_non_oom_batch_without_reserve
    elif baseline_non_oom and baseline_attempt["status"] == "accepted":
        for candidate in upward_candidates:
            if candidate <= safe_batch_size:
                continue
            attempt, non_oom, is_oom = run_probe(
                candidate,
                direction="upward",
                allow_reserve_override=False,
                allow_pre_probe_reserve_block=False,
            )
            attempts.append(attempt)
            if not non_oom or is_oom:
                break
            safe_batch_size = int(candidate)
    else:
        attempts.append(
            {
                "batch_size": int(configured_batch_size),
                "direction": "upward",
                "status": "skipped_due_to_baseline_reserve_headroom",
            }
        )

    if safe_batch_size == configured_batch_size and baseline_attempt["status"] == "accepted":
        resolved_reason = "no_change"
    elif safe_batch_size > configured_batch_size:
        resolved_reason = "scaled_up"
    elif safe_batch_size < configured_batch_size:
        resolved_reason = "fallback_after_oom"
    else:
        resolved_reason = "kept_configured_batch_without_reserve_headroom"

    return {
        "label": label,
        "configured_batch_size": int(configured_batch_size),
        "resolved_batch_size": int(safe_batch_size),
        "minimum_batch_size": int(minimum_batch_size),
        "resolved_reason": resolved_reason,
        "attempts": attempts,
    }


def auto_tune_batch_sizes(
    config: dict[str, Any],
    *,
    device: torch.device,
    train_rows: int,
    eval_rows: int,
    train_probe: Callable[[int], dict[str, Any] | None],
    eval_probe: Callable[[int], dict[str, Any] | None],
    minimum_train_batch_size: int = 1,
    minimum_eval_batch_size: int = 1,
) -> dict[str, Any]:
    base_batch_size = max(1, int(config["training"]["batch_size"]))
    base_eval_batch_size = max(1, int(config["training"].get("eval_batch_size") or base_batch_size))
    auto_batch_tune = resolve_auto_batch_tune_config(config)
    report: dict[str, Any] = {
        "enabled": bool(auto_batch_tune["enabled"]),
        "applied": False,
        "reason": None,
        "configured_batch_size": int(base_batch_size),
        "configured_eval_batch_size": int(base_eval_batch_size),
        "resolved_batch_size": int(base_batch_size),
        "resolved_eval_batch_size": int(base_eval_batch_size),
        "gpu_reserve_mb": int(auto_batch_tune["gpu_reserve_mb"]),
        "ram_reserve_mb": int(auto_batch_tune["ram_reserve_mb"]),
        "max_train_batch_multiplier": int(auto_batch_tune["max_train_batch_multiplier"]),
        "max_eval_batch_multiplier": int(auto_batch_tune["max_eval_batch_multiplier"]),
        "train_probe": None,
        "eval_probe": None,
    }
    if not auto_batch_tune["enabled"]:
        report["reason"] = "disabled"
        return report
    if device.type != "cuda":
        report["reason"] = "cuda_unavailable"
        return report

    available_ram_bytes = get_available_physical_memory_bytes()
    report["available_ram_bytes"] = (
        int(available_ram_bytes) if available_ram_bytes is not None else None
    )
    ram_reserve_bytes = int(auto_batch_tune["ram_reserve_mb"]) * 1024 * 1024
    if available_ram_bytes is not None and available_ram_bytes <= ram_reserve_bytes:
        report["reason"] = "ram_reserve_not_met"
        return report

    reserve_bytes = int(auto_batch_tune["gpu_reserve_mb"]) * 1024 * 1024
    train_probe_report = _probe_single_batch_size(
        label="train",
        base_batch_size=base_batch_size,
        max_multiplier=int(auto_batch_tune["max_train_batch_multiplier"]),
        max_rows=int(train_rows),
        reserve_bytes=reserve_bytes,
        probe_fn=train_probe,
        minimum_batch_size=minimum_train_batch_size,
    )
    eval_probe_report = _probe_single_batch_size(
        label="eval",
        base_batch_size=base_eval_batch_size,
        max_multiplier=int(auto_batch_tune["max_eval_batch_multiplier"]),
        max_rows=int(eval_rows),
        reserve_bytes=reserve_bytes,
        probe_fn=eval_probe,
        minimum_batch_size=minimum_eval_batch_size,
    )
    report["train_probe"] = train_probe_report
    report["eval_probe"] = eval_probe_report
    report["resolved_batch_size"] = int(train_probe_report["resolved_batch_size"])
    report["resolved_eval_batch_size"] = int(eval_probe_report["resolved_batch_size"])
    report["applied"] = (
        report["resolved_batch_size"] != report["configured_batch_size"]
        or report["resolved_eval_batch_size"] != report["configured_eval_batch_size"]
    )
    report["reason"] = "applied" if report["applied"] else "no_change"
    return report


def resolve_positive_class_weight(config: dict[str, Any], positives: float, negatives: float) -> float:
    override = config["training"].get("positive_class_weight")
    ratio = max(1.0, negatives / max(positives, 1.0))
    if override is None:
        return ratio
    if isinstance(override, str):
        mode = override.strip().lower()
        if mode == "auto":
            return ratio
        if mode == "sqrt_auto":
            return max(1.0, math.sqrt(ratio))
        raise ValueError(f"Unsupported positive_class_weight mode: {override}")
    return max(0.0, float(override))


def resolve_selection_split_name(config: dict[str, Any]) -> str:
    return str(config.get("training", {}).get("selection_split_name", "inner_val")).strip() or "inner_val"


def resolve_threshold_selection_policy(config: dict[str, Any]) -> str:
    policy = str(config.get("training", {}).get("threshold_selection_policy", "f1_plateau")).strip().lower()
    if policy != "f1_plateau":
        raise ValueError(f"Unsupported threshold_selection_policy: {policy}")
    return policy


def score_quantiles(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {}
    quantiles = np.quantile(scores, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    labels = ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    return {label: float(value) for label, value in zip(labels, quantiles)}


def split_score_diagnostics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    preds = (scores >= threshold).astype(np.int64)
    output: dict[str, Any] = {
        "rows": int(len(scores)),
        "threshold": float(threshold),
        "predicted_positive_rate": float(preds.mean()) if preds.size else 0.0,
        "score_quantiles": score_quantiles(scores),
        "label_groups": {},
    }
    for label_value, label_name in ((0, "benign"), (1, "attack")):
        mask = y_true == label_value
        label_scores = scores[mask]
        label_preds = preds[mask]
        output["label_groups"][label_name] = {
            "rows": int(mask.sum()),
            "predicted_positive_rate": float(label_preds.mean()) if label_preds.size else 0.0,
            "score_quantiles": score_quantiles(label_scores),
        }
    return output


def build_score_diagnostics_payload(
    *,
    val_y: np.ndarray,
    val_scores: np.ndarray,
    test_y: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    pre_val_scores: np.ndarray | None = None,
    pre_test_scores: np.ndarray | None = None,
    pre_threshold: float | None = None,
    calibration_mode: str = "none",
    temperature: float = 1.0,
) -> dict[str, Any]:
    pre_val_scores = val_scores if pre_val_scores is None else pre_val_scores
    pre_test_scores = test_scores if pre_test_scores is None else pre_test_scores
    pre_threshold = float(threshold if pre_threshold is None else pre_threshold)
    post_val = split_score_diagnostics(val_y, val_scores, threshold)
    post_test = split_score_diagnostics(test_y, test_scores, threshold)
    pre_calibration = {
        "threshold": pre_threshold,
        "val": split_score_diagnostics(val_y, pre_val_scores, pre_threshold),
        "test": split_score_diagnostics(test_y, pre_test_scores, pre_threshold),
    }
    post_calibration = {
        "threshold": float(threshold),
        "val": post_val,
        "test": post_test,
    }
    return {
        "threshold": float(threshold),
        "calibration_mode": calibration_mode,
        "temperature": float(temperature),
        "val": post_val,
        "test": post_test,
        "pre_calibration": pre_calibration,
        "post_calibration": post_calibration,
    }


def apply_temperature_to_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-3)
    return logits / safe_temperature


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: np.ndarray,
    *,
    fit_split: str = "inner_val",
) -> dict[str, float]:
    if logits.numel() == 0:
        return {
            "mode": "temperature_scaling",
            "fit_split": str(fit_split),
            "temperature": 1.0,
            "pre_calibration_nll": 0.0,
            "post_calibration_nll": 0.0,
        }

    logits_cpu = logits.detach().cpu().to(torch.float32)
    targets_tensor = torch.as_tensor(targets, dtype=torch.float32)
    pre_nll = float(
        nn.functional.binary_cross_entropy_with_logits(logits_cpu, targets_tensor).item()
    )
    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.clamp(torch.exp(log_temperature), min=0.05, max=10.0)
        scaled_logits = logits_cpu / temperature
        loss = nn.functional.binary_cross_entropy_with_logits(scaled_logits, targets_tensor)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        learned_temperature = float(
            torch.clamp(torch.exp(log_temperature.detach()), min=0.05, max=10.0).item()
        )
    except RuntimeError:
        learned_temperature = 1.0

    scaled_logits = logits_cpu / max(learned_temperature, 1e-3)
    post_nll = float(
        nn.functional.binary_cross_entropy_with_logits(scaled_logits, targets_tensor).item()
    )
    if not math.isfinite(post_nll) or post_nll > pre_nll + 1e-8:
        learned_temperature = 1.0
        post_nll = pre_nll
    return {
        "mode": "temperature_scaling",
        "fit_split": str(fit_split),
        "temperature": float(learned_temperature),
        "pre_calibration_nll": pre_nll,
        "post_calibration_nll": post_nll,
    }


def _write_json(path: Path, payload: Any) -> None:
    _write_text_file(path, json.dumps(payload, indent=2), encoding="utf-8")


def run_tab_transformer(
    config: dict[str, Any],
    preprocessor: TabularPreprocessor,
    feature_schema: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment_dir: Path,
    *,
    train_arrays: ArrayTriplet | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    val_arrays: ArrayTriplet | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    test_arrays: ArrayTriplet | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    focus_attack_families: list[str] | tuple[str, ...] | None = None,
    model_type: str = "tab_transformer",
) -> TrainingResult:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_triplet = arrays_to_triplet(train_arrays or prepare_arrays(preprocessor, train_df))
    val_triplet = arrays_to_triplet(val_arrays or prepare_arrays(preprocessor, val_df))
    test_triplet = arrays_to_triplet(test_arrays or prepare_arrays(preprocessor, test_df))
    train_xc, train_xn, train_y = train_triplet.x_categ, train_triplet.x_cont, train_triplet.y
    val_xc, val_xn, val_y = val_triplet.x_categ, val_triplet.x_cont, val_triplet.y
    test_xc, test_xn, test_y = test_triplet.x_categ, test_triplet.x_cont, test_triplet.y

    configured_batch_size = max(1, int(config["training"]["batch_size"]))
    configured_eval_batch_size = max(
        1,
        int(config["training"].get("eval_batch_size") or configured_batch_size),
    )
    selection_split_name = resolve_selection_split_name(config)
    threshold_selection_policy = resolve_threshold_selection_policy(config)

    epoch_metric = resolve_epoch_selection_metric(config)
    epoch_metric_artifact_name = epoch_selection_metric_label(epoch_metric)

    train_xc_tensor = tensor_from_array(train_xc, torch.long, device)
    train_xn_tensor = tensor_from_array(train_xn, torch.float32, device)
    train_y_tensor = tensor_from_array(train_y, torch.float32, device)
    val_xc_tensor = tensor_from_array(val_xc, torch.long, device)
    val_xn_tensor = tensor_from_array(val_xn, torch.float32, device)
    test_xc_tensor = tensor_from_array(test_xc, torch.long, device)
    test_xn_tensor = tensor_from_array(test_xn, torch.float32, device)

    continuous_mean_std = preprocessor.continuous_mean_std_tensor()
    model = build_tab_transformer(
        categorical_cardinalities=feature_schema["categorical_cardinalities"],
        num_continuous=len(preprocessor.continuous_columns),
        model_config=config["tab_transformer"],
        continuous_mean_std=continuous_mean_std,
    ).to(device)

    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    pos_weight_value = resolve_positive_class_weight(config, positives, negatives)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"[TAB] positive_class_weight={pos_weight_value:.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    scaler = GradScaler(device.type, enabled=use_cuda)

    train_size = int(train_xc_tensor.size(0))

    def infer_logits(
        xc_tensor: torch.Tensor,
        xn_tensor: torch.Tensor,
        *,
        active_eval_batch_size: int,
    ) -> torch.Tensor:
        if xc_tensor.size(0) == 0:
            return torch.empty(0, device="cpu")
        nonlocal eval_batch_size
        effective_batch_size = max(1, int(active_eval_batch_size))
        while True:
            logits_chunks: list[torch.Tensor] = []
            try:
                with torch.no_grad():
                    for start_index in range(0, xc_tensor.size(0), effective_batch_size):
                        end_index = min(start_index + effective_batch_size, xc_tensor.size(0))
                        with autocast(device_type=device.type, enabled=use_cuda):
                            logits_chunk = model(
                                xc_tensor[start_index:end_index],
                                xn_tensor[start_index:end_index],
                            ).squeeze(-1)
                        logits_chunks.append(logits_chunk.detach().cpu())
                if effective_batch_size != int(active_eval_batch_size):
                    fallback_record = {
                        "requested_eval_batch_size": int(active_eval_batch_size),
                        "resolved_eval_batch_size": int(effective_batch_size),
                        "rows": int(xc_tensor.size(0)),
                        "reason": "runtime_oom_fallback",
                    }
                    runtime_eval_batch_fallbacks.append(fallback_record)
                    eval_batch_size = min(int(eval_batch_size), int(effective_batch_size))
                    auto_batch_tune_report.setdefault("runtime_eval_batch_fallbacks", []).append(
                        fallback_record
                    )
                    auto_batch_tune_report["resolved_eval_batch_size"] = int(eval_batch_size)
                    print(
                        "[TAB] runtime_eval_batch_fallback "
                        f"requested={int(active_eval_batch_size)} resolved={int(effective_batch_size)} "
                        f"rows={int(xc_tensor.size(0))}"
                    )
                if not logits_chunks:
                    return torch.empty(0, device="cpu")
                return torch.cat(logits_chunks)
            except RuntimeError as exc:
                if not use_cuda or not _is_cuda_oom(exc) or effective_batch_size <= 1:
                    raise
                next_batch_size = max(1, effective_batch_size // 2)
                if next_batch_size == effective_batch_size:
                    raise
                logits_chunks.clear()
                _cleanup_cuda_after_probe()
                effective_batch_size = next_batch_size

    def choose_threshold_for_selection(
        y_true_values: np.ndarray,
        score_values: np.ndarray,
    ) -> dict[str, Any]:
        return choose_threshold_with_plateau(
            y_true_values,
            score_values,
            beta=DEFAULT_SELECTION_BETA,
        )

    def probe_train_batch_size(candidate_batch_size: int) -> dict[str, Any]:
        if not use_cuda or train_size == 0:
            return {"extra_peak_bytes": 0}
        current_batch_size = min(max(1, int(candidate_batch_size)), train_size)
        indices = torch.arange(current_batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        batch_xc = train_xc_tensor[indices]
        batch_xn = train_xn_tensor[indices]
        batch_labels = train_y_tensor[indices]
        with autocast(device_type=device.type, enabled=use_cuda):
            logits = model(batch_xc, batch_xn).squeeze(-1)
            loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        extra_peak_bytes = max(
            0,
            int(torch.cuda.max_memory_allocated(device) - baseline_allocated),
        )
        del batch_xc, batch_xn, batch_labels, logits, loss_values, loss
        _cleanup_cuda_after_probe()
        free_cuda_bytes_after_probe, _ = torch.cuda.mem_get_info()
        return {
            "extra_peak_bytes": extra_peak_bytes,
            "free_cuda_bytes_after_probe": int(free_cuda_bytes_after_probe),
        }

    eval_probe_xc_tensor = val_xc_tensor if val_xc_tensor.size(0) >= test_xc_tensor.size(0) else test_xc_tensor
    eval_probe_xn_tensor = val_xn_tensor if val_xc_tensor.size(0) >= test_xc_tensor.size(0) else test_xn_tensor

    def probe_eval_batch_size(candidate_batch_size: int) -> dict[str, Any]:
        if not use_cuda or eval_probe_xc_tensor.size(0) == 0:
            return {"extra_peak_bytes": 0}
        current_batch_size = min(max(1, int(candidate_batch_size)), int(eval_probe_xc_tensor.size(0)))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        model.eval()
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_cuda):
                _ = model(
                    eval_probe_xc_tensor[:current_batch_size],
                    eval_probe_xn_tensor[:current_batch_size],
                ).squeeze(-1)
        torch.cuda.synchronize(device)
        free_cuda_bytes_after_probe, _ = torch.cuda.mem_get_info()
        extra_peak_bytes = max(
            0,
            int(torch.cuda.max_memory_allocated(device) - baseline_allocated),
        )
        _cleanup_cuda_after_probe()
        free_cuda_bytes_after_probe, _ = torch.cuda.mem_get_info()
        return {
            "extra_peak_bytes": extra_peak_bytes,
            "free_cuda_bytes_after_probe": int(free_cuda_bytes_after_probe),
        }

    auto_batch_tune_report = auto_tune_batch_sizes(
        config,
        device=device,
        train_rows=train_size,
        eval_rows=max(int(val_xc_tensor.size(0)), int(test_xc_tensor.size(0))),
        train_probe=probe_train_batch_size,
        eval_probe=probe_eval_batch_size,
        minimum_train_batch_size=1,
    )
    batch_size = int(auto_batch_tune_report["resolved_batch_size"])
    eval_batch_size = int(auto_batch_tune_report["resolved_eval_batch_size"])

    if use_cuda:
        staged_bytes = (
            train_xc_tensor.element_size() * train_xc_tensor.numel()
            + train_xn_tensor.element_size() * train_xn_tensor.numel()
            + train_y_tensor.element_size() * train_y_tensor.numel()
            + val_xc_tensor.element_size() * val_xc_tensor.numel()
            + val_xn_tensor.element_size() * val_xn_tensor.numel()
            + test_xc_tensor.element_size() * test_xc_tensor.numel()
            + test_xn_tensor.element_size() * test_xn_tensor.numel()
        )
        print(
            "[TAB] staged_tensors_on_gpu="
            f"{staged_bytes / (1024 ** 3):.2f} GiB "
            f"configured_batch_size={configured_batch_size} resolved_batch_size={batch_size} "
            f"configured_eval_batch_size={configured_eval_batch_size} resolved_eval_batch_size={eval_batch_size}"
        )

    configured_total_batches = max(1, math.ceil(train_size / batch_size))
    max_batches_per_epoch = resolve_max_batches_per_epoch(config)
    total_batches = configured_total_batches
    if max_batches_per_epoch is not None:
        total_batches = min(total_batches, max_batches_per_epoch)
    best_state = None
    best_f1 = -1.0
    best_f_beta = -1.0
    best_epoch = 0
    best_threshold_selection: dict[str, Any] | None = None
    last_threshold_selection: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    runtime_eval_batch_fallbacks: list[dict[str, Any]] = []
    patience = int(config["training"]["patience"])
    remaining = patience
    epochs_completed = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if use_cuda and epoch > 1:
            _cleanup_cuda_after_probe()

        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        samples_seen = 0
        permutation = torch.randperm(train_size, device=device)
        for batch_index, start_index in enumerate(range(0, train_size, batch_size), start=1):
            if batch_index > total_batches:
                break
            end_index = min(start_index + batch_size, train_size)
            indices = permutation[start_index:end_index]
            batch_xc = train_xc_tensor.index_select(0, indices)
            batch_xn = train_xn_tensor.index_select(0, indices)
            batch_labels = train_y_tensor.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(batch_xc, batch_xn).squeeze(-1)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * batch_labels.size(0)
            samples_seen += int(batch_labels.size(0))

        model.eval()
        val_logits = infer_logits(val_xc_tensor, val_xn_tensor, active_eval_batch_size=eval_batch_size)
        model.train()
        val_scores_np = torch.sigmoid(val_logits).cpu().numpy()
        del val_logits
        if use_cuda:
            _cleanup_cuda_after_probe()
        threshold_selection = choose_threshold_for_selection(val_y, val_scores_np)
        last_threshold_selection = dict(threshold_selection)
        val_f1 = float(threshold_selection["f1"])
        val_f_beta = float(threshold_selection["f_beta"])
        val_recall = float(threshold_selection["recall"])
        selection_score = epoch_selection_score(threshold_selection, epoch_metric)
        scheduler.step(selection_score)
        train_loss = running_loss / max(samples_seen, 1)
        epoch_duration = time.perf_counter() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_threshold": float(threshold_selection["threshold"]),
                "val_best_f1": val_f1,
                "val_best_f_beta": val_f_beta,
                "val_recall": val_recall,
                "val_selection_score": selection_score,
                "epoch_selection_metric": epoch_metric_artifact_name,
                "lr": current_lr,
                "duration_seconds": epoch_duration,
            }
        )
        epochs_completed = epoch
        print(
            f"[TAB][epoch {epoch:02d}] loss={train_loss:.6f} "
            f"val_best_f1={val_f1:.6f} val_best_f_beta={val_f_beta:.6f} "
            f"val_recall={val_recall:.6f} selection_score={selection_score:.6f} "
            f"metric={epoch_metric_artifact_name} lr={current_lr:.7f} "
            f"duration={epoch_duration:.1f}s"
        )
        if is_better_epoch_selection(
            threshold_selection,
            best_threshold_selection,
            metric=epoch_metric,
        ):
            best_f1 = float(val_f1)
            best_f_beta = float(val_f_beta)
            best_epoch = epoch
            best_threshold_selection = dict(threshold_selection)
            remaining = patience
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            remaining -= 1

        if remaining <= 0:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if last_threshold_selection is not None:
            best_threshold_selection = dict(last_threshold_selection)
            best_f1 = float(last_threshold_selection["f1"])
            best_f_beta = float(last_threshold_selection["f_beta"])
            best_epoch = epochs_completed

    model.load_state_dict(best_state)
    model.eval()
    final_val_logits = infer_logits(val_xc_tensor, val_xn_tensor, active_eval_batch_size=eval_batch_size)
    if val_xc_tensor.data_ptr() == test_xc_tensor.data_ptr() and val_xn_tensor.data_ptr() == test_xn_tensor.data_ptr():
        test_logits = final_val_logits.clone()
    else:
        test_logits = infer_logits(test_xc_tensor, test_xn_tensor, active_eval_batch_size=eval_batch_size)
    pre_val_scores = torch.sigmoid(final_val_logits).cpu().numpy()
    pre_test_scores = torch.sigmoid(test_logits).cpu().numpy()
    calibration = fit_temperature_scaling(
        final_val_logits,
        val_y,
        fit_split=selection_split_name,
    )
    temperature = float(calibration["temperature"])
    val_scores = torch.sigmoid(apply_temperature_to_logits(final_val_logits, temperature)).cpu().numpy()
    test_scores = torch.sigmoid(apply_temperature_to_logits(test_logits, temperature)).cpu().numpy()
    pre_threshold_selection = choose_threshold_for_selection(
        val_y,
        pre_val_scores,
    )
    threshold_selection = choose_threshold_for_selection(
        val_y,
        val_scores,
    )
    threshold = float(threshold_selection["threshold"])
    selection_score_name = "val_f_beta" if epoch_metric == "f_beta" else "val_f1"
    selection_metrics = build_selection_metrics(
        val_y,
        val_scores,
        threshold,
        best_epoch=best_epoch,
        beta=DEFAULT_SELECTION_BETA,
        epoch_selection_metric=epoch_metric_artifact_name,
        threshold_selection_rule=str(threshold_selection["selection_rule"]),
        calibration_mode=str(calibration["mode"]),
        temperature=temperature,
    )
    selection_metrics["selection_score"] = float(selection_metrics[selection_score_name])
    selection_metrics["resolved_batch_size"] = batch_size
    selection_metrics["resolved_eval_batch_size"] = eval_batch_size
    selection_metrics["auto_batch_tune"] = auto_batch_tune_report
    selection_metrics["configured_total_batches"] = configured_total_batches
    selection_metrics["effective_total_batches"] = total_batches
    selection_metrics["max_batches_per_epoch"] = max_batches_per_epoch
    selection_metrics["selection_score_name"] = selection_score_name
    selection_metrics["selection_split_name"] = selection_split_name
    selection_metrics["threshold_selection_policy"] = threshold_selection_policy
    selection_metrics["best_val_f1_any_threshold"] = float(
        threshold_selection.get("best_f1", selection_metrics["val_f1"])
    )
    selection_metrics["best_val_f_beta_any_threshold"] = float(
        threshold_selection.get("best_f_beta", selection_metrics["val_f_beta"])
    )
    score_diagnostics = build_score_diagnostics_payload(
        val_y=val_y,
        val_scores=val_scores,
        test_y=test_y,
        test_scores=test_scores,
        threshold=threshold,
        pre_val_scores=pre_val_scores,
        pre_test_scores=pre_test_scores,
        pre_threshold=float(pre_threshold_selection["threshold"]),
        calibration_mode=str(calibration["mode"]),
        temperature=temperature,
    )
    metrics = classification_metrics(test_y, test_scores, threshold)
    metrics["best_epoch"] = best_epoch
    metrics["model_type"] = model_type

    torch.save(
        {
            "model_type": model_type,
            "state_dict": model.state_dict(),
            "model_config": config["tab_transformer"],
            "continuous_mean_std": preprocessor.continuous_mean_std_tensor().tolist(),
            "threshold": threshold,
            "temperature": temperature,
            "calibration_mode": calibration["mode"],
        },
        _filesystem_path_str(experiment_dir / "model.pt"),
    )
    _write_json(experiment_dir / "metrics.json", metrics)
    _write_json(experiment_dir / "selection_metrics.json", selection_metrics)
    _write_json(experiment_dir / "calibration.json", calibration)
    _write_json(experiment_dir / "score_diagnostics.json", score_diagnostics)

    benign_label = str(config.get("dataset", {}).get("benign_label", "BENIGN"))
    family_metrics = None
    if "label_original" in test_df.columns:
        family_metrics = build_attack_family_metrics(
            test_df["label_original"].astype(str).to_numpy(),
            test_y,
            test_scores,
            threshold,
            focus_families=focus_attack_families,
            benign_label=benign_label,
        )
        if family_metrics is not None:
            _write_json(experiment_dir / "family_metrics.json", family_metrics)

    _write_json(
        experiment_dir / "classification_report.json",
        metrics["classification_report"],
    )
    _write_json(
        experiment_dir / "threshold.json",
        {
            "threshold": threshold,
            "threshold_selection_rule": threshold_selection["selection_rule"],
            "threshold_selection_policy": threshold_selection_policy,
            "calibration_mode": calibration["mode"],
            "temperature": temperature,
            "epoch_selection_metric": epoch_metric_artifact_name,
            "selection_split_name": selection_split_name,
            "resolved_batch_size": batch_size,
            "resolved_eval_batch_size": eval_batch_size,
        },
    )
    _write_json(experiment_dir / "training_history.json", history)
    save_roc_curve(test_y, test_scores, experiment_dir / "roc_curve.png")
    save_confusion_matrix(test_y, test_scores, threshold, experiment_dir / "confusion_matrix.png")
    if use_cuda:
        _cleanup_cuda_after_probe()
    return TrainingResult(
        best_epoch=best_epoch,
        best_f1=best_f1,
        epochs_completed=epochs_completed,
    )


def _save_supervised_model_outputs(
    *,
    experiment_dir: Path,
    model_type: str,
    best_epoch: int,
    val_y: np.ndarray,
    val_scores: np.ndarray,
    test_y: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    selection_metrics = build_selection_metrics(
        val_y,
        val_scores,
        threshold,
        best_epoch=best_epoch,
    )
    score_diagnostics = build_score_diagnostics_payload(
        val_y=val_y,
        val_scores=val_scores,
        test_y=test_y,
        test_scores=test_scores,
        threshold=threshold,
    )
    metrics = classification_metrics(test_y, test_scores, threshold)
    metrics["model_type"] = model_type
    metrics["best_epoch"] = int(best_epoch)
    _write_json(experiment_dir / "metrics.json", metrics)
    _write_json(experiment_dir / "selection_metrics.json", selection_metrics)
    _write_json(experiment_dir / "calibration.json", {"mode": "none", "temperature": 1.0})
    _write_json(experiment_dir / "score_diagnostics.json", score_diagnostics)
    _write_json(experiment_dir / "classification_report.json", metrics["classification_report"])
    _write_json(experiment_dir / "threshold.json", {"threshold": float(threshold)})
    _write_json(experiment_dir / "training_history.json", history)
    save_roc_curve(test_y, test_scores, experiment_dir / "roc_curve.png")
    save_confusion_matrix(test_y, test_scores, threshold, experiment_dir / "confusion_matrix.png")
    return metrics


def _train_dense_torch_baseline(
    *,
    config: dict[str, Any],
    experiment_dir: Path,
    model_type: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_rows: int,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    model_payload: dict[str, Any],
) -> None:
    model = model.to(device)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    patience = int(config["training"]["patience"])
    remaining = patience
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).squeeze(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_labels)

        model.eval()
        val_batches = []
        with torch.no_grad():
            for batch_features, _ in val_loader:
                logits = model(batch_features.to(device)).squeeze(-1)
                val_batches.append(torch.sigmoid(logits).cpu().numpy())
        val_scores = np.concatenate(val_batches)
        _, val_f1 = choose_best_threshold(val_y, val_scores)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(train_rows, 1),
                "val_best_f1": float(val_f1),
            }
        )
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_batches = []
    test_batches = []
    with torch.no_grad():
        for batch_features, _ in val_loader:
            val_batches.append(torch.sigmoid(model(batch_features.to(device)).squeeze(-1)).cpu().numpy())
        for batch_features, _ in test_loader:
            test_batches.append(torch.sigmoid(model(batch_features.to(device)).squeeze(-1)).cpu().numpy())
    val_scores = np.concatenate(val_batches)
    test_scores = np.concatenate(test_batches)
    threshold, _ = choose_best_threshold(val_y, val_scores)
    torch.save(
        {
            "model_type": model_type,
            "state_dict": model.state_dict(),
            "model_config": model_payload,
        },
        _filesystem_path_str(experiment_dir / "model.pt"),
    )
    _save_supervised_model_outputs(
        experiment_dir=experiment_dir,
        model_type=model_type,
        best_epoch=best_epoch,
        val_y=val_y,
        val_scores=val_scores,
        test_y=test_y,
        test_scores=test_scores,
        threshold=threshold,
        history=history,
    )


def _train_embedding_torch_baseline(
    *,
    config: dict[str, Any],
    experiment_dir: Path,
    model_type: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_rows: int,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    model_payload: dict[str, Any],
) -> None:
    model = model.to(device)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    patience = int(config["training"]["patience"])
    remaining = patience
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        for batch_xc, batch_xn, batch_labels in train_loader:
            batch_xc = batch_xc.to(device)
            batch_xn = batch_xn.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_xc, batch_xn).squeeze(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_labels)

        model.eval()
        val_batches = []
        with torch.no_grad():
            for batch_xc, batch_xn, _ in val_loader:
                logits = model(batch_xc.to(device), batch_xn.to(device)).squeeze(-1)
                val_batches.append(torch.sigmoid(logits).cpu().numpy())
        val_scores = np.concatenate(val_batches)
        _, val_f1 = choose_best_threshold(val_y, val_scores)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(train_rows, 1),
                "val_best_f1": float(val_f1),
            }
        )
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_batches = []
    test_batches = []
    with torch.no_grad():
        for batch_xc, batch_xn, _ in val_loader:
            val_batches.append(
                torch.sigmoid(model(batch_xc.to(device), batch_xn.to(device)).squeeze(-1)).cpu().numpy()
            )
        for batch_xc, batch_xn, _ in test_loader:
            test_batches.append(
                torch.sigmoid(model(batch_xc.to(device), batch_xn.to(device)).squeeze(-1)).cpu().numpy()
            )
    val_scores = np.concatenate(val_batches)
    test_scores = np.concatenate(test_batches)
    threshold, _ = choose_best_threshold(val_y, val_scores)
    torch.save(
        {
            "model_type": model_type,
            "state_dict": model.state_dict(),
            "model_config": model_payload,
        },
        _filesystem_path_str(experiment_dir / "model.pt"),
    )
    _save_supervised_model_outputs(
        experiment_dir=experiment_dir,
        model_type=model_type,
        best_epoch=best_epoch,
        val_y=val_y,
        val_scores=val_scores,
        test_y=test_y,
        test_scores=test_scores,
        threshold=threshold,
        history=history,
    )


def run_mlp(
    config: dict[str, Any],
    preprocessor: TabularPreprocessor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment_dir: Path,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_xc, train_xn, train_y = prepare_arrays(preprocessor, train_df)
    val_xc, val_xn, val_y = prepare_arrays(preprocessor, val_df)
    test_xc, test_xn, test_y = prepare_arrays(preprocessor, test_df)
    train_features = flatten_tabular_arrays(train_xc, train_xn)
    val_features = flatten_tabular_arrays(val_xc, val_xn)
    test_features = flatten_tabular_arrays(test_xc, test_xn)
    batch_size = int(config["training"]["batch_size"])
    eval_batch_size = int(config["training"].get("eval_batch_size") or batch_size)
    model_cfg = {
        "input_features": int(train_features.shape[1]),
        "hidden_sizes": list(config.get("mlp", {}).get("hidden_sizes", [256, 128, 64])),
        "dropout": float(config.get("mlp", {}).get("dropout", 0.2)),
    }
    _train_dense_torch_baseline(
        config=config,
        experiment_dir=experiment_dir,
        model_type="mlp",
        model=DenseMLP(**model_cfg),
        train_loader=DataLoader(SequenceDataset(train_features, train_y), batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(SequenceDataset(val_features, val_y), batch_size=eval_batch_size, shuffle=False),
        test_loader=DataLoader(SequenceDataset(test_features, test_y), batch_size=eval_batch_size, shuffle=False),
        train_rows=int(train_features.shape[0]),
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        device=device,
        model_payload=model_cfg,
    )


def run_embedding_mlp(
    config: dict[str, Any],
    preprocessor: TabularPreprocessor,
    feature_schema: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment_dir: Path,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_xc, train_xn, train_y = prepare_arrays(preprocessor, train_df)
    val_xc, val_xn, val_y = prepare_arrays(preprocessor, val_df)
    test_xc, test_xn, test_y = prepare_arrays(preprocessor, test_df)
    batch_size = int(config["training"]["batch_size"])
    eval_batch_size = int(config["training"].get("eval_batch_size") or batch_size)
    model_cfg = {
        "categorical_cardinalities": list(feature_schema["categorical_cardinalities"]),
        "num_continuous": len(preprocessor.continuous_columns),
        "embedding_dim": int(config.get("embedding_mlp", {}).get("embedding_dim", 16)),
        "hidden_sizes": list(config.get("embedding_mlp", {}).get("hidden_sizes", [256, 128, 64])),
        "dropout": float(config.get("embedding_mlp", {}).get("dropout", 0.2)),
    }
    model = EmbeddingMLP(**model_cfg)
    model_payload = dict(model_cfg)
    model_payload["embedding_dims"] = list(model.embedding_dims)
    _train_embedding_torch_baseline(
        config=config,
        experiment_dir=experiment_dir,
        model_type="embedding_mlp",
        model=model,
        train_loader=DataLoader(TabDataset(train_xc, train_xn, train_y), batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(TabDataset(val_xc, val_xn, val_y), batch_size=eval_batch_size, shuffle=False),
        test_loader=DataLoader(TabDataset(test_xc, test_xn, test_y), batch_size=eval_batch_size, shuffle=False),
        train_rows=int(train_y.shape[0]),
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        device=device,
        model_payload=model_payload,
    )


def run_tab_transformer_no_engineered(
    config: dict[str, Any],
    preprocessor: TabularPreprocessor,
    feature_schema: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment_dir: Path,
) -> tuple[TabularPreprocessor, dict[str, Any]]:
    engineered_categorical = set(ENGINEERED_CATEGORICAL_COLUMNS)
    engineered_continuous = set(ENGINEERED_CONTINUOUS_COLUMNS)
    categorical_columns = [
        column for column in preprocessor.categorical_columns if column not in engineered_categorical
    ]
    continuous_columns = [
        column for column in preprocessor.continuous_columns if column not in engineered_continuous
    ]
    ablated_preprocessor = clone_preprocessor_with_columns(
        preprocessor,
        categorical_columns,
        continuous_columns,
    )
    ablated_schema = subset_feature_schema(feature_schema, categorical_columns, continuous_columns)
    train_triplet = arrays_to_triplet(prepare_arrays(preprocessor, train_df))
    val_triplet = arrays_to_triplet(prepare_arrays(preprocessor, val_df))
    test_triplet = arrays_to_triplet(prepare_arrays(preprocessor, test_df))
    run_tab_transformer(
        config,
        ablated_preprocessor,
        ablated_schema,
        train_df,
        val_df,
        test_df,
        experiment_dir,
        train_arrays=subset_triplet_columns(preprocessor, train_triplet, categorical_columns, continuous_columns),
        val_arrays=subset_triplet_columns(preprocessor, val_triplet, categorical_columns, continuous_columns),
        test_arrays=subset_triplet_columns(preprocessor, test_triplet, categorical_columns, continuous_columns),
        model_type="tab_transformer_no_engineered",
    )
    return ablated_preprocessor, ablated_schema


def _tab_transformer_scores(
    model: nn.Module,
    x_categ: np.ndarray,
    x_cont: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    x_categ_tensor = torch.as_tensor(x_categ, dtype=torch.long)
    x_cont_tensor = torch.as_tensor(x_cont, dtype=torch.float32)
    with torch.no_grad():
        for start_index in range(0, x_categ_tensor.size(0), max(1, int(batch_size))):
            end_index = min(start_index + max(1, int(batch_size)), x_categ_tensor.size(0))
            logits = model(
                x_categ_tensor[start_index:end_index].to(device),
                x_cont_tensor[start_index:end_index].to(device),
            ).squeeze(-1)
            logits = apply_temperature_to_logits(logits, temperature)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def run_feature_mask_ablation(
    config: dict[str, Any],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    artifact_dir = Path(config["project"]["artifacts_dir"]) / "current"
    output_dir = ensure_dir(Path(config["project"]["reports_dir"]) / "ablations" / "feature_masks" / "latest")
    artifact_preprocessor = TabularPreprocessor.load(artifact_dir / "preprocessor.joblib")
    artifact_schema = json.loads((artifact_dir / "feature_schema.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(_filesystem_path_str(artifact_dir / "model.pt"), map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, temperature = load_tab_transformer_checkpoint(
        checkpoint=checkpoint,
        categorical_cardinalities=artifact_schema["categorical_cardinalities"],
        num_continuous=len(artifact_preprocessor.continuous_columns),
        device=device,
    )
    model.to(device)
    val_xc, val_xn, val_y = prepare_arrays(artifact_preprocessor, val_df)
    test_xc, test_xn, test_y = prepare_arrays(artifact_preprocessor, test_df)
    eval_batch_size = int(config["training"].get("eval_batch_size") or config["training"]["batch_size"])
    val_scores = _tab_transformer_scores(
        model,
        val_xc,
        val_xn,
        device=device,
        batch_size=eval_batch_size,
        temperature=temperature,
    )
    threshold, _ = choose_best_threshold(val_y, val_scores)
    baseline_scores = _tab_transformer_scores(
        model,
        test_xc,
        test_xn,
        device=device,
        batch_size=eval_batch_size,
        temperature=temperature,
    )
    baseline_metrics = classification_metrics(test_y, baseline_scores, threshold)
    categorical_groups = {
        "port": ["destination_port_bucket", "is_well_known_port", "destination_port_band"],
        "tcp_flags": [
            "tcp_flag_signature",
            "fwd_psh_flags",
            "fwd_urg_flags",
            "fin_flag_count",
            "syn_flag_count",
            "rst_flag_count",
            "psh_flag_count",
            "ack_flag_count",
            "urg_flag_count",
            "cwe_flag_count",
            "ece_flag_count",
        ],
        "window_state": ["window_state"],
        "payload_profile": ["payload_profile"],
    }
    continuous_groups = {
        "port": ["destination_port"],
        "tcp_flags": [],
        "window_state": ["init_win_bytes_forward", "init_win_bytes_backward"],
        "payload_profile": [
            "total_length_of_fwd_packets",
            "total_length_of_bwd_packets",
            "total_bytes",
            "bytes_per_packet",
            "packet_length_mean",
            "packet_length_mean_log1p",
            "average_packet_size",
            "avg_fwd_segment_size",
            "avg_bwd_segment_size",
            "subflow_fwd_bytes",
            "subflow_bwd_bytes",
            "act_data_pkt_fwd",
        ],
    }
    categorical_index = {
        column: index for index, column in enumerate(artifact_preprocessor.categorical_columns)
    }
    continuous_index = {
        column: index for index, column in enumerate(artifact_preprocessor.continuous_columns)
    }
    mask_results: dict[str, Any] = {}
    for group_name in ("port", "tcp_flags", "window_state", "payload_profile"):
        masked_xc = test_xc.copy()
        masked_xn = test_xn.copy()
        masked_categorical = [
            column for column in categorical_groups[group_name] if column in categorical_index
        ]
        masked_continuous = [
            column for column in continuous_groups[group_name] if column in continuous_index
        ]
        for column in masked_categorical:
            masked_xc[:, categorical_index[column]] = 0
        for column in masked_continuous:
            masked_xn[:, continuous_index[column]] = 0.0
        masked_scores = _tab_transformer_scores(
            model,
            masked_xc,
            masked_xn,
            device=device,
            batch_size=eval_batch_size,
            temperature=temperature,
        )
        masked_metrics = classification_metrics(test_y, masked_scores, threshold)
        mask_results[group_name] = {
            "masked_categorical_columns": masked_categorical,
            "masked_continuous_columns": masked_continuous,
            "metrics": masked_metrics,
            "delta_f1": float(masked_metrics["f1"] - baseline_metrics["f1"]),
            "delta_roc_auc": float(masked_metrics["roc_auc"] - baseline_metrics["roc_auc"]),
        }
    payload = {
        "model_type": "feature_mask_ablation",
        "source_model": str(artifact_dir),
        "threshold": threshold,
        "archived_threshold": float(metrics_payload["threshold"]),
        "threshold_source": "loadable_model_validation_best_f1",
        "temperature": float(temperature),
        "baseline": baseline_metrics,
        "masks": mask_results,
    }
    _write_json(output_dir / "metrics.json", payload)


def run_lstm(
    config: dict[str, Any],
    preprocessor: TabularPreprocessor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment_dir: Path,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_xc, train_xn, train_y = prepare_arrays(preprocessor, train_df)
    val_xc, val_xn, val_y = prepare_arrays(preprocessor, val_df)
    test_xc, test_xn, test_y = prepare_arrays(preprocessor, test_df)
    train_features = np.concatenate([train_xc.astype(np.float32), train_xn], axis=1)
    val_features = np.concatenate([val_xc.astype(np.float32), val_xn], axis=1)
    test_features = np.concatenate([test_xc.astype(np.float32), test_xn], axis=1)
    train_loader = DataLoader(
        SequenceDataset(train_features, train_y),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(val_features, val_y),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(test_features, test_y),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    model_cfg = {
        "input_features": int(train_features.shape[1]),
        "hidden_size": config["lstm"]["hidden_size"],
        "num_layers": config["lstm"]["num_layers"],
        "dropout": config["lstm"]["dropout"],
    }
    model = FeatureSequenceLSTM(**model_cfg).to(device)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    patience = config["training"]["patience"]
    remaining = patience

    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).squeeze(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_labels)

        model.eval()
        scores = []
        with torch.no_grad():
            for batch_features, _ in val_loader:
                val_logits = model(batch_features.to(device)).squeeze(-1)
                scores.append(torch.sigmoid(val_logits).cpu().numpy())
        val_scores = np.concatenate(scores)
        _, val_f1 = choose_best_threshold(val_y, val_scores)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(len(train_features), 1),
                "val_best_f1": val_f1,
            }
        )
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    val_batches = []
    test_batches = []
    with torch.no_grad():
        for batch_features, _ in val_loader:
            val_batches.append(torch.sigmoid(model(batch_features.to(device)).squeeze(-1)).cpu().numpy())
        for batch_features, _ in test_loader:
            test_batches.append(torch.sigmoid(model(batch_features.to(device)).squeeze(-1)).cpu().numpy())
    val_scores = np.concatenate(val_batches)
    test_scores = np.concatenate(test_batches)
    threshold, _ = choose_best_threshold(val_y, val_scores)
    selection_metrics = build_selection_metrics(
        val_y,
        val_scores,
        threshold,
        best_epoch=best_epoch,
    )
    score_diagnostics = build_score_diagnostics_payload(
        val_y=val_y,
        val_scores=val_scores,
        test_y=test_y,
        test_scores=test_scores,
        threshold=threshold,
    )
    metrics = classification_metrics(test_y, test_scores, threshold)
    metrics["model_type"] = "lstm"
    metrics["best_epoch"] = best_epoch

    torch.save(
        {"model_type": "lstm", "state_dict": model.state_dict(), "model_config": model_cfg},
        _filesystem_path_str(experiment_dir / "model.pt"),
    )
    _write_json(experiment_dir / "metrics.json", metrics)
    _write_json(experiment_dir / "selection_metrics.json", selection_metrics)
    _write_json(experiment_dir / "score_diagnostics.json", score_diagnostics)
    _write_json(experiment_dir / "classification_report.json", metrics["classification_report"])
    _write_json(experiment_dir / "threshold.json", {"threshold": threshold})
    _write_json(experiment_dir / "training_history.json", history)
    save_roc_curve(test_y, test_scores, experiment_dir / "roc_curve.png")
    save_confusion_matrix(test_y, test_scores, threshold, experiment_dir / "confusion_matrix.png")


def run_isolation_forest(
    config: dict[str, Any],
    preprocessor: TabularPreprocessor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment_dir: Path,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    train_xc, train_xn, train_y = prepare_arrays(preprocessor, train_df)
    val_xc, val_xn, val_y = prepare_arrays(preprocessor, val_df)
    test_xc, test_xn, test_y = prepare_arrays(preprocessor, test_df)

    train_features = np.concatenate([train_xc.astype(np.float32), train_xn], axis=1)
    val_features = np.concatenate([val_xc.astype(np.float32), val_xn], axis=1)
    test_features = np.concatenate([test_xc.astype(np.float32), test_xn], axis=1)

    benign_only = train_features[train_y == 0]
    model = build_isolation_forest(random_state=config["dataset"]["random_state"])
    model.fit(benign_only)
    val_scores = (-model.decision_function(val_features)).astype(np.float32)
    test_scores = (-model.decision_function(test_features)).astype(np.float32)

    val_conf = 1.0 / (1.0 + np.exp(-val_scores))
    test_conf = 1.0 / (1.0 + np.exp(-test_scores))
    threshold, _ = choose_best_threshold(val_y, val_conf)
    selection_metrics = build_selection_metrics(
        val_y,
        val_conf,
        threshold,
        best_epoch=0,
    )
    score_diagnostics = build_score_diagnostics_payload(
        val_y=val_y,
        val_scores=val_conf,
        test_y=test_y,
        test_scores=test_conf,
        threshold=threshold,
    )
    metrics = classification_metrics(test_y, test_conf, threshold)
    metrics["model_type"] = "isolation_forest"
    metrics["best_epoch"] = 0

    joblib.dump(model, _filesystem_path_str(experiment_dir / "model.pt"))
    _write_json(experiment_dir / "metrics.json", metrics)
    _write_json(experiment_dir / "selection_metrics.json", selection_metrics)
    _write_json(experiment_dir / "score_diagnostics.json", score_diagnostics)
    _write_json(experiment_dir / "classification_report.json", metrics["classification_report"])
    _write_json(experiment_dir / "threshold.json", {"threshold": threshold})
    _write_json(experiment_dir / "training_history.json", [])
    save_roc_curve(test_y, test_conf, experiment_dir / "roc_curve.png")
    save_confusion_matrix(test_y, test_conf, threshold, experiment_dir / "confusion_matrix.png")


def save_common_artifacts(
    preprocessor: TabularPreprocessor,
    feature_schema: dict[str, Any],
    config: dict[str, Any],
    model_name: str,
    experiment_dir: Path,
) -> None:
    preprocessor.save(_filesystem_path_str(experiment_dir / "preprocessor.joblib"))
    _write_text_file(
        experiment_dir / "feature_schema.json",
        json.dumps(feature_schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics = json.loads(_read_text_file(experiment_dir / "metrics.json", encoding="utf-8"))
    selection_metrics = json.loads(
        _read_text_file(experiment_dir / "selection_metrics.json", encoding="utf-8")
    )
    model_card = {
        "model_type": model_name,
        "raw_data_dir": config["project"]["raw_data_dir"],
        "categorical_columns": preprocessor.categorical_columns,
        "continuous_columns": preprocessor.continuous_columns,
        "summary_metrics": {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
        },
        "selection_summary_metrics": {
            "val_f1": selection_metrics["val_f1"],
            "val_f_beta": selection_metrics.get("val_f_beta"),
            "val_precision": selection_metrics["val_precision"],
            "val_recall": selection_metrics["val_recall"],
            "val_average_precision": selection_metrics.get("val_average_precision"),
            "val_roc_auc": selection_metrics["val_roc_auc"],
            "selection_score": selection_metrics["selection_score"],
            "epoch_selection_metric": selection_metrics.get("epoch_selection_metric"),
            "threshold_selection_rule": selection_metrics.get("threshold_selection_rule"),
            "calibration_mode": selection_metrics.get("calibration_mode"),
            "temperature": selection_metrics.get("temperature"),
        },
    }
    _write_json(experiment_dir / "model_card.json", model_card)


def train_model(config_path: str, model_name: str) -> None:
    config = load_config(config_path)
    project = config["project"]
    processed_dir = Path(project["processed_dir"])
    reports_root = ensure_dir(Path(project["reports_dir"]) / model_name / "latest")

    preprocessor, feature_schema, train_df, val_df, test_df = read_processed_assets(processed_dir)
    report_preprocessor = preprocessor
    report_feature_schema = feature_schema

    if model_name == "tab_transformer":
        run_tab_transformer(config, preprocessor, feature_schema, train_df, val_df, test_df, reports_root)
    elif model_name == "tab_transformer_no_engineered":
        report_preprocessor, report_feature_schema = run_tab_transformer_no_engineered(
            config,
            preprocessor,
            feature_schema,
            train_df,
            val_df,
            test_df,
            reports_root,
        )
    elif model_name == "feature_mask_ablation":
        run_feature_mask_ablation(config, val_df, test_df)
        return
    elif model_name == "mlp":
        run_mlp(config, preprocessor, train_df, val_df, test_df, reports_root)
    elif model_name == "embedding_mlp":
        run_embedding_mlp(config, preprocessor, feature_schema, train_df, val_df, test_df, reports_root)
    elif model_name == "lstm":
        run_lstm(config, preprocessor, train_df, val_df, test_df, reports_root)
    elif model_name == "isolation_forest":
        run_isolation_forest(config, preprocessor, train_df, val_df, test_df, reports_root)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    save_common_artifacts(report_preprocessor, report_feature_schema, config, model_name, reports_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "tab_transformer",
            "tab_transformer_no_engineered",
            "feature_mask_ablation",
            "mlp",
            "embedding_mlp",
            "lstm",
            "isolation_forest",
        ],
    )
    args = parser.parse_args()
    train_model(args.config, args.model)


if __name__ == "__main__":
    main()
