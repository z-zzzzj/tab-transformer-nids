from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_SELECTION_BETA = 1.5
DEFAULT_THRESHOLD_PLATEAU_TOLERANCE = 0.001


def _filesystem_path_str(path: str | Path) -> str:
    raw_path = os.path.abspath(os.fspath(path))
    if os.name != "nt":
        return raw_path
    if raw_path.startswith("\\\\?\\") or raw_path.startswith("\\\\.\\"):
        return raw_path
    if raw_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw_path[2:]
    return "\\\\?\\" + raw_path


def _safe_roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = np.unique(np.asarray(y_true))
    if labels.size < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def _safe_average_precision_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = np.unique(np.asarray(y_true))
    if labels.size < 2:
        return float(labels[0]) if labels.size == 1 else 0.0
    return float(average_precision_score(y_true, scores))


def fbeta_from_precision_recall(
    precision: np.ndarray,
    recall: np.ndarray,
    beta: float,
) -> np.ndarray:
    beta_sq = float(beta) ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return numerator / np.clip(denominator, a_min=1e-12, a_max=None)


def choose_threshold_with_plateau(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    beta: float = DEFAULT_SELECTION_BETA,
    plateau_tolerance: float = DEFAULT_THRESHOLD_PLATEAU_TOLERANCE,
) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        threshold = 0.5
        preds = (scores >= threshold).astype(int)
        precision_value = float(precision_score(y_true, preds, zero_division=0))
        recall_value = float(recall_score(y_true, preds, zero_division=0))
        f1_value = float(f1_score(y_true, preds, zero_division=0))
        f_beta_value = float(
            fbeta_from_precision_recall(
                np.asarray([precision_value], dtype=np.float64),
                np.asarray([recall_value], dtype=np.float64),
                beta,
            )[0]
        )
        return {
            "threshold": float(threshold),
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
            "f_beta": f_beta_value,
            "predicted_positive_rate": float(preds.mean()) if preds.size else 0.0,
            "plateau_size": 1,
            "best_f1": f1_value,
            "best_f_beta": f_beta_value,
            "selection_rule": "fallback_threshold_0.5",
        }

    precision_values = precision[:-1].astype(np.float64)
    recall_values = recall[:-1].astype(np.float64)
    threshold_values = thresholds.astype(np.float64)
    f1_values = fbeta_from_precision_recall(precision_values, recall_values, beta=1.0)
    f_beta_values = fbeta_from_precision_recall(precision_values, recall_values, beta=beta)

    best_f1 = float(np.max(f1_values))
    plateau_floor = best_f1 - float(max(0.0, plateau_tolerance))
    plateau_indices = np.flatnonzero(f1_values >= plateau_floor)
    if plateau_indices.size == 0:
        plateau_indices = np.asarray([int(np.nanargmax(f1_values))], dtype=np.int64)

    chosen_index = min(
        plateau_indices.tolist(),
        key=lambda index: (
            -float(recall_values[index]),
            float(threshold_values[index]),
            -float(f_beta_values[index]),
            -float(precision_values[index]),
            int(index),
        ),
    )

    positive_count = float(np.sum(np.asarray(y_true) == 1))
    predicted_positive_counts = (recall_values * positive_count) / np.clip(
        precision_values,
        a_min=1e-12,
        a_max=None,
    )
    predicted_positive_rate = np.clip(
        predicted_positive_counts / max(float(len(scores)), 1.0),
        a_min=0.0,
        a_max=1.0,
    )
    return {
        "threshold": float(threshold_values[chosen_index]),
        "precision": float(precision_values[chosen_index]),
        "recall": float(recall_values[chosen_index]),
        "f1": float(f1_values[chosen_index]),
        "f_beta": float(f_beta_values[chosen_index]),
        "predicted_positive_rate": float(predicted_positive_rate[chosen_index]),
        "plateau_size": int(plateau_indices.size),
        "best_f1": best_f1,
        "best_f_beta": float(np.max(f_beta_values)),
        "selection_rule": (
            f"f1_plateau_tol_{float(max(0.0, plateau_tolerance)):.4f}"
            "_recall_first_lower_threshold_tiebreak"
        ),
    }


def choose_threshold_max_f1(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        threshold = 0.5
        preds = (scores >= threshold).astype(int)
        precision_value = float(precision_score(y_true, preds, zero_division=0))
        recall_value = float(recall_score(y_true, preds, zero_division=0))
        f1_value = float(f1_score(y_true, preds, zero_division=0))
        return {
            "threshold": float(threshold),
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
            "best_f1": f1_value,
            "selection_rule": "fallback_threshold_0.5",
        }

    precision_values = precision[:-1].astype(np.float64)
    recall_values = recall[:-1].astype(np.float64)
    threshold_values = thresholds.astype(np.float64)
    f1_values = fbeta_from_precision_recall(precision_values, recall_values, beta=1.0)
    best_f1 = float(np.max(f1_values))
    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0.0, atol=1e-12))
    chosen_index = min(
        best_indices.tolist(),
        key=lambda index: (
            -float(precision_values[index]),
            -float(recall_values[index]),
            float(threshold_values[index]),
            int(index),
        ),
    )
    return {
        "threshold": float(threshold_values[chosen_index]),
        "precision": float(precision_values[chosen_index]),
        "recall": float(recall_values[chosen_index]),
        "f1": float(f1_values[chosen_index]),
        "best_f1": best_f1,
        "selection_rule": "max_f1_precision_recall_lower_threshold",
    }


def choose_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    plateau_tolerance: float = DEFAULT_THRESHOLD_PLATEAU_TOLERANCE,
) -> tuple[float, float]:
    selected = choose_threshold_with_plateau(
        y_true,
        scores,
        beta=DEFAULT_SELECTION_BETA,
        plateau_tolerance=plateau_tolerance,
    )
    return float(selected["threshold"]), float(selected["f1"])


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    beta: float = DEFAULT_SELECTION_BETA,
) -> dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    precision_value = float(precision_score(y_true, preds, zero_division=0))
    recall_value = float(recall_score(y_true, preds, zero_division=0))
    f1_value = float(f1_score(y_true, preds, zero_division=0))
    f_beta_value = float(
        fbeta_from_precision_recall(
            np.asarray([precision_value], dtype=np.float64),
            np.asarray([recall_value], dtype=np.float64),
            beta,
        )[0]
    )
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "f_beta": f_beta_value,
        "f_beta_beta": float(beta),
        "average_precision": _safe_average_precision_score(y_true, scores),
        "predicted_positive_rate": float(preds.mean()) if preds.size else 0.0,
        "roc_auc": _safe_roc_auc_score(y_true, scores),
        "classification_report": classification_report(y_true, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
    }
    return metrics


def build_selection_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    best_epoch: int,
    beta: float = DEFAULT_SELECTION_BETA,
    epoch_selection_metric: str = "val_f1",
    threshold_selection_rule: str = "max_f1",
    calibration_mode: str = "none",
    temperature: float = 1.0,
) -> dict[str, Any]:
    metrics = classification_metrics(y_true, scores, threshold, beta=beta)
    return {
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "val_precision": float(metrics["precision"]),
        "val_recall": float(metrics["recall"]),
        "val_f1": float(metrics["f1"]),
        "val_f_beta": float(metrics["f_beta"]),
        "val_average_precision": float(metrics["average_precision"]),
        "val_roc_auc": float(metrics["roc_auc"]),
        "selection_score": float(metrics["f1"]),
        "epoch_selection_metric": str(epoch_selection_metric),
        "threshold_selection_rule": str(threshold_selection_rule),
        "calibration_mode": str(calibration_mode),
        "temperature": float(temperature),
    }


def build_attack_family_metrics(
    family_labels: Sequence[Any] | np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    focus_families: Sequence[str] | None = None,
    benign_label: str = "BENIGN",
) -> dict[str, Any] | None:
    if family_labels is None:
        return None
    labels = np.asarray(list(family_labels), dtype=object)
    targets = np.asarray(y_true)
    score_values = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.size != targets.size or targets.size != score_values.size:
        return None

    focus_set = {str(family) for family in (focus_families or [])}
    benign_label_normalized = str(benign_label)
    preds = (score_values >= float(threshold)).astype(np.int64)
    families: dict[str, Any] = {}

    for raw_family in sorted({str(value) for value in labels.tolist()}):
        family_mask = labels == raw_family
        family_rows = int(np.sum(family_mask))
        family_targets = targets[family_mask]
        positive_rows = int(np.sum(family_targets == 1))
        if positive_rows <= 0 or raw_family == benign_label_normalized:
            continue

        family_scores = score_values[family_mask]
        family_preds = preds[family_mask]
        tp = int(np.sum((family_preds == 1) & (family_targets == 1)))
        fn = int(np.sum((family_preds == 0) & (family_targets == 1)))
        families[raw_family] = {
            "rows": family_rows,
            "tp": tp,
            "fn": fn,
            "recall": float(tp / max(positive_rows, 1)),
            "score_p01": float(np.quantile(family_scores, 0.01)) if family_scores.size else None,
            "score_p05": float(np.quantile(family_scores, 0.05)) if family_scores.size else None,
            "score_p50": float(np.quantile(family_scores, 0.50)) if family_scores.size else None,
            "focus_family": raw_family in focus_set,
        }

    return {
        "available": True,
        "focus_attack_families": list(focus_families or []),
        "threshold": float(threshold),
        "family_count": len(families),
        "families": families,
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    os.makedirs(_filesystem_path_str(destination.parent), exist_ok=True)
    with open(_filesystem_path_str(destination), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, indent=2))


def save_roc_curve(y_true: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    destination = Path(path)
    os.makedirs(_filesystem_path_str(destination.parent), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_filesystem_path_str(destination), dpi=150)
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, scores: np.ndarray, threshold: float, path: str | Path) -> None:
    destination = Path(path)
    os.makedirs(_filesystem_path_str(destination.parent), exist_ok=True)
    preds = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, preds, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(_filesystem_path_str(destination), dpi=150)
    plt.close()
