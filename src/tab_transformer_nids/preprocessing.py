from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ENGINEERED_CATEGORICAL_COLUMNS = [
    "destination_port_band",
    "flow_duration_bin",
    "total_packets_bin",
    "total_bytes_bin",
    "flow_bytes_per_s_bin",
    "flow_packets_per_s_bin",
    "packet_length_mean_bin",
    "packet_length_std_bin",
    "iat_mean_bin",
    "iat_cv_bin",
    "direction_ratio_bin",
    "header_payload_bin",
    "tcp_flag_signature",
    "window_state",
    "payload_profile",
]

ENGINEERED_CONTINUOUS_COLUMNS = [
    "total_packets",
    "total_bytes",
    "bytes_per_packet",
    "packet_length_range",
    "iat_cv",
    "header_payload_ratio",
    "direction_ratio_log1p",
    "flow_duration_log1p",
    "flow_bytes_per_s_log1p",
    "flow_packets_per_s_log1p",
    "packet_length_mean_log1p",
]

SEMANTIC_DEDUPE_EXCLUDED_COLUMNS = {
    "row_id",
    "source_file",
    "source_dataset",
}

SEEN_FAMILY_NO_LEAK_SPLIT_STRATEGY = "seen_family_no_leak"
DEFAULT_RARE_FAMILY_TRAIN_ONLY_THRESHOLD = 100
SEEN_FAMILY_GROUP_COLUMNS = ("source_file", "label_original")


def normalize_column_name(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = cleaned.replace("/", "_per_")
    cleaned = cleaned.replace(".", "_")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def normalize_label(label: Any) -> str:
    if pd.isna(label):
        return "UNKNOWN"
    text = str(label).strip()
    text = text.replace("\ufffd", "-")
    return text


def bucketize_port(value: Any, top_ports: set[int]) -> str:
    try:
        port = int(float(value))
    except (TypeError, ValueError):
        return "UNKNOWN"
    if port < 0:
        return "UNKNOWN"
    if port in top_ports:
        return str(port)
    return "OTHER"


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if pd.api.types.is_float_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="float")
        elif pd.api.types.is_integer_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="integer")
    return df


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return series.astype(np.float32)


def _bucket_numeric(series: pd.Series, edges: list[float], prefix: str) -> pd.Series:
    labels = [f"{prefix}_{index:02d}" for index in range(len(edges) + 1)]
    bucketed = pd.cut(
        series.astype(np.float64),
        bins=[-np.inf, *edges, np.inf],
        labels=labels,
        right=False,
    )
    output = bucketed.astype("object").where(series.notna(), "MISSING").astype(str)
    return output


def _log1p_nonnegative(series: pd.Series) -> pd.Series:
    clipped = series.clip(lower=0).fillna(0.0).astype(np.float32)
    return pd.Series(np.log1p(clipped), index=series.index, dtype=np.float32)


def augment_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    dst_port = _numeric_series(output, "destination_port")
    total_fwd_packets = _numeric_series(output, "total_fwd_packets")
    total_bwd_packets = _numeric_series(output, "total_backward_packets")
    total_fwd_bytes = _numeric_series(output, "total_length_of_fwd_packets")
    total_bwd_bytes = _numeric_series(output, "total_length_of_bwd_packets")
    flow_duration = _numeric_series(output, "flow_duration")
    flow_bytes_per_s = _numeric_series(output, "flow_bytes_per_s")
    flow_packets_per_s = _numeric_series(output, "flow_packets_per_s")
    packet_length_mean = _numeric_series(output, "packet_length_mean")
    packet_length_std = _numeric_series(output, "packet_length_std")
    packet_length_max = _numeric_series(output, "max_packet_length")
    packet_length_min = _numeric_series(output, "min_packet_length")
    flow_iat_mean = _numeric_series(output, "flow_iat_mean")
    flow_iat_std = _numeric_series(output, "flow_iat_std")
    down_per_up_ratio = _numeric_series(output, "down_per_up_ratio")
    fwd_header_length = _numeric_series(output, "fwd_header_length")
    bwd_header_length = _numeric_series(output, "bwd_header_length")
    init_win_bytes_forward = _numeric_series(output, "init_win_bytes_forward")
    init_win_bytes_backward = _numeric_series(output, "init_win_bytes_backward")
    act_data_pkt_fwd = _numeric_series(output, "act_data_pkt_fwd")
    syn_flag_count = _numeric_series(output, "syn_flag_count")
    ack_flag_count = _numeric_series(output, "ack_flag_count")
    rst_flag_count = _numeric_series(output, "rst_flag_count")
    psh_flag_count = _numeric_series(output, "psh_flag_count")
    urg_flag_count = _numeric_series(output, "urg_flag_count")
    fin_flag_count = _numeric_series(output, "fin_flag_count")
    ece_flag_count = _numeric_series(output, "ece_flag_count")
    cwe_flag_count = _numeric_series(output, "cwe_flag_count")

    total_packets = total_fwd_packets.fillna(0.0) + total_bwd_packets.fillna(0.0)
    total_bytes = total_fwd_bytes.fillna(0.0) + total_bwd_bytes.fillna(0.0)
    bytes_per_packet = total_bytes / (total_packets + 1.0)
    packet_length_range = (packet_length_max - packet_length_min).fillna(0.0)
    iat_cv = flow_iat_std.fillna(0.0) / (flow_iat_mean.abs().fillna(0.0) + 1.0)
    header_payload_ratio = (
        fwd_header_length.fillna(0.0) + bwd_header_length.fillna(0.0)
    ) / (total_bytes + 1.0)
    direction_ratio = (total_fwd_bytes.fillna(0.0) + 1.0) / (total_bwd_bytes.fillna(0.0) + 1.0)

    output["total_packets"] = total_packets.astype(np.float32)
    output["total_bytes"] = total_bytes.astype(np.float32)
    output["bytes_per_packet"] = bytes_per_packet.astype(np.float32)
    output["packet_length_range"] = packet_length_range.astype(np.float32)
    output["iat_cv"] = iat_cv.astype(np.float32)
    output["header_payload_ratio"] = header_payload_ratio.astype(np.float32)
    output["direction_ratio_log1p"] = _log1p_nonnegative(direction_ratio)
    output["flow_duration_log1p"] = _log1p_nonnegative(flow_duration)
    output["flow_bytes_per_s_log1p"] = _log1p_nonnegative(flow_bytes_per_s)
    output["flow_packets_per_s_log1p"] = _log1p_nonnegative(flow_packets_per_s)
    output["packet_length_mean_log1p"] = _log1p_nonnegative(packet_length_mean)

    output["destination_port_band"] = _bucket_numeric(
        dst_port,
        [0.0, 1024.0, 49152.0],
        "port_band",
    )
    output["flow_duration_bin"] = _bucket_numeric(
        flow_duration,
        [10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0],
        "flow_duration",
    )
    output["total_packets_bin"] = _bucket_numeric(
        total_packets,
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
        "total_packets",
    )
    output["total_bytes_bin"] = _bucket_numeric(
        total_bytes,
        [64.0, 256.0, 1_024.0, 4_096.0, 16_384.0, 65_536.0, 262_144.0, 1_048_576.0],
        "total_bytes",
    )
    output["flow_bytes_per_s_bin"] = _bucket_numeric(
        flow_bytes_per_s,
        [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0],
        "flow_bytes_rate",
    )
    output["flow_packets_per_s_bin"] = _bucket_numeric(
        flow_packets_per_s,
        [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
        "flow_packets_rate",
    )
    output["packet_length_mean_bin"] = _bucket_numeric(
        packet_length_mean,
        [1.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1_024.0],
        "packet_mean",
    )
    output["packet_length_std_bin"] = _bucket_numeric(
        packet_length_std,
        [1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
        "packet_std",
    )
    output["iat_mean_bin"] = _bucket_numeric(
        flow_iat_mean,
        [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
        "iat_mean",
    )
    output["iat_cv_bin"] = _bucket_numeric(
        iat_cv,
        [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "iat_cv",
    )
    output["direction_ratio_bin"] = _bucket_numeric(
        direction_ratio,
        [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        "direction_ratio",
    )
    output["header_payload_bin"] = _bucket_numeric(
        header_payload_ratio,
        [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "header_payload",
    )

    flag_signature = np.select(
        [
            rst_flag_count.fillna(0.0) > 0,
            (syn_flag_count.fillna(0.0) > 0) & (ack_flag_count.fillna(0.0) > 0),
            syn_flag_count.fillna(0.0) > 0,
            (psh_flag_count.fillna(0.0) > 0) & (ack_flag_count.fillna(0.0) > 0),
            psh_flag_count.fillna(0.0) > 0,
            ack_flag_count.fillna(0.0) > 0,
            (
                fin_flag_count.fillna(0.0)
                + urg_flag_count.fillna(0.0)
                + ece_flag_count.fillna(0.0)
                + cwe_flag_count.fillna(0.0)
            )
            > 0,
        ],
        [
            "rst_seen",
            "syn_ack",
            "syn_only",
            "psh_ack",
            "psh_only",
            "ack_only",
            "rare_flag_combo",
        ],
        default="no_flags",
    )
    output["tcp_flag_signature"] = pd.Series(flag_signature, index=output.index, dtype="object")

    window_state = np.select(
        [
            (init_win_bytes_forward < 0) | (init_win_bytes_backward < 0),
            (init_win_bytes_forward == 0) & (init_win_bytes_backward == 0),
            init_win_bytes_forward >= 32_768,
            init_win_bytes_backward >= 32_768,
        ],
        [
            "missing_window",
            "zero_window",
            "forward_large_window",
            "backward_large_window",
        ],
        default="standard_window",
    )
    output["window_state"] = pd.Series(window_state, index=output.index, dtype="object")

    payload_profile = np.select(
        [
            (total_bytes <= 0) & (total_packets > 0),
            act_data_pkt_fwd.fillna(0.0) <= 0,
            total_bytes < 256.0,
            total_bytes < 4_096.0,
            total_bytes < 65_536.0,
        ],
        [
            "control_only",
            "no_forward_payload",
            "tiny_payload",
            "small_payload",
            "medium_payload",
        ],
        default="large_payload",
    )
    output["payload_profile"] = pd.Series(payload_profile, index=output.index, dtype="object")

    if "down_per_up_ratio" in output.columns:
        output["down_per_up_ratio"] = down_per_up_ratio.astype(np.float32)
    return downcast_numeric(output)

class TabularPreprocessor:
    def __init__(self, categorical_columns: list[str], continuous_columns: list[str]):
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.category_maps: dict[str, dict[str, int]] = {}
        self.categorical_cardinalities: list[int] = []
        self.numeric_fill_values: dict[str, float] = {}
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        working_df = augment_behavioral_features(df)
        categ_frame = working_df[self.categorical_columns].copy()
        for column in self.categorical_columns:
            values = (
                categ_frame[column]
                .fillna("UNKNOWN")
                .astype(str)
                .replace({"nan": "UNKNOWN"})
            )
            uniques = sorted(set(values.tolist()))
            mapping = {value: idx for idx, value in enumerate(uniques)}
            self.category_maps[column] = mapping
            self.categorical_cardinalities.append(len(mapping))

        cont_frame = working_df[self.continuous_columns].replace([np.inf, -np.inf], np.nan)
        for column in self.continuous_columns:
            median = float(cont_frame[column].median()) if len(cont_frame) else 0.0
            if math.isnan(median):
                median = 0.0
            self.numeric_fill_values[column] = median
        filled = cont_frame.fillna(self.numeric_fill_values).astype(np.float32)
        if self.continuous_columns:
            self.scaler.fit(filled.values)
        else:
            self.scaler.mean_ = np.array([], dtype=np.float64)
            self.scaler.var_ = np.array([], dtype=np.float64)
            self.scaler.scale_ = np.array([], dtype=np.float64)
            self.scaler.n_features_in_ = 0
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        working_df = augment_behavioral_features(df)
        categ_arrays: list[np.ndarray] = []
        for column in self.categorical_columns:
            mapping = self.category_maps[column]
            values = (
                working_df[column]
                .fillna("UNKNOWN")
                .astype(str)
                .replace({"nan": "UNKNOWN"})
                .map(mapping)
                .fillna(-1)
                .astype(np.int64)
            )
            categ_arrays.append(values.to_numpy())
        if categ_arrays:
            x_categ = np.vstack(categ_arrays).T.astype(np.int64)
        else:
            x_categ = np.zeros((len(df), 0), dtype=np.int64)

        cont_frame = (
            working_df[self.continuous_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.numeric_fill_values)
            .astype(np.float32)
        )
        if self.continuous_columns:
            x_cont = self.scaler.transform(cont_frame.values).astype(np.float32)
        else:
            x_cont = np.zeros((len(df), 0), dtype=np.float32)
        return x_categ, x_cont

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def continuous_mean_std_tensor(self) -> np.ndarray:
        # Continuous features are standardized before entering the model, so the
        # model-side normalization tensor must reflect zero-mean/unit-std inputs.
        means = np.zeros(len(self.continuous_columns), dtype=np.float32)
        stds = np.ones(len(self.continuous_columns), dtype=np.float32)
        return np.stack([means, stds], axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorical_columns": self.categorical_columns,
            "continuous_columns": self.continuous_columns,
            "category_maps": self.category_maps,
            "categorical_cardinalities": self.categorical_cardinalities,
            "numeric_fill_values": self.numeric_fill_values,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_var": self.scaler.var_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TabularPreprocessor":
        instance = cls(
            categorical_columns=payload["categorical_columns"],
            continuous_columns=payload["continuous_columns"],
        )
        instance.category_maps = payload["category_maps"]
        instance.categorical_cardinalities = payload["categorical_cardinalities"]
        instance.numeric_fill_values = {
            key: float(value) for key, value in payload["numeric_fill_values"].items()
        }
        instance.scaler.mean_ = np.array(payload["scaler_mean"], dtype=np.float64)
        instance.scaler.var_ = np.array(payload["scaler_var"], dtype=np.float64)
        instance.scaler.scale_ = np.array(payload["scaler_scale"], dtype=np.float64)
        instance.scaler.n_features_in_ = len(instance.continuous_columns)
        return instance

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_dict(), destination)

    @classmethod
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        payload = joblib.load(path)
        return cls.from_dict(payload)


def infer_dataset_name(file_name: str) -> str:
    stem = Path(file_name).stem
    return normalize_column_name(stem.replace("pcap_iscx", ""))


def resolve_semantic_dedupe_columns(
    frame: pd.DataFrame,
    strong_identifier_columns: list[str] | None = None,
    extra_excluded_columns: list[str] | None = None,
) -> list[str]:
    excluded = set(SEMANTIC_DEDUPE_EXCLUDED_COLUMNS)
    for column in strong_identifier_columns or []:
        excluded.add(normalize_column_name(column))
    for column in extra_excluded_columns or []:
        excluded.add(normalize_column_name(column))
    dedupe_columns = [column for column in frame.columns if column not in excluded]
    if not dedupe_columns:
        raise ValueError("No semantic dedupe columns remain after excluding identifier columns.")
    return dedupe_columns


def semantic_deduplicate_frame(
    frame: pd.DataFrame,
    *,
    strong_identifier_columns: list[str] | None = None,
    extra_excluded_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], int]:
    dedupe_columns = resolve_semantic_dedupe_columns(
        frame,
        strong_identifier_columns=strong_identifier_columns,
        extra_excluded_columns=extra_excluded_columns,
    )
    deduped = frame.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
    removed_rows = int(len(frame) - len(deduped))
    return deduped, dedupe_columns, removed_rows


def rank_top_ports(port_counts: pd.Series, top_k: int) -> list[int]:
    counts: dict[int, int] = {}
    for value, count in port_counts.items():
        try:
            port = int(float(value))
        except (TypeError, ValueError):
            continue
        counts[port] = counts.get(port, 0) + int(count)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [port for port, _ in ranked[:top_k]]


def choose_top_ports_from_frame(
    frame: pd.DataFrame,
    top_k: int,
    column: str = "destination_port",
) -> list[int]:
    if column not in frame.columns:
        return []
    return rank_top_ports(frame[column].value_counts(dropna=True), top_k)


def scan_top_ports(files: list[Path], max_rows_per_file: int | None, top_k: int) -> list[int]:
    counts = pd.Series(dtype="int64")
    for file_path in files:
        frame = pd.read_csv(file_path, usecols=[" Destination Port"], nrows=max_rows_per_file)
        port_counts = frame[" Destination Port"].value_counts(dropna=True)
        counts = counts.add(port_counts, fill_value=0)
    return rank_top_ports(counts, top_k)


def load_raw_frame(
    file_path: Path,
    max_rows_per_file: int | None,
    benign_label: str,
) -> pd.DataFrame:
    frame = pd.read_csv(file_path, nrows=max_rows_per_file, low_memory=False)
    frame.columns = [normalize_column_name(column) for column in frame.columns]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame["label_original"] = frame["label"].map(normalize_label)
    frame["target_binary"] = (frame["label_original"] != benign_label).astype(np.int8)
    frame["source_file"] = file_path.name
    frame["source_dataset"] = infer_dataset_name(file_path.name)
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    return downcast_numeric(frame)


def apply_port_bucket_features(frame: pd.DataFrame, top_ports: set[int]) -> pd.DataFrame:
    if "destination_port" in frame.columns:
        frame["destination_port_bucket"] = frame["destination_port"].map(
            lambda value: bucketize_port(value, top_ports)
        )
        frame["is_well_known_port"] = frame["destination_port"].map(
            lambda value: "true"
            if pd.notna(value) and float(value) >= 0 and float(value) < 1024
            else "false"
        )
    else:
        frame["destination_port_bucket"] = "UNKNOWN"
        frame["is_well_known_port"] = "false"
        frame["destination_port"] = -1
    return downcast_numeric(frame)


def load_and_prepare_raw_frame(
    file_path: Path,
    top_ports: set[int],
    max_rows_per_file: int | None,
    benign_label: str,
) -> pd.DataFrame:
    frame = load_raw_frame(
        file_path=file_path,
        max_rows_per_file=max_rows_per_file,
        benign_label=benign_label,
    )
    return apply_port_bucket_features(frame, top_ports)


def choose_continuous_columns(
    df: pd.DataFrame,
    exclude: set[str],
    categorical_columns: list[str],
) -> list[str]:
    result: list[str] = []
    for column in df.columns:
        if column in exclude or column in categorical_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            result.append(column)
    return result


def _split_frame_random(
    df: pd.DataFrame,
    random_state: int,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target_binary"],
    )
    adjusted_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val,
        random_state=random_state,
        stratify=train_val_df["target_binary"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _allocate_contiguous_split_counts(
    group_size: int,
    test_size: float,
    val_size: float,
) -> tuple[int, int, int]:
    if group_size <= 0:
        return 0, 0, 0
    if group_size == 1:
        return 1, 0, 0

    remaining_eval_budget = group_size - 1
    test_rows = 0
    if test_size > 0 and remaining_eval_budget > 0:
        test_rows = int(round(group_size * float(test_size)))
        if group_size >= 3:
            test_rows = max(1, test_rows)
        test_rows = min(test_rows, remaining_eval_budget)
        remaining_eval_budget -= test_rows

    val_rows = 0
    if val_size > 0 and remaining_eval_budget > 0:
        val_rows = int(round(group_size * float(val_size)))
        if group_size - test_rows >= 3:
            val_rows = max(1, val_rows)
        val_rows = min(val_rows, remaining_eval_budget)
        remaining_eval_budget -= val_rows

    train_rows = group_size - val_rows - test_rows
    if train_rows <= 0:
        raise ValueError(
            "Contiguous split allocation must leave at least one row in train for every group."
        )
    return train_rows, val_rows, test_rows


def _split_frame_seen_family_no_leak(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    rare_family_train_only_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_columns = [
        column
        for column in (*SEEN_FAMILY_GROUP_COLUMNS, "target_binary", "row_id")
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "seen_family_no_leak split requires columns: "
            + ", ".join(sorted(missing_columns))
        )

    ordered_df = df.sort_values(
        [*SEEN_FAMILY_GROUP_COLUMNS, "row_id"],
        kind="mergesort",
    )
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group_df in ordered_df.groupby(list(SEEN_FAMILY_GROUP_COLUMNS), sort=True, dropna=False):
        group_df = group_df.sort_values("row_id", kind="mergesort")
        is_attack_group = bool(int(group_df["target_binary"].iloc[0]) == 1)
        train_only = is_attack_group and len(group_df) < int(rare_family_train_only_threshold)
        if train_only:
            train_parts.append(group_df)
            continue

        train_rows, val_rows, test_rows = _allocate_contiguous_split_counts(
            len(group_df),
            test_size=test_size,
            val_size=val_size,
        )
        train_parts.append(group_df.iloc[:train_rows])
        if val_rows > 0:
            val_parts.append(group_df.iloc[train_rows : train_rows + val_rows])
        if test_rows > 0:
            test_parts.append(group_df.iloc[train_rows + val_rows : train_rows + val_rows + test_rows])

    empty = df.iloc[0:0].copy()
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else empty.copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else empty.copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else empty.copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_frame(
    df: pd.DataFrame,
    random_state: int,
    test_size: float,
    val_size: float,
    split_strategy: str = "random",
    group_column: str | None = None,
    rare_family_train_only_threshold: int = DEFAULT_RARE_FAMILY_TRAIN_ONLY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = (split_strategy or "random").strip().lower()
    if strategy == "random":
        if group_column:
            raise ValueError(
                "split_group_column is only supported by a future non-random split strategy"
            )
        return _split_frame_random(df, random_state, test_size, val_size)
    if strategy == SEEN_FAMILY_NO_LEAK_SPLIT_STRATEGY:
        if group_column:
            raise ValueError(
                "split_group_column is not used by the seen_family_no_leak split strategy"
            )
        return _split_frame_seen_family_no_leak(
            df,
            test_size=test_size,
            val_size=val_size,
            rare_family_train_only_threshold=rare_family_train_only_threshold,
        )
    raise ValueError(f"Unsupported split_strategy: {split_strategy}")


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(destination, compression="gzip")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_pickle(path, compression="gzip")


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
