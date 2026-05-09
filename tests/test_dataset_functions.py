from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.scripts.build_dataset import build_dataset
from tab_transformer_nids.preprocessing import (
    choose_top_ports_from_frame,
    load_and_prepare_raw_frame,
    load_dataframe,
    scan_top_ports,
)


def test_scan_ports_and_prepare_frame(tmp_path: Path) -> None:
    csv_path = tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    frame = pd.DataFrame(
        {
            " Destination Port": [80, 443, 8080],
            " Flow Duration": [1, 2, 3],
            " Flow Packets/s": [10.0, 20.0, 30.0],
            "Flow Bytes/s": [100.0, 200.0, 300.0],
            " SYN Flag Count": [0, 1, 2],
            " RST Flag Count": [0, 0, 1],
            " Label": ["BENIGN", "DDoS", "BENIGN"],
        }
    )
    frame.to_csv(csv_path, index=False)

    top_ports = scan_top_ports([csv_path], max_rows_per_file=None, top_k=2)
    prepared = load_and_prepare_raw_frame(
        file_path=csv_path,
        top_ports=set(top_ports),
        max_rows_per_file=None,
        benign_label="BENIGN",
    )

    assert top_ports == [80, 443]
    assert "destination_port_bucket" in prepared.columns
    assert prepared["target_binary"].tolist() == [0, 1, 0]
    assert prepared["source_file"].iloc[0] == csv_path.name


def test_build_dataset_uses_train_only_port_stats_and_excludes_source_dataset(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    logs_dir = tmp_path / "logs"
    raw_dir.mkdir()

    csv_path = raw_dir / "Monday-WorkingHours.pcap_ISCX.csv"
    frame = pd.DataFrame(
        {
            " Destination Port": [80, 80, 80, 80, 80, 443, 443, 443, 443, 443, 443, 443],
            " Flow Duration": list(range(1, 13)),
            " Flow Packets/s": [10.0 + index for index in range(12)],
            "Flow Bytes/s": [100.0 + index for index in range(12)],
            " SYN Flag Count": [index % 3 for index in range(12)],
            " RST Flag Count": [index % 2 for index in range(12)],
            " Label": ["BENIGN", "DDoS"] * 6,
        }
    )
    frame.to_csv(csv_path, index=False)

    config = {
        "project": {
            "root_dir": str(tmp_path),
            "raw_data_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "reports_dir": str(tmp_path / "reports"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "logs_dir": str(logs_dir),
        },
        "dataset": {
            "random_state": 42,
            "test_size": 0.2,
            "val_size": 0.2,
            "split_strategy": "random",
            "split_group_column": None,
            "port_top_k": 1,
            "max_rows_per_file": None,
            "benign_label": "BENIGN",
            "drop_duplicate_rows": False,
            "strong_identifier_columns": [],
            "categorical_candidates": [
                "destination_port_bucket",
                "is_well_known_port",
                "source_dataset",
            ],
            "continuous_exclude": ["target_binary", "label_original", "source_file", "row_id"],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    build_dataset(str(config_path))

    feature_schema = json.loads(
        (processed_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    dataset_profile = json.loads(
        (processed_dir / "dataset_profile.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (processed_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    train_df = load_dataframe(processed_dir / "train.pkl.gz")

    assert "destination_port_bucket" in feature_schema["categorical_columns"]
    assert "is_well_known_port" in feature_schema["categorical_columns"]
    assert feature_schema["source_dataset_feature_enabled"] is False
    assert "source_dataset" not in feature_schema["categorical_columns"]
    assert "source_dataset" not in feature_schema["continuous_columns"]
    assert split_manifest["split_strategy"] == "random"
    assert split_manifest["dedupe_scope"] == "before_split"
    assert split_manifest["dedupe_enabled"] is False

    full_top_ports = scan_top_ports([csv_path], max_rows_per_file=None, top_k=1)
    train_top_ports = choose_top_ports_from_frame(train_df, top_k=1)

    assert full_top_ports == [443]
    assert train_top_ports == [80]
    assert dataset_profile["top_ports"] == [80]
    assert dataset_profile["top_ports_source_split"] == "train"
    assert dataset_profile["top_ports"] != full_top_ports
    assert train_df.loc[train_df["destination_port"] == 443, "destination_port_bucket"].eq(
        "OTHER"
    ).all()


def test_build_dataset_deduplicates_before_random_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    logs_dir = tmp_path / "logs"
    raw_dir.mkdir()

    rows = []
    for index in range(4):
        rows.append(
            {
                " Destination Port": 80 + index,
                " Flow Duration": 1 + index,
                " Flow Packets/s": 10.0 + index,
                "Flow Bytes/s": 100.0 + index,
                " SYN Flag Count": 0,
                " RST Flag Count": 0,
                " Label": "BENIGN",
            }
        )
    for index in range(4):
        rows.append(
            {
                " Destination Port": 443 + index,
                " Flow Duration": 10 + index,
                " Flow Packets/s": 20.0 + index,
                "Flow Bytes/s": 200.0 + index,
                " SYN Flag Count": 1,
                " RST Flag Count": 0,
                " Label": "DDoS",
            }
        )
    pd.DataFrame(rows).to_csv(
        raw_dir / "Monday-WorkingHours.pcap_ISCX.csv",
        index=False,
    )
    pd.DataFrame(rows).to_csv(
        raw_dir / "Tuesday-WorkingHours.pcap_ISCX.csv",
        index=False,
    )

    config = {
        "project": {
            "root_dir": str(tmp_path),
            "raw_data_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "reports_dir": str(tmp_path / "reports"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "logs_dir": str(logs_dir),
        },
        "dataset": {
            "random_state": 42,
            "test_size": 0.25,
            "val_size": 0.25,
            "split_strategy": "random",
            "split_group_column": None,
            "port_top_k": 4,
            "max_rows_per_file": None,
            "benign_label": "BENIGN",
            "drop_duplicate_rows": True,
            "strong_identifier_columns": [],
            "categorical_candidates": ["destination_port_bucket", "is_well_known_port"],
            "continuous_exclude": ["target_binary", "label_original", "source_file", "row_id"],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    build_dataset(str(config_path))

    split_manifest = json.loads((processed_dir / "split_manifest.json").read_text(encoding="utf-8"))
    combined = pd.concat(
        [
            load_dataframe(processed_dir / "train.pkl.gz"),
            load_dataframe(processed_dir / "val.pkl.gz"),
            load_dataframe(processed_dir / "test.pkl.gz"),
        ],
        ignore_index=True,
    )

    assert split_manifest["dedupe_enabled"] is True
    assert split_manifest["dedupe_scope"] == "before_split"
    assert split_manifest["dedupe_rows_removed"] == 8
    assert len(combined) == 8
    assert combined["target_binary"].tolist().count(0) == 4
    assert combined["target_binary"].tolist().count(1) == 4

