from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tab_transformer_nids.preprocessing import (
    ENGINEERED_CATEGORICAL_COLUMNS,
    TabularPreprocessor,
    augment_behavioral_features,
    apply_port_bucket_features,
    choose_continuous_columns,
    choose_top_ports_from_frame,
    load_raw_frame,
    resolve_semantic_dedupe_columns,
    save_dataframe,
    semantic_deduplicate_frame,
    split_frame,
    write_json,
)
from tab_transformer_nids.settings import ensure_dir, load_config


def build_dataset(config_path: str) -> None:
    config = load_config(config_path)
    project = config["project"]
    dataset_cfg = config["dataset"]
    raw_dir = Path(project["raw_data_dir"])
    processed_dir = ensure_dir(project["processed_dir"])
    split_strategy = dataset_cfg.get("split_strategy", "random")
    split_group_column = dataset_cfg.get("split_group_column")
    include_source_dataset_feature = bool(
        dataset_cfg.get("include_source_dataset_feature", False)
    )

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    benign_label = dataset_cfg["benign_label"]
    for file_path in csv_files:
        frame = load_raw_frame(
            file_path=file_path,
            max_rows_per_file=dataset_cfg.get("max_rows_per_file"),
            benign_label=benign_label,
        )
        frames.append(frame)
    dataset_df = pd.concat(frames, ignore_index=True)

    drop_identifier_columns = [
        column
        for column in dataset_cfg.get("strong_identifier_columns", [])
        if column in dataset_df.columns
    ]
    if drop_identifier_columns:
        dataset_df = dataset_df.drop(columns=drop_identifier_columns)

    if dataset_cfg.get("drop_duplicate_rows", True):
        dataset_df, dedupe_columns, dedupe_rows_removed = semantic_deduplicate_frame(
            dataset_df,
            strong_identifier_columns=drop_identifier_columns,
        )
    else:
        dedupe_columns = resolve_semantic_dedupe_columns(
            dataset_df,
            strong_identifier_columns=drop_identifier_columns,
        )
        dedupe_rows_removed = 0

    train_df, val_df, test_df = split_frame(
        dataset_df,
        random_state=dataset_cfg["random_state"],
        test_size=dataset_cfg["test_size"],
        val_size=dataset_cfg["val_size"],
        split_strategy=split_strategy,
        group_column=split_group_column,
        rare_family_train_only_threshold=dataset_cfg.get("rare_family_train_only_threshold", 100),
    )

    top_ports = choose_top_ports_from_frame(train_df, dataset_cfg["port_top_k"])
    top_ports_set = set(top_ports)
    train_df = apply_port_bucket_features(train_df, top_ports_set)
    val_df = apply_port_bucket_features(val_df, top_ports_set)
    test_df = apply_port_bucket_features(test_df, top_ports_set)
    train_df = augment_behavioral_features(train_df)
    val_df = augment_behavioral_features(val_df)
    test_df = augment_behavioral_features(test_df)

    categorical_candidates = list(dataset_cfg["categorical_candidates"]) + list(
        ENGINEERED_CATEGORICAL_COLUMNS
    )
    if not include_source_dataset_feature:
        categorical_candidates = [
            column for column in categorical_candidates if column != "source_dataset"
        ]
    categorical_columns = list(
        dict.fromkeys(column for column in categorical_candidates if column in train_df.columns)
    )
    exclude = set(dataset_cfg["continuous_exclude"]) | {"label"}
    if not include_source_dataset_feature:
        exclude.add("source_dataset")
    continuous_candidates = choose_continuous_columns(train_df, exclude, categorical_columns)
    low_cardinality_limit = int(dataset_cfg.get("low_cardinality_categorical_max_unique", 16))
    promoted_low_cardinality_numeric_columns: list[str] = []
    dropped_constant_columns: list[str] = []
    for column in continuous_candidates:
        series = train_df[column].replace([float("inf"), float("-inf")], pd.NA)
        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 1:
            dropped_constant_columns.append(column)
        elif unique_count <= low_cardinality_limit:
            promoted_low_cardinality_numeric_columns.append(column)
    categorical_columns = list(
        dict.fromkeys(categorical_columns + promoted_low_cardinality_numeric_columns)
    )
    continuous_columns = [
        column
        for column in continuous_candidates
        if column not in promoted_low_cardinality_numeric_columns
        and column not in dropped_constant_columns
    ]

    preprocessor = TabularPreprocessor(categorical_columns, continuous_columns)
    preprocessor.fit(train_df)
    preprocessor.save(processed_dir / "preprocessor.joblib")

    save_dataframe(train_df, processed_dir / "train.pkl.gz")
    save_dataframe(val_df, processed_dir / "val.pkl.gz")
    save_dataframe(test_df, processed_dir / "test.pkl.gz")

    label_map = {"BENIGN": 0, "ATTACK": 1}
    split_manifest = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "random_state": dataset_cfg["random_state"],
        "test_size": dataset_cfg["test_size"],
        "val_size": dataset_cfg["val_size"],
        "split_strategy": split_strategy,
        "dedupe_enabled": bool(dataset_cfg.get("drop_duplicate_rows", True)),
        "dedupe_scope": "before_split",
        "dedupe_columns": dedupe_columns,
        "dedupe_rows_removed": dedupe_rows_removed,
        "strong_identifier_columns_removed": drop_identifier_columns,
    }
    if split_group_column:
        split_manifest["split_group_column"] = split_group_column
    dataset_profile = {
        "raw_dir": str(raw_dir),
        "source_files": [path.name for path in csv_files],
        "top_ports": top_ports,
        "top_ports_source_split": "train",
        "total_rows": int(len(dataset_df)),
        "target_distribution": {
            "benign": int((dataset_df["target_binary"] == 0).sum()),
            "attack": int((dataset_df["target_binary"] == 1).sum()),
        },
        "dedupe": {
            "enabled": bool(dataset_cfg.get("drop_duplicate_rows", True)),
            "scope": "before_split",
            "columns": dedupe_columns,
            "rows_removed": dedupe_rows_removed,
            "strong_identifier_columns_removed": drop_identifier_columns,
        },
        "label_original_counts": {
            key: int(value)
            for key, value in dataset_df["label_original"].value_counts().to_dict().items()
        },
    }
    feature_schema = {
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "categorical_cardinalities": preprocessor.categorical_cardinalities,
        "category_maps": preprocessor.category_maps,
        "continuous_mean_std": preprocessor.continuous_mean_std_tensor().tolist(),
        "source_dataset_feature_enabled": include_source_dataset_feature,
        "engineered_categorical_columns": ENGINEERED_CATEGORICAL_COLUMNS,
        "promoted_low_cardinality_numeric_columns": promoted_low_cardinality_numeric_columns,
        "dropped_constant_columns": dropped_constant_columns,
        "processed_files": {
            "train": str(processed_dir / "train.pkl.gz"),
            "val": str(processed_dir / "val.pkl.gz"),
            "test": str(processed_dir / "test.pkl.gz"),
        },
    }

    write_json(feature_schema, processed_dir / "feature_schema.json")
    write_json(label_map, processed_dir / "label_map.json")
    write_json(split_manifest, processed_dir / "split_manifest.json")
    write_json(dataset_profile, processed_dir / "dataset_profile.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_dataset(args.config)


if __name__ == "__main__":
    main()
