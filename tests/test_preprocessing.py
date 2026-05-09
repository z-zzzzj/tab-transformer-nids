from __future__ import annotations

import numpy as np
import pandas as pd

from tab_transformer_nids.preprocessing import (
    TabularPreprocessor,
    normalize_column_name,
    normalize_label,
    semantic_deduplicate_frame,
)


def test_normalize_helpers() -> None:
    assert normalize_column_name(" Flow Bytes/s ") == "flow_bytes_per_s"
    assert normalize_label("Web Attack \ufffd XSS") == "Web Attack - XSS"


def test_preprocessor_unknown_category_round_trip(tmp_path) -> None:
    train_df = pd.DataFrame(
        {
            "destination_port_bucket": ["80", "443", "OTHER"],
            "is_well_known_port": ["true", "true", "false"],
            "flow_duration": [1.0, 2.0, 3.0],
            "target_binary": [0, 1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "destination_port_bucket": ["9999"],
            "is_well_known_port": ["true"],
            "flow_duration": [2.5],
            "target_binary": [1],
        }
    )

    preprocessor = TabularPreprocessor(
        categorical_columns=["destination_port_bucket", "is_well_known_port"],
        continuous_columns=["flow_duration"],
    )
    x_categ_train, x_cont_train = preprocessor.fit_transform(train_df)
    x_categ_test, x_cont_test = preprocessor.transform(test_df)

    assert x_categ_train.shape == (3, 2)
    assert x_cont_train.shape == (3, 1)
    assert x_categ_test[0, 0] == -1
    assert np.isfinite(x_cont_test).all()

    path = tmp_path / "preprocessor.joblib"
    preprocessor.save(path)
    loaded = TabularPreprocessor.load(path)
    loaded_categ, loaded_cont = loaded.transform(test_df)
    assert loaded_categ.tolist() == x_categ_test.tolist()
    assert np.allclose(loaded_cont, x_cont_test)


def test_semantic_deduplicate_frame_ignores_source_columns_but_keeps_label_columns() -> None:
    frame = pd.DataFrame(
        {
            "source_file": ["train.csv", "test.csv", "test.csv"],
            "source_dataset": ["monday", "friday", "friday"],
            "row_id": [0, 99, 100],
            "destination_port": [80, 80, 80],
            "flow_duration": [1.0, 1.0, 1.0],
            "label_original": ["BENIGN", "BENIGN", "DDoS"],
            "target_binary": [0, 0, 1],
        }
    )

    deduped, dedupe_columns, removed_rows = semantic_deduplicate_frame(frame)

    assert removed_rows == 1
    assert "source_file" not in dedupe_columns
    assert "source_dataset" not in dedupe_columns
    assert "row_id" not in dedupe_columns
    assert "label_original" in dedupe_columns
    assert "target_binary" in dedupe_columns
    assert deduped["label_original"].tolist() == ["BENIGN", "DDoS"]
