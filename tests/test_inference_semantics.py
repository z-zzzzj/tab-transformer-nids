from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from streaming.runtime import ModelRuntime
from tab_transformer_nids.inference import InferenceBundle


class FakeIsolationForest:
    def __init__(self, attack_probability: float) -> None:
        self.raw_score = math.log(attack_probability / (1.0 - attack_probability))

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], -self.raw_score, dtype=np.float32)


def test_bundle_confidence_is_predicted_class_probability_for_benign() -> None:
    bundle = object.__new__(InferenceBundle)
    bundle.model_type = "isolation_forest"
    bundle.threshold = 0.5
    bundle.model = FakeIsolationForest(attack_probability=0.2)
    bundle._transform_records = lambda records: (
        np.zeros((len(records), 0), dtype=np.int64),
        np.zeros((len(records), 0), dtype=np.float32),
        pd.DataFrame(records),
    )

    response = bundle.predict_records([{"destination_port": 80, "label_original": "BENIGN"}])[0]

    assert response.prediction == "benign"
    assert response.attack_probability == pytest.approx(0.2, abs=1e-6)
    assert response.confidence == pytest.approx(0.8, abs=1e-6)
    assert response.raw_score == pytest.approx(math.log(0.2 / 0.8), abs=1e-6)


def test_fallback_confidence_keeps_predicted_class_probability() -> None:
    runtime = object.__new__(ModelRuntime)
    runtime.threshold = 0.5

    response = runtime._heuristic_response(
        {
            "destination_port": 80,
            "flow_packets_per_s": 0,
            "flow_bytes_per_s": 0,
            "syn_flag_count": 0,
            "rst_flag_count": 0,
        }
    )

    assert response.prediction == "benign"
    assert response.attack_probability == pytest.approx(response.raw_score)
    assert response.confidence == pytest.approx(1.0 - response.attack_probability)
