from __future__ import annotations

import json
import time

import pandas as pd
import pytest
from django.test import Client


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("NIDS_ARTIFACT_DIR", str(tmp_path / "missing_artifacts"))
    replay_path = tmp_path / "test.pkl.gz"
    frame = pd.DataFrame(
        [
            {
                "label_original": "DDoS",
                "destination_port": 8080,
                "flow_packets_per_s": 50000.0,
                "flow_bytes_per_s": 600000.0,
                "syn_flag_count": 8,
                "rst_flag_count": 1,
                "destination_port_bucket": "OTHER",
                "is_well_known_port": "false",
                "source_dataset": "friday_workinghours_afternoon_ddos",
            }
        ]
    )
    frame.to_pickle(replay_path, compression="gzip")
    monkeypatch.setenv("NIDS_REPLAY_PICKLE", str(replay_path))

    import streaming.runtime as runtime_module

    runtime_module._runtime = None
    yield Client()
    runtime_module._runtime = None


def test_health_endpoint(client: Client) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["model_loaded"] is False


def test_infer_endpoint(client: Client) -> None:
    payload = {
        "destination_port": 8080,
        "flow_packets_per_s": 75000.0,
        "flow_bytes_per_s": 900000.0,
        "syn_flag_count": 10,
        "rst_flag_count": 2,
        "label_original": "DDoS",
    }
    response = client.post(
        "/api/v1/infer",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"attack", "benign"}
    assert "confidence" in body
    assert body["original_label"] == "DDoS"


def test_replay_endpoints(client: Client) -> None:
    start = client.post(
        "/api/v1/replay/start",
        data=json.dumps({"rate_per_second": 100, "max_events": 1}),
        content_type="application/json",
    )
    assert start.status_code == 200

    time.sleep(0.1)
    events = client.get("/api/v1/replay/events?since=0&limit=10")
    assert events.status_code == 200
    payload = events.json()
    assert payload["ok"] is True
    assert len(payload["events"]) >= 1

