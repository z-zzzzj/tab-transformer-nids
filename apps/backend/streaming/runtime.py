from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from tab_transformer_nids.contracts import AlertEvent, InferenceResponse
from tab_transformer_nids.inference import InferenceBundle


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
        if math.isfinite(num):
            return num
    except (TypeError, ValueError):
        return None
    return None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


class EventBuffer:
    def __init__(self, maxlen: int = 5000) -> None:
        self._items: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._cursor = 0
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> int:
        with self._lock:
            event_copy = dict(event)
            event_copy["_cursor"] = self._cursor
            self._items.append(event_copy)
            self._cursor += 1
            return self._cursor

    def list_since(self, since: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            events = [item for item in self._items if int(item.get("_cursor", -1)) >= since]
            clipped = events[:limit]
            next_cursor = since if not clipped else int(clipped[-1]["_cursor"]) + 1
            cleaned = [{k: v for k, v in item.items() if k != "_cursor"} for item in clipped]
            return cleaned, next_cursor


class ModelRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.threshold = 0.5
        self.model_loaded = False
        self.model_name = "fallback-heuristic"
        self.bundle: InferenceBundle | None = None
        self.artifact_dir = self._resolve_artifact_dir()
        self._load_threshold()
        self._load_bundle()

    def _resolve_artifact_dir(self) -> Path:
        env_dir = os.getenv("NIDS_ARTIFACT_DIR")
        if env_dir:
            return Path(env_dir)
        current = Path(__file__).resolve()
        project_root = current.parents[3]
        for candidate in (
            project_root / "artifacts" / "current",
            project_root / "artifacts_smoke" / "current",
        ):
            if candidate.exists():
                return candidate
        return project_root / "artifacts"

    def _load_threshold(self) -> None:
        threshold_path = self.artifact_dir / "threshold.json"
        if not threshold_path.exists():
            return
        try:
            data = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold = float(data.get("threshold", 0.5))
            if 0 <= threshold <= 1:
                self.threshold = threshold
        except Exception:
            pass

    def _load_bundle(self) -> None:
        required = [
            self.artifact_dir / "model.pt",
            self.artifact_dir / "preprocessor.joblib",
            self.artifact_dir / "feature_schema.json",
            self.artifact_dir / "model_card.json",
            self.artifact_dir / "threshold.json",
        ]
        if not all(path.exists() for path in required):
            return
        try:
            self.bundle = InferenceBundle(self.artifact_dir)
            self.model_loaded = True
            self.model_name = self.bundle.model_card.get("model_type", "trained-model")
            self.threshold = float(self.bundle.threshold)
        except Exception:
            self.bundle = None
            self.model_loaded = False
            self.model_name = "fallback-heuristic"

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.model_loaded,
            "name": self.model_name,
            "threshold": self.threshold,
            "artifact_dir": str(self.artifact_dir),
            "mode": "trained" if self.model_loaded else "fallback",
        }

    def infer_payload(self, payload: dict[str, Any]) -> InferenceResponse:
        return self.infer_payloads([payload])[0]

    def infer_payloads(self, payloads: list[dict[str, Any]]) -> list[InferenceResponse]:
        with self._lock:
            if self.bundle is not None:
                return self.bundle.predict_records(payloads)
            return [self._heuristic_response(payload) for payload in payloads]

    def _heuristic_response(self, payload: dict[str, Any]) -> InferenceResponse:
        score = self._heuristic_score(payload)
        prediction = "attack" if score >= self.threshold else "benign"
        original_label = payload.get("label_original") or payload.get("Label")
        protocol = payload.get("Protocol") or payload.get("protocol")
        dst_port = payload.get("Destination Port") or payload.get("destination_port")
        return InferenceResponse(
            prediction=prediction,
            confidence=float(score if prediction == "attack" else 1 - score),
            raw_score=float(score),
            threshold=float(self.threshold),
            original_label=str(original_label) if original_label is not None else None,
            display_fields={
                "protocol": str(protocol) if protocol is not None else None,
                "destination_port": dst_port,
            },
        )

    def _heuristic_score(self, payload: dict[str, Any]) -> float:
        keys = {
            "flow_packets_s": ["Flow Packets/s", "flow_packets_s", "flow_packets_per_s"],
            "flow_bytes_s": ["Flow Bytes/s", "flow_bytes_s", "flow_bytes_per_s"],
            "syn_count": [" SYN Flag Count", "SYN Flag Count", "syn_flag_count"],
            "rst_count": [" RST Flag Count", "RST Flag Count", "rst_flag_count"],
        }
        values: dict[str, float] = {}
        for out_key, candidates in keys.items():
            found = None
            for key in candidates:
                found = _safe_float(payload.get(key))
                if found is not None:
                    break
            values[out_key] = found or 0.0

        dst_port = _safe_float(payload.get("Destination Port") or payload.get("destination_port")) or 0.0
        port_boost = 0.0 if dst_port in {80, 443, 53} else 0.15
        linear = (
            0.000002 * values["flow_packets_s"]
            + 0.0000002 * values["flow_bytes_s"]
            + 0.02 * values["syn_count"]
            + 0.03 * values["rst_count"]
            + port_boost
            - 1.5
        )
        score = 1.0 / (1.0 + math.exp(-linear))
        return max(0.0, min(1.0, score))


@dataclass
class ReplayController:
    model_runtime: ModelRuntime
    event_buffer: EventBuffer
    running: bool = False
    rate_per_second: float = 24.0
    batch_size: int = 48
    emitted: int = 0
    last_error: str | None = None
    _thread: threading.Thread | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _records_cache: list[dict[str, Any]] | None = field(default=None, init=False, repr=False)
    _records_cache_path: Path | None = field(default=None, init=False, repr=False)
    _records_cache_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        threading.Thread(
            target=self._warm_replay_cache,
            daemon=True,
            name="replay-cache-warmup",
        ).start()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "rate_per_second": self.rate_per_second,
            "batch_size": self.batch_size,
            "events_emitted": self.emitted,
            "last_error": self.last_error,
        }

    def start(self, options: dict[str, Any]) -> tuple[bool, str]:
        with self._lock:
            if self.running:
                return False, "Replay already running"
            self.rate_per_second = max(1.0, float(options.get("rate_per_second", self.rate_per_second)))
            self.batch_size = max(1, int(options.get("batch_size", self.batch_size)))
            max_events = int(options.get("max_events", 1000))
            source_files = options.get("source_files")
            self._stop_event.clear()
            self.last_error = None
            self.emitted = 0
            self._thread = threading.Thread(
                target=self._run,
                args=(max_events, source_files),
                daemon=True,
                name="replay-worker",
            )
            self.running = True
            self._thread.start()
            return True, "Replay started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self.running:
                return False, "Replay not running"
            self._stop_event.set()
            self.running = False
            return True, "Replay stopping"

    def _run(self, max_events: int, source_files: list[str] | None) -> None:
        try:
            batch: list[dict[str, Any]] = []
            for record in self._record_stream(source_files):
                if self._stop_event.is_set() or self.emitted + len(batch) >= max_events:
                    break
                batch.append(record)
                if len(batch) >= self.batch_size:
                    self._emit_batch(batch)
                    batch = []
            if batch and not self._stop_event.is_set():
                self._emit_batch(batch)
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            with self._lock:
                self.running = False

    def _emit_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        started = time.perf_counter()
        responses = self.model_runtime.infer_payloads(records)
        for record, response in zip(records, responses, strict=False):
            if self._stop_event.is_set():
                break
            event = self._to_alert_event(response=response, payload=record)
            self.event_buffer.append(event)
            self.emitted += 1
            self._push_websocket(event)

        target_duration = len(records) / max(self.rate_per_second, 1.0)
        remaining = target_duration - (time.perf_counter() - started)
        if remaining > 0 and not self._stop_event.is_set():
            time.sleep(remaining)

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _resolve_processed_pickle(self) -> Path:
        project_root = self._project_root()
        return Path(os.getenv("NIDS_REPLAY_PICKLE", project_root / "ml" / "data" / "processed" / "test.pkl.gz"))

    def _warm_replay_cache(self) -> None:
        processed_pickle = self._resolve_processed_pickle()
        if not processed_pickle.exists():
            return
        try:
            self._load_processed_records(processed_pickle)
        except Exception:
            return

    def _load_processed_records(self, processed_pickle: Path) -> list[dict[str, Any]]:
        with self._records_cache_lock:
            if self._records_cache is not None and self._records_cache_path == processed_pickle:
                return self._records_cache

            frame = pd.read_pickle(processed_pickle, compression="gzip")
            records = [
                {
                    key: (None if pd.isna(value) else _json_safe(value))
                    for key, value in record.items()
                }
                for record in frame.to_dict(orient="records")
            ]
            self._records_cache = records
            self._records_cache_path = processed_pickle
            return records

    def _record_stream(self, source_files: list[str] | None) -> Any:
        project_root = self._project_root()
        processed_pickle = self._resolve_processed_pickle()
        if processed_pickle.exists():
            records = self._load_processed_records(processed_pickle)
            if source_files:
                source_set = set(source_files)
                records = [record for record in records if record.get("source_file") in source_set]
            for record in records:
                yield record
            return

        env_raw = os.getenv("NIDS_REPLAY_DATA_DIR")
        if env_raw:
            raw_dir = Path(env_raw)
        else:
            candidate_a = project_root / "data" / "raw"
            candidate_b = project_root.parent / "data" / "raw"
            raw_dir = candidate_a if candidate_a.exists() else candidate_b
        if source_files:
            files = [raw_dir / name for name in source_files]
        else:
            files = sorted(raw_dir.glob("*.csv"))
        columns = [
            " Label",
            " Destination Port",
            " Flow Packets/s",
            "Flow Bytes/s",
            " SYN Flag Count",
            " RST Flag Count",
            " Protocol",
        ]
        for csv_path in files:
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    slim = {key: row.get(key) for key in columns if key in row}
                    slim["source_file"] = csv_path.name
                    slim["label_original"] = slim.get(" Label")
                    yield slim

    def _to_alert_event(self, response: InferenceResponse, payload: dict[str, Any]) -> dict[str, Any]:
        if response.prediction == "benign":
            severity = "info"
        elif response.confidence >= 0.9:
            severity = "critical"
        else:
            severity = "warning"
        event = AlertEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            prediction=response.prediction,
            confidence=response.confidence,
            severity=severity,
            original_label=response.original_label,
            raw_score=response.raw_score,
            threshold=response.threshold,
            display_fields=(
                response.display_fields.model_dump()
                if hasattr(response.display_fields, "model_dump")
                else response.display_fields
            ),
            payload=payload,
        )
        return event.model_dump()

    def _push_websocket(self, event: dict[str, Any]) -> None:
        channel_layer = get_channel_layer()
        if channel_layer is None:
            return
        async_to_sync(channel_layer.group_send)(
            "alerts",
            {
                "type": "alert.event",
                "event": event,
            },
        )


@dataclass
class RuntimeContainer:
    model_runtime: ModelRuntime
    event_buffer: EventBuffer
    replay_controller: ReplayController


_runtime: RuntimeContainer | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> RuntimeContainer:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        model_runtime = ModelRuntime()
        event_buffer = EventBuffer(maxlen=10000)
        replay_controller = ReplayController(model_runtime=model_runtime, event_buffer=event_buffer)
        _runtime = RuntimeContainer(
            model_runtime=model_runtime,
            event_buffer=event_buffer,
            replay_controller=replay_controller,
        )
        return _runtime
