from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class InferDisplayFields(BaseModel):
    protocol: str | None = None
    destination_port: int | float | str | None = None


class InferenceResponse(BaseModel):
    prediction: Literal["attack", "benign"]
    confidence: float
    attack_probability: float | None = None
    raw_score: float
    threshold: float
    original_label: str | None = None
    display_fields: InferDisplayFields | dict[str, Any] = Field(default_factory=InferDisplayFields)


class AlertEvent(BaseModel):
    event_id: str
    timestamp: str
    prediction: Literal["attack", "benign"]
    confidence: float
    attack_probability: float | None = None
    severity: Literal["info", "warning", "critical"]
    original_label: str | None = None
    raw_score: float
    threshold: float
    display_fields: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
