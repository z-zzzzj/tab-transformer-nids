from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from tab_transformer_nids.contracts import InferDisplayFields, InferenceResponse
from tab_transformer_nids.preprocessing import TabularPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class InferenceBundle:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.preprocessor = TabularPreprocessor.load(self.artifact_dir / "preprocessor.joblib")
        self.feature_schema = json.loads((self.artifact_dir / "feature_schema.json").read_text(encoding="utf-8"))
        self.threshold = json.loads((self.artifact_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]
        self.metrics = json.loads((self.artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        self.model_card = json.loads((self.artifact_dir / "model_card.json").read_text(encoding="utf-8"))
        self.model_type = self.model_card["model_type"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = 1.0
        self.model = self._load_model()

    def _load_model(self) -> Any:
        model_path = self.artifact_dir / "model.pt"
        if self.model_type == "isolation_forest":
            return joblib.load(model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        if self.model_type == "tab_transformer":
            from ml.models.tab_transformer_model import load_tab_transformer_checkpoint

            model, self.temperature = load_tab_transformer_checkpoint(
                checkpoint=checkpoint,
                categorical_cardinalities=self.feature_schema["categorical_cardinalities"],
                num_continuous=len(self.preprocessor.continuous_columns),
                device=self.device,
            )
            return model
        elif self.model_type == "lstm":
            from ml.models.lstm_baseline import FeatureSequenceLSTM

            model = FeatureSequenceLSTM(**checkpoint["model_config"])
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _transform_records(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        frame = pd.DataFrame(records)
        for column in self.preprocessor.categorical_columns + self.preprocessor.continuous_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        x_categ, x_cont = self.preprocessor.transform(frame)
        return x_categ, x_cont, frame

    def predict_records(self, records: list[dict[str, Any]]) -> list[InferenceResponse]:
        x_categ, x_cont, frame = self._transform_records(records)
        if self.model_type == "isolation_forest":
            features = np.concatenate([x_categ.astype(np.float32), x_cont], axis=1)
            raw_scores = (-self.model.decision_function(features)).astype(np.float32)
            confidences = 1.0 / (1.0 + np.exp(-raw_scores))
        elif self.model_type == "lstm":
            with torch.no_grad():
                tensor = torch.tensor(
                    np.concatenate([x_categ.astype(np.float32), x_cont], axis=1),
                    dtype=torch.float32,
                    device=self.device,
                )
                logits = self.model(tensor).squeeze(-1)
                confidences = torch.sigmoid(logits).cpu().numpy()
                raw_scores = logits.cpu().numpy()
        else:
            with torch.no_grad():
                logits = self.model(
                    torch.tensor(x_categ, dtype=torch.long, device=self.device),
                    torch.tensor(x_cont, dtype=torch.float32, device=self.device),
                ).squeeze(-1)
                if self.temperature != 1.0:
                    logits = logits / self.temperature
                confidences = torch.sigmoid(logits).cpu().numpy()
                raw_scores = logits.cpu().numpy()

        responses: list[InferenceResponse] = []
        for index, confidence in enumerate(confidences):
            prediction = "attack" if float(confidence) >= self.threshold else "benign"
            destination_port = None
            if "destination_port" in frame.columns and pd.notna(frame.iloc[index].get("destination_port")):
                destination_port = int(float(frame.iloc[index]["destination_port"]))
            original_label = frame.iloc[index].get("label_original")
            if pd.isna(original_label):
                original_label = None
            response = InferenceResponse(
                prediction=prediction,
                confidence=float(confidence),
                raw_score=float(raw_scores[index]),
                threshold=float(self.threshold),
                original_label=original_label,
                display_fields=InferDisplayFields(
                    protocol=str(frame.iloc[index].get("protocol")) if "protocol" in frame.columns else None,
                    destination_port=destination_port,
                ),
            )
            responses.append(response)
        return responses
