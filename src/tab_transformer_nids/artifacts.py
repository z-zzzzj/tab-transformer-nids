from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


STANDARD_ARTIFACTS = [
    "model.pt",
    "preprocessor.joblib",
    "feature_schema.json",
    "threshold.json",
    "metrics.json",
    "selection_metrics.json",
    "model_card.json",
]


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def package_artifacts(source_dir: str | Path, target_dir: str | Path, extra_files: list[str] | None = None) -> Path:
    source = Path(source_dir)
    target = Path(target_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for name in STANDARD_ARTIFACTS + (extra_files or []):
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)
    return target
