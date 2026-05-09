from __future__ import annotations

import argparse
import json
from pathlib import Path

from tab_transformer_nids.artifacts import package_artifacts, write_json
from tab_transformer_nids.settings import ensure_dir, load_config


def package_model(config_path: str, model_name: str) -> None:
    config = load_config(config_path)
    reports_root = Path(config["project"]["reports_dir"]) / model_name / "latest"
    artifacts_root = ensure_dir(Path(config["project"]["artifacts_dir"]) / "current")
    packaged = package_artifacts(reports_root, artifacts_root)
    write_json(
        {
            "active_model": model_name,
            "source_experiment_dir": str(reports_root),
            "artifact_dir": str(packaged),
        },
        artifacts_root / "manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["tab_transformer", "mlp", "embedding_mlp", "lstm", "isolation_forest"],
    )
    args = parser.parse_args()
    package_model(args.config, args.model)


if __name__ == "__main__":
    main()
