"""
Model versioning. train.py writes weights/model_metadata.json after a
training run completes; inference.py reads it at startup and threads
`model_version` through every analysis result, so a stored analysis always
records which model produced it (useful once weights get retrained and
old/new results need distinguishing -- e.g. this project's v1.0.0 -> v1.1.0
retrain to fix a LAMBDA_QUALITY-driven generalization issue, see
ml_training/train.py's docstring).
"""

import json
from pathlib import Path
from typing import Optional, TypedDict

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
METADATA_PATH = WEIGHTS_DIR / "model_metadata.json"

UNKNOWN_VERSION = "unknown"


class ModelMetadata(TypedDict):
    version: str
    trained_at: str
    val_macro_f1: float
    lambda_quality: float


def write_metadata(metadata: ModelMetadata, path: Path = METADATA_PATH) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_version(path: Path = METADATA_PATH) -> str:
    """Returns the model version string, or UNKNOWN_VERSION if no metadata file exists yet."""
    if not path.exists():
        return UNKNOWN_VERSION
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("version", UNKNOWN_VERSION)
    except (json.JSONDecodeError, OSError):
        return UNKNOWN_VERSION


def read_metadata(path: Path = METADATA_PATH) -> Optional[ModelMetadata]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
