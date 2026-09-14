"""
Confirms the real, committed model artifacts (backend/app/ml/weights/) load
correctly and produce outputs of the expected shape/range on a fixed sample
image -- not a mock, the actual trained checkpoint from ml_training/train.py.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.classical_features import FEATURE_NAMES  # noqa: E402
from app.ml.cnn_model import ISSUE_HEADS  # noqa: E402
from app.ml.inference import LABEL_ACCEPTABLE, LABEL_DEFECTIVE, LABEL_DEGRADED  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLEAN_IMAGE = REPO_ROOT / "sample_images" / "clean" / "I15.png"
SAMPLE_CORRUPTED_IMAGE = REPO_ROOT / "sample_images" / "corrupted" / "I15_10_05.png"

VALID_LABELS = {LABEL_ACCEPTABLE, LABEL_DEGRADED, LABEL_DEFECTIVE}
VALID_SEVERITIES = {"low", "medium", "high"}


class TestModelLoading:
    def test_engine_reports_loaded(self, _loaded_inference_engine):
        assert _loaded_inference_engine.is_loaded
        assert _loaded_inference_engine.model is not None
        assert _loaded_inference_engine.scaler is not None
        assert _loaded_inference_engine.anomaly_detector is not None

    def test_model_is_on_cpu(self, _loaded_inference_engine):
        assert _loaded_inference_engine.device.type == "cpu"


class TestAnalyzeImageOutputShape:
    def test_clean_sample_image(self, _loaded_inference_engine):
        image = Image.open(SAMPLE_CLEAN_IMAGE)
        result = _loaded_inference_engine.analyze_image(image)

        assert set(result.keys()) == {
            "quality_score",
            "quality_label",
            "issues",
            "image_stats",
            "gradcam_available",
        }
        assert 0.0 <= result["quality_score"] <= 100.0
        assert result["quality_label"] in VALID_LABELS
        assert result["gradcam_available"] is True

        assert isinstance(result["issues"], list)
        for issue in result["issues"]:
            assert set(issue.keys()) == {"type", "severity", "confidence"}
            assert issue["severity"] in VALID_SEVERITIES
            assert 0.0 <= issue["confidence"] <= 1.0

        assert set(result["image_stats"].keys()) == set(FEATURE_NAMES)
        for value in result["image_stats"].values():
            assert isinstance(value, float)

    def test_corrupted_sample_image_scores_lower_than_clean(self, _loaded_inference_engine):
        clean_result = _loaded_inference_engine.analyze_image(Image.open(SAMPLE_CLEAN_IMAGE))
        corrupted_result = _loaded_inference_engine.analyze_image(Image.open(SAMPLE_CORRUPTED_IMAGE))
        assert corrupted_result["quality_score"] < clean_result["quality_score"]

    def test_issue_types_are_known_heads_or_composite_defect(self, _loaded_inference_engine):
        result = _loaded_inference_engine.analyze_image(Image.open(SAMPLE_CORRUPTED_IMAGE))
        known_types = set(ISSUE_HEADS) | {"potential_defect"}
        for issue in result["issues"]:
            assert issue["type"] in known_types
