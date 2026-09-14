"""
Singleton-style inference wrapper. One InferenceEngine is created and
.load()ed once, in main.py's lifespan handler at process startup -- never
per-request. Exposes analyze_image(pil_image) -> dict, returning everything
needed to build an AnalyzeResponse.
"""

from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from app.core.config import get_settings
from app.ml.anomaly import AnomalyDetector
from app.ml.classical_features import FEATURE_NAMES, extract_feature_vector
from app.ml.cnn_model import ISSUE_HEADS, QUALITY_SCORE_HEAD, HybridQualityModel

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# quality_label thresholds (0-100 scale, higher = better -- see
# ml_training/data_gen/build_labels_from_kadid.py for the same scale on the
# training side). Matches the assessment brief's example response shape
# ({"quality_score": 82, "quality_label": "ACCEPTABLE"}).
QUALITY_LABEL_ACCEPTABLE_MIN = 70.0
QUALITY_LABEL_DEGRADED_MIN = 40.0
LABEL_ACCEPTABLE = "ACCEPTABLE"
LABEL_DEGRADED = "DEGRADED"
LABEL_DEFECTIVE = "DEFECTIVE"

# A head's probability must clear this to be reported as a detected issue
# at all (matches the 0.5 threshold evaluate.py reports metrics at).
ISSUE_PROBABILITY_THRESHOLD = 0.5
# Severity bands applied to a head's probability once it has cleared
# ISSUE_PROBABILITY_THRESHOLD.
SEVERITY_LOW_MAX = 0.65
SEVERITY_MEDIUM_MAX = 0.85

# Composite "potential visual defect" logic, per BUILD_SPEC.md §1.5: high
# corruption probability OR the Isolation Forest flagging the image as
# anomalous relative to the clean-image training distribution. Not a 7th
# model head -- computed here from the other two signals.
DEFECT_CORRUPTION_THRESHOLD = 0.5
DEFECT_ISSUE_TYPE = "potential_defect"


def _severity_for_probability(prob: float) -> str:
    if prob < SEVERITY_LOW_MAX:
        return "low"
    if prob < SEVERITY_MEDIUM_MAX:
        return "medium"
    return "high"


def _quality_label(score: float) -> str:
    if score >= QUALITY_LABEL_ACCEPTABLE_MIN:
        return LABEL_ACCEPTABLE
    if score >= QUALITY_LABEL_DEGRADED_MIN:
        return LABEL_DEGRADED
    return LABEL_DEFECTIVE


def _anomaly_confidence(anomaly_score: float) -> float:
    """
    IsolationForest's decision_function isn't a probability -- it's centered
    at 0 (the fitted inlier/outlier boundary), more negative = more
    anomalous. Linearly squashed onto [0, 1] for consistency with the other
    heads' sigmoid-confidence fields: 0.5 at the boundary, -> 1.0 for
    strongly negative (anomalous) scores, -> 0.0 for strongly positive
    (typical) scores.
    """
    return float(np.clip(0.5 - anomaly_score, 0.0, 1.0))


class InferenceEngine:
    def __init__(self) -> None:
        # Render has no GPU -- explicit per BUILD_SPEC.md, not just "whatever's available".
        self.device = torch.device("cpu")
        self.model: Optional[HybridQualityModel] = None
        self.scaler = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None and self.anomaly_detector is not None

    def load(self) -> None:
        settings = get_settings()
        try:
            self.model = HybridQualityModel.from_pretrained(
                settings.MODEL_WEIGHTS_PATH, num_classical_features=len(FEATURE_NAMES)
            ).to(self.device)
            self.scaler = joblib.load(settings.FEATURE_SCALER_PATH)
            self.anomaly_detector = AnomalyDetector.load(settings.ANOMALY_MODEL_PATH)
        except Exception as e:  # noqa: BLE001
            self.load_error = str(e)
            raise

    def analyze_image(self, pil_image: Image.Image) -> Dict:
        if not self.is_loaded:
            raise RuntimeError("InferenceEngine.load() must succeed before analyze_image() can be called.")

        resized = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        raw_features = extract_feature_vector(resized)
        scaled_features = self.scaler.transform(raw_features.reshape(1, -1))[0]
        classical_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        image_tensor = TF.normalize(TF.to_tensor(resized), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor, classical_tensor)

        head_probs = {name: float(outputs[name].item()) for name in ISSUE_HEADS}
        quality_score = float(outputs[QUALITY_SCORE_HEAD].item())

        issues: List[Dict] = [
            {"type": name, "severity": _severity_for_probability(prob), "confidence": round(prob, 4)}
            for name, prob in head_probs.items()
            if prob >= ISSUE_PROBABILITY_THRESHOLD
        ]

        anomaly_score = self.anomaly_detector.score(scaled_features)
        anomaly_conf = _anomaly_confidence(anomaly_score)
        is_defect = (head_probs["corruption"] >= DEFECT_CORRUPTION_THRESHOLD) or (anomaly_conf > 0.5)
        if is_defect:
            defect_confidence = max(head_probs["corruption"], anomaly_conf)
            issues.append(
                {
                    "type": DEFECT_ISSUE_TYPE,
                    "severity": _severity_for_probability(defect_confidence),
                    "confidence": round(defect_confidence, 4),
                }
            )

        image_stats = {name: float(val) for name, val in zip(FEATURE_NAMES, raw_features)}

        return {
            "quality_score": round(quality_score, 2),
            "quality_label": _quality_label(quality_score),
            "issues": issues,
            "image_stats": image_stats,
            "gradcam_available": True,
        }


_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Module-level singleton accessor -- main.py's lifespan calls .load() on this once at startup."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
