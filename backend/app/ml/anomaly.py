"""
Isolation Forest anomaly detector over classical image-quality features
(classical_features.extract_feature_vector), used to flag "potential visual
defect" as a composite/anomaly signal independent of the 5 supervised issue
heads in cnn_model.py.

Fit on classical feature vectors from clean-labeled training images only, so
that anything unusual relative to that learned "normal" distribution --
including defect types never seen during supervised training -- can be
flagged as anomalous, per BUILD_SPEC.md's anomaly-detection requirement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

DEFAULT_CONTAMINATION = 0.05
DEFAULT_RANDOM_STATE = 42

# sklearn's IsolationForest.decision_function() is centered so that 0 is
# the boundary it fit between inlier and outlier during training (given
# `contamination`): negative scores are anomalous, positive scores are
# normal. This is the default threshold; callers may pass a different one
# to trade off precision/recall against the "potential visual defect" label.
DEFAULT_ANOMALY_THRESHOLD = 0.0


class AnomalyDetector:
    """Thin, serializable wrapper around sklearn.ensemble.IsolationForest."""

    def __init__(
        self,
        contamination: float = DEFAULT_CONTAMINATION,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self._model = IsolationForest(contamination=contamination, random_state=random_state)
        self._is_fitted = False

    def fit(self, clean_feature_vectors: np.ndarray) -> "AnomalyDetector":
        """
        clean_feature_vectors: array of shape (n_clean_samples, n_features),
        one classical feature vector per clean-labeled training image.
        """
        self._model.fit(clean_feature_vectors)
        self._is_fitted = True
        return self

    def score(self, feature_vector: np.ndarray) -> float:
        """
        Anomaly score for a single feature vector (shape (n_features,)).
        Lower (more negative) = more anomalous; see DEFAULT_ANOMALY_THRESHOLD.
        """
        self._require_fitted()
        return float(self._model.decision_function(feature_vector.reshape(1, -1))[0])

    def is_anomalous(self, feature_vector: np.ndarray, threshold: float = DEFAULT_ANOMALY_THRESHOLD) -> bool:
        return self.score(feature_vector) < threshold

    def save(self, path: Union[str, Path]) -> None:
        self._require_fitted()
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnomalyDetector":
        instance = cls.__new__(cls)
        instance._model = joblib.load(path)
        instance._is_fitted = True
        return instance

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector must be fit() or load()ed before scoring.")
