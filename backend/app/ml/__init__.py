"""
ML package -- single source of truth for this project's model architecture
and feature extraction, shared between the backend (for inference) and
ml_training/ (for training).

Submodules:
  - classical_features.py: OpenCV/NumPy image-quality feature extraction.
  - cnn_model.py: the hybrid MobileNetV3-Small + classical-features model.
  - anomaly.py: the Isolation Forest "potential visual defect" detector.

IMPORT NOTE: ml_training/ is a separate top-level package (a sibling of
backend/, not nested under it). Rather than an editable pip install of
backend/, ml_training/ scripts import these modules via a plain sys.path
insertion at the top of each script, e.g.:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.ml.cnn_model import HybridQualityModel

This was chosen over packaging backend/ as an installable distribution
because there's exactly one internal consumer (ml_training/) and no need to
publish or version backend/ independently -- adding a pyproject.toml/setup.py
solely to satisfy that one import would be unneeded packaging overhead.

A single, still-to-be-implemented inference function that loads the
exported CNN weights + anomaly detector artifact and runs the full
prediction pipeline will live here once ml_training/export_weights.py
produces those artifacts.
"""
