"""
ML inference wrapper.

Will load the exported hybrid MobileNetV3-Small CNN weights (5 sigmoid
issue heads + quality-score regression head), the classical OpenCV/NumPy
feature extractors (Laplacian variance, luma histogram stats, noise
estimate, contrast, JPEG blockiness), and the Isolation Forest anomaly
detector produced by ml_training/export_weights.py, and expose a single
CPU-only inference function consumed by backend/app/services/. Not
implemented yet.
"""
