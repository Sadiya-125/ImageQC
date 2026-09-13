"""
Training script.

Will train the hybrid MobileNetV3-Small transfer-learning CNN with 5
independent sigmoid heads (blur, underexposure, overexposure, noise,
corruption) plus a regression head for the 0-100 overall quality score,
fused with the classical image-quality features from dataset.py. Also
trains the separate Isolation Forest anomaly detector on classical features
from clean/mild images. Runs locally on an RTX 3050 (4GB VRAM) — batch size
and mixed precision should respect that budget. Writes checkpoints for
evaluate.py and export_weights.py to consume. Not implemented yet.
"""
