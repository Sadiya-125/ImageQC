"""
PyTorch Dataset/DataLoader definitions for the ImageQC training pipeline.

Will load images and labels from ml_training/data_gen/labels.csv (built by
data_gen/build_labels_from_kadid.py), respect the reference-image-level
train/val/test split in ml_training/data_gen/kadid10k/split.csv, and apply
preprocessing/augmentation appropriate for MobileNetV3-Small transfer
learning. Also responsible for extracting the classical OpenCV/NumPy
image-quality features (Laplacian variance for sharpness, luma histogram
stats for exposure, noise estimate, contrast, JPEG blockiness) that get
fused with the CNN and used to train the Isolation Forest anomaly detector.
Not implemented yet.
"""
