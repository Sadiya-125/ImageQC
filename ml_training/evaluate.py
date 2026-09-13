"""
Evaluation script.

Will report, on the KADID-10k test split: classification metrics (accuracy,
precision, recall, F1, ROC-AUC, confusion matrix) per issue head, regression
error metrics for the quality score, and anomaly-detection metrics for the
Isolation Forest. Will also evaluate against the fully separate real-world
holdout set (ml_training/data_gen/real_world_holdout/) as a second,
independent test set — this holdout must never be mixed into the KADID-10k
train/val/test split. Surfaces failure cases and uncertain predictions for
the written evaluation discussion. Not implemented yet.
"""
