# Evaluation Results

Metrics below come from running `ml_training/evaluate.py` against the checkpoint produced by `ml_training/train.py` (best validation macro-F1 during training: 0.3053). Two independent evaluation sets are reported, clearly separated:

1. **KADID-10k held-out test split** -- reference images never seen during training or validation (see `ml_training/data_gen/kadid10k/split.csv`), but drawn from the same synthetic-distortion distribution as training data.

2. **Real-world holdout set**: not present at evaluation time (`ml_training/data_gen/real_world_holdout/` does not exist -- this is a manual, user-captured step per BUILD_SPEC.md's Prompt 1b). Everything below is therefore evidence of fitting the KADID-10k distortion distribution, **not** evidence of real-world generalization; re-run this script after adding that holdout set.

## KADID-10k Test Split

n = 1008 images, 0.5 probability threshold.

### Per-issue-head classification

| Head | Precision | Recall | F1 | ROC-AUC | Support (pos/total) | Confusion Matrix |
| --- | --- | --- | --- | --- | --- | --- |
| blur | 0.396 | 0.583 | 0.472 | 0.910 | 72/1008 | TN=872 FP=64 FN=30 TP=42 |
| underexposure | 0.000 | 0.000 | 0.000 | 0.632 | 24/1008 | TN=984 FP=0 FN=24 TP=0 |
| overexposure | 0.000 | 0.000 | 0.000 | 0.703 | 24/1008 | TN=984 FP=0 FN=24 TP=0 |
| noise | 0.774 | 0.400 | 0.527 | 0.889 | 120/1008 | TN=874 FP=14 FN=72 TP=48 |
| corruption | 0.431 | 0.473 | 0.451 | 0.698 | 264/1008 | TN=579 FP=165 FN=139 TP=125 |

### Quality score regression

MAE = 20.47 (0-100 scale) | SROCC = 0.520 | PLCC = 0.517

### Anomaly detector ("potential visual defect" flag)

Evaluated against `corruption == 1` as the proxy ground truth for "known-corrupted" (KADID-10k has no separate "visual defect" label). Precision = 0.387, Recall = 0.091, F1 = 0.147 (62/1008 flagged anomalous, 264 truly corrupted).

## Failure Cases

The 8 worst-predicted KADID-10k test images per head (images + true/predicted labels) are saved to [`ml_training/notebooks/failure_cases/README.md`](ml_training/notebooks/failure_cases/README.md).

## Limitations

- **Underexposure and overexposure heads never cross the 0.5 decision threshold (F1 = 0.000 for both, on this test split), but they are not clueless: ROC-AUC is 0.632 (underexposure) and 0.703 (overexposure), both well above the 0.5 random baseline.** That combination -- real ranking signal, zero predictions past threshold -- is the signature of class-imbalance-induced threshold miscalibration, not a head that learned nothing. Root cause: both categories are ~2.4% of the KADID-10k-derived label distribution (243/10,206 images each -- see ml_training/data_gen/build_labels_from_kadid.py's printed class balance), and train.py's loss is unweighted BCE per BUILD_SPEC.md §1.5's exact loss spec, so the learned decision boundary sits below 0.5 for every test image even though the model ranks truly-underexposed/overexposed images higher than clean ones. A lower decision threshold, positive-class loss weighting, or a weighted DataLoader sampler would all be reasonable next experiments -- deliberately not applied here rather than silently patched in.
- **quality_score's regression loss dominates the classification loss during training** (quality_loss stayed roughly 10-20x larger than the summed issue_loss across all 8 epochs -- see the train.py log), exactly the risk BUILD_SPEC.md §1.5 flagged for starting LAMBDA_QUALITY at 1.0. It wasn't retuned mid-run per that same section's instruction to document rather than silently adjust; a lower LAMBDA_QUALITY is a reasonable next experiment.
- **Training labels come from KADID-10k's algorithmically-applied distortion filters (Gaussian blur kernels, JPEG re-encoding, synthetic brightness shifts, etc.), not organically-occurring camera defects.** A real out-of-focus phone photo, a real underexposed low-light shot, or real sensor noise at high ISO don't look pixel-for-pixel identical to KADID-10k's synthetic equivalents, even where the underlying physical phenomenon is similar -- this is the standard NR-IQA synthetic-to-real domain gap.
- **KADID-10k applies exactly one distortion type per image (never combinations)**, so almost every training label vector is one-hot across the 5 issue heads (per BUILD_SPEC.md §2.4). Real photos are frequently multi-issue (e.g. dark AND noisy, since noise increases as sensor gain/ISO rises to compensate for low light) -- the model has never seen a genuine co-occurring-issue training example, only the architectural capacity (independent sigmoid heads) to predict them at inference time.
- **No real-world generalization evidence in this report** -- see the note under "Real-World Holdout Set" above. The KADID-10k test-split numbers alone should not be read as evidence the model works on genuine photos.
