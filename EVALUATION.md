# Evaluation Results

Metrics below come from running `ml_training/evaluate.py` against the checkpoint produced by `ml_training/train.py` (best validation macro-F1 during training: 0.5127, model version **v1.1.0**). Two independent evaluation sets are reported, clearly separated:

1. **KADID-10k held-out test split** -- reference images never seen during training or validation (see `ml_training/data_gen/kadid10k/split.csv`), but drawn from the same synthetic-distortion distribution as training data.
2. **Real-world generalization test** -- not the formal labeled holdout from BUILD_SPEC.md's Prompt 1b (still not captured), but a real, informal test against 5 genuine document photos. It found a genuine, unresolved failure mode -- see that section below.

## Model version history: v1.0.0 → v1.1.0

After v1.0.0 shipped, real usage surfaced a failure: images structurally unlike KADID-10k's 81 natural-photography references (ID cards, documents, signatures) were scored confidently DEFECTIVE regardless of actual condition. Diagnosis (full writeup in `ml_training/train.py`'s docstring): v1.0.0's `quality_loss` outweighed the summed `issue_loss` by ~10.2x across all 8 epochs -- a risk BUILD_SPEC.md §1.5 explicitly flagged when it set the starting `LAMBDA_QUALITY=1.0` and asked for a revisit, not a silent retune, if that happened. v1.1.0 is that revisit: `LAMBDA_QUALITY` lowered to 0.1 (~= the observed issue/quality loss ratio), same data/architecture/epoch budget, retrained from scratch, with per-head confidence also calibrated post-hoc via temperature scaling (`ml_training/calibrate.py`).

| Metric (KADID-10k test split) | v1.0.0 | v1.1.0 | Change |
| --- | --- | --- | --- |
| Best val macro-F1 (training) | 0.3053 | 0.5127 | **+68%** |
| blur F1 / AUC | 0.472 / 0.910 | 0.627 / 0.959 | better |
| underexposure F1 / AUC | 0.000 / 0.632 | 0.000 / 0.769 | AUC better, still below threshold |
| overexposure F1 / AUC | 0.000 / 0.703 | 0.450 / 0.878 | **F1 unstuck from 0** |
| noise F1 / AUC | 0.527 / 0.889 | 0.625 / 0.910 | better |
| corruption F1 / AUC | 0.451 / 0.698 | 0.498 / 0.693 | roughly flat |
| quality_score MAE | 20.47 | 22.60 | **worse** |
| quality_score SROCC / PLCC | 0.520 / 0.517 | 0.449 / 0.449 | **worse** |

**Read honestly, not cherry-picked**: rebalancing the loss toward classification bought a large, real improvement in 4 of 5 issue heads -- the actual required detection capabilities -- at a real cost to `quality_score` regression accuracy (MAE +2.13, correlation -0.07). Deliberate trade-off, not a wash: the assessment's required capabilities center on the issue/defect categories, with `quality_score` as one summary number. To recover regression accuracy specifically, the next experiment would be an intermediate `LAMBDA_QUALITY` (0.3-0.5), not reverting to 1.0.

`underexposure` still never crosses the 0.5 threshold at 2.4% prevalence -- rebalancing the two loss *terms* didn't address the *class* imbalance within the issue heads. That needs positive-class weighting or a weighted sampler, a reasonable next experiment not applied here.

## KADID-10k Test Split

n = 1008 images, 0.5 probability threshold.

### Per-issue-head classification

| Head | Precision | Recall | F1 | ROC-AUC | Support (pos/total) | Confusion Matrix |
| --- | --- | --- | --- | --- | --- | --- |
| blur | 0.603 | 0.653 | 0.627 | 0.959 | 72/1008 | TN=905 FP=31 FN=25 TP=47 |
| underexposure | 0.000 | 0.000 | 0.000 | 0.769 | 24/1008 | TN=984 FP=0 FN=24 TP=0 |
| overexposure | 0.562 | 0.375 | 0.450 | 0.878 | 24/1008 | TN=977 FP=7 FN=15 TP=9 |
| noise | 0.594 | 0.658 | 0.625 | 0.910 | 120/1008 | TN=834 FP=54 FN=41 TP=79 |
| corruption | 0.403 | 0.652 | 0.498 | 0.693 | 264/1008 | TN=489 FP=255 FN=92 TP=172 |

### Quality score regression

MAE = 22.60 (0-100 scale) | SROCC = 0.449 | PLCC = 0.449

### Anomaly detector ("potential visual defect" flag)

Evaluated against `corruption == 1` as the proxy ground truth (KADID-10k has no separate "visual defect" label). Precision = 0.387, Recall = 0.091, F1 = 0.147 (62/1008 flagged anomalous, 264 truly corrupted). Unchanged from v1.0.0 -- the Isolation Forest is fit on classical (non-learned) features, which the `LAMBDA_QUALITY` change doesn't touch.

## Failure Cases

The 8 worst-predicted KADID-10k test images per head are saved, with true/predicted labels, to [`ml_training/notebooks/failure_cases/README.md`](ml_training/notebooks/failure_cases/README.md). The counterpart -- real cases the model gets right -- is [`ml_training/notebooks/accepted_cases/README.md`](ml_training/notebooks/accepted_cases/README.md).

## Real-world generalization test (informal, but real)

5 genuine photos of ID documents (Aadhaar card, PAN card, a signature, two passport-style photos) were run through the live app -- not BUILD_SPEC.md Prompt 1b's formal labeled holdout (still not captured), but real evidence from real usage. All 5 scored `DEFECTIVE` (quality_score 2-20) under **both** model versions, despite being ordinary, adequately-captured photos -- a false-positive failure, not a correct catch.

**v1.1.0 made this *more* confident, not less**: the driving issue head's confidence rose from 0.58-0.86 (v1.0.0) to 0.94-0.99 (v1.1.0) on the same 5 images. Sharper in-distribution decision boundaries -- exactly what the `LAMBDA_QUALITY` fix produced -- extrapolate more confidently on out-of-distribution input, not more cautiously. Rebalancing a loss term and fixing a training-distribution coverage gap are different problems; v1.1.0 solved the first, not the second.

Root cause: the CNN backbone is fine-tuned on exactly 81 unique reference photos, all natural photography -- no document/text/card-style content anywhere in KADID-10k. A model this size, tuned on this little visual diversity, was never going to generalize to a structurally different domain regardless of loss weighting.

One piece of the system already partially catches this, unprompted: the classical-feature Isolation Forest (§1.4) flagged 4 of the 5 images as anomalous on its own:

| Image | Anomaly score | Flagged anomalous |
| --- | --- | --- |
| Aadhaar card | -0.102 | Yes |
| PAN card | +0.038 | No (borderline) |
| Signature | -0.136 | Yes |
| Passport photo 1 | -0.058 | Yes |
| Passport photo 2 | -0.093 | Yes |

This signal exists today but isn't surfaced distinctly from the ordinary `DEFECTIVE` verdict -- the API/UI can't currently distinguish "this looks low-quality" from "this looks unlike anything in training." A distinct "outside assessed domain, confidence reduced" state (rather than a confident DEFECTIVE) is a natural next step, not implemented in this submission.

## Limitations

- **Underexposure never crosses the 0.5 decision threshold** despite real ranking signal (AUC 0.769) -- see "Model version history" above for why rebalancing the loss didn't fix this specific class imbalance.
- **The v1.0.0 → v1.1.0 change traded some `quality_score` regression accuracy for large classification gains** -- see "Model version history" above for the full numbers and reasoning.
- **Real-world generalization on document/ID-card-style images is a confirmed, unresolved failure** -- see "Real-world generalization test" above. KADID-10k test-split numbers in this document are evidence only within the natural-photography domain KADID-10k represents, not for document-style images.
- **Training labels come from KADID-10k's algorithmically-applied distortion filters (Gaussian blur kernels, JPEG re-encoding, synthetic brightness shifts), not organically-occurring camera defects.** A real out-of-focus phone photo or real sensor noise at high ISO doesn't look pixel-for-pixel identical to KADID-10k's synthetic equivalent, even where the underlying physical phenomenon is similar -- the standard NR-IQA synthetic-to-real domain gap.
- **KADID-10k applies exactly one distortion type per image**, so almost every training label vector is one-hot across the 5 issue heads (per BUILD_SPEC.md §2.4). Real photos are frequently multi-issue (dark AND noisy is common, since noise rises with sensor gain in low light) -- the model has the architectural capacity to predict co-occurring issues (independent sigmoid heads) but has never seen a training example of one.
