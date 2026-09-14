# Accepted Cases

Real, verified examples from the KADID-10k **test split** (never trained/validated
on) where v1.1.0 correctly scores the image `ACCEPTABLE` with no issues flagged --
the counterpart to [`../failure_cases/`](../failure_cases/), which documents where
the model gets it wrong. Both are included so the model's real behavior can be
inspected directly rather than taken on faith from aggregate metrics.

| Image | True quality_score | Predicted | Issues |
| --- | --- | --- | --- |
| `I21_16_01.png` | 90.0 | 88.98, ACCEPTABLE | none |
| `I75_16_01.png` | 88.3 | 70.45, ACCEPTABLE | none |
| `I15_16_01.png` | 86.8 | 79.72, ACCEPTABLE | none |

**One honest caveat, found while assembling this folder**: not every clean-labeled
image scores this well -- several other pristine/mild-distortion test images from
the same candidate pool (`I52_16_01.png`, `I61_16_01.png`, `I72_16_01.png`,
`I76_16_01.png`) scored `DEFECTIVE`/`DEGRADED` despite having no ground-truth issues
either, consistent with `quality_score`'s MAE=22.60 and the `corruption` head's
mediocre precision (0.403) reported in [`EVALUATION.md`](../../../EVALUATION.md).
This folder shows real successes, not a cherry-picked claim that the model is
reliable on every clean image.
