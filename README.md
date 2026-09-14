# ImageQC — AI-Powered Image Quality & Defect Detection

A full-stack application that accepts an uploaded image and automatically evaluates its
visual quality: an overall 0–100 quality score, an ACCEPTABLE / DEGRADED / DEFECTIVE
label, and detection of six issue categories — blur, underexposure, overexposure, noise,
corruption, and potential visual defect. No external AI/vision APIs are used anywhere;
every prediction comes from a model trained specifically for this project.

**Status:**

| | URL |
| --- | --- |
| Backend (Render) | **[https://imageqc.onrender.com](https://imageqc.onrender.com)** — live, migrated, smoke-tested |
| Frontend (Vercel) | deploy with the steps in [§8 Deployment](#8-deployment) |

## Table of contents

1. [Architecture](#1-architecture)
2. [Model: design, data, training, architecture](#2-model-design-data-training-architecture)
3. [Evaluation results](#3-evaluation-results)
4. [Local setup](#4-local-setup)
5. [API documentation](#5-api-documentation)
6. [How inference works in production](#6-how-inference-works-in-production)
7. [Assessment criteria coverage](#7-assessment-criteria-coverage)
8. [Deployment](#8-deployment)
9. [Bonus / optional work](#9-bonus--optional-work-brief-13)
10. [Submission checklist](#10-submission-checklist-brief-12)

## 1. Architecture

```
                     ┌─────────────────────┐
                     │   Browser (user)    │
                     └──────────┬──────────┘
                                │ HTTPS
                     ┌──────────▼──────────┐
                     │  Next.js Frontend   │   deployed to Vercel
                     │  (App Router, TS,   │   frontend/
                     │  Tailwind, shadcn)  │
                     └──────────┬──────────┘
                                │ REST (NEXT_PUBLIC_API_URL)
                     ┌──────────▼──────────┐
                     │   FastAPI Backend   │   deployed to Render (Docker, CPU-only)
                     │   backend/app/      │   live: imageqc.onrender.com
                     │                     │
                     │  api/  →  ml/       │   loads model ONCE at startup (lifespan),
                     │  core/    inference │   not per-request — see §6
                     │  models/  cnn_model │
                     │           anomaly   │
                     │           gradcam   │
                     └──────────┬──────────┘
                                │ SQLAlchemy async (asyncpg)
                     ┌──────────▼──────────┐
                     │   Neon Postgres     │   pooled conn (app) / direct conn (Alembic)
                     │   analyses          │
                     │   analysis_issues   │
                     └─────────────────────┘

   ml_training/  (offline, RTX 3050 laptop GPU — never runs in production)
   ┌─────────────────────────────────────────────────────────────────┐
   │ data_gen/ → dataset.py → train.py → evaluate.py                  │
   │ KADID-10k (real benchmark) ──labels──► HybridQualityModel        │
   │                                        (MobileNetV3-Small +      │
   │                                         classical CV features)   │
   │                                        + IsolationForest         │
   │  outputs: backend/app/ml/weights/{mobilenetv3_iqa.pt,            │
   │           feature_scaler.joblib, anomaly_iforest.joblib,         │
   │           model_metadata.json, calibration.json}                │
   │  (committed to git — see backend/app/ml/, and §2 below)          │
   └─────────────────────────────────────────────────────────────────┘
```

`backend/app/ml/` (`classical_features.py`, `cnn_model.py`, `anomaly.py`, `gradcam.py`) is
the single source of truth for the model architecture and feature extraction — both
`ml_training/` (training) and the backend (inference) import from there, so training and
production can never silently drift out of sync.

## 2. Model: design, data, training, architecture

### 2.1 Why this design

Three constraints shaped every decision here: training happens on a **4GB laptop GPU**
(RTX 3050), inference runs on a **CPU-only, 512MB free-tier Render container**, and **no
external AI/vision APIs are allowed** — everything has to be a model trained specifically
for this project, small enough to fine-tune on a small GPU and fast enough for sub-second
CPU inference. That points at one answer: a small ImageNet-pretrained CNN backbone
(**MobileNetV3-Small**, ~1.1M trainable params after dropping its classifier), fused with
cheap classical CV features for interpretability and for the signals (exposure, basic
noise) that don't need learning at all. Five independent **sigmoid** heads — not one
softmax — because real degraded images are very often multi-issue at once (underexposed
*and* noisy is common, since noise rises with sensor gain in low light); softmax would
force one label per image and misrepresent that. A separate **Isolation Forest**, fit on
classical features from clean images, covers the anomaly-detection angle and feeds the
"potential visual defect" composite signal alongside the corruption head.

### 2.2 Dataset: KADID-10k

Training data is the real, publicly downloadable **KADID-10k** benchmark (Lin, Hosu &
Saupe, 2019; mirrored on Hugging Face at `chaofengc/IQA-PyTorch-Datasets`) — 81 pristine
reference images (Pixabay-licensed), each degraded with 25 distortion types × 5 severity
levels = 10,125 distorted images, plus the 81 originals. It maps closely onto this
project's 6 categories, built deterministically by
`ml_training/data_gen/build_labels_from_kadid.py`:

| Category | KADID-10k distortion types (level ≥ 3 sets the label) |
| --- | --- |
| Blur | Gaussian blur, Lens blur, Motion blur |
| Underexposure | Darken |
| Overexposure | Brighten |
| Noise | White noise, White noise in color component, Impulse noise, Multiplicative noise, Denoise |
| Corruption | JPEG, JPEG2000, Color quantization, Color block — **plus any distortion type at level 5**, regardless of its own category |
| Clean | The 81 pristine references, plus level 1–2 of any mapped type |

`Mean shift` (type 18) is deliberately excluded from both exposure categories: its
per-level luma shift measured on real images isn't a clean under/over split (brighter at
levels 1–2, darker at 4–5, unchanged at level 3), which doesn't fit a single severity
threshold — excluded rather than guessed.

Two things in the original build spec were corrected after inspecting the real archive,
not assumed from documentation:

- **DMOS direction.** The spec assumed DMOS 1–5 with higher = worse. The real data (and
  KADID-10k's own database page) confirm the opposite — DMOS is MOS-like, **higher =
  better** (mean DMOS falls monotonically from 4.08 at level 1 to 2.01 at level 5 across
  all 10,125 distorted images). `quality_score` is derived accordingly:
  `quality_score = 100 * (dmos - 1) / 4` (pristine references, never rated, get 100.0
  directly).
- **Metadata schema.** `dmos.csv` ships only 4 columns (`dist_img`, `ref_img`, `dmos`,
  `var`) — distortion type and severity level are parsed from the filename itself
  (`I<ref>_<type>_<level>.png`), not read from separate columns as first assumed.

**Splits** are at the *reference-image* level (65/8/8 of the 81 references → train/val/
test), never at the distorted-variant level — since all 125 variants of one reference
share the same scene content, splitting at the variant level would leak content across
train/test and inflate metrics. This is standard KADID-10k protocol. A hand-captured
real-world holdout set (genuine, non-filtered defects) was planned as a second,
independent evaluation set (`ml_training/data_gen/real_world_holdout/`, wired into
`evaluate.py` and picked up automatically if present) but wasn't captured during this
build; §3 below instead reports a smaller, informal real-world test run through the live
API.

**Known limitation, stated plainly:** KADID-10k applies exactly one distortion type per
image, so almost every training label vector is one-hot across the 5 issue heads. Real
photos are frequently multi-issue (dark *and* noisy is common) — the architecture has the
capacity to predict co-occurring issues (independent sigmoid heads, not a single softmax)
but has never seen a training example of one, so multi-issue confidence is less validated
than single-issue detection.

### 2.3 Architecture (exact shapes)

- **Backbone**: `torchvision.models.mobilenet_v3_small` (ImageNet-pretrained), classifier
  dropped, `features` → `AdaptiveAvgPool2d(1)` → a **576-d** pooled embedding (verified
  against the real torchvision module, not assumed from the paper).
- **Classical features branch** (`backend/app/ml/classical_features.py`): an **8-d**
  vector — `laplacian_variance`, `mean_luma`, `pct_clipped_low`, `pct_clipped_high`,
  `luma_skew`, `noise_estimate` (Immerkær's high-frequency-residual method),
  `contrast_std`, `jpeg_blockiness` — standardized via a `StandardScaler` fit on the
  training split only, then `Linear(8,32) → BatchNorm1d → ReLU → Linear(32,32) → ReLU`.
- **Fusion trunk**: concatenate 576-d + 32-d = 608-d →
  `Linear(608,256) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(256,128) → ReLU → Dropout(0.2)`.
- **Heads**, all reading the shared 128-d representation: 5 independent
  `Linear(128,1) → Sigmoid` issue heads (blur, underexposure, overexposure, noise,
  corruption) plus 1 `Linear(128,1) → Sigmoid × 100` regression head for `quality_score`.
  ~1.1M trainable parameters total.
- **"Potential visual defect"** is not a 7th head — computed post-hoc as
  `corruption_prob > 0.5 OR isolation_forest.is_anomalous(classical_features)`.
- **Quality label thresholds** (`backend/app/ml/inference.py`): `score >= 70` →
  ACCEPTABLE, `40 <= score < 70` → DEGRADED, else DEFECTIVE. An issue is reported once its
  head clears **0.5**, with severity bands low (<0.65) / medium (<0.85) / high (≥0.85).
- **Explainability**: Grad-CAM (`backend/app/ml/gradcam.py`) against the CNN branch's
  last convolutional layer, exposed via `GET /api/analyses/{id}/gradcam?head=<issue>` and
  the frontend's "View Grad-CAM heatmap" button, plus the classical feature values shown
  in a readable panel rather than a raw JSON dump.

### 2.4 Training

`ml_training/train.py`, run for real on an **RTX 3050 (4GB VRAM) laptop GPU** — two-phase
(frozen backbone → last 3 blocks unfrozen), mixed precision, 8 epochs total, ~21–35
minutes wall-clock. Loss: `Σ BCE(issue_head_i)` + `λ_quality * SmoothL1(quality_score)`.
Current model is **v1.1.0** — see §3's "Model version history" for the real v1.0.0 →
v1.1.0 story, numbers, and honest trade-offs. Confidence is calibrated post-hoc via
per-head temperature scaling (`ml_training/calibrate.py` fits it on the validation split,
`backend/app/ml/calibration.py` applies it at inference — the `confidence` values
returned by the API are calibrated, not raw sigmoid outputs).

## 3. Evaluation results

Metrics below come from running `ml_training/evaluate.py` against the checkpoint produced
by `ml_training/train.py` (best validation macro-F1 during training: 0.5127, model version
**v1.1.0**). Two independent evaluation sets are reported, clearly separated:

1. **KADID-10k held-out test split** — reference images never seen during training or
   validation (`ml_training/data_gen/kadid10k/split.csv`), but drawn from the same
   synthetic-distortion distribution as training data.
2. **Real-world generalization test** — not the formal labeled holdout from §2.2 (still
   not captured), but a real, informal test against 5 genuine document photos. It found a
   genuine, unresolved failure mode — see below.

### Model version history: v1.0.0 → v1.1.0

After v1.0.0 shipped, real usage surfaced a failure: images structurally unlike
KADID-10k's 81 natural-photography references (ID cards, documents, signatures) were
scored confidently DEFECTIVE regardless of actual condition. Diagnosis (full derivation in
`ml_training/train.py`'s docstring): v1.0.0's `quality_loss` outweighed the summed
`issue_loss` by ~10.2x across all 8 epochs — a risk the original spec flagged when it set
the starting `λ_quality=1.0` and asked for a revisit, not a silent retune, if that
happened. v1.1.0 is that revisit: `λ_quality` lowered to 0.1 (≈ the observed issue/quality
loss ratio), same data/architecture/epoch budget, retrained from scratch, with per-head
confidence also calibrated post-hoc via temperature scaling.

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

**Read honestly, not cherry-picked**: rebalancing the loss toward classification bought a
large, real improvement in 4 of 5 issue heads — the actual required detection
capabilities — at a real cost to `quality_score` regression accuracy (MAE +2.13,
correlation -0.07). Deliberate trade-off, not a wash: the assessment's required
capabilities center on the issue/defect categories, with `quality_score` as one summary
number. To recover regression accuracy specifically, the next experiment would be an
intermediate `λ_quality` (0.3–0.5), not reverting to 1.0.

`underexposure` still never crosses the 0.5 threshold at 2.4% prevalence — rebalancing the
two loss *terms* didn't address the *class* imbalance within the issue heads. That needs
positive-class weighting or a weighted sampler, a reasonable next experiment not applied
here.

### KADID-10k test split (n = 1008, 0.5 probability threshold)

| Head | Precision | Recall | F1 | ROC-AUC | Support (pos/total) | Confusion Matrix |
| --- | --- | --- | --- | --- | --- | --- |
| blur | 0.603 | 0.653 | 0.627 | 0.959 | 72/1008 | TN=905 FP=31 FN=25 TP=47 |
| underexposure | 0.000 | 0.000 | 0.000 | 0.769 | 24/1008 | TN=984 FP=0 FN=24 TP=0 |
| overexposure | 0.562 | 0.375 | 0.450 | 0.878 | 24/1008 | TN=977 FP=7 FN=15 TP=9 |
| noise | 0.594 | 0.658 | 0.625 | 0.910 | 120/1008 | TN=834 FP=54 FN=41 TP=79 |
| corruption | 0.403 | 0.652 | 0.498 | 0.693 | 264/1008 | TN=489 FP=255 FN=92 TP=172 |

**Quality score regression**: MAE = 22.60 (0–100 scale), SROCC = 0.449, PLCC = 0.449.

**Anomaly detector** ("potential visual defect" flag), evaluated against `corruption == 1`
as the proxy ground truth (KADID-10k has no separate "visual defect" label): precision =
0.387, recall = 0.091, F1 = 0.147 (62/1008 flagged anomalous, 264 truly corrupted).
Unchanged from v1.0.0 — the Isolation Forest is fit on classical (non-learned) features,
which the `λ_quality` change doesn't touch.

**Failure cases**: the 8 worst-predicted test images per head, with true/predicted labels,
are saved to [`ml_training/notebooks/failure_cases/README.md`](ml_training/notebooks/failure_cases/README.md).
The counterpart — real cases the model gets right — is
[`ml_training/notebooks/accepted_cases/README.md`](ml_training/notebooks/accepted_cases/README.md).

### Real-world generalization test (informal, but real)

5 genuine photos of ID documents (Aadhaar card, PAN card, a signature, two passport-style
photos) were run through the live app — not §2.2's formal labeled holdout (still not
captured), but real evidence from real usage. All 5 scored `DEFECTIVE` (quality_score
2–20) under **both** model versions, despite being ordinary, adequately-captured photos —
a false-positive failure, not a correct catch.

**v1.1.0 made this *more* confident, not less**: the driving issue head's confidence rose
from 0.58–0.86 (v1.0.0) to 0.94–0.99 (v1.1.0) on the same 5 images. Sharper
in-distribution decision boundaries — exactly what the `λ_quality` fix produced —
extrapolate more confidently on out-of-distribution input, not more cautiously.
Rebalancing a loss term and fixing a training-distribution coverage gap are different
problems; v1.1.0 solved the first, not the second.

Root cause: the CNN backbone is fine-tuned on exactly 81 unique reference photos, all
natural photography — no document/text/card-style content anywhere in KADID-10k. A model
this size, tuned on this little visual diversity, was never going to generalize to a
structurally different domain regardless of loss weighting.

One piece of the system already partially catches this, unprompted: the classical-feature
Isolation Forest (§2.1) flagged 4 of the 5 images as anomalous on its own:

| Image | Anomaly score | Flagged anomalous |
| --- | --- | --- |
| Aadhaar card | -0.102 | Yes |
| PAN card | +0.038 | No (borderline) |
| Signature | -0.136 | Yes |
| Passport photo 1 | -0.058 | Yes |
| Passport photo 2 | -0.093 | Yes |

This signal exists today but isn't surfaced distinctly from the ordinary `DEFECTIVE`
verdict — the API/UI can't currently distinguish "this looks low-quality" from "this looks
unlike anything in training." A distinct "outside assessed domain, confidence reduced"
state (rather than a confident DEFECTIVE) is a natural next step, not implemented in this
submission.

### Limitations

- **Underexposure never crosses the 0.5 decision threshold** despite real ranking signal
  (AUC 0.769) — see "Model version history" above for why rebalancing the loss didn't fix
  this specific class imbalance.
- **The v1.0.0 → v1.1.0 change traded some `quality_score` regression accuracy for large
  classification gains** — see "Model version history" above for the full numbers.
- **Real-world generalization on document/ID-card-style images is a confirmed, unresolved
  failure** — see "Real-world generalization test" above. KADID-10k test-split numbers
  above are evidence only within the natural-photography domain KADID-10k represents, not
  for document-style images.
- **Training labels come from KADID-10k's algorithmically-applied distortion filters**
  (Gaussian blur kernels, JPEG re-encoding, synthetic brightness shifts), not organically-
  occurring camera defects — the standard NR-IQA synthetic-to-real domain gap.
- **KADID-10k applies exactly one distortion type per image** — see §2.2's "Known
  limitation" for the multi-issue co-occurrence gap this leaves in training data.

## 4. Local setup

Requires Docker Desktop, Node.js 20+, and Python 3.11+.

```bash
git clone <this-repo> && cd "IIITH Assessment"

# 1. Local Postgres + backend, via Docker
docker compose up -d db backend
# db on localhost:5433, backend on localhost:8000 (override the host port with
# BACKEND_HOST_PORT=xxxx if 8000 is already taken on your machine)

# 2. Run the database migration (first time only)
cd backend
python -m venv .venv && source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt   # requirements.txt + pytest
cp .env.example .env                  # defaults already match docker-compose
alembic upgrade head

# 3. Frontend
cd ../frontend
cp .env.local.example .env.local      # points at http://localhost:8000
npm install
npm run dev                           # http://localhost:3000
```

Confirm it's alive: `curl http://localhost:8000/health` → `{"status":"ok","model_loaded":true}`.

**Running the backend without Docker** (faster iteration while coding): `docker compose
up -d db`, then from `backend/` with its venv active, `uvicorn app.main:app --reload`.
Both paths use the exact same code — Docker is only slower to iterate against because it
rebuilds the image.

**Running the test suite:**
```bash
cd backend
pytest tests/ -v
```
40 tests, all passing (real run — see `tests/`): classical-feature unit tests against
synthetic inputs, model-loading/output-shape tests against the real committed weights,
and API integration tests against a real Postgres test database (auto-created), covering
analyze/batch-analyze, pagination, delete, 404s, and the Grad-CAM/image endpoints. CI runs
the same suite on every push (`.github/workflows/ci.yml`), plus frontend lint/typecheck/
build.

## 5. API documentation

Base URL: `http://localhost:8000` locally, or `https://imageqc.onrender.com` (live).
Interactive docs (Swagger UI, auto-generated by FastAPI) are also available at `/docs` on
any running instance.

### `GET /health`
```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```
Returns `503` with `model_loaded: false` if the model failed to load at startup.

### `POST /api/analyze`
Multipart image upload → runs the full ML pipeline → persists the result → returns it.
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@sample_images/corrupted/I15_10_05.png;type=image/png"
```
```json
{
  "id": "088b363b-1593-41e0-9ddc-25a7893d16e5",
  "filename": "I15_10_05.png",
  "quality_score": 9.55,
  "quality_label": "DEFECTIVE",
  "issues": [
    {"type": "corruption", "severity": "low", "confidence": 0.6357},
    {"type": "potential_defect", "severity": "low", "confidence": 0.6357}
  ],
  "image_stats": {"laplacian_variance": 757.66, "mean_luma": 109.10, "...": "..."},
  "gradcam_available": true,
  "model_version": "1.1.0",
  "created_at": "2026-09-14T14:07:05.973494Z"
}
```
`400` for an unreadable/corrupt file, `413` for a file over `MAX_UPLOAD_MB` (default 10).
Both are real, tested error paths (see `backend/tests/test_api.py`), not just documented
intent.

### `POST /api/analyze/batch`
Same pipeline, up to 10 files in one request. Each file is validated/analyzed/persisted
independently — one bad file in the batch doesn't fail the others.
```bash
curl -X POST http://localhost:8000/api/analyze/batch \
  -F "files=@sample_images/clean/I15.png;type=image/png" \
  -F "files=@sample_images/corrupted/I15_10_05.png;type=image/png"
```
```json
{"results": [
  {"filename": "I15.png", "success": true, "result": {"...": "full AnalyzeResponse"}, "error": null},
  {"filename": "I15_10_05.png", "success": true, "result": {"...": "full AnalyzeResponse"}, "error": null}
]}
```

### `GET /api/analyses?page=1&page_size=20`
Paginated history, most recent first. Excludes the raw image and image_stats to keep the
payload small.
```bash
curl "http://localhost:8000/api/analyses?page=1&page_size=10"
```
```json
{"items": [{"id": "...", "filename": "...", "quality_score": 71.94, "quality_label": "ACCEPTABLE", "created_at": "..."}], "total": 3, "page": 1, "page_size": 10}
```

### `GET /api/analyses/{id}`
Full detail for one past analysis (issues + image_stats included).
```bash
curl http://localhost:8000/api/analyses/088b363b-1593-41e0-9ddc-25a7893d16e5
```
`404` if the id doesn't exist.

### `DELETE /api/analyses/{id}`
Deletes an analysis and its issues.
```bash
curl -X DELETE http://localhost:8000/api/analyses/088b363b-1593-41e0-9ddc-25a7893d16e5
# 204 No Content
```

### `GET /api/analyses/{id}/image`
The original uploaded bytes, as stored (used by the frontend to render past analyses).
```bash
curl http://localhost:8000/api/analyses/088b363b-1593-41e0-9ddc-25a7893d16e5/image -o original.png
```

### `GET /api/analyses/{id}/gradcam?head=<blur|underexposure|overexposure|noise|corruption>`
Regenerates a Grad-CAM heatmap overlay on demand from the stored image (not pre-computed/
stored — keeps storage small) and streams it back as a PNG.
```bash
curl "http://localhost:8000/api/analyses/088b363b-1593-41e0-9ddc-25a7893d16e5/gradcam?head=corruption" -o heatmap.png
```
`400` for an unrecognized `head` value.

## 6. How inference works in production

- **CPU-only, explicitly.** `backend/app/ml/inference.py` hard-codes
  `torch.device("cpu")` — Render's free/starter tiers have no GPU, and the model was
  chosen (MobileNetV3-Small, ~1.1M trainable params) specifically to be CPU-fast.
- **Loaded once, not per-request.** `main.py`'s FastAPI `lifespan` handler calls
  `InferenceEngine.load()` exactly once at process startup (loading the CNN checkpoint,
  the fitted feature scaler, and the fitted Isolation Forest); every request reuses the
  same in-memory model via a module-level singleton
  (`app.ml.inference.get_inference_engine()`). If loading fails, the process stays up
  and `/health` reports `503` instead of crash-looping.
- **Blocking inference is offloaded from the event loop.** `analyze_image()` and Grad-CAM
  generation are CPU-bound PyTorch calls; both run via `asyncio.to_thread(...)` so one
  slow inference doesn't serialize concurrent requests behind it.
- **Real measured latency: ~53ms/image** (52.9ms average over 20 warmed-up runs, measured
  directly rather than estimated) — comfortably under any reasonable request timeout even
  before accounting for network overhead.
- **Model artifacts are committed to git** (`backend/app/ml/weights/*.pt`/`*.joblib`/
  `*.json`, ~5.5MB total) specifically so a fresh clone or a Render build has something to
  load without needing to re-run training — see the comment in `.gitignore` for why this
  is an intentional exception to the usual "don't commit binaries" rule.
- **Docker image**: `backend/Dockerfile`, `python:3.11-slim` base, CPU-only PyTorch wheels
  (`--index-url https://download.pytorch.org/whl/cpu`, avoiding ~2GB of unused CUDA
  libraries), non-root user, `HEALTHCHECK` against `/health`. Built and run for real via
  `docker compose up` — **530MB** final image size.

## 7. Assessment criteria coverage

Cross-referencing this repo against the brief's §14 weighted criteria, so a reviewer can
find each piece quickly:

| Criterion (weight) | Where to look |
| --- | --- |
| Computer vision understanding & feature reasoning (15%) | `backend/app/ml/classical_features.py` (Laplacian variance, exposure stats, Immerkær noise estimator, contrast, JPEG blockiness — each with a docstring citing the method); §2 above |
| AI/ML/Deep Learning implementation (25%) | `backend/app/ml/cnn_model.py` (hybrid MobileNetV3-Small architecture), `backend/app/ml/anomaly.py` (Isolation Forest), `ml_training/train.py` (real two-phase training run, RTX 3050); §2 above |
| Model evaluation & experimental rigor (15%) | §3 above — per-head precision/recall/F1/ROC-AUC/confusion matrices, regression MAE/SROCC/PLCC, anomaly-detector precision/recall, failure cases, and an honest limitations section |
| Backend/API implementation (15%) | `backend/app/` (FastAPI, async SQLAlchemy, Alembic, validation, structured errors); §5 above; `backend/tests/test_api.py` |
| Frontend functionality & usability (10%) | `frontend/` (Next.js App Router, upload/analyze flow, history with delete, detail view, Grad-CAM viewer, responsive, dark/light theme) |
| Deployment & reproducibility (10%) | `backend/Dockerfile`, `docker-compose.yml`, `render.yaml`; §4, §6, §8 above |
| Code quality & documentation (10%) | This README, docstrings throughout `backend/app/ml/` and `ml_training/`, `backend/tests/` (40 passing tests), consistent typed frontend (`frontend/lib/types.ts`) |

## 8. Deployment

Neon (database) → Render (backend, via the `render.yaml` Blueprint) → Vercel (frontend).
Follow the steps in order — each one produces a value the next step needs. **The backend
below is already live**; what remains is the Vercel step for the frontend.

### 8.1 Create the Neon project and get both connection strings

1. Sign up / log in at [neon.tech](https://neon.tech) and create a new project (any
   region close to your Render region is fine — this repo defaults Render to `oregon`,
   see `render.yaml`).
2. In the Neon dashboard, open **Connection Details** for the project's default database.
   Neon gives you two connection strings:
   - **Pooled connection** (host contains `-pooler`, routes through PgBouncer) →
     `DATABASE_URL`. The app's actual runtime traffic uses this.
   - **Direct connection** (no `-pooler` in the host) → `DIRECT_URL`. Alembic migrations
     use this only — PgBouncer's pooled connections don't reliably support the
     session-level features Alembic's DDL transactions rely on (see
     `backend/app/alembic/env.py`).
3. Both strings look like `postgresql://user:password@host/dbname?sslmode=require&channel_binding=require`.
   This project's SQLAlchemy setup expects the `asyncpg` driver explicitly, and asyncpg
   does **not** understand libpq-style `sslmode=`/`channel_binding=` query params
   (confirmed by testing directly — the connection fails with those params, succeeds with
   `ssl=require`). Rewrite both strings before using them anywhere in this app:
   ```
   postgresql://user:pass@host/db?sslmode=require&channel_binding=require
     ->
   postgresql+asyncpg://user:pass@host/db?ssl=require
   ```
4. Save both rewritten strings somewhere private (a password manager, not a committed
   file) — you'll paste them into `backend/.env` and into Render's dashboard next.

### 8.2 Run Alembic migrations against `DIRECT_URL`, before the first deploy

Do this once, before Render (or any client) ever serves traffic against a fresh database —
it has no tables yet, and the backend will fail every request until they exist.

```bash
cd backend
cp .env.example .env
# edit .env: paste your real (asyncpg-scheme) DATABASE_URL and DIRECT_URL from 8.1
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
alembic upgrade head
```

Confirm it worked: `alembic current` should print the latest revision id. **Already done
for the live database** — `alembic current` there reports `50d9509d427d (head)`.

### 8.3 Deploy the backend to Render

1. Push this repo to GitHub.
2. In the Render dashboard: **New** → **Web Service**, connect the repo, Language =
   Docker.
3. Set **Dockerfile Path** to `backend/Dockerfile` and **Docker Build Context Directory**
   to `backend` (the Dockerfile's `COPY . .` needs its context scoped to `backend/`, not
   the whole monorepo) — leave **Root Directory** blank.
4. Set **Health Check Path** to `/health` (not Render's default `/healthz`).
5. Pick the **Free** plan, matching `render.yaml`.
6. Add environment variables:

   | Variable | Value |
   | --- | --- |
   | `DATABASE_URL` | Neon **pooled** connection string, `postgresql+asyncpg://...` |
   | `DIRECT_URL` | Neon **direct** connection string, `postgresql+asyncpg://...` |
   | `CORS_ORIGINS` | Your Vercel frontend's URL, e.g. `https://imageqc.vercel.app` (comma-separate for more than one) |
   | `MODEL_WEIGHTS_PATH` | `app/ml/weights/mobilenetv3_iqa.pt` (default) |
   | `ANOMALY_MODEL_PATH` | `app/ml/weights/anomaly_iforest.joblib` (default) |
   | `MAX_UPLOAD_MB` | `10` (or your preferred limit) |
7. Deploy. Render builds `backend/Dockerfile` (CPU-only PyTorch, ~500MB image) — first
   build takes several minutes given PyTorch/OpenCV's size.

Alternatively, use Render's **Blueprint** flow (**New** → **Blueprint**), which reads
`render.yaml` and pre-fills most of the above automatically, prompting only for the
`sync: false` env vars in the table.

**Free tier note:** free web services spin down after ~15 minutes of no traffic and
cold-start on the next request (10–30+ seconds, since the container has to boot Python,
load PyTorch, and load the model weights before `/health` goes green). This is expected
free-tier behavior, not a bug — the frontend's loading state acknowledges it ("a little
longer if the server has been idle").

### 8.4 Deploy the frontend to Vercel

1. In the Vercel dashboard: **Add New** → **Project**, import this repo, set **Root
   Directory** to `frontend/` (this is a monorepo — Vercel needs to know the Next.js app
   isn't at the repo root).
2. Add one environment variable: `NEXT_PUBLIC_API_URL` = the Render backend URL from §8.3
   (`https://imageqc.onrender.com`, no trailing slash).
3. Deploy. Vercel auto-detects Next.js and needs no other configuration — this repo
   intentionally has no `frontend/vercel.json`, since there's no custom header/redirect/
   rewrite behavior to declare.
4. Once live, take the Vercel URL and go back to Render's dashboard to set
   `CORS_ORIGINS` to match it exactly.

### 8.5 Post-deploy smoke test

- [x] `curl https://imageqc.onrender.com/health` → `{"status":"ok","model_loaded":true}`
- [x] Upload a real photo via the API and confirm the result: quality score, label,
      issues, image stats — all render correctly, `model_version` reports `1.1.0`
- [x] Grad-CAM heatmap endpoint returns a real PNG
- [x] History (paginated list) endpoint shows uploaded analyses
- [x] Detail endpoint + original-image endpoint both return the stored data
- [x] Non-image file upload returns a clean `400`, not a crash
- [ ] Open the live Vercel URL and repeat the same checks through the actual UI, once
      deployed

If the backend was asleep (free tier), expect the *first* request to take noticeably
longer — that's the cold start described in §8.3, not a failure.

## 9. Bonus / optional work (brief §13)

| Item | Status |
| --- | --- |
| Quality heatmaps / localization | **Done** — Grad-CAM (§2, §5) |
| Automated backend tests | **Done** — 40 tests, `backend/tests/`, real Postgres, real model weights |
| Batch image analysis | **Done** — `POST /api/analyze/batch`, §5 above |
| Model versioning | **Done** — `backend/app/ml/model_metadata.py`, `model_version` column + API field; this submission ships v1.1.0 (see §3's "Model version history" for the real v1.0.0 → v1.1.0 story) |
| Confidence calibration | **Done** — per-head temperature scaling, `ml_training/calibrate.py` + `backend/app/ml/calibration.py`, fit on the validation split |
| Performance optimization for concurrent requests | **Done** — found and fixed a real bug: `analyze_image()` and Grad-CAM are blocking, CPU-bound calls that were sitting directly in `async def` routes, serializing concurrent requests behind each other's inference time; moved both to `asyncio.to_thread` |
| CI/CD | **Done** — `.github/workflows/ci.yml`: backend tests against a real Postgres service container, frontend lint/typecheck/build |
| Monitoring / logging | **Done** — structured JSON logging (`backend/app/core/logging_config.py`): every request (method/path/status/latency) and every analysis result (score, label, issues, model version) |

## 10. Submission checklist (brief §12)

- [x] Complete source code — frontend (`frontend/`), backend (`backend/`), AI/ML (`ml_training/`, `backend/app/ml/`)
- [x] README with setup, model/training, API, and deployment instructions — this file
- [x] Database setup instructions — §4 above, and §8.1–8.2
- [x] API documentation / example requests — §5 above
- [x] Evaluation results and technical explanation — §3 above
- [x] Sample images demonstrating different quality conditions — `sample_images/` (3 real KADID-10k **test-split** images per category, 18 total)
- [x] Docker / Docker Compose configuration — `backend/Dockerfile`, `docker-compose.yml`
- [x] Deployed URL — backend live at `https://imageqc.onrender.com`; frontend pending Vercel deploy (§8.4)
