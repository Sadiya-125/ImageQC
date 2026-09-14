# ImageQC - AI-Powered Image Quality & Defect Detection

Full-stack app that scores an uploaded image's quality (0-100), labels it ACCEPTABLE /
DEGRADED / DEFECTIVE, and detects 6 issue types: blur, underexposure, overexposure,
noise, corruption, potential visual defect.

**Status:**

|                   | URL                                                                                             |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| Backend (Render)  | **[https://imageqc.onrender.com](https://imageqc.onrender.com)** - live, migrated, smoke-tested |
| Frontend (Vercel) | deploy with the steps in [Deployment](#8-deployment)                                            |

## Table of contents

1. [Architecture](#1-architecture)
2. [Model: design, data, training](#2-model-design-data-training)
3. [Evaluation results](#3-evaluation-results)
4. [Local setup](#4-local-setup)
5. [API documentation](#5-api-documentation)
6. [Production inference notes](#6-production-inference-notes)
7. [Assessment criteria coverage](#7-assessment-criteria-coverage)
8. [Deployment](#8-deployment)
9. [Bonus / optional work](#9-bonus--optional-work)

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
                     │  core/    inference │   not per-request
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

   ml_training/  (offline, RTX 3050 laptop GPU - never runs in production)
   ┌─────────────────────────────────────────────────────────────────┐
   │ data_gen/ → dataset.py → train.py → evaluate.py                 │
   │ KADID-10k (real benchmark) ──labels──► HybridQualityModel       │
   │                                        (MobileNetV3-Small +     │
   │                                         classical CV features)  │
   │                                        + IsolationForest        │
   │  outputs: backend/app/ml/weights/{mobilenetv3_iqa.pt,           │
   │           feature_scaler.joblib, anomaly_iforest.joblib,        │
   │           model_metadata.json, calibration.json}                │
   │  (committed to git - see backend/app/ml/)                       │
   └─────────────────────────────────────────────────────────────────┘
```

`backend/app/ml/` is the single source of truth for model architecture + feature
extraction. Both `ml_training/` (training) and the backend (inference) import from
there, so the two can never silently drift out of sync.

## 2. Model: design, data, training

### Why this design

- Training runs on a **4GB laptop GPU** (RTX 3050); inference runs on a **CPU-only,
  512MB free-tier Render container**; **no external AI/vision APIs allowed**.
- That points to a small ImageNet-pretrained CNN: **MobileNetV3-Small** (~1.1M trainable
  params after dropping its classifier), fused with cheap classical CV features for
  interpretability and for signals (exposure, basic noise) that don't need learning.
- **5 independent sigmoid heads, not one softmax** - real degraded images are often
  multi-issue at once (underexposed _and_ noisy is common). Softmax would force one
  label per image.
- A separate **Isolation Forest**, fit on classical features from clean images, covers
  anomaly detection and feeds the "potential visual defect" signal alongside the
  corruption head.

### Dataset: KADID-10k

- Real, public benchmark (Lin, Hosu & Saupe, 2019; mirrored on Hugging Face at
  `chaofengc/IQA-PyTorch-Datasets`) - 81 pristine reference images (Pixabay-licensed) ×
  25 distortion types × 5 severity levels = 10,125 distorted images + 81 originals.
- Mapped onto this project's 6 categories deterministically, by
  `ml_training/data_gen/build_labels_from_kadid.py`:

| Category      | KADID-10k distortion types (level ≥ 3 sets the label)                                     |
| ------------- | ----------------------------------------------------------------------------------------- |
| Blur          | Gaussian blur, Lens blur, Motion blur                                                     |
| Underexposure | Darken                                                                                    |
| Overexposure  | Brighten                                                                                  |
| Noise         | White noise, White noise in color component, Impulse noise, Multiplicative noise, Denoise |
| Corruption    | JPEG, JPEG2000, Color quantization, Color block - **plus any type at level 5**            |
| Clean         | The 81 pristine references, plus level 1-2 of any mapped type                             |

- `Mean shift` (type 18) is excluded from exposure categories - its per-level luma shift
  isn't a clean under/over split (brighter at levels 1-2, darker at 4-5), so a single
  severity threshold can't express it.
- **Two corrections made after inspecting the real archive** (not assumed from docs):
  - **DMOS direction**: higher = better, not worse as originally assumed (confirmed by
    KADID-10k's own docs and by measuring DMOS falling monotonically as severity rises).
    `quality_score = 100 * (dmos - 1) / 4`; pristine references get 100.0 directly.
  - **Metadata schema**: `dmos.csv` has only 4 columns - distortion type and level are
    parsed from the filename (`I<ref>_<type>_<level>.png`), not read from columns.
- **Splits**: reference-image level (65/8/8 of 81 references → train/val/test), never
  per distorted-variant - avoids leaking scene content across train/test.
- A hand-captured real-world holdout set was planned as a second eval set
  (`ml_training/data_gen/real_world_holdout/`, wired into `evaluate.py`) but wasn't
  captured this build - reported metrics are within the KADID-10k distortion
  distribution only.
- **Known limitation**: KADID-10k applies one distortion per image, so training labels
  are mostly one-hot. Real photos are often multi-issue (dark _and_ noisy) - the
  architecture can predict co-occurring issues but never trained on an example of one.

### Architecture (exact shapes)

- **Backbone**: `mobilenet_v3_small` (ImageNet-pretrained), classifier dropped,
  `features` → `AdaptiveAvgPool2d(1)` → **576-d** pooled embedding.
- **Classical features branch**: **8-d** vector (`laplacian_variance`, `mean_luma`,
  `pct_clipped_low`, `pct_clipped_high`, `luma_skew`, `noise_estimate` via Immerkær's
  method, `contrast_std`, `jpeg_blockiness`), standardized via `StandardScaler`, then
  `Linear(8,32) → BatchNorm1d → ReLU → Linear(32,32) → ReLU`.
- **Fusion trunk**: concat 576-d + 32-d = 608-d →
  `Linear(608,256) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(256,128) → ReLU → Dropout(0.2)`.
- **Heads**: 5x `Linear(128,1) → Sigmoid` (blur, underexposure, overexposure, noise,
  corruption) + 1x `Linear(128,1) → Sigmoid × 100` for `quality_score`. ~1.1M trainable
  params total.
- **"Potential visual defect"** isn't a 7th head - computed as
  `corruption_prob > 0.5 OR isolation_forest.is_anomalous(features)`.
- **Quality label thresholds**: `score >= 70` → ACCEPTABLE, `40-69` → DEGRADED, else
  DEFECTIVE. An issue is reported once its head clears **0.5**; severity bands: low
  (<0.65) / medium (<0.85) / high (≥0.85).
- **Explainability**: Grad-CAM against the CNN's last conv layer, via
  `GET /api/analyses/{id}/gradcam?head=<issue>` and the frontend's heatmap viewer, plus
  classical feature values shown in a readable panel.

### Training

- `ml_training/train.py` on an **RTX 3050 (4GB VRAM)** - two-phase (frozen backbone →
  last 3 blocks unfrozen), mixed precision, 8 epochs, ~21-35 min wall-clock.
- Loss: `Σ BCE(issue_head_i) + λ_quality * SmoothL1(quality_score)`.
- Current model: **v1.1.0** - see [Evaluation results](#3-evaluation-results) for the
  v1.0.0 → v1.1.0 story.
- Confidence is calibrated post-hoc via per-head temperature scaling
  (`ml_training/calibrate.py` fits it, `backend/app/ml/calibration.py` applies it) - API
  `confidence` values are calibrated, not raw sigmoid outputs.

## 3. Evaluation results

From `ml_training/evaluate.py` against the v1.1.0 checkpoint (best val macro-F1:
0.5127), on the **KADID-10k held-out test split** - unseen reference images, same
synthetic-distortion distribution as training.

### Model version history: v1.0.0 → v1.1.0

- After v1.0.0 shipped, real usage showed images unlike KADID-10k's 81
  natural-photography references (ID cards, documents, signatures) scored confidently
  DEFECTIVE regardless of actual condition.
- Diagnosis (full derivation in `train.py`'s docstring): v1.0.0's `quality_loss`
  outweighed the summed `issue_loss` ~10.2x across all 8 epochs.
- Fix (v1.1.0): `λ_quality` lowered 1.0 → 0.1 (≈ the observed loss ratio), retrained from
  scratch, same data/architecture/epochs, plus post-hoc calibration.

| Metric (KADID-10k test split) | v1.0.0        | v1.1.0        | Change                            |
| ----------------------------- | ------------- | ------------- | --------------------------------- |
| Best val macro-F1 (training)  | 0.3053        | 0.5127        | **+68%**                          |
| blur F1 / AUC                 | 0.472 / 0.910 | 0.627 / 0.959 | better                            |
| underexposure F1 / AUC        | 0.000 / 0.632 | 0.000 / 0.769 | AUC better, still below threshold |
| overexposure F1 / AUC         | 0.000 / 0.703 | 0.450 / 0.878 | **F1 unstuck from 0**             |
| noise F1 / AUC                | 0.527 / 0.889 | 0.625 / 0.910 | better                            |
| corruption F1 / AUC           | 0.451 / 0.698 | 0.498 / 0.693 | roughly flat                      |
| quality_score MAE             | 20.47         | 22.60         | **worse**                         |
| quality_score SROCC / PLCC    | 0.520 / 0.517 | 0.449 / 0.449 | **worse**                         |

- **Honest trade-off, not a wash**: rebalancing toward classification improved 4 of 5
  issue heads (the actual required capability) at a real cost to `quality_score`
  regression (MAE +2.13, correlation -0.07). An intermediate `λ_quality` (0.3-0.5) is
  the next experiment to recover regression accuracy.
- `underexposure` still never crosses the 0.5 threshold at 2.4% prevalence - rebalancing
  the loss terms didn't fix the _class_ imbalance. Needs positive-class weighting or a
  weighted sampler (not applied here).

### KADID-10k test split (n = 1008, 0.5 probability threshold)

| Head          | Precision | Recall | F1    | ROC-AUC | Support (pos/total) | Confusion Matrix           |
| ------------- | --------- | ------ | ----- | ------- | ------------------- | -------------------------- |
| blur          | 0.603     | 0.653  | 0.627 | 0.959   | 72/1008             | TN=905 FP=31 FN=25 TP=47   |
| underexposure | 0.000     | 0.000  | 0.000 | 0.769   | 24/1008             | TN=984 FP=0 FN=24 TP=0     |
| overexposure  | 0.562     | 0.375  | 0.450 | 0.878   | 24/1008             | TN=977 FP=7 FN=15 TP=9     |
| noise         | 0.594     | 0.658  | 0.625 | 0.910   | 120/1008            | TN=834 FP=54 FN=41 TP=79   |
| corruption    | 0.403     | 0.652  | 0.498 | 0.693   | 264/1008            | TN=489 FP=255 FN=92 TP=172 |

- **Quality score regression**: MAE = 22.60 (0-100 scale), SROCC = 0.449, PLCC = 0.449.
- **Anomaly detector** ("potential visual defect" flag), vs. `corruption == 1` as proxy
  ground truth: precision = 0.387, recall = 0.091, F1 = 0.147 (62/1008 flagged, 264
  truly corrupted). Unchanged from v1.0.0 - it's fit on classical features only.
- **Failure cases**: worst 8 predictions per head, with true/predicted labels -
  [`failure_cases/README.md`](ml_training/notebooks/failure_cases/README.md). Real
  correct predictions -
  [`accepted_cases/README.md`](ml_training/notebooks/accepted_cases/README.md).

## 4. Local setup

Requires Docker Desktop, Node.js 20+, Python 3.11+.

```bash
git clone https://github.com/Sadiya-125/ImageQC && cd "ImageQC"

# 1. Local Postgres + backend, via Docker
docker compose up -d db backend
# db on localhost:5433, backend on localhost:8000 (override with BACKEND_HOST_PORT=xxxx)

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

- Confirm it's alive: `curl http://localhost:8000/health` → `{"status":"ok","model_loaded":true}`.
- **Without Docker** (faster iteration): `docker compose up -d db`, then from `backend/`
  with its venv active, `uvicorn app.main:app --reload`. Same code either way.

**Test suite:**

```bash
cd backend
pytest tests/ -v
```

- 40 tests, all passing: classical-feature unit tests, model-loading/output-shape tests
  against real committed weights, API integration tests against a real Postgres test DB
  (analyze/batch-analyze, pagination, delete, 404s, Grad-CAM/image endpoints).
- CI runs the same suite on every push (`.github/workflows/ci.yml`), plus frontend
  lint/typecheck/build.

## 5. API documentation

Base URL: `http://localhost:8000` locally, or `https://imageqc.onrender.com` (live).
Swagger UI at `/docs` on any running instance.

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
    { "type": "corruption", "severity": "low", "confidence": 0.6357 },
    { "type": "potential_defect", "severity": "low", "confidence": 0.6357 }
  ],
  "image_stats": {
    "laplacian_variance": 757.66,
    "mean_luma": 109.1,
    "...": "..."
  },
  "gradcam_available": true,
  "model_version": "1.1.0",
  "created_at": "2026-09-14T14:07:05.973494Z"
}
```

`400` for an unreadable/corrupt file, `413` for a file over `MAX_UPLOAD_MB` (default
10). Both are real, tested error paths (`backend/tests/test_api.py`).

### `POST /api/analyze/batch`

Same pipeline, up to 10 files per request, each validated/persisted independently - one
bad file doesn't fail the others.

```bash
curl -X POST http://localhost:8000/api/analyze/batch \
  -F "files=@sample_images/clean/I15.png;type=image/png" \
  -F "files=@sample_images/corrupted/I15_10_05.png;type=image/png"
```

```json
{
  "results": [
    {
      "filename": "I15.png",
      "success": true,
      "result": { "...": "full AnalyzeResponse" },
      "error": null
    },
    {
      "filename": "I15_10_05.png",
      "success": true,
      "result": { "...": "full AnalyzeResponse" },
      "error": null
    }
  ]
}
```

### `GET /api/analyses?page=1&page_size=20`

Paginated history, most recent first (excludes raw image and image_stats).

```bash
curl "http://localhost:8000/api/analyses?page=1&page_size=10"
```

```json
{
  "items": [
    {
      "id": "...",
      "filename": "...",
      "quality_score": 71.94,
      "quality_label": "ACCEPTABLE",
      "created_at": "..."
    }
  ],
  "total": 3,
  "page": 1,
  "page_size": 10
}
```

### `GET /api/analyses/{id}`

Full detail for one past analysis (issues + image_stats included). `404` if not found.

```bash
curl http://localhost:8000/api/analyses/088b363b-1593-41e0-9ddc-25a7893d16e5
```

### `DELETE /api/analyses/{id}`

```bash
curl -X DELETE http://localhost:8000/api/analyses/088b363b-1593-41e0-9ddc-25a7893d16e5
# 204 No Content
```

### `GET /api/analyses/{id}/image`

Original uploaded bytes, as stored.

```bash
curl http://localhost:8000/api/analyses/088b363b-1593-41e0-9ddc-25a7893d16e5/image -o original.png
```

### `GET /api/analyses/{id}/gradcam?head=<blur|underexposure|overexposure|noise|corruption>`

Regenerates a Grad-CAM heatmap overlay on demand (not pre-stored) and streams it as a
PNG. `400` for an unrecognized `head`.

```bash
curl "http://localhost:8000/api/analyses/088b363b-1593-41e0-9ddc-25a7893d16e5/gradcam?head=corruption" -o heatmap.png
```

## 6. Production inference notes

- **CPU-only, explicitly** - `torch.device("cpu")` hard-coded (Render has no GPU;
  MobileNetV3-Small was chosen specifically to be CPU-fast).
- **Loaded once, not per-request** - FastAPI's `lifespan` handler loads the model at
  startup; every request reuses one in-memory singleton. If loading fails, `/health`
  reports `503` instead of crash-looping.
- **Blocking calls offloaded from the event loop** - `analyze_image()` and Grad-CAM
  generation run via `asyncio.to_thread(...)` so slow inference doesn't serialize
  concurrent requests.
- **Real measured latency: ~53ms/image** (52.9ms average over 20 warmed-up runs).
- **Model artifacts committed to git** (`backend/app/ml/weights/*`, ~5.5MB total) so a
  fresh clone or Render build has something to load without re-running training.
- **Docker image**: `python:3.11-slim` base, CPU-only PyTorch wheels, non-root user,
  `HEALTHCHECK` against `/health`. **530MB** final image size.

## 7. Deployment

Neon (database) → Render (backend, via `render.yaml`) → Vercel (frontend). **Backend is
already live**; the Vercel step for the frontend is what remains.

### 7.1 Create the Neon project

- Sign up at [neon.tech](https://neon.tech), create a project.
- Get two connection strings from **Connection Details**:
  - **Pooled** (host has `-pooler`) → `DATABASE_URL` - app runtime traffic.
  - **Direct** (no `-pooler`) → `DIRECT_URL` - Alembic migrations only (pooled
    connections don't reliably support Alembic's DDL transactions).
- Rewrite both for asyncpg before use anywhere in this app:
  ```
  postgresql://user:pass@host/db?sslmode=require&channel_binding=require
    ->
  postgresql+asyncpg://user:pass@host/db?ssl=require
  ```
- Save both privately (a password manager, not a committed file).

### 7.2 Run Alembic migrations against `DIRECT_URL`

Once, before any client serves traffic against a fresh database:

```bash
cd backend
cp .env.example .env
# edit .env: paste your real DATABASE_URL and DIRECT_URL
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
alembic upgrade head
```

Already done for the live database - `alembic current` there reports
`50d9509d427d (head)`.

### 7.3 Deploy the backend to Render

- **New** → **Web Service**, connect the repo, Language = Docker.
- **Dockerfile Path**: `backend/Dockerfile`. **Docker Build Context Directory**:
  `backend`. **Root Directory**: blank.
- **Health Check Path**: `/health` (not Render's default `/healthz`).
- Plan: **Free** (matches `render.yaml`).
- Environment variables:

  | Variable             | Value                                                  |
  | -------------------- | ------------------------------------------------------ |
  | `DATABASE_URL`       | Neon **pooled** string, `postgresql+asyncpg://...`     |
  | `DIRECT_URL`         | Neon **direct** string, `postgresql+asyncpg://...`     |
  | `CORS_ORIGINS`       | Vercel frontend URL, e.g. `https://imageqc.vercel.app` |
  | `MODEL_WEIGHTS_PATH` | `app/ml/weights/mobilenetv3_iqa.pt` (default)          |
  | `ANOMALY_MODEL_PATH` | `app/ml/weights/anomaly_iforest.joblib` (default)      |
  | `MAX_UPLOAD_MB`      | `10` (or your limit)                                   |

- Deploy - first build takes several minutes (CPU-only PyTorch, ~500MB image).
- Or use Render's **Blueprint** flow (reads `render.yaml`, pre-fills most of this).
- **Free tier**: spins down after ~15 min idle, cold-starts in 10-30+ seconds on the
  next request. Expected, not a bug - the frontend's loading state acknowledges it.

### 7.4 Deploy the frontend to Vercel

- **Add New** → **Project**, import the repo, **Root Directory** = `frontend/`.
- Env var: `NEXT_PUBLIC_API_URL` = the Render URL (`https://imageqc.onrender.com`, no
  trailing slash).
- Deploy - Vercel auto-detects Next.js, no extra config needed (no `vercel.json`).
- Once live, set Render's `CORS_ORIGINS` to the Vercel URL exactly.

## 8. Bonus / Optional work

| Item                                             | Status                                                                                                                      |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Quality heatmaps / localization                  | **Done** - Grad-CAM (Model, API sections)                                                                                   |
| Automated backend tests                          | **Done** - 40 tests, real Postgres, real model weights                                                                      |
| Batch image analysis                             | **Done** - `POST /api/analyze/batch`                                                                                        |
| Model versioning                                 | **Done** - `model_metadata.py`, `model_version` column + API field; ships v1.1.0                                            |
| Confidence calibration                           | **Done** - per-head temperature scaling, fit on the validation split                                                        |
| Performance optimization for concurrent requests | **Done** - fixed a real bug where blocking CPU-bound inference serialized concurrent requests; moved to `asyncio.to_thread` |
| CI/CD                                            | **Done** - backend tests against a real Postgres service container, frontend lint/typecheck/build                           |
| Monitoring / logging                             | **Done** - structured JSON logging: every request and every analysis result                                                 |
