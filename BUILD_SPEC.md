# AI-Powered Image Quality & Defect Detection — Build Spec & Claude Code Prompt Chain

This document is written to be pasted straight into Claude Code, section by section, as a working session. It contains: the system architecture, the model R&D decision (with reasoning and exact layer-by-layer specs), the dataset plan (built on the real, publicly downloadable KADID-10k benchmark — no synthetic data generation), the full repo layout, and an ordered chain of prompts that take you from your current partially-scaffolded folder to a deployed app. Read the "Decisions" sections (0–3) once yourself before you start — everything after that is written to be handed to the agent.

**Current state assumed by Prompt 0**: `frontend/` and `backend/` are already siblings at the repo root, `backend/` is empty and ready for the FastAPI scaffold, and `BUILD_SPEC.md` sits at the root. `frontend/` also already contains `AGENTS.md` and `CLAUDE.md` from earlier scaffolding — Prompt 0 has Claude Code check these before touching anything. If your structure looks different by the time you start, skim Prompt 0 and adjust the specifics; the intent (verify the layout, then scaffold `backend/` and the rest around the existing `frontend/`) is what matters.

---

## 0. Constraints that shaped every decision

- **Your GPU**: RTX 3050 laptop, 4GB VRAM. This is enough to train a small CNN or fine-tune a lightweight transfer-learning backbone at modest batch sizes (16–32) with mixed precision. It is _not_ enough to train anything ResNet-50-class or larger comfortably, and it's not meant to be — the assessment explicitly asks for a **lightweight** model, so this constraint and the assignment's own preference point the same direction.
- **Render free/starter web service**: 512MB RAM on free tier, CPU-only, cold starts after 15 min idle. Even on a paid Render tier you'd be on CPU unless you pay for a GPU instance, which is unnecessary here. This means: **inference in production runs on CPU**, so the model must be small enough for sub-second CPU inference (a few MFLOPs–low hundreds of MFLOPs, a few million parameters at most).
- **No external AI APIs allowed** — so no calling out to OpenAI/Google Vision/etc. Everything must be your own trained model + classical CV features running locally in the container.
- **Frontend on Vercel, backend on Render, DB on Neon** — three separate deployment targets, so the backend must be a fully standalone REST API with CORS configured for the Vercel domain, and the DB must be reached over the network (pooled connection) rather than assumed local.

These four constraints converge on one answer: **a small transfer-learning CNN (2–5M params), trained on your GPU, exported to a CPU-fast inference path, combined with classical CV features for interpretability and for the issue types that don't need learning (exposure, basic noise estimation).** Full reasoning below.

---

## 1. Model architecture decision (the R&D)

### 1.1 What the field actually uses for this problem

Image Quality Assessment (IQA) research gives a clear, converging answer for "detect blur/noise/exposure/corruption without a reference image" (this is **No-Reference IQA**, or **NR-IQA**):

- The standard **benchmark datasets** for synthetic-distortion NR-IQA are **KADID-10k** (81 pristine images × 25 distortion types × 5 severity levels = 10,125 images covering blur, noise, compression, brightness/exposure, color, and sharpness/contrast issues) and **KonIQ-10k** (10,073 real-world "in-the-wild" images with crowd-sourced quality ratings covering noise, JPEG artifacts, blur, wrong exposure, over-saturation). KADID-10k in particular is built almost exactly around your six required categories.
- The standard **model class** for lightweight, deployable IQA/classification in this space is a **small transfer-learning CNN** — MobileNetV3-Small or EfficientNet-Lite/B0 pretrained on ImageNet, fine-tuned on the quality task. MobileNetV3-Small-1.0 is ~2.5M parameters and ~56–66 MFLOPs — small enough to run inference on CPU in well under 100ms, and small enough to fine-tune on a 4GB GPU with room to spare. This is a well-established, defensible, "field-standard" choice, not an exotic one — recent lightweight IQA challenges (e.g. VQualA 2025) explicitly cap models at 5M params / 0.5 GFLOPs, which is the same size class.
- Purely classical (hand-crafted-feature) approaches are explicitly marked insufficient for full credit by the assessment itself ("a traditional computer-vision-only solution is not sufficient for full credit"), which rules out a features-plus-SVM-only design as your _primary_ method — but classical features remain valuable as **inputs alongside the CNN** (hybrid approach) and as the _entire_ mechanism for exposure and basic noise estimation, where they are actually more reliable and more explainable than a learned model would be at this data scale.

### 1.2 The chosen architecture: hybrid, multi-head, multi-label

**Backbone**: MobileNetV3-Small (`torchvision.models.mobilenet_v3_small`, ImageNet-pretrained), used as a frozen-then-fine-tuned feature extractor.

**Heads** (this is a multi-label problem — an image can be simultaneously blurry AND underexposed AND noisy):

- One shared MobileNetV3-Small trunk → global pooled feature vector (576-d — see §1.5 for the exact layer-by-layer spec).
- Concatenate the pooled CNN features with a small vector of **classical, cheaply-computed image statistics** (this is your explicit "hybrid approach"):
  - Laplacian variance (blur/sharpness signal)
  - Mean luma + luma histogram skew (exposure)
  - Percentage of pixels clipped at 0/255 (over/under-exposure)
  - Noise estimate via wavelet-based or high-frequency-residual noise estimator
  - Global contrast (std of luma)
  - JPEG blockiness estimate (for "corruption")
- Concatenated vector → small fully connected trunk (256 → 128) → **5 independent sigmoid output heads**, one per issue type (blur, underexposure, overexposure, noise, corruption), each producing a probability; plus a **regression head** producing the overall `quality_score` (0–100), trained against the DMOS-derived target from KADID-10k.
- "Potential visual defect" (the 6th required category) is handled as a **derived/composite flag**: triggered when corruption probability is high OR when an anomaly-detection check (see 1.4) flags the image as statistically unlike the training distribution of "normal" photos. This is deliberately not folded into the same softmax as the others because "defect" in the assessment's framing is closer to an anomaly/catch-all than a distinct visual distortion class — treating it as anomaly detection is explicitly named as an acceptable formulation in the brief, so this gives you two of the four suggested AI approaches (classical-feature+learned-model hybrid, _and_ anomaly detection) in one coherent design instead of bolting on a second unrelated model.

**Why multi-label sigmoid heads instead of one softmax classifier**: real degraded images very often have more than one problem at once (underexposed AND noisy is extremely common — noise increases as exposure/ISO drops). A softmax forces one label per image and will misrepresent reality; independent sigmoid heads let each issue be predicted on its own axis with its own confidence, which also maps directly onto the required JSON response shape (`issues: [{type, severity, confidence}]`).

**Why hybrid instead of pure CNN**: (a) it satisfies the assessment's explicit hybrid-approach option, (b) it gives you free, robust, already-interpretable signals for exposure and blur that a 10k-image fine-tune might otherwise learn noisily, (c) it gives you natural material for the Explainability section (§10 of the brief) without needing Grad-CAM as your _only_ explanation mechanism — though you should still implement Grad-CAM on the CNN branch since it's explicitly named in the brief and is genuinely informative for spatial defect localization (also covers the optional "quality heatmaps / localization" bonus in one implementation).

### 1.3 Framework: PyTorch

PyTorch over TensorFlow because: torchvision ships MobileNetV3-Small with ImageNet weights ready to load in one line, Grad-CAM implementations for torchvision models are extremely well-trodden, TorchScript/ONNX export for fast CPU inference is straightforward, and it is the more common choice in current NR-IQA literature (easier to find reference implementations if you get stuck). TorchServe is unnecessary here — plain PyTorch inference wrapped in FastAPI is enough at this scale and this is more transparent for the "document how the model is loaded and inference performed" deployment requirement.

### 1.4 Anomaly detection component

Train a small **Isolation Forest** (or a simple autoencoder reconstruction-error check — Isolation Forest is faster to justify and evaluate for a 48-hour scope) on the classical feature vector (the same 6–8 statistics above) extracted from a large sample of _clean/pristine_ images only. At inference, compute the anomaly score for the incoming image's feature vector; a high anomaly score contributes to the `defect` flag and its confidence. This is cheap, fast on CPU, easy to evaluate (contamination rate, precision/recall against known-corrupted holdout images), and easy to explain in the write-up — exactly proportioned to a 48-hour assessment rather than over-engineered.

### 1.5 Full architecture specification (exact shapes and layers)

This is the concrete spec Prompt 2 will implement — pin these numbers now so training and inference agree.

**Backbone — MobileNetV3-Small, ImageNet-pretrained (`torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)`)**:

- Input: RGB image resized to 224×224, normalized with standard ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`).
- Use `model.features` (the convolutional trunk) followed by `nn.AdaptiveAvgPool2d(1)` and flatten — this drops the original 1000-class ImageNet classifier entirely. MobileNetV3-Small's final feature map before pooling has **576 channels** (the last inverted-residual block output, after the 1×1 expansion conv — verify this against the actual torchvision source when implementing, since minor version differences can shift it, but 576 is correct for the standard 1.0-width small variant), so the pooled embedding is a **576-d vector**.
- Total backbone parameters: ~1.0M (feature layers only, classifier dropped) out of the model's full ~2.5M — the classifier head you're discarding accounts for most of the rest.

**Classical-features branch**:

- Input: an 7-dimensional feature vector (Laplacian variance, mean luma, shadow-clip %, highlight-clip %, luma histogram skew, noise estimate, contrast std — pin this exact ordered list as a module-level constant, e.g. `FEATURE_NAMES = ["laplacian_var", "mean_luma", "shadow_clip_pct", "highlight_clip_pct", "luma_skew", "noise_estimate", "contrast_std"]`, since training and inference must agree on order byte-for-byte).
- Standardize each feature (zero mean, unit variance) using statistics computed once from the training set and saved alongside the model weights (a `feature_scaler.joblib`, `sklearn.preprocessing.StandardScaler`) — raw pixel-intensity-scale features and 0–1-scale features must not be fed to a linear layer unnormalized.
- `Linear(7, 32) → BatchNorm1d(32) → ReLU → Linear(32, 32) → ReLU` → a 32-d classical embedding.

**Fusion trunk**:

- Concatenate: 576-d CNN embedding + 32-d classical embedding = **608-d fused vector**.
- `Linear(608, 256) → BatchNorm1d(256) → ReLU → Dropout(0.3) → Linear(256, 128) → ReLU → Dropout(0.2)` → a 128-d shared representation.

**Output heads** (all read from the same 128-d shared representation, each is its own small `nn.Module` so Grad-CAM and per-head loss weighting stay clean):

- 5 issue heads: `Linear(128, 1) → Sigmoid`, one each for `blur`, `underexposure`, `overexposure`, `noise`, `corruption`. Output: independent probability per head, not summing to 1.
- 1 regression head: `Linear(128, 1)` (no output activation — regress directly to a 0–100-scaled target, or use `Sigmoid` then multiply by 100 if you want the output range hard-bounded; document whichever you pick).
- The "defect" flag is _not_ a 7th head — it's computed post-hoc in `inference.py` as `defect = (corruption_prob > threshold) OR isolation_forest.is_anomalous(classical_features)`, combining the corruption head's own signal with the independent anomaly detector (§1.4) as specified.

**Total trainable parameters**: roughly 1.0M (unfrozen backbone layers, once you unfreeze the last few blocks in phase 2 of training) + ~0.02M (classical branch) + ~0.19M (fusion trunk) + ~0.001M (heads) ≈ **1.2–1.5M parameters actively trained**, well inside what a 4GB GPU handles easily at batch size 16–32, and well inside fast-CPU-inference range at deployment.

**Loss function** (exact, for Prompt 3 to implement without re-deriving):

```
total_loss = Σ_i BCE(issue_head_i_pred, issue_head_i_true)  for i in [blur, underexposure, overexposure, noise, corruption]
           + λ * SmoothL1Loss(quality_score_pred, quality_score_true)     # Huber loss, robust to DMOS outliers
```

Start with `λ = 1.0` (all 5 BCE terms summed, not averaged, so the regression term is comparable in magnitude to the combined classification signal) and treat `λ` as a tunable documented in `train.py`'s docstring — if the regression loss dominates or is dominated during early training, adjust and note why in EVALUATION.md rather than silently tuning it away.

**Optimizer/schedule**: AdamW, phase 1 (frozen backbone) lr=1e-3 for the new layers only, phase 2 (last 2–3 backbone blocks unfrozen) lr=1e-5 for backbone / 1e-4 for the rest, cosine annealing, weight decay 1e-4. These are reasonable starting points, not sacred — `train.py` should log enough for you to see if they need adjusting on the real KADID-10k-derived data.

### 1.6 Model size/latency budget (what you're targeting)

| Component                                   | Params                                               | Approx CPU latency (single image, Render-class CPU) |
| ------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| MobileNetV3-Small backbone                  | ~1.5M (feature layers only, classifier head dropped) | ~15–40ms                                            |
| Classical feature extraction (OpenCV/NumPy) | n/a                                                  | ~5–15ms                                             |
| FC heads + Isolation Forest scoring         | <0.1M                                                | <2ms                                                |
| **Total per-image inference**               | **~1.6M params**                                     | **~30–80ms**, well under any reasonable timeout     |

This comfortably fits Render's free-tier 512MB RAM (a fp32 MobileNetV3-Small state dict is a few MB; a quantized version is smaller still) and gives snappy response times without a GPU.

---

## 2. Dataset plan

### 2.1 Primary approach: the real KADID-10k dataset (no synthesis)

Use **KADID-10k** (Kim, Lin, Hosu & Saupe, 2019) as-is. This is a real, publicly downloadable, academically standard benchmark — not something you generate — and it maps almost one-to-one onto the brief's six required categories, which is exactly why it's the right choice here rather than a fallback.

**What it actually is**: 81 pristine reference images (sourced from Pixabay, released under the Pixabay License — free to use, edit, and redistribute, so there is no licensing question for an assessment submission), each degraded with **25 distortion types at 5 severity levels**, giving 81 × 25 × 5 = **10,125 distorted images total**. Every image carries a crowdsourced **DMOS** (Difference Mean Opinion Score, scale 1–5, where _higher DMOS = lower quality_ — note the direction, it's inverted from MOS) from 30 independent raters per image.

**Where to get it**: it's mirrored as a ready-to-download archive on Hugging Face at `chaofengc/IQA-PyTorch-Datasets` (file `kadid10k.tgz`), which is the fastest path — no manual request/access-gate, no scraping, just a direct download. (The original authors' site, `database.mmsp-kn.de`, is the canonical source if you want to cite it that way, but the HF mirror is the practical download path.)

**The 25 distortion types and how they map onto your 6 required categories** (this mapping is the actual R&D deliverable here — it's what you'll put in your README to justify the label scheme):

| Your required category          | KADID-10k distortion types that map to it                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Blur                            | #01 Gaussian blur, #02 Lens blur, #03 Motion blur                                                                                                                                                                                                                                                                                                                                                                               |
| Underexposure                   | #17 Darken (and the darker levels of #19 Mean shift)                                                                                                                                                                                                                                                                                                                                                                            |
| Overexposure                    | #18 Brighten (and the brighter levels of #19 Mean shift)                                                                                                                                                                                                                                                                                                                                                                        |
| Noise                           | #04 Additive Gaussian noise, #05 Additive noise in color components, #06 Impulse noise, #07 Multiplicative noise, #08 Denoise (residual/under-denoised artifacts), #09 Brighten... _(exact numbering varies slightly by source table — verify against the `dmos.csv` / distortion-name column that ships in the archive itself, don't hardcode the code numbers from memory; the archive's own metadata file is authoritative)_ |
| Corruption / severe degradation | #14 JPEG compression, #15 JPEG2000 compression, #24/#25 high-intensity color quantization/color block distortions at severity level 5, and any distortion type at its most severe (level 5) level generally                                                                                                                                                                                                                     |
| Clean / acceptable              | The 81 pristine reference images themselves, plus level-1 (mildest) variants of the more forgiving distortion types                                                                                                                                                                                                                                                                                                             |

Do this mapping **in code, driven by the actual filenames and the distortion-type column in the shipped metadata CSV** — every KADID-10k image filename encodes which reference image, which distortion type, and which level it is (pattern `I<ref>_<distortion>_<level>.png`), so the mapping table above becomes a lookup dict from distortion-type-name to your 6 category labels, not something you re-derive by hand per image.

**Multi-label ground truth**: since KADID-10k applies exactly _one_ distortion type per image (not combinations), your 5 sigmoid heads' ground truth is mostly one-hot per image at training time — this is a real limitation (see §2.4) but the DMOS severity level still gives you a genuine per-image quality gradient, and real-world "combination" cases (e.g. a photo that's both dark and noisy) are then handled at _inference_ time by the fact that each head is trained and thresholded independently — nothing in the architecture requires combination examples to exist in training data, it's the sigmoid-per-head design itself (§1.2) that makes multi-issue detection possible at inference, not combination training samples.

### 2.2 Deriving your labels from KADID-10k's fields

Write a single deterministic label-mapping script (`ml_training/data_gen/build_labels_from_kadid.py`) that:

- Reads the archive's shipped metadata (`dmos.csv` or equivalent — inspect the actual extracted archive to confirm the exact filename/columns, don't assume) containing, per image: reference id, distortion type, level, DMOS, and DMOS standard deviation.
- Applies the distortion-type → your-6-category mapping table above to produce multi-hot binary labels (an image can only carry one "cause" label directly from KADID-10k's own scheme, but you additionally derive a `clean` vs `not-clean` binary from the DMOS threshold — a level-1, high-DMOS-quality image of a "blur" type is still very mildly blurred, so you're free to threshold: e.g. treat level 1–2 of a distortion as _not_ triggering that issue's binary label, only levels 3–5 do — document this threshold explicitly since it directly determines your class balance).
- Converts DMOS (1–5, higher = worse) into your `quality_score` (0–100, higher = better) via a documented linear rescaling: `quality_score = 100 * (5 - dmos) / 4`. State this formula plainly in the README/EVALUATION.md — a reviewer checking your work should be able to verify it from the raw DMOS values in one line.
- Writes out a single `labels.csv` in the same schema your `ml_training/dataset.py` (Prompt 3) expects, so the rest of the training pipeline doesn't need to know or care that the ultimate source was KADID-10k rather than something synthesized.

### 2.3 A necessary supplement: a small real-world holdout set (not for training)

KADID-10k's distortions, despite being carefully calibrated, are still synthetic — applied via filters, not genuine defective camera captures. To satisfy the brief's "evaluation should use unseen images and provide evidence of generalization" in a way that's actually convincing (and not just unseen-but-still-synthetic), collect a small **manually-captured real-world holdout set** yourself: 20–40 photos you deliberately take with genuine out-of-focus blur, genuine underexposure (dim room, no flash), genuine overexposure (backlit/direct sun), genuine sensor noise (high ISO, low light), and a few intentionally corrupted files (truncate a JPEG mid-save, or re-save at quality=3). Label these by hand (you know what you did to each one) and hold them out entirely from training — use them only in `evaluate.py` as a second, independent test set alongside the KADID-10k test split. This is cheap (an afternoon with your phone camera) and is the single highest-leverage thing you can do for the "evidence of generalization" criterion, since it's real camera sensor behavior, not a filter.

### 2.4 Known limitation to document honestly (don't hide this — name it)

Because KADID-10k applies one distortion type per image, your training data under-represents realistic **co-occurring** degradations (e.g., real low-light photos are simultaneously dark _and_ noisy, because raising ISO to compensate for underexposure is what _causes_ the noise — the two aren't independent in real cameras the way they are in this dataset's construction). Say this plainly in EVALUATION.md's limitations section: the model's per-head thresholds are calibrated on single-issue examples, so multi-issue confidence calibration (an image flagged for both noise and underexposure simultaneously) is less validated than single-issue detection, and your real-world holdout set (§2.3) is exactly the evidence you'll cite for how well that generalizes in practice.

### 2.5 Splits

Split KADID-10k **at the reference-image level** (81 pristine images → e.g. 65 train / 8 val / 8 test, roughly 80/10/10), never at the distorted-variant level — since all 125 variants of one reference image share the same underlying scene content, splitting at the variant level would leak content across train/test and inflate your reported metrics. This is the standard protocol used throughout the KADID-10k literature (train/test splits are consistently reported as content-disjoint in every paper using this dataset), so following it isn't just good practice, it's what makes your numbers comparable to published baselines if you want to cite any for context. The real-world holdout set from §2.3 is a third, fully separate evaluation set, never mixed into this split.

### 2.6 Isolation Forest training set

Fit the anomaly detector's classical-feature vectors on the **pristine reference images plus level-1 (mildest) distorted variants only** (i.e., what you're treating as "clean" per the threshold in §2.2); evaluate its flagging behavior against the level-4/level-5 (most severe) distorted images and against the deliberately-corrupted items in your real-world holdout set.

---

## 3. System architecture

```
┌─────────────────────────┐        HTTPS/JSON        ┌──────────────────────────────┐
│   Next.js frontend       │ ───────────────────────▶ │  FastAPI backend (Render)     │
│   (Vercel)                │ ◀─────────────────────── │  - /api/analyze (POST image)  │
│   - Upload UI             │                           │  - /api/analyses (GET history)│
│   - Result display         │                           │  - /api/analyses/{id}         │
│   - History view           │                           │  - /health                    │
└─────────────────────────┘                           │  - inference: PyTorch model    │
                                                          │    + classical CV features    │
                                                          │    + Isolation Forest          │
                                                          └───────────────┬────────────────┘
                                                                          │ asyncpg (pooled)
                                                                          ▼
                                                          ┌──────────────────────────────┐
                                                          │  Neon Postgres (serverless)   │
                                                          │  - analyses table              │
                                                          │  - issues table (1:N)          │
                                                          └──────────────────────────────┘
```

- **Frontend (Next.js, Vercel)**: App Router, TypeScript, Tailwind. Talks to the backend only via `NEXT_PUBLIC_API_URL` env var — no direct DB access from the frontend, everything goes through the API.
- **Backend (FastAPI, Render, Docker)**: single container, model weights baked into the image (or downloaded on startup from a release asset if you want a slimmer image — bake-in is simpler and more reliable for a 512MB-RAM free instance since it avoids a startup network dependency). SQLAlchemy 2.0 async + asyncpg talking to Neon's **pooled** connection string. Alembic for migrations, run against Neon's **direct** (unpooled) connection string.
- **DB (Neon Postgres)**: two tables — `analyses` (one row per uploaded image: filename, quality_score, quality_label, image stats, timestamps) and `analysis_issues` (one row per detected issue: type, severity, confidence, FK to analysis). Store the uploaded image itself either as a small compressed blob in Postgres (simplest for a 48-hour assessment, fine at this scale) or, if you want to demonstrate more production maturity, on Render's local disk under a volume — Postgres blob storage is simpler and more portable, recommended default.

---

## 4. Repository layout

```
imageqc/
├── frontend/                      # Next.js app, deployed to Vercel
│   ├── app/
│   │   ├── page.tsx                # upload + analyze flow
│   │   ├── history/page.tsx        # past analyses list
│   │   └── analyses/[id]/page.tsx  # single analysis detail
│   ├── components/
│   │   ├── UploadDropzone.tsx
│   │   ├── QualityScoreGauge.tsx
│   │   ├── IssueBadgeList.tsx
│   │   ├── GradCamOverlay.tsx
│   │   └── HistoryTable.tsx
│   ├── lib/api.ts                  # typed fetch wrapper for backend
│   ├── .env.local.example
│   ├── next.config.ts
│   └── package.json
│
├── backend/                        # FastAPI app, deployed to Render
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, router mounting
│   │   ├── api/
│   │   │   ├── analyze.py          # POST /api/analyze
│   │   │   └── analyses.py         # GET /api/analyses, /api/analyses/{id}
│   │   ├── core/
│   │   │   ├── config.py           # pydantic-settings, env vars
│   │   │   └── db.py                # async engine/session
│   │   ├── models/
│   │   │   ├── orm.py               # SQLAlchemy models
│   │   │   └── schemas.py           # Pydantic request/response models
│   │   ├── ml/
│   │   │   ├── classical_features.py  # blur/exposure/noise/contrast/jpeg-blockiness fns
│   │   │   ├── cnn_model.py           # MobileNetV3 hybrid multi-head architecture
│   │   │   ├── anomaly.py             # Isolation Forest wrapper
│   │   │   ├── gradcam.py             # Grad-CAM for the CNN branch
│   │   │   ├── inference.py           # loads weights once, orchestrates full pipeline
│   │   │   └── weights/
│   │   │       └── mobilenetv3_iqa.pt
│   │   └── alembic/                 # migrations, run against Neon direct URL
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   └── tests/
│       ├── test_classical_features.py
│       ├── test_api.py
│       └── test_inference.py
│
├── ml_training/                    # NOT deployed — your local GPU training pipeline
│   ├── data_gen/
│   │   ├── download_kadid10k.py         # pulls kadid10k.tgz, extracts it
│   │   ├── build_labels_from_kadid.py   # maps KADID-10k's 25 distortion types
│   │   │                                 # + DMOS onto your 6 categories + quality_score
│   │   └── real_world_holdout/          # your own 20-40 hand-captured, hand-labeled
│   │       └── labels.csv               # real defective/clean photos (§2.3) — never
│   │                                     # used in training, only in evaluate.py
│   ├── dataset.py                       # PyTorch Dataset/DataLoader
│   ├── train.py                         # trains on RTX 3050, mixed precision
│   ├── evaluate.py                      # metrics, confusion matrices, failure cases
│   ├── export_weights.py                # saves final .pt for backend/app/ml/weights/
│   └── notebooks/
│       └── error_analysis.ipynb
│
├── sample_images/                  # required by submission checklist §12
│   ├── clean/
│   ├── blurry/
│   ├── underexposed/
│   ├── overexposed/
│   ├── noisy/
│   └── corrupted/
│
├── docker-compose.yml              # local: backend + local Postgres (Neon-compatible)
├── render.yaml                     # Render Blueprint for backend
├── vercel.json                     # optional Vercel config for frontend
├── README.md
└── EVALUATION.md                   # metrics, confusion matrix, limitations, failure cases
```

---

## 5. Backend API contract

```
GET  /health
     → 200 { "status": "ok", "model_loaded": true }

POST /api/analyze
     multipart/form-data, field "file"
     → 200 {
         "id": "uuid",
         "filename": "photo.jpg",
         "quality_score": 82,
         "quality_label": "ACCEPTABLE",   # ACCEPTABLE | DEGRADED | DEFECTIVE
         "issues": [
           {"type": "noise", "severity": "low", "confidence": 0.71},
           {"type": "underexposure", "severity": "medium", "confidence": 0.55}
         ],
         "image_stats": {
           "sharpness_laplacian_var": 143.2,
           "mean_luma": 61.4,
           "clipped_shadow_pct": 4.1,
           "clipped_highlight_pct": 0.0,
           "noise_estimate": 3.8,
           "contrast_std": 39.7
         },
         "gradcam_available": true,
         "created_at": "2026-09-13T..."
       }
     → 400 invalid/unreadable file, 413 too large, 422 validation error

GET  /api/analyses?limit=20&offset=0
     → 200 { "items": [ ...summaries... ], "total": 137 }

GET  /api/analyses/{id}
     → 200 full record including issues + stats
     → 404 not found

GET  /api/analyses/{id}/gradcam
     → 200 image/png (heatmap overlay) — bonus/localization feature
```

`quality_label` thresholding (document this explicitly in the README): e.g. `score >= 80` → ACCEPTABLE, `50 <= score < 80` → DEGRADED, `score < 50` → DEFECTIVE, with `DEFECTIVE` also force-triggered whenever the anomaly/corruption head fires above its threshold regardless of score.

---

## 6. Deployment specifics (verified current as of Sept 2026)

- **Render**: connect the GitHub repo, choose "Docker" as environment, point at `backend/Dockerfile`. Free tier gives 512MB RAM and CPU-only compute, and free web services **sleep after 15 minutes of inactivity** — the first request after sleep costs a 30–60s cold start while the container and model reload, which you should mention in your README as a known limitation of the free tier (a paid instance removes this). Set `PORT` env var handling (`uvicorn app.main:app --host 0.0.0.0 --port $PORT`) since Render assigns the port dynamically. Use a `render.yaml` Blueprint so the whole service definition (build command, env vars, health check path `/health`) is versioned in the repo rather than clicked together in the dashboard.
- **Neon**: create a project, get two connection strings from the dashboard's Connect widget — the **pooled** one (hostname contains `-pooler`, routes through PgBouncer, supports high concurrency) goes into the backend's `DATABASE_URL` for normal app traffic; the **direct** (non-pooled) one goes into `DIRECT_URL` and is used only for Alembic migrations, since pooled connections can be unreliable for schema changes. Keep `sslmode=require&channel_binding=require` in both — Neon requires TLS. Use `postgresql+asyncpg://...` scheme for the SQLAlchemy async engine.
- **Vercel**: standard Next.js deploy, set `NEXT_PUBLIC_API_URL` to the Render backend's public URL in Vercel's project env vars. Configure CORS on the FastAPI backend (`fastapi.middleware.cors.CORSMiddleware`) to explicitly allow the Vercel domain(s) — including the Vercel preview-deployment wildcard domain if you want PR previews to work against the same backend.
- **Docker Compose (local)**: `backend` service + a local `postgres:16` container that mimics Neon for local dev (same schema, same asyncpg driver) so you're not burning Neon connections while iterating locally; swap `DATABASE_URL` to the real Neon URL only for the deployed environment.

---

## 7. Evaluation plan (brief §9)

Report, per issue-type head: precision, recall, F1 (since these are independent binary classifications, not one multi-class problem), plus an overall confusion matrix per head and ROC-AUC per head since your outputs are probabilities. For the regression head (`quality_score`), report MAE and Pearson/Spearman correlation against the DMOS-derived ground-truth score — SROCC/PLCC are literally the standard metrics in the NR-IQA papers you're drawing this design from (§1.1), so using them also signals you did the reading, and lets you contextualize your numbers against published KADID-10k baselines if you want to cite any. For the anomaly detector, report precision/recall of the "defect" flag against your held-out corrupted set (both the KADID-10k level-5 subset and your real-world holdout), and separately show a handful of **failure cases** (false positives on unusual-but-valid photos — e.g. intentionally artistic high-contrast/silhouette shots — and false negatives on subtle defects) with a short discussion, since the brief explicitly asks for this.

---

## 8. The Claude Code prompt chain

Copy these prompts into Claude Code **in order**, one at a time, letting each complete (and reviewing output) before moving to the next. Each prompt is self-contained but assumes the previous steps' files exist.

### Prompt 0 — Verify the folder structure, then scaffold the rest

```
The repo root ("IIITH Assessment") should now have frontend/ and backend/
as siblings, with backend/ empty and ready for the FastAPI project, and
BUILD_SPEC.md at the root. Before building anything:

1. Show me a tree listing (a couple levels deep) of the repo root to confirm:
   - frontend/ and backend/ are siblings at the true root
   - backend/ is empty
   - BUILD_SPEC.md sits at the root, not nested
   If any of this isn't true, stop and tell me what you actually find instead
   of proceeding on an assumption.
2. frontend/ contains AGENTS.md and CLAUDE.md already (likely generated by
   whatever scaffolding tool or earlier session created the Next.js project).
   Open both and show me their contents — don't overwrite, delete, or merge
   them into anything yet. If either contains project-specific context worth
   preserving (as opposed to generic boilerplate), tell me what you found so
   we can decide together whether it should inform anything here; otherwise
   leave them exactly as they are.
3. Confirm `npm install` still runs cleanly in frontend/ from its current
   location (just check, don't reinstall unless something is actually
   broken).

Now read BUILD_SPEC.md in full as your standing context for this entire
project (I will refer back to it, do not ask me to repeat it):

ARCHITECTURE: Next.js 14+ (App Router, TypeScript, Tailwind) frontend
(already scaffolded, at ./frontend/, deployed to Vercel); FastAPI
(Python 3.11+) backend (to be built at ./backend/, currently empty) deployed
to Render via Docker; Neon serverless Postgres as the database, accessed via
SQLAlchemy 2.0 async + asyncpg, pooled connection for the app and direct
connection for Alembic migrations. The ML model is a hybrid MobileNetV3-Small
transfer-learning CNN (PyTorch) with 5 independent sigmoid heads (blur,
underexposure, overexposure, noise, corruption) plus a regression head for an
overall 0-100 quality score, fused with classical OpenCV/NumPy image-quality
features (Laplacian variance for sharpness, luma histogram stats for
exposure, noise estimation, contrast, JPEG blockiness), plus a separate
Isolation Forest anomaly detector trained on classical features from
clean/mild images to flag "potential visual defect" as a composite/anomaly
signal. No external AI/vision APIs are used anywhere. Training data comes
from the real, publicly downloadable KADID-10k dataset (81 pristine images ×
25 distortion types × 5 severity levels, mapped onto our 6 required
categories) plus a small hand-captured real-world holdout set — no synthetic
degradation pipeline is being built; the labels come from a real published
benchmark. Training happens locally on an RTX 3050 (4GB VRAM) laptop GPU;
inference in production runs on CPU only (Render free tier, 512MB RAM), so
the deployed model must be small and CPU-fast.

Now build out the rest of the top-level scaffold around what already exists
(folders and placeholder files only for anything not yet built — do not
touch frontend/'s existing contents at all in this step):

imageqc/ (repo root — frontend/ and backend/ already siblings here)
├── frontend/              (already exists — leave entirely as is)
├── backend/               (empty — scaffold FastAPI app package structure,
│                          requirements.txt, Dockerfile placeholder here —
│                          do not implement routes yet)
├── ml_training/            (data_gen/, dataset.py, train.py, evaluate.py,
│                          export_weights.py as empty files with docstrings
│                          describing their intended contents)
├── sample_images/         (clean/, blurry/, underexposed/, overexposed/,
│                          noisy/, corrupted/ — empty dirs with .gitkeep)
├── docker-compose.yml     (placeholder)
├── render.yaml            (placeholder)
├── BUILD_SPEC.md          (already exists — leave as is)
├── README.md              (top-level project description only, we'll fill in
│                          setup instructions at the end)
└── .gitignore             (Python + Node + model weights + .env — check
                           frontend/'s existing .gitignore first and don't
                           duplicate what it already covers for that folder)

Confirm whether a git repo already exists at the root; if not, initialize one
there (not inside frontend/ or backend/ individually — one repo for the whole
project). Do not implement business logic in this step — this is scaffolding
and verification only. Confirm the final structure with a tree listing when
done.
```

### Prompt 1 — Dataset acquisition and label mapping (real KADID-10k, no synthesis)

```
Build ml_training/data_gen/download_kadid10k.py and
ml_training/data_gen/build_labels_from_kadid.py.

download_kadid10k.py: downloads kadid10k.tgz from the Hugging Face dataset
mirror at chaofengc/IQA-PyTorch-Datasets (this is a real, publicly hosted
archive of the KADID-10k benchmark — 81 pristine reference images degraded
with 25 distortion types at 5 severity levels each, 10,125 images total,
sourced originally from Pixabay-licensed photos, so there is no licensing
concern for our use), extracts it into ml_training/data_gen/kadid10k/, and
verifies the extraction produced the expected image count and the shipped
metadata file (locate and confirm the exact metadata filename and its
columns once extracted — it should contain, per image, at minimum: reference
image id, distortion type, distortion level, and DMOS score with its standard
deviation; print the actual column names you find so we can confirm before
writing the mapping logic against them, since I don't want you guessing the
schema).

build_labels_from_kadid.py: using the actual extracted metadata, build our
project's labels.csv with one row per KADID-10k image containing: filename,
reference_id, kadid_distortion_type (raw), kadid_distortion_level (1-5),
dmos, our 5 binary issue labels (blur, underexposure, overexposure, noise,
corruption), and our derived quality_score.

Implement the distortion-type -> our-category mapping as an explicit,
readable dict at the top of the file (don't scatter magic numbers through the
logic), following this starting mapping — but VERIFY it against the actual
distortion-type names/numbers found in the extracted metadata rather than
trusting this list blindly, since numbering can vary slightly by source, and
tell me if what you find differs from this:
  blur types      -> Gaussian blur, Lens blur, Motion blur
  underexposure   -> Darken (and low end of Mean shift if present)
  overexposure    -> Brighten (and high end of Mean shift if present)
  noise types     -> Gaussian noise, color-component noise, impulse noise,
                     multiplicative noise, any denoising-artifact type present
  corruption      -> JPEG compression, JPEG2000 compression, color
                     quantization/color-block distortion, and (regardless of
                     type) any image at severity level 5
  clean           -> all 81 pristine reference images, plus level-1 variants
                     of types not otherwise flagged as an issue at level 1

Apply a documented severity threshold: only levels 3-5 of a mapped distortion
type set that category's binary label to 1; levels 1-2 leave it at 0 (state
this threshold as a named constant, not a magic number inline, since it's a
real modeling decision that affects class balance and needs to be citable in
our writeup).

Convert DMOS (1-5 scale, higher = WORSE, note the direction) to our
quality_score (0-100 scale, higher = BETTER) via:
  quality_score = 100 * (5 - dmos) / 4
Implement exactly this formula, don't substitute a different rescaling.

Split at the REFERENCE-IMAGE level (not per-distorted-variant) into roughly
80/10/10 train/val/test over the 81 reference images, and write this split
to ml_training/data_gen/kadid10k/split.csv — assert in code that no
reference_id appears in more than one split, don't just intend this.

After running both scripts, print: total images processed, class balance per
issue label (this will likely be imbalanced since not every distortion type
maps to our categories - show me the real numbers so we can decide together
if anything needs class-weighting in the loss function later), and split
sizes. Then actually run this end-to-end now and show me the real output, not
a description of what it would show.
```

### Prompt 1b — Real-world holdout set (do this yourself, not the agent)

```
This step is manual, not a Claude Code prompt: using your own phone or
camera, capture 20-40 photos covering genuine (not filtered) versions of
each category — a few intentionally out-of-focus shots, a few underexposed
(dim room, no flash) and a few overexposed (backlit/direct sun) shots, a few
genuinely noisy shots (bump ISO way up in a dark room), and a few files you
deliberately corrupt afterward (truncate a JPEG mid-copy, or re-save a couple
at very low JPEG quality), plus a handful of clean, well-exposed control
shots. Drop them in ml_training/data_gen/real_world_holdout/images/ and
hand-write ml_training/data_gen/real_world_holdout/labels.csv with the same
schema as the main labels.csv (you know the true labels since you staged
each shot — this is your ground truth, not something to infer). Once this
exists, tell Claude Code to wire it into evaluate.py as a second, fully
separate test set — never mix it into the KADID-10k train/val/test split.
```

### Prompt 2 — PyTorch model architecture

```
Build backend/app/ml/classical_features.py, backend/app/ml/cnn_model.py, and
backend/app/ml/anomaly.py. These same files will later be imported by both
ml_training/ (for training) and the backend (for inference), so keep them
dependency-light (torch, torchvision, opencv-python, numpy, scikit-learn —
nothing web-framework-specific) and put them under backend/app/ml/ as the
single source of truth; ml_training/ will import from there via a relative
path or editable install, your choice, document which.

classical_features.py: implement pure functions, each taking a numpy/PIL
image and returning a float or small dict:
- laplacian_variance(img) -> sharpness/blur signal
- exposure_stats(img) -> mean luma, % pixels clipped near 0, % clipped near
  255, luma histogram skew
- noise_estimate(img) -> a wavelet-based or high-frequency-residual noise
  estimator (implement a real, documented method — e.g. the standard
  fast noise estimation via Laplacian-of-Gaussian residual in flat regions,
  cite the approach in a docstring)
- contrast_std(img) -> global luma standard deviation
- jpeg_blockiness(img) -> 8x8 block-edge discontinuity measure
- extract_feature_vector(img) -> combines all of the above into a single
  fixed-length numpy vector, with a documented, stable feature order (this
  order must stay identical between training and inference, so define it
  once as a module-level constant list of feature names)
Include unit-testable pure functions with clear docstrings and type hints.

cnn_model.py: implement the hybrid multi-head model:
- Load torchvision.models.mobilenet_v3_small(weights=...IMAGENET1K...),
  strip the classifier, keep the feature extractor + pooling, producing a
  576-d (or whatever MobileNetV3-Small's actual pooled dim is — verify and
  use the real number) embedding.
- A small classical-features MLP branch (input = len(feature vector) from
  classical_features.py, e.g. 6-8 dims -> 32-d embedding via 1-2 FC layers
  with ReLU/BatchNorm).
- Concatenate CNN embedding + classical embedding -> shared FC trunk
  (e.g. -> 256 -> 128 with dropout).
- 5 independent sigmoid output heads (one Linear(128,1)+sigmoid each) for
  blur, underexposure, overexposure, noise, corruption.
- 1 regression head (Linear(128,1), no activation or sigmoid*100) for
  quality_score.
- Expose a clean forward(image_tensor, classical_features_tensor) -> dict of
  named outputs, and a from_pretrained(weights_path) classmethod for
  inference-time loading.
- Also implement a get_gradcam_target_layer() helper that returns the correct
  MobileNetV3-Small layer to hook for Grad-CAM (needed later).

anomaly.py: implement a thin wrapper around sklearn.ensemble.IsolationForest
that: (a) fits on an array of classical feature vectors from clean-labeled
training images only, (b) exposes score(feature_vector) -> anomaly score and
is_anomalous(feature_vector, threshold) -> bool, (c) supports save/load via
joblib so training and inference share one serialized artifact.

Do not train anything yet — this prompt is architecture only. After writing,
run a quick smoke test: instantiate the model with random weights, run a
dummy 224x224 image through the full forward pass, and print output shapes
for every head to confirm the architecture is wired correctly end to end.
```

### Prompt 3 — Training script and run

```
Build ml_training/dataset.py (PyTorch Dataset reading
ml_training/data_gen/kadid10k/labels.csv + split.csv from Prompt 1, applying
standard ImageNet normalization + light augmentation for train split only,
and computing the classical feature vector for each sample at __getitem__
time using backend/app/ml/classical_features.py — fit and save the
StandardScaler from §1.5 on the training split only, then apply it to val/
test, don't fit on the full dataset) and ml_training/train.py.

train.py requirements:
- Detect and use CUDA if available (torch.cuda.is_available()), targeting
  this RTX 3050 (4GB VRAM) — use batch size 16-32, mixed precision
  (torch.cuda.amp) to fit comfortably in 4GB, and print device info at
  startup so I can confirm it's using the GPU.
- Two-phase training: (1) freeze the MobileNetV3 backbone, train only the
  classical-feature branch + fusion trunk + heads for a few epochs so the
  new heads stabilize before touching pretrained weights; (2) unfreeze the
  last 2-3 backbone blocks and fine-tune end-to-end at a lower learning rate.
  Document both phases' hyperparameters in the script.
- Loss: BCE loss per sigmoid head (summed or averaged, your call, document
  it) + MSE (or Huber) loss for the regression head, combined with
  documented weighting.
- Log per-epoch train/val loss and per-head val F1 to stdout, and save the
  best checkpoint (by val macro-F1 across the 5 heads) to
  backend/app/ml/weights/mobilenetv3_iqa.pt, plus the fitted Isolation
  Forest to backend/app/ml/weights/anomaly_iforest.joblib.
- Keep total training time reasonable for a 4GB GPU and a 48-hour assessment
  window — default to a modest number of epochs (document the number and
  why) rather than an open-ended run, and support resuming from checkpoint.

After writing the script, actually run training now and show me real loss/
metric curves as they log, not simulated output. If CUDA isn't available in
this environment, fall back to CPU automatically, tell me clearly, and reduce
epochs/dataset size so a real (if smaller) training run still completes here
— I'll do the full GPU run locally afterward using the same script.
```

### Prompt 4 — Evaluation and explainability

```
Build ml_training/evaluate.py and backend/app/ml/gradcam.py.

evaluate.py: load the best checkpoint, run inference over the held-out
KADID-10k test split AND, separately, over
ml_training/data_gen/real_world_holdout/ if it exists (report both sets'
metrics side by side, clearly labeled — the real-world numbers are your
actual evidence of generalization beyond synthetic-style distortions, say
this explicitly in the output), and produce:
- Per-issue-head: precision, recall, F1, ROC-AUC, and a confusion matrix
  (at a documented probability threshold, e.g. 0.5, plus report the
  threshold-independent ROC-AUC too).
- Regression head: MAE and Spearman/Pearson correlation (SROCC/PLCC) against
  the DMOS-derived ground-truth quality_score.
- Anomaly detector: precision/recall of the defect flag against the held-out
  corrupted-image subset.
- A small "failure cases" report: pull out the 5-10 worst-predicted test
  images per head (largest error/lowest confidence-correctness) and save
  them with their true vs predicted labels to
  ml_training/notebooks/failure_cases/ as images + a markdown summary.
- Write all of the above into EVALUATION.md at the repo root, formatted and
  ready to submit as-is, including a short written discussion of limitations
  (training labels come from KADID-10k's filter-applied distortions rather
  than organically-occurring camera defects, single-issue-per-image training
  data under-represents realistic co-occurring degradations per §2.4, real-
  world generalization is tested on a small manual holdout rather than a
  large independent benchmark, etc — be honest and specific, not generic
  boilerplate).

gradcam.py: implement Grad-CAM against the MobileNetV3-Small branch using the
target layer from get_gradcam_target_layer(), producing a heatmap overlay
image for a given input image and a given output head (so I can visualize
"why did it think this was blurry" vs "why did it think this was noisy"
separately). Expose a single generate_gradcam(model, image, head_name) ->
PIL.Image function. Test it on 2-3 sample images from each issue category and
save the overlays to ml_training/notebooks/gradcam_samples/ so I can review
quality before wiring it into the API.

Run both scripts now against the checkpoint from the previous step and show
me the real EVALUATION.md content and a couple of the generated Grad-CAM
overlay images.
```

### Prompt 5 — Backend API

```
Build the full FastAPI backend in backend/app/:

core/config.py — pydantic-settings reading DATABASE_URL, DIRECT_URL (for
migrations only), CORS_ORIGINS (comma-separated), MODEL_WEIGHTS_PATH,
ANOMALY_MODEL_PATH, MAX_UPLOAD_MB, all from environment variables with
sensible local-dev defaults in a checked-in .env.example (no real secrets).

core/db.py — SQLAlchemy 2.0 async engine using postgresql+asyncpg://, built
from DATABASE_URL (the Neon pooled connection string in production), with
pool_pre_ping=True, an async_sessionmaker, and a FastAPI dependency
get_db() yielding an AsyncSession per request.

models/orm.py — two tables:
- analyses: id (uuid pk), filename, content_type, file_size_bytes,
  quality_score (float), quality_label (str), image_stats (JSONB),
  image_data (bytea, the uploaded image itself, stored compressed),
  created_at (timestamptz default now)
- analysis_issues: id (uuid pk), analysis_id (fk -> analyses.id, cascade
  delete), issue_type (str), severity (str), confidence (float)
Set up Alembic (backend/app/alembic/) configured to read DIRECT_URL (not the
pooled URL) for migrations, generate the initial migration, and show me the
generated SQL.

models/schemas.py — Pydantic response/request models matching exactly the
API contract: AnalyzeResponse, AnalysisSummary, AnalysisDetail, IssueOut,
ImageStatsOut, PaginatedAnalyses.

ml/inference.py — a singleton-style loader (load once at app startup via
FastAPI lifespan, not per-request) that: loads the CNN checkpoint, the
Isolation Forest, puts the model in eval mode on CPU (torch.device("cpu")
explicitly, since Render has no GPU), and exposes a single
analyze_image(pil_image: Image) -> dict function returning everything needed
to build an AnalyzeResponse (quality_score, quality_label via the documented
thresholds, per-head issues list, image_stats, and whether gradcam is
available for this result).

api/analyze.py — POST /api/analyze: accept multipart file upload, validate
it's a real, readable image (reject non-images and oversized files with
proper 400/413 responses, not a 500), run analyze_image(), persist the
analysis + its issues to the DB in one transaction, return the
AnalyzeResponse. Handle corrupt/unreadable "image" files gracefully (PIL will
throw — catch it and return a clean 400 with a helpful message, this is an
explicit requirement in the brief).

api/analyses.py — GET /api/analyses (paginated list, most recent first,
excluding the raw image_data blob from the summary payload for response
size), GET /api/analyses/{id} (full detail), GET /api/analyses/{id}/gradcam
(regenerate and stream back a Grad-CAM PNG for a stored analysis — re-run
Grad-CAM on the stored image_data on demand rather than storing every
possible heatmap, to keep storage small).

main.py — wire it all together: FastAPI app, lifespan handler that loads the
model once, CORSMiddleware configured from CORS_ORIGINS, mount the routers,
add GET /health that reports {"status": "ok", "model_loaded": <bool>} and
returns 503 if the model failed to load, and proper exception handlers so
unhandled errors return structured JSON with correct status codes instead of
leaking stack traces.

requirements.txt — pin real, compatible versions for fastapi, uvicorn,
sqlalchemy, asyncpg, alembic, pydantic-settings, python-multipart, torch
(CPU-only build — use the +cpu index so the Docker image doesn't pull CUDA
libraries it'll never use on Render), torchvision, opencv-python-headless
(headless, not opencv-python, since there's no display in the container),
scikit-learn, joblib, Pillow.

After building, run the backend locally against a local Postgres (via
docker-compose, which you'll also write in this step) and actually exercise
POST /api/analyze with a couple of real sample images from sample_images/,
showing me real response JSON, then GET /api/analyses to confirm persistence
works end to end.
```

### Prompt 6 — Frontend

```
Build the Next.js frontend in frontend/, App Router + TypeScript + Tailwind.

lib/api.ts — a small typed fetch client wrapping the backend contract
(analyzeImage(file), listAnalyses(limit, offset), getAnalysis(id),
getGradcamUrl(id)), reading the backend base URL from
process.env.NEXT_PUBLIC_API_URL, with clear error handling that distinguishes
network failures from 4xx validation errors from 5xx server errors (the UI
needs to show different messages for each, per the brief's "handle loading,
success, error states" requirement).

app/page.tsx — main upload + analyze flow:
- Drag-and-drop + click-to-browse upload (component: UploadDropzone.tsx),
  client-side validation of file type/size before even hitting the API.
- Loading state while analysis runs (this can take a couple seconds,
  especially on a cold Render instance — show a message acknowledging that).
- Result display once analysis returns: the uploaded image, an overall
  quality score (component: QualityScoreGauge.tsx — a simple radial/bar
  gauge, color-coded by quality_label), a list of detected issues with
  severity and confidence (component: IssueBadgeList.tsx), and the raw
  image_stats in a readable panel (not just a JSON dump).
- A button to view the Grad-CAM heatmap overlay for the result (component:
  GradCamOverlay.tsx), lazy-loaded only when clicked since it's an extra
  request.
- Clear error states: invalid file, upload failure, backend unreachable,
  backend returned an error — each with a distinct, actionable message.

app/history/page.tsx — paginated table/list of past analyses (component:
HistoryTable.tsx) with quality_label badges and a link into each detail page,
loading and empty states handled.

app/analyses/[id]/page.tsx — full detail view for one past analysis, same
components reused from the main flow.

Responsive layout throughout (mobile usable, not just desktop), reasonably
polished with Tailwind but keep this proportional to a 10%-weighted
requirement — clean and functional over elaborate.

Set NEXT_PUBLIC_API_URL in frontend/.env.local.example pointing at
http://localhost:8000 for local dev against the Dockerized backend.

After building, run it locally against the backend from the previous step and
actually walk through: upload an image, see the result render, check the
history page shows it, open the detail page. Show me it working, not just
that it compiles.
```

### Prompt 7 — Containerization

```
Write backend/Dockerfile: multi-stage build, CPU-only PyTorch (use the
official +cpu wheel index so we don't ship unused CUDA libraries and bloat
the image), copy in the trained model weights from backend/app/ml/weights/,
non-root user, expose the port via $PORT env var (Render assigns this
dynamically) with a CMD that reads it correctly
(uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}), and a HEALTHCHECK
hitting /health. Keep the final image as small as reasonably possible given
we're bundling PyTorch + OpenCV + model weights — use python:3.11-slim as the
base and clean up apt/pip caches in the same layer they're created.

Update docker-compose.yml at the repo root to run: the backend container
(built from backend/Dockerfile, or a dev-mode volume-mounted variant for fast
iteration — your call, document which), plus a local postgres:16 service with
a persisted volume and the same schema Alembic would create, wired together
via a documented DATABASE_URL. Do NOT include the frontend in
docker-compose — it's deployed separately to Vercel and typically run with
`next dev` locally, not containerized, but note this choice in a comment.

Build the image now and actually run `docker compose up`, then re-run the
same POST /api/analyze smoke test from Prompt 5 against the containerized
backend to confirm the Docker build actually works end to end, not just that
it builds without error.
```

### Prompt 8 — Deployment configs

```
Write render.yaml at the repo root as a Render Blueprint defining the backend
as a Docker web service: point it at backend/Dockerfile, set the health check
path to /health, and declare (without values) the required env vars
(DATABASE_URL, DIRECT_URL, CORS_ORIGINS, MODEL_WEIGHTS_PATH,
ANOMALY_MODEL_PATH, MAX_UPLOAD_MB) as sync:false entries so they're prompted
for in the Render dashboard rather than committed.

Write frontend/vercel.json only if any non-default Vercel config is actually
needed (custom headers, redirects) — Next.js App Router usually needs zero
config on Vercel, so if there's nothing to add, tell me that explicitly
instead of writing a no-op file.

Write a DEPLOYMENT.md at the repo root walking through, in order: (1)
creating the Neon project and getting both pooled and direct connection
strings, (2) running Alembic migrations against the direct URL from a local
machine before first deploy, (3) creating the Render Blueprint deploy from
render.yaml and filling in the env vars including the Neon pooled URL and the
Vercel domain for CORS_ORIGINS, (4) deploying the frontend to Vercel and
setting NEXT_PUBLIC_API_URL to the live Render backend URL, (5) a
post-deploy smoke test checklist (hit /health, upload a real image through
the live frontend, confirm it appears in /history). Note the free-tier cold-
start behavior explicitly so it's not mistaken for a bug during review.
```

### Prompt 9 — Tests, README, and submission packaging

```
Write backend/tests/ (pytest): test_classical_features.py (unit tests against
known synthetic inputs — e.g. assert a heavily blurred image has lower
laplacian_variance than a sharp one, assert an all-black image is flagged as
underexposed), test_api.py (integration tests against a test DB or sqlite/
async test fixture — exercise POST /api/analyze with a real sample image and
a deliberately corrupt file, GET /api/analyses pagination), test_inference.py
(confirm the model loads and produces outputs of the expected shape/range on
a fixed sample image). Run the full suite now and show me it passing.

Copy 2-3 representative images per category from ml_training's test split
into sample_images/{clean,blurry,underexposed,overexposed,noisy,corrupted}/
so the submission includes real demonstration images as required.

Write the final root README.md covering: project overview, architecture
diagram (ASCII is fine), local setup (docker-compose up, migration steps),
how the model was trained and where (RTX 3050 locally, the real KADID-10k
dataset mapped onto our 6 categories plus a hand-captured real-world holdout,
MobileNetV3-Small hybrid architecture — summarize, link to EVALUATION.md for
full detail), API documentation with example curl requests
for every endpoint, how inference works in production (CPU-only, model
loaded once at startup, ~X ms per image — pull the real number from your
test runs), deployment URLs once you have them, and a short "Assessment
Criteria Coverage" table cross-referencing this README/EVALUATION.md/repo
structure against the brief's §14 weighted criteria so a reviewer can find
each piece quickly.

Finally, list every file changed/created in this session and confirm nothing
required by brief §12 (source code all three layers, README, DB setup
instructions, API docs, evaluation results, sample images, Docker config,
deployed URL placeholder) is missing.
```

---

## 9. Bonus items — where they fall out of this design for free vs. need extra work

| Bonus (brief §13)                                | Status in this plan                                                                                                                                                         |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Batch image analysis                             | Not built by default — add a `POST /api/analyze/batch` accepting multiple files as a follow-up prompt if time allows                                                        |
| Quality heatmaps / localization                  | **Included** — Grad-CAM (Prompt 4 + 5) covers this directly                                                                                                                 |
| Confidence calibration / uncertainty             | Partially included — sigmoid outputs are used as confidence; add temperature scaling post-hoc as a follow-up if time allows (cheap, well-documented technique)              |
| Model versioning                                 | Easy add — store a `model_version` string alongside weights, add to `analyses` table and API response; not in the base chain, add as Prompt 10 if time allows               |
| Automated tests                                  | **Included** — Prompt 9                                                                                                                                                     |
| Performance optimization for concurrent requests | Partially included — model is loaded once at startup (not per-request), CPU inference is fast enough (~30-80ms) for reasonable concurrency on Render's default worker setup |
| CI/CD                                            | Not built by default — add a GitHub Actions workflow (lint + test on PR, optional auto-deploy hook to Render) as a follow-up prompt                                         |
| Monitoring/logging                               | Basic structured logging can be added to `main.py` cheaply; full observability (Sentry, etc.) is out of scope for 48 hours unless time is left over                         |

If you have time left after Prompt 9, prioritize in this order: **model versioning → CI/CD → batch analysis → confidence calibration → monitoring**, since the first two are cheap and visibly signal engineering maturity to a reviewer, while the rest are diminishing returns for the time they cost.

---

## 10. A note on running this yourself

Prompts 1, 1b, and 3–4 are the only ones that need your GPU/your camera. Prompt 1's download step and Prompt 2's architecture smoke-test are CPU-fine. Everything from Prompt 5 onward runs fine on Claude Code's own environment (or your CPU) since it's application code, not model training. The realistic sequencing: run Prompt 0 (folder fix + scaffold) anywhere, do Prompt 1 (KADID-10k download + label mapping) and Prompt 1b (your real-world photos) on your machine, run Prompts 2–4 on your machine so training actually uses the RTX 3050, then Prompts 5–9 can run either locally or in whatever environment you're driving Claude Code from — just make sure the trained weights and `feature_scaler.joblib` from Prompts 3/4 land in `backend/app/ml/weights/` before Prompt 5 needs to load them.
