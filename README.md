# ImageQC — AI-Powered Image Quality & Defect Detection

A full-stack application that accepts an uploaded image and automatically
evaluates its visual quality: overall 0-100 quality score, an
ACCEPTABLE/DEGRADED/DEFECTIVE-style label, and detection of six issue
categories — blur/insufficient sharpness, underexposure, overexposure,
noise, corruption/severe degradation, and potential visual defect (anomaly).

No external AI or vision APIs are used. Quality assessment is driven by a
hybrid model: a MobileNetV3-Small transfer-learning CNN (PyTorch) with
independent sigmoid heads per issue plus a quality-score regression head,
fused with classical OpenCV/NumPy image-quality features (sharpness,
exposure, noise, contrast, JPEG blockiness), plus a separate Isolation
Forest anomaly detector trained on those classical features.

## Architecture

- **frontend/** — Next.js (App Router, TypeScript, Tailwind), deployed to Vercel.
- **backend/** — FastAPI (Python 3.11+), deployed to Render via Docker.
- **Database** — Neon serverless Postgres, via SQLAlchemy 2.0 async + asyncpg.
- **ml_training/** — dataset preparation, training, evaluation, and weight
  export for the hybrid CNN + classical features + anomaly detector, trained
  on the real, publicly available KADID-10k benchmark plus a small
  hand-captured real-world holdout set.

See [BUILD_SPEC.md](./BUILD_SPEC.md) for full architectural detail.

## Status

Project scaffolding in progress. Setup, training, API, and deployment
instructions will be added here as each part is built.
