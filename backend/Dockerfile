FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0 are required by opencv-python-headless at import time
# even though it's the headless build (it still links against libGL).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch/torchvision come from the CPU-only wheel index -- Render has no GPU,
# and the default PyPI wheels pull ~2GB of CUDA libraries that would only
# bloat the image and slow cold starts for nothing. Every dependency here
# ships a prebuilt manylinux wheel for cp311/x86_64 (torch, opencv-python-headless,
# scipy, scikit-learn, asyncpg, psycopg2-binary all do), so no C compiler
# toolchain is needed -- deliberately not pulling in build-essential (or a
# multi-stage build to discard it again) just to leave it unused.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 \
    && pip install --no-cache-dir -r requirements.txt

# Includes backend/app/ml/weights/{mobilenetv3_iqa.pt,anomaly_iforest.joblib,
# feature_scaler.joblib} -- the trained artifacts ml_training/train.py
# produced, committed to the repo specifically so this COPY (and therefore a
# fresh deploy) has something to load. See root .gitignore's comment on why
# those three (and only those three) are tracked.
COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Render assigns $PORT dynamically at runtime; 8000 is only the local-dev
# fallback (matches docker-compose.yml's default).
ENV PORT=8000
EXPOSE 8000

# No curl in this image (deliberately -- see the dependency note above);
# Python's stdlib does the same job without adding a package just for this.
# (Render's own health monitoring uses render.yaml's healthCheckPath from
# outside the container regardless -- this HEALTHCHECK is for `docker ps`
# and `docker compose` status locally.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
