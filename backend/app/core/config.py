"""
Application settings, read from environment variables (with a checked-in
.env.example documenting each one) via pydantic-settings. Local dev picks up
backend/.env (gitignored) automatically if present; in production
(Render/Docker) these are real environment variables, not a file.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = BACKEND_DIR / "app" / "ml" / "weights"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Neon's pooled connection string in production (via PgBouncer); a
    # local docker-compose Postgres by default for local dev.
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/imageqc"

    # Neon's *direct* (non-pooled) connection string, used only by Alembic
    # migrations (see backend/alembic.ini / backend/app/alembic/env.py) --
    # PgBouncer's pooled connections don't reliably support the session-level
    # features Alembic's DDL transactions rely on. Defaults to the same
    # local Postgres for local dev, where pooling isn't in play at all.
    DIRECT_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/imageqc"

    # Comma-separated list of allowed CORS origins (the deployed Vercel
    # frontend URL(s) in production).
    CORS_ORIGINS: str = "http://localhost:3000"

    MODEL_WEIGHTS_PATH: str = str(DEFAULT_WEIGHTS_DIR / "mobilenetv3_iqa.pt")
    ANOMALY_MODEL_PATH: str = str(DEFAULT_WEIGHTS_DIR / "anomaly_iforest.joblib")
    FEATURE_SCALER_PATH: str = str(DEFAULT_WEIGHTS_DIR / "feature_scaler.joblib")

    MAX_UPLOAD_MB: int = 10

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
