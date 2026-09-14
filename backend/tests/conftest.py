"""
Shared pytest fixtures.

Integration tests run against a REAL Postgres test database, not SQLite --
the ORM uses Postgres-specific column types (JSONB, native UUID; see
app/models/orm.py) that SQLite can't represent faithfully, and this project
already runs a local Postgres via docker-compose for dev, so reusing it (on
a separate `imageqc_test` database, auto-created if missing, never touching
dev data) is both more faithful to production and requires no extra
infrastructure. Override TEST_DATABASE_URL to point at a different Postgres
instance (e.g. in CI).

Test isolation is via truncating both tables after each test (see
_truncate_after_test below), not transaction rollback -- a shared,
rolled-back transaction was tried first, but asyncpg connections aren't
safe for concurrent use from more than one coroutine, and sharing one
connection between the test's own assertions and the app's request handling
hit exactly that ("another operation is in progress"). Each session below
gets its own pooled connection instead.
"""

import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.db import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.ml.inference import get_inference_engine  # noqa: E402
from app.models.orm import Base  # noqa: E402

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5433/imageqc_test",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLEAN_IMAGE = REPO_ROOT / "sample_images" / "clean" / "I15.png"
SAMPLE_CORRUPTED_IMAGE = REPO_ROOT / "sample_images" / "corrupted" / "I15_10_05.png"


def _target_db_name(url: str) -> str:
    return urlsplit(url.replace("postgresql+asyncpg://", "postgresql://")).path.lstrip("/")


def _maintenance_dsn(url: str) -> str:
    """Same server as TEST_DATABASE_URL, but the always-present `postgres` db, for CREATE DATABASE."""
    plain = url.replace("postgresql+asyncpg://", "postgresql://")
    parts = urlsplit(plain)
    return urlunsplit(parts._replace(path="/postgres"))


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_database():
    """Creates the imageqc_test database on the local Postgres server if it doesn't exist yet."""

    async def _create_if_missing() -> None:
        db_name = _target_db_name(TEST_DATABASE_URL)
        conn = await asyncpg.connect(_maintenance_dsn(TEST_DATABASE_URL))
        try:
            exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
            if not exists:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
        finally:
            await conn.close()

    asyncio.run(_create_if_missing())


@pytest_asyncio.fixture(scope="session")
async def engine(_ensure_test_database):
    eng = create_async_engine(TEST_DATABASE_URL)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _loaded_inference_engine():
    """Loads the real, committed model weights once for the whole test session."""
    inference_engine = get_inference_engine()
    if not inference_engine.is_loaded:
        inference_engine.load()
    return inference_engine


@pytest_asyncio.fixture
async def db_session(engine):
    """
    A standalone session for tests to query/assert with directly, on its
    own connection (never shared with the app's requests below) --
    asyncpg connections aren't safe for concurrent use from more than one
    coroutine at a time, so each session gets its own pooled connection
    rather than everyone joining a single shared one.
    """
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
async def _truncate_after_test(engine):
    """Isolation between tests: wipe both tables after each test runs."""
    yield
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE analysis_issues, analyses RESTART IDENTITY CASCADE"))


@pytest_asyncio.fixture
async def client(engine):
    async def _override_get_db():
        async_session = async_sessionmaker(engine, expire_on_commit=False)
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
