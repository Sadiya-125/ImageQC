"""
SQLAlchemy 2.0 async engine/session setup. One engine is created at import
time (cheap -- it doesn't connect until first use) and reused for the life
of the process; get_db() is a FastAPI dependency yielding one AsyncSession
per request.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
