"""
Async database session management.

Will create the SQLAlchemy 2.0 async engine (asyncpg) against Neon's pooled
connection string, plus an async session factory and a FastAPI dependency
that yields a request-scoped session. Not implemented yet.
"""
