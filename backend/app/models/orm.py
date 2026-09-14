"""
SQLAlchemy ORM models -- the two persisted tables backing the analysis
history feature. See backend/app/alembic/ for the migration that creates
these, and backend/app/models/schemas.py for the Pydantic API shapes built
from them.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    quality_label: Mapped[str] = mapped_column(String, nullable=False)
    # Classical image-quality feature values (see classical_features.FEATURE_NAMES).
    image_stats: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # The uploaded file's original bytes (already compressed -- JPEG/PNG/etc
    # -- as received; not re-encoded, not stored as raw pixels).
    image_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    issues: Mapped[list["AnalysisIssue"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", lazy="selectin"
    )


class AnalysisIssue(Base):
    __tablename__ = "analysis_issues"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    issue_type: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    analysis: Mapped["Analysis"] = relationship(back_populates="issues")
