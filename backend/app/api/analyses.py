"""
GET /api/analyses -- paginated list, most recent first, excluding
image_data/image_stats from the payload to keep it small.
GET /api/analyses/{id} -- full detail.
GET /api/analyses/{id}/image -- the original uploaded bytes, as stored.
GET /api/analyses/{id}/gradcam -- regenerates a Grad-CAM overlay from the
stored image_data on demand and streams it back as a PNG, rather than
storing every possible per-head heatmap up front.
"""

import io
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.ml.cnn_model import ISSUE_HEADS
from app.ml.gradcam import generate_gradcam
from app.ml.inference import get_inference_engine
from app.models.orm import Analysis
from app.models.schemas import AnalysisDetail, AnalysisSummary, IssueOut, PaginatedAnalyses

router = APIRouter()

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


@router.get("/analyses", response_model=PaginatedAnalyses)
async def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    db: AsyncSession = Depends(get_db),
) -> PaginatedAnalyses:
    total = (await db.execute(select(func.count()).select_from(Analysis))).scalar_one()

    stmt = select(Analysis).order_by(Analysis.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()

    return PaginatedAnalyses(
        items=[AnalysisSummary.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/analyses/{analysis_id}", response_model=AnalysisDetail)
async def get_analysis(analysis_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> AnalysisDetail:
    analysis = await db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    return AnalysisDetail(
        id=analysis.id,
        filename=analysis.filename,
        content_type=analysis.content_type,
        file_size_bytes=analysis.file_size_bytes,
        quality_score=analysis.quality_score,
        quality_label=analysis.quality_label,
        issues=[IssueOut(type=i.issue_type, severity=i.severity, confidence=i.confidence) for i in analysis.issues],
        image_stats=analysis.image_stats,
        created_at=analysis.created_at,
    )


@router.get("/analyses/{analysis_id}/image")
async def get_analysis_image(analysis_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Response:
    analysis = await db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return Response(content=analysis.image_data, media_type=analysis.content_type)


@router.get("/analyses/{analysis_id}/gradcam")
async def get_analysis_gradcam(
    analysis_id: uuid.UUID,
    head: str = Query(..., description=f"One of: {', '.join(ISSUE_HEADS)}"),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    if head not in ISSUE_HEADS:
        raise HTTPException(status_code=400, detail=f"head must be one of {ISSUE_HEADS}, got {head!r}")

    analysis = await db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    engine = get_inference_engine()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded; try again shortly.")

    pil_image = Image.open(io.BytesIO(analysis.image_data)).convert("RGB")
    overlay = generate_gradcam(engine.model, pil_image, head, scaler=engine.scaler)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
