"""
POST /api/analyze -- accepts a multipart image upload, validates it's a
real, readable, size-limited image, runs it through the ML pipeline, and
persists the analysis + its issues in one DB transaction.
"""

import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import get_db
from app.ml.inference import get_inference_engine
from app.models.orm import Analysis, AnalysisIssue
from app.models.schemas import AnalyzeResponse, IssueOut

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, status_code=201)
async def analyze(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    settings = get_settings()
    engine = get_inference_engine()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded; try again shortly.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes exceeds the {settings.MAX_UPLOAD_MB}MB limit.",
        )

    try:
        pil_image = Image.open(io.BytesIO(content))
        pil_image.load()  # force full decode now -- surfaces truncated/corrupt data immediately, not lazily later
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a readable image: {e}") from e

    result = engine.analyze_image(pil_image)

    analysis = Analysis(
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        file_size_bytes=len(content),
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        image_stats=result["image_stats"],
        image_data=content,
        issues=[
            AnalysisIssue(issue_type=i["type"], severity=i["severity"], confidence=i["confidence"])
            for i in result["issues"]
        ],
    )

    db.add(analysis)
    await db.commit()
    await db.refresh(analysis, attribute_names=["id", "created_at", "issues"])

    return AnalyzeResponse(
        id=analysis.id,
        filename=analysis.filename,
        quality_score=analysis.quality_score,
        quality_label=analysis.quality_label,
        issues=[IssueOut(type=i.issue_type, severity=i.severity, confidence=i.confidence) for i in analysis.issues],
        image_stats=result["image_stats"],
        gradcam_available=result["gradcam_available"],
        created_at=analysis.created_at,
    )
