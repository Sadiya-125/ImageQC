"""
POST /api/analyze -- accepts a multipart image upload, validates it's a
real, readable, size-limited image, runs it through the ML pipeline, and
persists the analysis + its issues in one DB transaction.

POST /api/analyze/batch -- the same pipeline over multiple files in one
request. Each file is validated/analyzed/persisted independently: one bad
file in the batch doesn't fail the others, it just reports its own error
alongside their successful results.
"""

import asyncio
import io
from typing import List, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import get_db
from app.core.logging_config import get_logger, log_with_fields
from app.ml.inference import InferenceEngine, get_inference_engine
from app.models.orm import Analysis, AnalysisIssue
from app.models.schemas import AnalyzeResponse, BatchAnalyzeItem, BatchAnalyzeResponse, IssueOut

router = APIRouter()
logger = get_logger("imageqc.analyze")

MAX_BATCH_SIZE = 10


async def _read_and_validate(file: UploadFile, settings: Settings) -> Tuple[bytes, Image.Image]:
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
    return content, pil_image


async def _analyze_and_persist(
    file: UploadFile, engine: InferenceEngine, db: AsyncSession, settings: Settings
) -> AnalyzeResponse:
    content, pil_image = await _read_and_validate(file, settings)
    # analyze_image() is a blocking, CPU-bound PyTorch call (~50ms). Run it
    # off the event loop thread so it doesn't stall every other in-flight
    # request for that duration -- otherwise concurrent requests would
    # serialize behind each other's inference time.
    result = await asyncio.to_thread(engine.analyze_image, pil_image)

    analysis = Analysis(
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        file_size_bytes=len(content),
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        model_version=result["model_version"],
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

    log_with_fields(
        logger,
        20,
        "analysis_complete",
        analysis_id=str(analysis.id),
        quality_score=analysis.quality_score,
        quality_label=analysis.quality_label,
        issue_types=[i["type"] for i in result["issues"]],
        model_version=analysis.model_version,
    )

    return AnalyzeResponse(
        id=analysis.id,
        filename=analysis.filename,
        quality_score=analysis.quality_score,
        quality_label=analysis.quality_label,
        issues=[IssueOut(type=i.issue_type, severity=i.severity, confidence=i.confidence) for i in analysis.issues],
        image_stats=result["image_stats"],
        gradcam_available=result["gradcam_available"],
        model_version=analysis.model_version,
        created_at=analysis.created_at,
    )


@router.post("/analyze", response_model=AnalyzeResponse, status_code=201)
async def analyze(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    settings = get_settings()
    engine = get_inference_engine()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded; try again shortly.")

    return await _analyze_and_persist(file, engine, db, settings)


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse, status_code=201)
async def analyze_batch(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
) -> BatchAnalyzeResponse:
    settings = get_settings()
    engine = get_inference_engine()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded; try again shortly.")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)} exceeds the {MAX_BATCH_SIZE}-file batch limit.",
        )

    items: List[BatchAnalyzeItem] = []
    for file in files:
        try:
            result = await _analyze_and_persist(file, engine, db, settings)
            items.append(BatchAnalyzeItem(filename=file.filename or "upload", success=True, result=result))
        except HTTPException as e:
            items.append(
                BatchAnalyzeItem(filename=file.filename or "upload", success=False, error=str(e.detail))
            )

    return BatchAnalyzeResponse(results=items)
