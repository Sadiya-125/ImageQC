"""
Pydantic request/response schemas -- the API's public contract. Field names
in ImageStatsOut intentionally mirror
backend/app/ml/classical_features.FEATURE_NAMES exactly, since that list is
this project's single source of truth for feature naming/order.
"""

import uuid
from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict


class IssueOut(BaseModel):
    type: str
    severity: str
    confidence: float


class ImageStatsOut(BaseModel):
    laplacian_variance: float
    mean_luma: float
    pct_clipped_low: float
    pct_clipped_high: float
    luma_skew: float
    noise_estimate: float
    contrast_std: float
    jpeg_blockiness: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    quality_score: float
    quality_label: str
    issues: List[IssueOut]
    image_stats: ImageStatsOut
    gradcam_available: bool
    created_at: datetime


class AnalysisSummary(BaseModel):
    """List-view row -- deliberately excludes image_data and image_stats to keep the payload small."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    quality_score: float
    quality_label: str
    created_at: datetime


class AnalysisDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    file_size_bytes: int
    quality_score: float
    quality_label: str
    issues: List[IssueOut]
    image_stats: ImageStatsOut
    created_at: datetime


class PaginatedAnalyses(BaseModel):
    items: List[AnalysisSummary]
    total: int
    page: int
    page_size: int
