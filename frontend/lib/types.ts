// Mirrors backend/app/models/schemas.py exactly -- keep these two in sync by hand.

export const ISSUE_TYPES = [
  "blur",
  "underexposure",
  "overexposure",
  "noise",
  "corruption",
] as const;

export type IssueType = (typeof ISSUE_TYPES)[number];

/** "potential_defect" is a composite signal, not a model head -- see backend/app/ml/inference.py. */
export type AnyIssueType = IssueType | "potential_defect";

export type Severity = "low" | "medium" | "high";

export type QualityLabel = "ACCEPTABLE" | "DEGRADED" | "DEFECTIVE";

export interface IssueOut {
  type: AnyIssueType;
  severity: Severity;
  confidence: number;
}

export interface ImageStatsOut {
  laplacian_variance: number;
  mean_luma: number;
  pct_clipped_low: number;
  pct_clipped_high: number;
  luma_skew: number;
  noise_estimate: number;
  contrast_std: number;
  jpeg_blockiness: number;
}

export interface AnalyzeResponse {
  id: string;
  filename: string;
  quality_score: number;
  quality_label: QualityLabel;
  issues: IssueOut[];
  image_stats: ImageStatsOut;
  gradcam_available: boolean;
  created_at: string;
}

export interface AnalysisSummary {
  id: string;
  filename: string;
  quality_score: number;
  quality_label: QualityLabel;
  created_at: string;
}

export interface AnalysisDetail {
  id: string;
  filename: string;
  content_type: string;
  file_size_bytes: number;
  quality_score: number;
  quality_label: QualityLabel;
  issues: IssueOut[];
  image_stats: ImageStatsOut;
  created_at: string;
}

export interface PaginatedAnalyses {
  items: AnalysisSummary[];
  total: number;
  page: number;
  page_size: number;
}
