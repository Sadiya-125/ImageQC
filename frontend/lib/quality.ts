import type { QualityLabel, Severity } from "@/lib/types";

interface QualityStyle {
  label: string;
  badgeClass: string;
  ringClass: string;
  textClass: string;
}

const QUALITY_STYLES: Record<QualityLabel, QualityStyle> = {
  ACCEPTABLE: {
    label: "Acceptable",
    badgeClass: "bg-quality-good text-quality-good-foreground border-transparent",
    ringClass: "stroke-quality-good",
    textClass: "text-quality-good",
  },
  DEGRADED: {
    label: "Degraded",
    badgeClass: "bg-quality-warn text-quality-warn-foreground border-transparent",
    ringClass: "stroke-quality-warn",
    textClass: "text-quality-warn",
  },
  DEFECTIVE: {
    label: "Defective",
    badgeClass: "bg-quality-bad text-quality-bad-foreground border-transparent",
    ringClass: "stroke-quality-bad",
    textClass: "text-quality-bad",
  },
};

export function qualityStyle(label: QualityLabel): QualityStyle {
  return QUALITY_STYLES[label];
}

const SEVERITY_ORDER: Record<Severity, number> = { high: 0, medium: 1, low: 2 };

export function severityRank(severity: Severity): number {
  return SEVERITY_ORDER[severity];
}

const SEVERITY_CLASSES: Record<Severity, string> = {
  high: "bg-quality-bad/15 text-quality-bad border-quality-bad/30",
  medium: "bg-quality-warn/20 text-quality-warn-foreground border-quality-warn/40 dark:text-quality-warn",
  low: "bg-muted text-muted-foreground border-border",
};

export function severityClass(severity: Severity): string {
  return SEVERITY_CLASSES[severity];
}

const ISSUE_LABELS: Record<string, string> = {
  blur: "Blur",
  underexposure: "Underexposure",
  overexposure: "Overexposure",
  noise: "Noise",
  corruption: "Corruption",
  potential_defect: "Potential Defect",
};

export function issueLabel(type: string): string {
  return ISSUE_LABELS[type] ?? type;
}
