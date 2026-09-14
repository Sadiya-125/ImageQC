import { qualityStyle } from "@/lib/quality";
import type { QualityLabel } from "@/lib/types";
import { cn } from "@/lib/utils";

interface QualityScoreGaugeProps {
  score: number;
  label: QualityLabel;
  size?: number;
}

const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
// A little under a full circle, leaving a visible gap at the bottom so the
// gauge reads as a gauge rather than a plain ring.
const ARC_FRACTION = 0.78;

export function QualityScoreGauge({ score, label, size = 176 }: QualityScoreGaugeProps) {
  const style = qualityStyle(label);
  const clamped = Math.max(0, Math.min(100, score));
  const arcLength = CIRCUMFERENCE * ARC_FRACTION;
  const filled = (clamped / 100) * arcLength;

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          viewBox="0 0 120 120"
          width={size}
          height={size}
          className="-rotate-[128deg]"
        >
          <circle
            cx="60"
            cy="60"
            r={RADIUS}
            fill="none"
            strokeWidth={12}
            strokeLinecap="round"
            className="stroke-muted"
            strokeDasharray={`${arcLength} ${CIRCUMFERENCE}`}
          />
          <circle
            cx="60"
            cy="60"
            r={RADIUS}
            fill="none"
            strokeWidth={12}
            strokeLinecap="round"
            className={cn(style.ringClass, "transition-[stroke-dasharray] duration-700 ease-out")}
            strokeDasharray={`${filled} ${CIRCUMFERENCE}`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-heading text-4xl font-bold tabular-nums">
            {Math.round(clamped)}
          </span>
          <span className="text-xs text-muted-foreground">/ 100</span>
        </div>
      </div>
      <span
        className={cn(
          "rounded-full px-3 py-1 text-sm font-semibold",
          style.badgeClass
        )}
      >
        {style.label}
      </span>
    </div>
  );
}
