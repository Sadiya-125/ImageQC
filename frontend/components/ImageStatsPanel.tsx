import type { ImageStatsOut } from "@/lib/types";

interface StatMeta {
  key: keyof ImageStatsOut;
  label: string;
  hint: string;
  format: (value: number) => string;
}

const STATS_META: StatMeta[] = [
  {
    key: "laplacian_variance",
    label: "Sharpness",
    hint: "Higher = sharper edges, lower = more blur",
    format: (v) => v.toFixed(1),
  },
  {
    key: "mean_luma",
    label: "Brightness",
    hint: "Average luma, 0 (black) to 255 (white)",
    format: (v) => v.toFixed(1),
  },
  {
    key: "contrast_std",
    label: "Contrast",
    hint: "Standard deviation of luma",
    format: (v) => v.toFixed(1),
  },
  {
    key: "noise_estimate",
    label: "Noise level",
    hint: "Estimated sensor/compression noise",
    format: (v) => v.toFixed(2),
  },
  {
    key: "pct_clipped_low",
    label: "Shadow clipping",
    hint: "Share of pixels crushed to near-black",
    format: (v) => `${(v * 100).toFixed(2)}%`,
  },
  {
    key: "pct_clipped_high",
    label: "Highlight clipping",
    hint: "Share of pixels blown out to near-white",
    format: (v) => `${(v * 100).toFixed(2)}%`,
  },
  {
    key: "luma_skew",
    label: "Brightness skew",
    hint: "Negative leans bright, positive leans dark",
    format: (v) => v.toFixed(3),
  },
  {
    key: "jpeg_blockiness",
    label: "Block artifacts",
    hint: "8x8 block-edge discontinuity",
    format: (v) => v.toFixed(3),
  },
];

interface ImageStatsPanelProps {
  stats: ImageStatsOut;
}

export function ImageStatsPanel({ stats }: ImageStatsPanelProps) {
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-4 sm:grid-cols-4">
      {STATS_META.map(({ key, label, hint, format }) => (
        <div key={key} title={hint} className="space-y-0.5">
          <dt className="text-xs text-muted-foreground">{label}</dt>
          <dd className="font-heading text-lg font-semibold tabular-nums">
            {format(stats[key])}
          </dd>
        </div>
      ))}
    </dl>
  );
}
