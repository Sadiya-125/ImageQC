import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { GradCamOverlay } from "@/components/GradCamOverlay";
import { ImageStatsPanel } from "@/components/ImageStatsPanel";
import { IssueBadgeList } from "@/components/IssueBadgeList";
import { QualityScoreGauge } from "@/components/QualityScoreGauge";
import type { IssueOut, ImageStatsOut, IssueType, QualityLabel } from "@/lib/types";

interface AnalysisResultViewProps {
  analysisId: string;
  imageUrl: string;
  filename: string;
  qualityScore: number;
  qualityLabel: QualityLabel;
  issues: IssueOut[];
  imageStats: ImageStatsOut;
}

export function AnalysisResultView({
  analysisId,
  imageUrl,
  filename,
  qualityScore,
  qualityLabel,
  issues,
  imageStats,
}: AnalysisResultViewProps) {
  const primaryIssueHead = issues.find((i) => i.type !== "potential_defect")?.type as
    | IssueType
    | undefined;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <Card className="overflow-hidden py-0">
        <div className="flex items-center justify-center bg-muted/50 p-4">
          {/* eslint-disable-next-line @next/next/no-img-element -- arbitrary user-uploaded/stored image, not a build-time asset */}
          <img
            src={imageUrl}
            alt={filename}
            className="max-h-[28rem] w-auto rounded-lg object-contain"
          />
        </div>
        <CardContent className="flex flex-wrap items-center justify-between gap-3 border-t py-4">
          <p className="truncate text-sm font-medium text-muted-foreground">{filename}</p>
          <GradCamOverlay analysisId={analysisId} defaultHead={primaryIssueHead ?? "blur"} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-heading text-base">Quality Assessment</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex justify-center">
            <QualityScoreGauge score={qualityScore} label={qualityLabel} />
          </div>
          <Separator />
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground">Detected issues</h3>
            <IssueBadgeList issues={issues} />
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="font-heading text-base">Image statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <ImageStatsPanel stats={imageStats} />
        </CardContent>
      </Card>
    </div>
  );
}
