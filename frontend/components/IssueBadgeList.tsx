import { CheckCircle2 } from "lucide-react";

import { issueLabel, severityClass, severityRank } from "@/lib/quality";
import type { IssueOut } from "@/lib/types";
import { cn } from "@/lib/utils";

interface IssueBadgeListProps {
  issues: IssueOut[];
}

export function IssueBadgeList({ issues }: IssueBadgeListProps) {
  if (issues.length === 0) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-quality-good/30 bg-quality-good/10 px-3.5 py-2.5 text-sm text-quality-good">
        <CheckCircle2 className="size-4 shrink-0" />
        <span className="font-medium">No issues detected</span>
      </div>
    );
  }

  const sorted = [...issues].sort((a, b) => severityRank(a.severity) - severityRank(b.severity));

  return (
    <ul className="flex flex-col gap-2">
      {sorted.map((issue, i) => (
        <li
          key={`${issue.type}-${i}`}
          className={cn(
            "flex items-center justify-between gap-3 rounded-lg border px-3.5 py-2.5 text-sm",
            severityClass(issue.severity)
          )}
        >
          <span className="font-medium">{issueLabel(issue.type)}</span>
          <span className="flex items-center gap-2 text-xs opacity-90">
            <span className="rounded-full border border-current/30 px-2 py-0.5 capitalize">
              {issue.severity}
            </span>
            <span className="tabular-nums">{Math.round(issue.confidence * 100)}% confidence</span>
          </span>
        </li>
      ))}
    </ul>
  );
}
