import Link from "next/link";
import { ImageOff } from "lucide-react";

import { qualityStyle } from "@/lib/quality";
import type { AnalysisSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface HistoryTableProps {
  items: AnalysisSummary[];
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function HistoryTable({ items }: HistoryTableProps) {
  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed py-16 text-center">
        <ImageOff className="size-8 text-muted-foreground" />
        <div>
          <p className="font-medium">No analyses yet</p>
          <p className="text-sm text-muted-foreground">
            Analyze an image and it&apos;ll show up here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>File</TableHead>
            <TableHead>Quality</TableHead>
            <TableHead className="text-right">Score</TableHead>
            <TableHead className="hidden text-right sm:table-cell">Analyzed</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => {
            const style = qualityStyle(item.quality_label);
            return (
              <TableRow key={item.id} className="group">
                <TableCell className="max-w-[12rem] truncate font-medium sm:max-w-sm">
                  <Link
                    href={`/analyses/${item.id}`}
                    className="hover:underline underline-offset-4"
                  >
                    {item.filename}
                  </Link>
                </TableCell>
                <TableCell>
                  <span
                    className={cn(
                      "inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold",
                      style.badgeClass
                    )}
                  >
                    {style.label}
                  </span>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {Math.round(item.quality_score)}
                </TableCell>
                <TableCell className="hidden text-right text-muted-foreground sm:table-cell">
                  {formatDate(item.created_at)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
