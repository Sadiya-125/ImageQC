"use client";

import * as React from "react";
import { AlertTriangle, ChevronLeft, ChevronRight, WifiOff } from "lucide-react";

import { ApiError, listAnalyses, NetworkError } from "@/lib/api";
import type { PaginatedAnalyses } from "@/lib/types";
import { HistoryTable } from "@/components/HistoryTable";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

const PAGE_SIZE = 10;

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string; network: boolean }
  | { status: "loaded"; data: PaginatedAnalyses };

/**
 * Keyed by `page` from the parent below, so a page change remounts this
 * (fresh "loading" initial state) instead of needing an effect to reset
 * state synchronously -- setState only ever happens inside the async
 * .then()/.catch() callbacks here.
 */
function HistoryContent({ page, onTotalChange }: { page: number; onTotalChange: (total: number) => void }) {
  const [state, setState] = React.useState<LoadState>({ status: "loading" });

  React.useEffect(() => {
    let cancelled = false;

    listAnalyses(PAGE_SIZE, page * PAGE_SIZE)
      .then((data) => {
        if (cancelled) return;
        setState({ status: "loaded", data });
        onTotalChange(data.total);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof NetworkError) {
          setState({ status: "error", message: err.message, network: true });
        } else if (err instanceof ApiError) {
          setState({ status: "error", message: err.message, network: false });
        } else {
          setState({
            status: "error",
            message: "Couldn't load analysis history.",
            network: false,
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [page, onTotalChange]);

  if (state.status === "loading") {
    return (
      <div className="space-y-3">
        <Skeleton className="h-11 w-full rounded-lg" />
        <Skeleton className="h-11 w-full rounded-lg" />
        <Skeleton className="h-11 w-full rounded-lg" />
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <Alert variant="destructive">
        {state.network ? <WifiOff className="size-4" /> : <AlertTriangle className="size-4" />}
        <AlertTitle>Couldn&apos;t load history</AlertTitle>
        <AlertDescription>{state.message}</AlertDescription>
      </Alert>
    );
  }

  return <HistoryTable items={state.data.items} />;
}

export default function HistoryPage() {
  const [page, setPage] = React.useState(0); // 0-indexed for offset math
  const [total, setTotal] = React.useState(0);
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-14">
      <div className="mb-8 space-y-2">
        <h1 className="font-heading text-3xl font-bold tracking-tight">History</h1>
        <p className="text-muted-foreground">Past analyses, most recent first.</p>
      </div>

      <div className="space-y-4">
        <HistoryContent page={page} onTotalChange={setTotal} />

        {total > PAGE_SIZE && (
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Page {page + 1} of {totalPages} &middot; {total} total
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                className="gap-1"
              >
                <ChevronLeft className="size-4" />
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page + 1 >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="gap-1"
              >
                Next
                <ChevronRight className="size-4" />
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
