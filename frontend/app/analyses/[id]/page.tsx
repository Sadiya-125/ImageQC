"use client";

import * as React from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { AlertTriangle, ArrowLeft, WifiOff } from "lucide-react";

import {
  ApiError,
  getAnalysis,
  getAnalysisImageUrl,
  NetworkError,
} from "@/lib/api";
import type { AnalysisDetail } from "@/lib/types";
import { AnalysisResultView } from "@/components/AnalysisResultView";
import { DeleteAnalysisButton } from "@/components/DeleteAnalysisButton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string; network: boolean; notFound: boolean }
  | { status: "loaded"; data: AnalysisDetail };

/** Keyed by `id` from the parent below, so switching ids remounts this with a fresh "loading" state. */
function AnalysisDetailContent({ id }: { id: string }) {
  const router = useRouter();
  const [state, setState] = React.useState<LoadState>({ status: "loading" });

  React.useEffect(() => {
    let cancelled = false;

    getAnalysis(id)
      .then((data) => {
        if (!cancelled) setState({ status: "loaded", data });
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof NetworkError) {
          setState({
            status: "error",
            message: err.message,
            network: true,
            notFound: false,
          });
        } else if (err instanceof ApiError) {
          setState({
            status: "error",
            message:
              err.status === 404 ? "This Analysis Doesn't Exist" : err.message,
            network: false,
            notFound: err.status === 404,
          });
        } else {
          setState({
            status: "error",
            message: "Couldn't Load this Analysis.",
            network: false,
            notFound: false,
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id]);

  if (state.status === "loading") {
    return (
      <Card>
        <CardContent className="space-y-4 py-8">
          <Skeleton className="h-72 w-full rounded-lg" />
          <Skeleton className="h-6 w-1/3" />
          <Skeleton className="h-6 w-1/2" />
        </CardContent>
      </Card>
    );
  }

  if (state.status === "error") {
    return (
      <Alert variant="destructive">
        {state.network ? (
          <WifiOff className="size-4" />
        ) : (
          <AlertTriangle className="size-4" />
        )}
        <AlertTitle>
          {state.notFound ? "Not found" : "Couldn't load analysis"}
        </AlertTitle>
        <AlertDescription>{state.message}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <DeleteAnalysisButton
          analysisId={state.data.id}
          filename={state.data.filename}
          onDeleted={() => router.push("/history")}
        />
      </div>
      <AnalysisResultView
        analysisId={state.data.id}
        imageUrl={getAnalysisImageUrl(state.data.id)}
        filename={state.data.filename}
        qualityScore={state.data.quality_score}
        qualityLabel={state.data.quality_label}
        issues={state.data.issues}
        imageStats={state.data.image_stats}
        modelVersion={state.data.model_version}
      />
    </div>
  );
}

export default function AnalysisDetailPage() {
  const params = useParams<{ id: string }>();

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-14">
      <Button
        variant="ghost"
        size="sm"
        className="mb-6 -ml-2 gap-1.5"
        render={<Link href="/history" />}
      >
        <ArrowLeft className="size-4" />
        Back to History
      </Button>

      <AnalysisDetailContent key={params.id} id={params.id} />
    </div>
  );
}
