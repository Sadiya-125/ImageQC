"use client";

import * as React from "react";
import { AlertTriangle, Loader2, RotateCcw, WifiOff } from "lucide-react";

import { analyzeImage, ApiError, NetworkError } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";
import { AnalysisResultView } from "@/components/AnalysisResultView";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

type FlowState =
  | { status: "idle" }
  | { status: "loading"; previewUrl: string; filename: string }
  | { status: "success"; result: AnalyzeResponse; previewUrl: string }
  | { status: "error"; message: string; kind: "validation" | "server" | "network"; previewUrl: string | null };

export default function AnalyzePage() {
  const [state, setState] = React.useState<FlowState>({ status: "idle" });

  const handleFile = React.useCallback(async (file: File) => {
    const previewUrl = URL.createObjectURL(file);
    setState({ status: "loading", previewUrl, filename: file.name });

    try {
      const result = await analyzeImage(file);
      setState({ status: "success", result, previewUrl });
    } catch (err) {
      if (err instanceof NetworkError) {
        setState({
          status: "error",
          kind: "network",
          message: err.message,
          previewUrl,
        });
      } else if (err instanceof ApiError) {
        setState({
          status: "error",
          kind: err.isServerError ? "server" : "validation",
          message: err.message,
          previewUrl,
        });
      } else {
        setState({
          status: "error",
          kind: "server",
          message: "Something unexpected went wrong. Please try again.",
          previewUrl,
        });
      }
    }
  }, []);

  const reset = React.useCallback(() => {
    setState({ status: "idle" });
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-14">
      <div className="mb-8 space-y-2 text-center sm:mb-10">
        <h1 className="font-heading text-3xl font-bold tracking-tight sm:text-4xl">
          Check your image&apos;s quality
        </h1>
        <p className="mx-auto max-w-xl text-muted-foreground">
          Upload a photo to get an instant AI-powered read on sharpness, exposure,
          noise, corruption, and potential defects.
        </p>
      </div>

      {state.status === "idle" && (
        <UploadDropzone onFileAccepted={handleFile} />
      )}

      {state.status === "loading" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 py-16 text-center">
            <Loader2 className="size-8 animate-spin text-primary" />
            <div className="space-y-1">
              <p className="font-medium">Analyzing {state.filename}&hellip;</p>
              <p className="text-sm text-muted-foreground">
                Usually just a couple of seconds — a little longer if the server has
                been idle and is waking back up.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {state.status === "error" && (
        <div className="space-y-4">
          <Alert variant={state.kind === "validation" ? "default" : "destructive"}>
            {state.kind === "network" ? (
              <WifiOff className="size-4" />
            ) : (
              <AlertTriangle className="size-4" />
            )}
            <AlertTitle>
              {state.kind === "network" && "Can't reach the server"}
              {state.kind === "validation" && "That upload didn't work"}
              {state.kind === "server" && "Server error"}
            </AlertTitle>
            <AlertDescription>{state.message}</AlertDescription>
          </Alert>
          <Button onClick={reset} variant="outline" className="gap-2">
            <RotateCcw className="size-4" />
            Try another image
          </Button>
        </div>
      )}

      {state.status === "success" && (
        <div className="space-y-6">
          <AnalysisResultView
            analysisId={state.result.id}
            imageUrl={state.previewUrl}
            filename={state.result.filename}
            qualityScore={state.result.quality_score}
            qualityLabel={state.result.quality_label}
            issues={state.result.issues}
            imageStats={state.result.image_stats}
          />
          <Button onClick={reset} variant="outline" className="gap-2">
            <RotateCcw className="size-4" />
            Analyze another image
          </Button>
        </div>
      )}
    </div>
  );
}
