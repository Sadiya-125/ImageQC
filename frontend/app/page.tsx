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
  | {
      status: "error";
      message: string;
      kind: "validation" | "server" | "network";
      previewUrl: string | null;
    };

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
          message: "Something Unexpected Went Wrong. Please Try Again.",
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
          Check Your Image&apos;s Quality
        </h1>
        <p className="mx-auto max-w-xl text-muted-foreground">
          Upload a Photo to Get an Instant AI-Powered Read On Sharpness,
          Exposure, Noise, Corruption, and Potential Defects.
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
                Usually Just a Couple of Seconds...
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {state.status === "error" && (
        <div className="space-y-4">
          <Alert
            variant={state.kind === "validation" ? "default" : "destructive"}
          >
            {state.kind === "network" ? (
              <WifiOff className="size-4" />
            ) : (
              <AlertTriangle className="size-4" />
            )}
            <AlertTitle>
              {state.kind === "network" && "Can't Reach the Server"}
              {state.kind === "validation" && "That Upload Didn't Work"}
              {state.kind === "server" && "Server Error"}
            </AlertTitle>
            <AlertDescription>{state.message}</AlertDescription>
          </Alert>
          <Button onClick={reset} variant="outline" className="gap-2">
            <RotateCcw className="size-4" />
            Try Another Image
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
            modelVersion={state.result.model_version}
          />
          <Button onClick={reset} variant="outline" className="gap-2">
            <RotateCcw className="size-4" />
            Analyze Another Image
          </Button>
        </div>
      )}
    </div>
  );
}
