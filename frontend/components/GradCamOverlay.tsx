"use client";

import * as React from "react";
import { Flame, Loader2 } from "lucide-react";

import { getGradcamUrl } from "@/lib/api";
import { issueLabel } from "@/lib/quality";
import { ISSUE_TYPES, type IssueType } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface GradCamOverlayProps {
  analysisId: string;
  defaultHead?: IssueType;
}

type LoadState = "idle" | "loading" | "loaded" | "error";

export function GradCamOverlay({
  analysisId,
  defaultHead = "blur",
}: GradCamOverlayProps) {
  const [open, setOpen] = React.useState(false);
  const [head, setHead] = React.useState<IssueType>(defaultHead);
  const [state, setState] = React.useState<LoadState>("idle");

  // Lazy: the URL (and therefore the request) is only computed once the
  // dialog has actually been opened at least once, or the head changes
  // while it's open -- never fetched just because the result rendered.
  const [imageUrl, setImageUrl] = React.useState<string | null>(null);

  const loadOverlay = React.useCallback(
    (forHead: IssueType) => {
      setState("loading");
      setImageUrl(getGradcamUrl(analysisId, forHead));
    },
    [analysisId],
  );

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next && state === "idle") loadOverlay(head);
      }}
    >
      <DialogTrigger render={<Button variant="outline" className="gap-2" />}>
        <Flame className="size-4" />
        View Grad-CAM Heatmap
      </DialogTrigger>
      <DialogContent className="scrollbar-hide flex max-h-[85vh] flex-col overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Grad-CAM Heatmap</DialogTitle>
          <DialogDescription>
            Highlights the Regions the Model Weighed Most Heavily When Scoring
            the Selected Issue.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-1 flex-col space-y-4">
          <Select
            value={head}
            onValueChange={(value) => {
              const next = value as IssueType;
              setHead(next);
              loadOverlay(next);
            }}
          >
            <SelectTrigger className="w-full">
              <SelectValue>{(value: IssueType) => issueLabel(value)}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {ISSUE_TYPES.map((type) => (
                <SelectItem key={type} value={type}>
                  {issueLabel(type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="relative aspect-square w-full overflow-hidden rounded-xl border bg-muted">
            {state === "loading" && (
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="size-6 animate-spin text-muted-foreground" />
              </div>
            )}
            {state === "error" && (
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="px-6 text-center text-sm text-destructive">
                  Couldn&apos;t Generate the Heatmap for this Image. Try Again
                  in a Moment.
                </p>
              </div>
            )}
            {imageUrl && (
              // eslint-disable-next-line @next/next/no-img-element -- dynamic, server-generated image; next/image optimization isn't useful here.
              <img
                src={imageUrl}
                alt={`Grad-CAM heatmap for ${issueLabel(head)}`}
                className={`h-full w-full object-contain ${state === "loaded" ? "" : "invisible"}`}
                onLoad={() => setState("loaded")}
                onError={() => setState("error")}
              />
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
