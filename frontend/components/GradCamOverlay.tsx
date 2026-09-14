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

export function GradCamOverlay({ analysisId, defaultHead = "blur" }: GradCamOverlayProps) {
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
    [analysisId]
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
        View Grad-CAM heatmap
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Grad-CAM heatmap</DialogTitle>
          <DialogDescription>
            Highlights the regions the model weighed most heavily when scoring the
            selected issue.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Select
            value={head}
            onValueChange={(value) => {
              const next = value as IssueType;
              setHead(next);
              loadOverlay(next);
            }}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ISSUE_TYPES.map((type) => (
                <SelectItem key={type} value={type}>
                  {issueLabel(type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="relative flex aspect-square w-full items-center justify-center overflow-hidden rounded-xl border bg-muted">
            {state === "loading" && (
              <Loader2 className="size-6 animate-spin text-muted-foreground" />
            )}
            {state === "error" && (
              <p className="px-6 text-center text-sm text-destructive">
                Couldn&apos;t generate the heatmap for this image. Try again in a moment.
              </p>
            )}
            {imageUrl && (
              // eslint-disable-next-line @next/next/no-img-element -- dynamic, server-generated image; next/image optimization isn't useful here.
              <img
                src={imageUrl}
                alt={`Grad-CAM heatmap for ${issueLabel(head)}`}
                className="h-full w-full object-contain"
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
