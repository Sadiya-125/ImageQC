"use client";

import * as React from "react";
import { Loader2, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { ApiError, deleteAnalysis, NetworkError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface DeleteAnalysisButtonProps {
  analysisId: string;
  filename: string;
  onDeleted: () => void;
  /** Compact icon-only trigger, for tight spaces like a table row. */
  iconOnly?: boolean;
}

export function DeleteAnalysisButton({
  analysisId,
  filename,
  onDeleted,
  iconOnly = false,
}: DeleteAnalysisButtonProps) {
  const [open, setOpen] = React.useState(false);
  const [deleting, setDeleting] = React.useState(false);

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await deleteAnalysis(analysisId);
      setOpen(false);
      toast.success(`Deleted "${filename}"`);
      onDeleted();
    } catch (err) {
      const message =
        err instanceof NetworkError || err instanceof ApiError
          ? err.message
          : "Couldn't delete this analysis. Please try again.";
      toast.error(message);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(next) => !deleting && setOpen(next)}>
      <DialogTrigger
        render={
          <Button
            variant="outline"
            size={iconOnly ? "icon-sm" : "sm"}
            className="gap-1.5 text-destructive hover:bg-destructive/10 hover:text-destructive"
            aria-label={iconOnly ? `Delete ${filename}` : undefined}
          />
        }
      >
        <Trash2 className="size-4" />
        {!iconOnly && "Delete"}
      </DialogTrigger>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Delete this Analysis?</DialogTitle>
          <DialogDescription>
            &ldquo;{filename}&rdquo; and Its Results will be Permanently
            Removed.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="gap-2 sm:gap-2">
          <DialogClose
            render={<Button variant="outline" disabled={deleting} />}
          >
            Cancel
          </DialogClose>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={deleting}
            className="gap-1.5"
          >
            {deleting && <Loader2 className="size-4 animate-spin" />}
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
