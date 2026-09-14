"use client";

import * as React from "react";
import { ImageUp, UploadCloud } from "lucide-react";

import { cn } from "@/lib/utils";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const DEFAULT_MAX_SIZE_MB = 10;

interface UploadDropzoneProps {
  onFileAccepted: (file: File) => void;
  disabled?: boolean;
  maxSizeMB?: number;
}

function validateFile(file: File, maxSizeMB: number): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return `"${file.name}" is a ${file.type || "unknown"} file. Please upload a JPEG, PNG, WebP, or BMP image.`;
  }
  const maxBytes = maxSizeMB * 1024 * 1024;
  if (file.size > maxBytes) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    return `"${file.name}" is ${sizeMB}MB, which is over the ${maxSizeMB}MB limit.`;
  }
  return null;
}

export function UploadDropzone({
  onFileAccepted,
  disabled = false,
  maxSizeMB = DEFAULT_MAX_SIZE_MB,
}: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = React.useState(false);
  const [validationError, setValidationError] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleFile = React.useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const error = validateFile(file, maxSizeMB);
      if (error) {
        setValidationError(error);
        return;
      }
      setValidationError(null);
      onFileAccepted(file);
    },
    [maxSizeMB, onFileAccepted]
  );

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (disabled) return;
          handleFile(e.dataTransfer.files?.[0]);
        }}
        className={cn(
          "flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-14 text-center transition-colors sm:py-20",
          disabled
            ? "cursor-not-allowed border-border/60 bg-muted/30 opacity-60"
            : "cursor-pointer border-border hover:border-primary/60 hover:bg-accent/30",
          isDragging && !disabled && "border-primary bg-accent/40"
        )}
      >
        <span className="flex size-14 items-center justify-center rounded-full bg-primary/10 text-primary">
          {isDragging ? <ImageUp className="size-7" /> : <UploadCloud className="size-7" />}
        </span>
        <div className="space-y-1">
          <p className="font-heading text-lg font-semibold">
            {isDragging ? "Drop it here" : "Drag & drop an image"}
          </p>
          <p className="text-sm text-muted-foreground">
            or click to browse — JPEG, PNG, WebP, BMP up to {maxSizeMB}MB
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          disabled={disabled}
          className="hidden"
          onChange={(e) => {
            handleFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>
      {validationError && (
        <p className="mt-3 text-sm font-medium text-destructive">{validationError}</p>
      )}
    </div>
  );
}
