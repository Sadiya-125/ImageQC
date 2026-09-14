import type {
  AnalysisDetail,
  AnalyzeResponse,
  IssueType,
  PaginatedAnalyses,
} from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * The backend returned a response (so the request reached it), but not a
 * successful one. `status` lets callers branch on 4xx (the request/file was
 * bad, actionable by the user) vs 5xx (something broke on the server,
 * actionable only by retrying/reporting).
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail?: string;

  constructor(status: number, detail: string | undefined, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  get isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }

  get isServerError(): boolean {
    return this.status >= 500;
  }
}

/** The request never reached the backend at all -- offline, DNS, CORS, connection refused, etc. */
export class NetworkError extends Error {
  constructor(message = "Could not reach the ImageQC server. Check your connection and try again.") {
    super(message);
    this.name = "NetworkError";
  }
}

async function parseErrorDetail(response: Response): Promise<string | undefined> {
  try {
    const body = (await response.clone().json()) as { detail?: string };
    return body?.detail;
  } catch {
    return undefined;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    // fetch() only rejects on network-level failure -- it resolves
    // normally for HTTP error statuses, which are handled below instead.
    throw new NetworkError();
  }

  if (!response.ok) {
    const detail = await parseErrorDetail(response);
    const message =
      detail ??
      (response.status >= 500
        ? "The server hit an unexpected error. Please try again shortly."
        : `Request failed (${response.status}).`);
    throw new ApiError(response.status, detail, message);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<AnalyzeResponse>("/api/analyze", {
    method: "POST",
    body: formData,
  });
}

export async function listAnalyses(
  limit = 20,
  offset = 0
): Promise<PaginatedAnalyses> {
  const page = Math.floor(offset / limit) + 1;
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(limit),
  });
  return request<PaginatedAnalyses>(`/api/analyses?${params.toString()}`);
}

export async function getAnalysis(id: string): Promise<AnalysisDetail> {
  return request<AnalysisDetail>(`/api/analyses/${id}`);
}

export function getGradcamUrl(id: string, head: IssueType): string {
  const params = new URLSearchParams({ head });
  return `${API_BASE_URL}/api/analyses/${id}/gradcam?${params.toString()}`;
}

export function getAnalysisImageUrl(id: string): string {
  return `${API_BASE_URL}/api/analyses/${id}/image`;
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  return request(`/health`);
}
