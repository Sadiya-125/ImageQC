"""
FastAPI application entrypoint: wires together config, DB, the ML inference
engine, and the API routers. The model is loaded exactly once, at process
startup, via the lifespan handler below -- never per-request.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import analyses, analyze
from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger, log_with_fields
from app.ml.inference import get_inference_engine

configure_logging()
logger = get_logger("imageqc")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_inference_engine()
    try:
        engine.load()
        log_with_fields(logger, 20, "model_loaded", model_version=engine.model_version, device=str(engine.device))
    except Exception as e:  # noqa: BLE001
        # A load failure doesn't crash the process -- /health reports it via
        # 503 so the container stays up (and inspectable) instead of
        # crash-looping.
        log_with_fields(logger, 40, "model_load_failed", error=str(e))
    yield


settings = get_settings()

app = FastAPI(title="ImageQC API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    log_with_fields(
        logger,
        20,
        "request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 1),
    )
    return response


app.include_router(analyze.router, prefix="/api", tags=["analyze"])
app.include_router(analyses.router, prefix="/api", tags=["analyses"])


@app.get("/health")
async def health():
    engine = get_inference_engine()
    if not engine.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_loaded": False, "error": engine.load_error},
        )
    return {"status": "ok", "model_loaded": True}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # FastAPI/Starlette already turn raised HTTPExceptions into structured
    # JSON with the right status code on their own; this catches everything
    # else so a bug never leaks a stack trace to the client.
    log_with_fields(logger, 40, "unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
