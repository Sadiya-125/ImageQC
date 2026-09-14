"""
FastAPI application entrypoint: wires together config, DB, the ML inference
engine, and the API routers. The model is loaded exactly once, at process
startup, via the lifespan handler below -- never per-request.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import analyses, analyze
from app.core.config import get_settings
from app.ml.inference import get_inference_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_inference_engine()
    try:
        engine.load()
        print("Model loaded successfully.")
    except Exception as e:  # noqa: BLE001
        # A load failure doesn't crash the process -- /health reports it via
        # 503 so the container stays up (and inspectable) instead of
        # crash-looping.
        print(f"WARNING: model failed to load at startup: {e}")
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
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
