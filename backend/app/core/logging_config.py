"""
Structured (JSON-lines) logging setup. Deliberately minimal for a 48-hour
scope -- this is "can I see what's happening from the logs" observability
(every request's method/path/status/duration, model load outcome, each
analysis's key result), not a full tracing/metrics stack (Sentry, OpenTelemetry,
etc). Render (and most container platforms) capture stdout directly, so
JSON-per-line here is immediately grep/jq-able in production without needing
a separate log shipper.
"""

import json
import logging
import sys
import time
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via logger.info("...", extra={...}) rides along.
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn's own access logger is redundant with the request-logging
    # middleware in main.py (which includes latency, uvicorn's doesn't) --
    # quieted to avoid duplicate lines per request.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_with_fields(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"extra_fields": fields})


class Timer:
    """Small helper: `with Timer() as t: ...` then `t.elapsed_ms`."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
