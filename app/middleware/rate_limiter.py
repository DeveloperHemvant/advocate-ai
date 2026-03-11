from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import get_settings


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Very simple in-memory IP-based rate limiter.
    """

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60) -> None:  # type: ignore[override]
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # type: ignore[override]
        settings = get_settings()
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Focus stricter limits on unified AI endpoint, lighter elsewhere
        max_req = getattr(settings, "rate_limit_per_minute", self.max_requests)
        if path == "/legal-ai":
            max_req = getattr(settings, "rate_limit_per_minute_legal_ai", max_req)

        now = time.time()
        window_start = now - self.window_seconds
        bucket = self._buckets.setdefault(client_ip, [])
        # Remove old entries
        bucket[:] = [ts for ts in bucket if ts >= window_start]
        if len(bucket) >= max_req:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down your requests."},
            )
        bucket.append(now)
        return await call_next(request)

