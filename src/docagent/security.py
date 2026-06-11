"""API hardening primitives — kept framework-free so they unit-test offline.

``web.py`` wraps these into FastAPI dependencies; the logic (fixed-window rate
limiting, constant-style key comparison) lives here as pure functions/classes so
it can be tested without spinning up the app or loading any models.
"""

import hmac
import os


def api_key_ok(provided: str | None) -> bool:
    """True if auth passes. Auth is OFF when ``$DOCAGENT_API_KEY`` is unset
    (dev-friendly); when set, the request must present the exact key."""
    expected = os.environ.get("DOCAGENT_API_KEY")
    if not expected:
        return True
    return provided is not None and hmac.compare_digest(provided, expected)


class RateLimiter:
    """Fixed-window, per-key request limiter (in-process).

    Pure and deterministic: the caller passes ``now`` (monotonic seconds), so the
    window logic is testable without sleeping or patching the clock. Good enough
    for a single-process server; use a shared store (e.g. Redis) when horizontally
    scaled.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float) -> bool:
        recent = [t for t in self._hits.get(key, []) if now - t < self.window]
        if len(recent) >= self.max_requests:
            self._hits[key] = recent  # keep pruned window; reject
            return False
        recent.append(now)
        self._hits[key] = recent
        return True
