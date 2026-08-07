"""Rate-limited HTTP client wrapper for crawl-style ingest (#415).

Sources with no bulk download are assembled by fanning out many small
requests against a live API (RxNav caps at 20 req/s per IP).  Crawl
plans receive a :class:`ThrottledClient` -- never a raw
:class:`httpx.Client` -- so that:

- every request is paced to the contract's ``rate_limit_rps`` budget
  via a monotonic-clock minimum interval between requests;
- every request routes through the redacting client from
  :func:`~moncpipelib.ingest._http.build_redacting_client` (the wrapper
  is constructed around it), so query strings / headers never appear in
  transport logs;
- the surface is read-only: only ``get`` is exposed.  Crawls do not
  mutate upstream state, and a narrow surface keeps plans from
  bypassing the throttle or the audit hooks.

Scope caveat (SECURITY.md, "Crawl-Assembled Ingest Payloads"): the
limiter is **per-process** while upstream caps are typically **per-IP**.
Concurrent Dagster runs behind one egress IP share no state here --
contracts should set conservative budgets and consumers should bound
run concurrency (Dagster tag-based concurrency limits).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx


class ThrottledClient:
    """Minimum-interval pacing wrapper around an :class:`httpx.Client`.

    Construct via :meth:`from_rate_limit` (the pattern does this from
    the contract's ``rate_limit_rps``).  The first request is not
    delayed; each subsequent request waits until at least
    ``min_interval_s`` has elapsed since the previous request was
    issued.

    ``clock`` / ``sleep`` are injectable for deterministic tests; the
    defaults are :func:`time.monotonic` and :func:`time.sleep`.
    """

    def __init__(
        self,
        client: httpx.Client,
        *,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval_s < 0:
            raise ValueError(f"min_interval_s must be >= 0; got {min_interval_s!r}")
        self._client = client
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None

    @classmethod
    def from_rate_limit(
        cls,
        client: httpx.Client,
        *,
        rate_limit_rps: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> ThrottledClient:
        """Build a wrapper enforcing at most ``rate_limit_rps`` requests/second.

        Raises:
            ValueError: When ``rate_limit_rps`` is not > 0.  (Contract
                validation rejects this earlier with a guidance-bearing
                message; this guard covers direct constructor misuse.)
        """
        if rate_limit_rps <= 0:
            raise ValueError(
                f"rate_limit_rps must be > 0 (requests per second); got {rate_limit_rps!r}"
            )
        return cls(client, min_interval_s=1.0 / rate_limit_rps, clock=clock, sleep=sleep)

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int | float] | None = None,
    ) -> httpx.Response:
        """Paced GET; raises on 4xx/5xx.

        Waits out the residual minimum interval (if any), issues the
        request through the wrapped redacting client, and calls
        ``raise_for_status()`` before returning -- crawl plans handle
        response *content*, not transport errors.

        Args:
            url: Target URL.  Never logged with its query string (the
                wrapped client's redacting hooks fire as usual).
            params: Optional query parameters, merged by httpx.

        Raises:
            httpx.HTTPStatusError: On a 4xx / 5xx response.
        """
        now = self._clock()
        if self._last_request_at is not None:
            wait = self._min_interval_s - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()
        response = self._client.get(url, params=dict(params) if params is not None else None)
        response.raise_for_status()
        return response
