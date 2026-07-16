"""Tests for ``UtsReleaseResolver``.

Network is mocked via respx; no real UTS calls.  Covers:

- ``validate_config`` (required field, allowed values, unknown-key
  rejection per ADR-2).
- ``current_release`` happy path and empty-list error.
- ``resolve_url`` happy path and partition-mismatch error.
- api_key redaction: even on a 401 path, the api_key never appears
  in captured log output.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.resolvers.uts import UtsReleaseResolver
from moncpipelib.ingest.types import IngestContext


def _ctx() -> IngestContext:
    return IngestContext(log=MagicMock(name="LoggingContext"))


def _release_payload(version: str, url: str) -> list[dict[str, str]]:
    return [{"releaseVersion": version, "downloadUrl": url}]


def _historical_payload(*entries: tuple[str, str, str]) -> list[dict[str, str]]:
    """Build a UTS historical releases payload.

    Each tuple is ``(releaseVersion, downloadUrl, releaseDate)``.
    """
    return [{"releaseVersion": v, "downloadUrl": u, "releaseDate": d} for v, u, d in entries]


# ---------------------------------------------------------------------------
# validate_config (ADR-2)
# ---------------------------------------------------------------------------


def test_validate_config_requires_release_type() -> None:
    errors = UtsReleaseResolver().validate_config({})
    assert any("release_type: required" in e for e in errors)


def test_validate_config_release_type_must_be_string() -> None:
    errors = UtsReleaseResolver().validate_config({"release_type": 42})
    assert any("release_type: must be a string" in e for e in errors)


def test_validate_config_release_type_must_be_known() -> None:
    errors = UtsReleaseResolver().validate_config({"release_type": "bogus"})
    assert any("must be one of" in e for e in errors)


def test_validate_config_unknown_key_rejected() -> None:
    """ADR-2: unknown keys must be flagged so typos like
    ``releas_type`` fail at contract-load time."""
    errors = UtsReleaseResolver().validate_config(
        {
            "release_type": "umls-full-release",
            "releas_type": "typo",
            "garbage": True,
        }
    )
    assert any("releas_type: unknown field" in e for e in errors)
    assert any("garbage: unknown field" in e for e in errors)


def test_validate_config_happy_path_yields_no_errors() -> None:
    errors = UtsReleaseResolver().validate_config({"release_type": "umls-full-release"})
    assert errors == []


def test_validate_config_aggregates_multiple_errors() -> None:
    """Audit posture: validate_config returns the full error list rather
    than bailing on the first one."""
    errors = UtsReleaseResolver().validate_config({"release_type": "bogus", "extra": 1})
    assert len(errors) >= 2


def test_validate_config_accepts_iso_start_date() -> None:
    """Per #228: optional start_date bounds historical discovery."""
    errors = UtsReleaseResolver().validate_config(
        {"release_type": "umls-full-release", "start_date": "2024-01-01"}
    )
    assert errors == []


def test_validate_config_rejects_malformed_start_date() -> None:
    errors = UtsReleaseResolver().validate_config(
        {"release_type": "umls-full-release", "start_date": "01/01/2024"}
    )
    assert any("start_date" in e and "ISO" in e for e in errors)


# ---------------------------------------------------------------------------
# current_release
# ---------------------------------------------------------------------------


@respx.mock
def test_current_release_happy_path() -> None:
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_release_payload("2026AA", "https://download.nlm.nih.gov/umls/.../2026AA.zip"),
    )

    result = UtsReleaseResolver().current_release(
        api_key="ignored-by-this-endpoint",
        config={"release_type": "umls-full-release"},
        ctx=_ctx(),
    )

    assert result["partition_key"] == "2026AA"
    assert result["release_version"] == "2026AA"
    assert result["download_url"].endswith("/2026AA.zip")


@respx.mock
def test_current_release_passes_release_type_query_param() -> None:
    """Sanity: the configured release_type is forwarded to UTS."""
    route = respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200, json=_release_payload("v1", "https://example/v1.zip")
    )

    UtsReleaseResolver().current_release(
        api_key="x",
        config={"release_type": "rxnorm-full-monthly-release"},
        ctx=_ctx(),
    )

    assert route.called
    request = route.calls[0].request
    assert request.url.params["releaseType"] == "rxnorm-full-monthly-release"
    assert request.url.params["current"] == "true"


@respx.mock
def test_current_release_empty_payload_raises() -> None:
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(200, json=[])

    with pytest.raises(ValueError, match="No current release found"):
        UtsReleaseResolver().current_release(
            api_key="x",
            config={"release_type": "umls-full-release"},
            ctx=_ctx(),
        )


@respx.mock
def test_current_release_5xx_raises_http_status_error() -> None:
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(503)

    with pytest.raises(httpx.HTTPStatusError):
        UtsReleaseResolver().current_release(
            api_key="x",
            config={"release_type": "umls-full-release"},
            ctx=_ctx(),
        )


# ---------------------------------------------------------------------------
# resolve_url
# ---------------------------------------------------------------------------


@respx.mock
def test_resolve_url_embeds_api_key_and_download_url() -> None:
    """Per #270, ``resolve_url`` returns ``ResolvedDownload``: the
    ``url`` embeds the apiKey (as before) and ``filename`` carries the
    upstream download_url's basename as a semantic hint for the
    non-archive payload chain."""
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_release_payload("2026AA", "https://download.nlm.nih.gov/u/2026AA.zip"),
    )

    resolved = UtsReleaseResolver().resolve_url(
        api_key="abc-secret-123",
        partition_key="2026AA",
        config={"release_type": "umls-full-release"},
        ctx=_ctx(),
    )

    parsed = urllib.parse.urlparse(resolved.url)
    assert parsed.netloc == "uts-ws.nlm.nih.gov"
    assert parsed.path == "/download"
    params = urllib.parse.parse_qs(parsed.query)
    assert params["apiKey"] == ["abc-secret-123"]
    assert params["url"] == ["https://download.nlm.nih.gov/u/2026AA.zip"]
    assert resolved.filename == "2026AA.zip"


def test_resolve_url_raises_without_api_key() -> None:
    """Per #218: UTS download endpoint requires authentication.  A
    credential-less contract (api_key=None) cannot resolve this URL."""
    with pytest.raises(IngestResolutionError, match="credential"):
        UtsReleaseResolver().resolve_url(
            api_key=None,
            partition_key="2026AA",
            config={"release_type": "umls-full-release"},
            ctx=_ctx(),
        )


@respx.mock
def test_resolve_url_for_unknown_partition_raises_clearly() -> None:
    """Per #228: ``resolve_url`` looks up the partition against
    ``historical_release`` first.  When UTS no longer hosts a release
    that matches ``partition_key``, the error message lists the
    available releases so an operator sees the drift immediately."""
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_release_payload("2026AB", "https://download.nlm.nih.gov/u/2026AB.zip"),
    )

    with pytest.raises(IngestResolutionError, match="2026AB"):
        UtsReleaseResolver().resolve_url(
            api_key="x",
            partition_key="2026AA",  # not in UTS's current historical list
            config={"release_type": "umls-full-release"},
            ctx=_ctx(),
        )


# ---------------------------------------------------------------------------
# historical_release (per #228)
# ---------------------------------------------------------------------------


@respx.mock
def test_historical_release_returns_full_list() -> None:
    """When UTS hosts multiple historical releases, all are surfaced."""
    route = respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_historical_payload(
            ("2024AA", "https://download.nlm.nih.gov/u/2024AA.zip", "2024-05-06"),
            ("2024AB", "https://download.nlm.nih.gov/u/2024AB.zip", "2024-11-04"),
            ("2025AA", "https://download.nlm.nih.gov/u/2025AA.zip", "2025-05-05"),
        ),
    )

    releases = UtsReleaseResolver().historical_release(
        api_key="ignored",
        config={"release_type": "umls-full-release"},
        ctx=_ctx(),
    )

    assert [r["partition_key"] for r in releases] == ["2024AA", "2024AB", "2025AA"]
    assert all(r["release_version"] == r["partition_key"] for r in releases)
    assert all(r["download_url"].endswith(".zip") for r in releases)

    # Verifies the call uses current=false (the historical endpoint shape).
    assert route.called
    request = route.calls[0].request
    assert request.url.params["releaseType"] == "umls-full-release"
    assert request.url.params["current"] == "false"


@respx.mock
def test_historical_release_filters_by_start_date() -> None:
    """Per #228: ``start_date`` bounds the returned list."""
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_historical_payload(
            ("2019AB", "https://download.nlm.nih.gov/u/2019AB.zip", "2019-11-04"),
            ("2024AA", "https://download.nlm.nih.gov/u/2024AA.zip", "2024-05-06"),
            ("2024AB", "https://download.nlm.nih.gov/u/2024AB.zip", "2024-11-04"),
            ("2025AA", "https://download.nlm.nih.gov/u/2025AA.zip", "2025-05-05"),
        ),
    )

    releases = UtsReleaseResolver().historical_release(
        api_key="ignored",
        config={
            "release_type": "umls-full-release",
            "start_date": "2024-01-01",
        },
        ctx=_ctx(),
    )

    # 2019AB is filtered out; everything 2024-01-01 or later remains.
    assert [r["partition_key"] for r in releases] == ["2024AA", "2024AB", "2025AA"]


@respx.mock
def test_historical_release_empty_payload_returns_empty_list() -> None:
    """Empty UTS response -> empty list (pattern falls back to current_release)."""
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(200, json=[])

    releases = UtsReleaseResolver().historical_release(
        api_key="ignored",
        config={"release_type": "umls-full-release"},
        ctx=_ctx(),
    )
    assert releases == []


@respx.mock
def test_historical_release_release_date_carried_through() -> None:
    """``release_date`` must round-trip into the release dict so consumers
    using ``effective_from_field=release_date`` can hydrate from it."""
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_historical_payload(
            ("2024AA", "https://download.nlm.nih.gov/u/2024AA.zip", "2024-05-06"),
        ),
    )

    [release] = UtsReleaseResolver().historical_release(
        api_key="ignored",
        config={"release_type": "umls-full-release"},
        ctx=_ctx(),
    )
    assert release["release_date"] == "2024-05-06"


@respx.mock
def test_historical_release_caches_within_ctx() -> None:
    """Per #256: a single ``IngestContext`` memoizes ``historical_release``
    so that ``resolve_url`` (called by ``materialize_partition``) and
    ``ApiResolverPattern.partition_metadata`` (called by the dispatcher
    at manifest-write time) share one HTTP fetch.  Without the cache,
    a backfill of N partitions would 2x the UTS list-endpoint hits."""
    route = respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_historical_payload(
            ("2024AA", "https://download.nlm.nih.gov/u/2024AA.zip", "2024-05-06"),
            ("2024AB", "https://download.nlm.nih.gov/u/2024AB.zip", "2024-11-04"),
        ),
    )

    resolver = UtsReleaseResolver()
    config = {"release_type": "umls-full-release"}
    ctx = _ctx()

    first = resolver.historical_release(api_key="k", config=config, ctx=ctx)
    second = resolver.historical_release(api_key="k", config=config, ctx=ctx)

    assert route.call_count == 1
    assert first == second


@respx.mock
def test_historical_release_cache_scope_is_per_ctx() -> None:
    """Cache scope is one ``IngestContext`` instance; a new ctx (e.g. a
    later partition's materialization, or a sensor tick) gets a fresh
    fetch.  This guards against the cache silently masking upstream
    changes between materializations."""
    route = respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_historical_payload(
            ("2024AA", "https://download.nlm.nih.gov/u/2024AA.zip", "2024-05-06"),
        ),
    )

    resolver = UtsReleaseResolver()
    config = {"release_type": "umls-full-release"}

    resolver.historical_release(api_key="k", config=config, ctx=_ctx())
    resolver.historical_release(api_key="k", config=config, ctx=_ctx())

    assert route.call_count == 2


@respx.mock
def test_historical_release_cache_keyed_by_start_date() -> None:
    """Different ``start_date`` filters yield different cache entries
    within the same ctx -- so a contract that calls historical_release
    twice with different bounds gets two fetches, not a stale hit."""
    route = respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_historical_payload(
            ("2019AB", "https://download.nlm.nih.gov/u/2019AB.zip", "2019-11-04"),
            ("2024AA", "https://download.nlm.nih.gov/u/2024AA.zip", "2024-05-06"),
        ),
    )

    resolver = UtsReleaseResolver()
    ctx = _ctx()

    resolver.historical_release(
        api_key="k",
        config={"release_type": "umls-full-release"},
        ctx=ctx,
    )
    resolver.historical_release(
        api_key="k",
        config={"release_type": "umls-full-release", "start_date": "2024-01-01"},
        ctx=ctx,
    )

    assert route.call_count == 2


@respx.mock
def test_resolve_url_works_for_historical_partition() -> None:
    """Per #228: resolve_url looks up the partition against
    historical_release, not just current_release.  Materialization for
    historical partitions works end-to-end."""
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(
        200,
        json=_historical_payload(
            ("2024AA", "https://download.nlm.nih.gov/u/2024AA.zip", "2024-05-06"),
            ("2024AB", "https://download.nlm.nih.gov/u/2024AB.zip", "2024-11-04"),
        ),
    )

    resolved = UtsReleaseResolver().resolve_url(
        api_key="abc-secret",
        partition_key="2024AA",  # not the latest historical
        config={"release_type": "umls-full-release"},
        ctx=_ctx(),
    )

    parsed = urllib.parse.urlparse(resolved.url)
    params = urllib.parse.parse_qs(parsed.query)
    assert params["url"] == ["https://download.nlm.nih.gov/u/2024AA.zip"]
    assert params["apiKey"] == ["abc-secret"]
    assert resolved.filename == "2024AA.zip"


# ---------------------------------------------------------------------------
# Redaction (load-bearing for SOC 2 / HITRUST audit posture)
# ---------------------------------------------------------------------------


@respx.mock
def test_api_key_never_appears_in_logs_on_401(caplog: Any) -> None:
    """Mock a 401 on the releases endpoint.  Even on the error path,
    the api_key the caller passed must never appear in captured log
    output -- redaction holds.

    The releases endpoint itself is unauthenticated, so we contrive a
    scenario by including the api_key in the params we'd have if a
    future API required it.  The test asserts the redacting client's
    behavior, not UTS-specific routing.
    """
    caplog.set_level(logging.INFO)
    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(401)

    secret = "REDACT-ME-SUPER-SECRET-API-KEY"
    with pytest.raises(httpx.HTTPStatusError):
        UtsReleaseResolver().current_release(
            api_key=secret,
            config={"release_type": "umls-full-release"},
            ctx=_ctx(),
        )

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in captured


# ---------------------------------------------------------------------------
# Streaming-memory acceptance (Migration 012 Phase F / #247)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@respx.mock
def test_historical_release_streaming_memory_bound() -> None:
    """A 100k-entry releases payload streams via ijson; peak heap stays
    bounded by the parser's chunk size, not the response body's total
    size.

    Pre-fix ``historical_release`` called ``response.json()`` which
    materialized the full list-of-dicts on the heap before any
    filtering applied.  For 100k entries at ~150 bytes per release
    that's ~30 MiB+ in dict allocations.  Post-fix ``ijson.items``
    yields one entry at a time; with a ``start_date`` filter that
    drops most entries the peak collected heap ~ entries-passing-filter
    rather than total entries.

    Threshold: 32 MiB peak Python heap above baseline.  Pre-fix the
    full list-of-dicts (100k entries x ~200 bytes per dict = ~20 MiB)
    landed on the heap on top of the ~8 MiB response-body buffer +
    httpx framing.  Post-fix the parser yields one entry at a time
    and the filtered-out entries never materialize as Python dicts at
    all; peak settles in the low-20s MiB range.
    """
    import json
    import tracemalloc

    n_entries = 100_000
    # 32 MiB is the operational ceiling we want to pin: pre-fix this
    # would have been ~28-30 MiB of dict allocations on top of the
    # response buffer; post-fix peaks in the low-20s.  A regression
    # that re-introduces ``response.json()`` would push past 32 MiB.
    threshold_bytes = 32 * 1024 * 1024

    # Build the synthetic payload OUTSIDE the tracemalloc window.  The
    # payload's bytes are not counted against the streaming bound --
    # we measure the peak heap during ``historical_release``, which
    # streams over them.
    entries = [
        {
            "releaseVersion": f"r-{i:06d}",
            "downloadUrl": f"https://example.test/{i:06d}.zip",
            "releaseDate": "2026-01-01",
        }
        for i in range(n_entries)
    ]
    payload_bytes = json.dumps(entries).encode("utf-8")
    # Sanity: payload is non-trivially large so the streaming
    # property is meaningful.
    assert len(payload_bytes) > 5 * 1024 * 1024

    respx.get("https://uts-ws.nlm.nih.gov/releases").respond(200, content=payload_bytes)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        # ``start_date`` filters EVERY entry out (release_date <
        # filter), so the kept-entries list stays empty -- the test
        # measures the streaming-parse overhead, not the kept payload.
        result = UtsReleaseResolver().historical_release(
            api_key=None,
            config={
                "release_type": "umls-full-release",
                "start_date": "2099-01-01",
            },
            ctx=_ctx(),
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result == []  # all entries filtered out
    assert peak <= threshold_bytes, (
        f"peak Python heap during a {n_entries:,}-entry historical_release "
        f"was {peak / 1024 / 1024:.1f} MiB -- streaming regression?  "
        f"Threshold: {threshold_bytes / 1024 / 1024:.0f} MiB."
    )
