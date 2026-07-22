"""Tests for ``CalendarReleaseResolver`` (per #218).

Covers:

- ``validate_config`` enforces required fields, bounded cadences,
  conditional ``anchor_dow`` (required for weekly, forbidden otherwise),
  ``anchor_tz`` validation via :mod:`zoneinfo`, and unknown-key rejection.
- ``current_release`` computes the correct cadence boundary for weekly
  (Sunday-anchored, Monday-anchored, wraparound), monthly, and quarterly.
- ``current_release`` honors ``anchor_tz``: a frozen UTC timestamp that
  spans the local-day boundary in another timezone yields different
  partition keys.
- ``current_release`` and ``resolve_url`` accept ``api_key=None`` (the
  resolver is forward-only and doesn't authenticate).
- The resolver is registered at import time alongside ``uts_release``.
- Forward-only: ``historical_release`` / ``list_releases`` are NOT
  implemented (per #218 Non-goals).
"""

from __future__ import annotations

import socket
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

from freezegun import freeze_time

from moncpipelib.ingest.resolvers import RESOLVERS
from moncpipelib.ingest.resolvers.calendar import CalendarReleaseResolver
from moncpipelib.ingest.types import IngestContext


def _ctx() -> IngestContext:
    return IngestContext(log=MagicMock(name="LoggingContext"), secrets=None)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_calendar_resolver_registered_at_import_time() -> None:
    assert "calendar" in RESOLVERS
    assert isinstance(RESOLVERS["calendar"], CalendarReleaseResolver)


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


def _good_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "start_date": "2024-01-01",
        "cadence": "weekly",
        "anchor_dow": "Sunday",
        "url": "https://upstream.example/feed.zip",
    }
    cfg.update(overrides)
    return cfg


def test_validate_config_happy_path() -> None:
    assert CalendarReleaseResolver().validate_config(_good_config()) == []


def test_validate_config_happy_path_monthly() -> None:
    cfg = _good_config(cadence="monthly")
    cfg.pop("anchor_dow")  # forbidden for monthly
    assert CalendarReleaseResolver().validate_config(cfg) == []


def test_validate_config_happy_path_quarterly_with_tz() -> None:
    cfg = _good_config(cadence="quarterly", anchor_tz="America/Chicago")
    cfg.pop("anchor_dow")
    assert CalendarReleaseResolver().validate_config(cfg) == []


def test_validate_config_accepts_date_object_for_start_date() -> None:
    """Per #218 Q1: start_date accepts ISO string or date object."""
    cfg = _good_config()
    cfg["start_date"] = date(2024, 1, 1)
    assert CalendarReleaseResolver().validate_config(cfg) == []


def test_validate_config_required_start_date() -> None:
    cfg = _good_config()
    del cfg["start_date"]
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("start_date" in e and "required" in e for e in errors)


def test_validate_config_invalid_start_date_format() -> None:
    cfg = _good_config(start_date="01/01/2024")
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("start_date" in e and "ISO" in e for e in errors)


def test_validate_config_invalid_start_date_type() -> None:
    cfg = _good_config(start_date=20240101)
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("start_date" in e for e in errors)


def test_validate_config_required_cadence() -> None:
    cfg = _good_config()
    del cfg["cadence"]
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("cadence" in e and "required" in e for e in errors)


def test_validate_config_invalid_cadence() -> None:
    cfg = _good_config(cadence="biweekly")
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("cadence" in e and "must be one of" in e for e in errors)


def test_validate_config_weekly_requires_anchor_dow() -> None:
    cfg = _good_config()
    del cfg["anchor_dow"]
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("anchor_dow" in e and "required" in e and "weekly" in e for e in errors)


def test_validate_config_invalid_anchor_dow() -> None:
    cfg = _good_config(anchor_dow="Funday")
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("anchor_dow" in e and "must be one of" in e for e in errors)


def test_validate_config_anchor_dow_forbidden_for_monthly() -> None:
    """Per #218 Q2: anchor_dow + cadence != weekly is an error, not silently ignored."""
    cfg = _good_config(cadence="monthly")
    # anchor_dow stays from the default
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("anchor_dow" in e and "forbidden" in e for e in errors)


def test_validate_config_anchor_dow_forbidden_for_quarterly() -> None:
    cfg = _good_config(cadence="quarterly")
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("anchor_dow" in e and "forbidden" in e for e in errors)


def test_validate_config_invalid_anchor_tz() -> None:
    cfg = _good_config(anchor_tz="Mars/Olympus_Mons")
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("anchor_tz" in e and "unknown timezone" in e for e in errors)


def test_validate_config_default_anchor_tz_is_implicit_utc() -> None:
    """Omitting anchor_tz is fine; default UTC applies at runtime."""
    cfg = _good_config()
    assert "anchor_tz" not in cfg
    assert CalendarReleaseResolver().validate_config(cfg) == []


def test_validate_config_required_url() -> None:
    cfg = _good_config()
    del cfg["url"]
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("url" in e and "required" in e for e in errors)


def test_validate_config_empty_url_rejected() -> None:
    cfg = _good_config(url="")
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("url" in e and "non-empty" in e for e in errors)


def test_validate_config_unknown_field_rejected() -> None:
    cfg = _good_config(extra_unknown="value")
    errors = CalendarReleaseResolver().validate_config(cfg)
    assert any("extra_unknown" in e and "unknown" in e for e in errors)


def test_validate_config_no_network_io() -> None:
    """ADR-2: validate_config makes no network calls.

    Patch socket.socket so any attempted connect fails the test.
    """

    def _no_sockets(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("validate_config opened a socket")

    with patch.object(socket, "socket", side_effect=_no_sockets):
        CalendarReleaseResolver().validate_config(_good_config())


# ---------------------------------------------------------------------------
# current_release: cadence boundary math
# ---------------------------------------------------------------------------


@freeze_time("2026-04-22 12:00:00")  # Wednesday
def test_current_release_weekly_sunday_anchor_wraparound() -> None:
    """Wednesday with anchor_dow=Sunday -> boundary = prior Sunday."""
    out = CalendarReleaseResolver().current_release(
        api_key=None,
        config=_good_config(anchor_dow="Sunday"),
        ctx=_ctx(),
    )
    assert out["partition_key"] == "2026-04-19"  # Sunday
    assert out["snapshot_date"] == "2026-04-19"
    assert out["url"] == "https://upstream.example/feed.zip"


@freeze_time("2026-04-19 12:00:00")  # Sunday
def test_current_release_weekly_anchor_is_today() -> None:
    """When today IS the anchor day, boundary = today."""
    out = CalendarReleaseResolver().current_release(
        api_key=None,
        config=_good_config(anchor_dow="Sunday"),
        ctx=_ctx(),
    )
    assert out["partition_key"] == "2026-04-19"


@freeze_time("2026-04-22 12:00:00")  # Wednesday
def test_current_release_weekly_monday_anchor() -> None:
    """Monday-anchored: boundary = prior Monday (or today if today is Monday)."""
    out = CalendarReleaseResolver().current_release(
        api_key=None,
        config=_good_config(anchor_dow="Monday"),
        ctx=_ctx(),
    )
    assert out["partition_key"] == "2026-04-20"  # Monday before


@freeze_time("2026-04-22 12:00:00")
def test_current_release_monthly() -> None:
    """Mid-month -> first of month."""
    cfg = _good_config(cadence="monthly")
    cfg.pop("anchor_dow")
    out = CalendarReleaseResolver().current_release(api_key=None, config=cfg, ctx=_ctx())
    assert out["partition_key"] == "2026-04-01"


@freeze_time("2026-05-15 12:00:00")
def test_current_release_quarterly_q2() -> None:
    """May -> Q2 start = April 1."""
    cfg = _good_config(cadence="quarterly")
    cfg.pop("anchor_dow")
    out = CalendarReleaseResolver().current_release(api_key=None, config=cfg, ctx=_ctx())
    assert out["partition_key"] == "2026-04-01"


@freeze_time("2026-12-31 12:00:00")
def test_current_release_quarterly_q4() -> None:
    """December -> Q4 start = October 1."""
    cfg = _good_config(cadence="quarterly")
    cfg.pop("anchor_dow")
    out = CalendarReleaseResolver().current_release(api_key=None, config=cfg, ctx=_ctx())
    assert out["partition_key"] == "2026-10-01"


@freeze_time("2026-01-01 00:30:00")
def test_current_release_quarterly_q1() -> None:
    """Jan 1 -> Q1 start = Jan 1 (boundary itself)."""
    cfg = _good_config(cadence="quarterly")
    cfg.pop("anchor_dow")
    out = CalendarReleaseResolver().current_release(api_key=None, config=cfg, ctx=_ctx())
    assert out["partition_key"] == "2026-01-01"


# ---------------------------------------------------------------------------
# anchor_tz behavior
# ---------------------------------------------------------------------------


@freeze_time("2026-04-19 03:30:00")
def test_current_release_anchor_tz_chicago_vs_utc_disagrees_on_day() -> None:
    """At 2026-04-19 03:30 UTC, it is 2026-04-18 22:30 in America/Chicago.

    With anchor_dow=Sunday:
    - UTC: today is Sunday 2026-04-19 -> partition_key = 2026-04-19
    - Chicago: today is Saturday 2026-04-18 -> partition_key = 2026-04-12 (prior Sunday)
    """
    cfg_utc = _good_config(anchor_dow="Sunday")  # default anchor_tz=UTC
    cfg_chi = _good_config(anchor_dow="Sunday", anchor_tz="America/Chicago")

    out_utc = CalendarReleaseResolver().current_release(None, cfg_utc, _ctx())
    out_chi = CalendarReleaseResolver().current_release(None, cfg_chi, _ctx())

    assert out_utc["partition_key"] == "2026-04-19"
    assert out_chi["partition_key"] == "2026-04-12"
    assert out_utc["partition_key"] != out_chi["partition_key"]


# ---------------------------------------------------------------------------
# resolve_url
# ---------------------------------------------------------------------------


def test_resolve_url_returns_config_url_verbatim() -> None:
    """Per #270 the resolver returns ``ResolvedDownload``; the calendar
    resolver has no semantic filename hint and sets ``filename=None``."""
    cfg = _good_config(url="https://upstream.example/feed.zip")
    out = CalendarReleaseResolver().resolve_url(
        api_key=None,
        partition_key="2026-04-19",
        config=cfg,
        ctx=_ctx(),
    )
    assert out.url == "https://upstream.example/feed.zip"
    assert out.filename is None


def test_resolve_url_ignores_api_key_and_partition_key() -> None:
    """Calendar resolver doesn't authenticate and the URL is static."""
    cfg = _good_config(url="https://upstream.example/feed.zip")
    out_a = CalendarReleaseResolver().resolve_url(None, "2026-04-19", cfg, _ctx())
    out_b = CalendarReleaseResolver().resolve_url("ignored", "2026-04-26", cfg, _ctx())
    assert out_a == out_b
    assert out_a.url == "https://upstream.example/feed.zip"
    assert out_a.filename is None


# ---------------------------------------------------------------------------
# Forward-only opt-out
# ---------------------------------------------------------------------------


def test_calendar_resolver_historical_release_returns_empty() -> None:
    """Per #218 Non-goals + #228 Protocol method: the calendar resolver
    opts out of historical discovery by returning ``[]`` from
    ``historical_release``.  ``ApiResolverPattern`` then falls back to
    ``current_release`` for forward-only synthesis."""
    resolver = CalendarReleaseResolver()
    assert resolver.historical_release(None, _good_config(), _ctx()) == []
