"""Tests for the shared cadence-boundary helper (``ingest/_cadence.py``, #463).

Covers:

- Refactor equivalence: ``cadence_boundary`` reproduces the pre-extraction
  ``CalendarReleaseResolver.current_release`` arithmetic exactly, swept over
  three years for monthly, quarterly, and all seven weekly anchors.
- ``daily`` cadence (new in this extraction) and its DST behavior across
  both the US spring-forward and fall-back transitions.
- ``lag_hours`` is applied to the aware datetime *before* truncation to a
  date -- the guard against the silent no-op of subtracting a timedelta
  from a ``date``.
- Input validation: naive ``now``, unknown cadence, weekly's ``anchor_dow``
  requirement, unknown ``anchor_dow``, non-weekly cadences ignoring a stray
  ``anchor_dow``, and an unknown ``anchor_tz`` propagating uncaught.
- ``KNOWN_CADENCES`` is a superset of the calendar resolver's validated
  cadence set (structural rule, not a hand-copied twin list).

The acceptance criterion that ``tests/test_calendar_release_resolver.py``
is a zero-line diff and still green is checked separately (not a test
function here) -- see the Step 1 executor brief.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest
from freezegun import freeze_time

from moncpipelib.ingest._cadence import KNOWN_CADENCES, WEEKDAY_BY_NAME, cadence_boundary
from moncpipelib.ingest.resolvers.calendar import _KNOWN_CADENCES

UTC = ZoneInfo("UTC")
NY = "America/New_York"


def _legacy_boundary(today: date, cadence: str, anchor_dow: str | None) -> date:
    """Reproduce the pre-extraction ``calendar.py:196-206`` arithmetic verbatim.

    Operates on an already-truncated ``today``, exactly as the old
    ``current_release`` did after ``datetime.now(tz=anchor_tz).date()``.
    """
    if cadence == "weekly":
        target = WEEKDAY_BY_NAME[str(anchor_dow)]
        delta = (today.weekday() - target) % 7
        return today - timedelta(days=delta)
    elif cadence == "monthly":
        return today.replace(day=1)
    elif cadence == "quarterly":
        quarter_start_month = ((today.month - 1) // 3) * 3 + 1
        return today.replace(month=quarter_start_month, day=1)
    else:  # pragma: no cover -- exhaustive over this test's cadence set
        raise ValueError(f"Unknown cadence {cadence!r}")


# ---------------------------------------------------------------------------
# Refactor equivalence -- an oracle, not an enumeration.
# ---------------------------------------------------------------------------


def test_cadence_boundary_matches_legacy_algorithm_over_three_years() -> None:
    start = date(2025, 1, 1)
    end = date(2027, 12, 31)
    day = start
    while day <= end:
        now = datetime(day.year, day.month, day.day, 12, 0, tzinfo=UTC)

        for cadence in ("monthly", "quarterly"):
            assert cadence_boundary(cadence=cadence, anchor_tz="UTC", now=now) == _legacy_boundary(
                day, cadence, None
            )

        for anchor_dow in WEEKDAY_BY_NAME:
            assert cadence_boundary(
                cadence="weekly", anchor_tz="UTC", anchor_dow=anchor_dow, now=now
            ) == _legacy_boundary(day, "weekly", anchor_dow)

        day += timedelta(days=1)


# NOTE: the brief's item 2, "test_calendar_release_resolver_tests_unchanged",
# is deliberately not a function here -- it is an acceptance criterion
# checked out-of-band: `git diff --stat origin/main --
# tests/test_calendar_release_resolver.py` must be empty, and
# `uv run pytest tests/test_calendar_release_resolver.py` must stay green.


# ---------------------------------------------------------------------------
# Daily + DST.
# ---------------------------------------------------------------------------


def test_daily_boundary_is_today_in_anchor_tz() -> None:
    now = datetime(2026, 8, 6, 2, 0, tzinfo=UTC)
    assert cadence_boundary(cadence="daily", anchor_tz=NY, now=now) == date(2026, 8, 5)
    assert cadence_boundary(cadence="daily", anchor_tz="UTC", now=now) == date(2026, 8, 6)


def test_daily_boundary_across_spring_forward() -> None:
    # America/New_York DST start is 2026-03-08 02:00 local.
    before = datetime(2026, 3, 8, 6, 30, tzinfo=UTC)  # 01:30 EST
    after = datetime(2026, 3, 8, 7, 30, tzinfo=UTC)  # 03:30 EDT
    assert cadence_boundary(cadence="daily", anchor_tz=NY, now=before) == date(2026, 3, 8)
    assert cadence_boundary(cadence="daily", anchor_tz=NY, now=after) == date(2026, 3, 8)


def test_daily_boundary_across_fall_back() -> None:
    # America/New_York DST end is 2026-11-01 02:00 local.
    edt = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)  # 01:30 EDT
    est = datetime(2026, 11, 1, 6, 30, tzinfo=UTC)  # 01:30 EST (repeated local hour)
    assert cadence_boundary(cadence="daily", anchor_tz=NY, now=edt) == date(2026, 11, 1)
    assert cadence_boundary(cadence="daily", anchor_tz=NY, now=est) == date(2026, 11, 1)


# ---------------------------------------------------------------------------
# Lag pre-truncation -- the guard against the no-op.
# ---------------------------------------------------------------------------


def test_lag_hours_shifts_daily_boundary_across_midnight() -> None:
    now = datetime(2026, 8, 6, 5, 0, tzinfo=UTC)  # 01:00 Eastern on the 6th
    assert cadence_boundary(cadence="daily", anchor_tz=NY, lag_hours=3, now=now) == date(2026, 8, 5)
    assert cadence_boundary(cadence="daily", anchor_tz=NY, lag_hours=0, now=now) == date(2026, 8, 6)


def test_lag_hours_shifts_monthly_boundary_across_month_end() -> None:
    now = datetime(2026, 9, 1, 1, 0, tzinfo=UTC)
    assert cadence_boundary(cadence="monthly", anchor_tz="UTC", lag_hours=6, now=now) == date(
        2026, 8, 1
    )
    assert cadence_boundary(cadence="monthly", anchor_tz="UTC", lag_hours=0, now=now) == date(
        2026, 9, 1
    )


def test_negative_lag_hours_rejected() -> None:
    now = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="lag_hours"):
        cadence_boundary(cadence="daily", anchor_tz="UTC", lag_hours=-1, now=now)


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


def test_naive_now_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        cadence_boundary(cadence="daily", anchor_tz="UTC", now=datetime(2026, 8, 6, 12, 0))


def test_unknown_cadence_raises_value_error() -> None:
    now = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError) as exc_info:
        cadence_boundary(cadence="fortnightly", anchor_tz="UTC", now=now)
    for known in sorted(KNOWN_CADENCES):
        assert known in str(exc_info.value)


def test_weekly_requires_anchor_dow() -> None:
    now = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="anchor_dow"):
        cadence_boundary(cadence="weekly", anchor_tz="UTC", anchor_dow=None, now=now)


def test_weekly_unknown_anchor_dow_raises() -> None:
    now = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        cadence_boundary(cadence="weekly", anchor_tz="UTC", anchor_dow="Funday", now=now)


def test_anchor_dow_ignored_for_non_weekly_cadences() -> None:
    now = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    for cadence in ("daily", "monthly", "quarterly"):
        with_dow = cadence_boundary(cadence=cadence, anchor_tz="UTC", anchor_dow="Friday", now=now)
        without_dow = cadence_boundary(cadence=cadence, anchor_tz="UTC", anchor_dow=None, now=now)
        assert with_dow == without_dow


def test_unknown_anchor_tz_propagates_zoneinfo_error() -> None:
    now = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    with pytest.raises(ZoneInfoNotFoundError):
        cadence_boundary(cadence="daily", anchor_tz="Mars/Olympus", now=now)


@freeze_time("2026-08-06 02:00:00")
def test_now_none_uses_current_clock_in_anchor_tz() -> None:
    assert cadence_boundary(cadence="daily", anchor_tz=NY) == date(2026, 8, 5)


def test_known_cadences_is_a_superset_of_calendar_resolver_cadences() -> None:
    assert _KNOWN_CADENCES <= KNOWN_CADENCES


def test_known_browser_export_cadences_is_a_subset_of_known_cadences() -> None:
    """Pre-merge review gate finding 5: ``KNOWN_BROWSER_EXPORT_CADENCES``
    (``contracts/loader.py`` -- what a ``browser_export`` contract author
    is allowed to write) must stay a subset of ``KNOWN_CADENCES`` (this
    module -- what ``cadence_boundary`` can actually compute). Same
    structural-subset shape as
    ``test_known_cadences_is_a_superset_of_calendar_resolver_cadences``
    above, for the loader's validated set instead of the resolver's.
    Without this, adding a cadence to ``KNOWN_BROWSER_EXPORT_CADENCES``
    that ``cadence_boundary`` cannot compute would validate a contract
    clean and only fail at discovery time on a live Dagster sensor tick,
    with the test suite green."""
    from moncpipelib.contracts.loader import KNOWN_BROWSER_EXPORT_CADENCES

    assert KNOWN_BROWSER_EXPORT_CADENCES <= KNOWN_CADENCES
