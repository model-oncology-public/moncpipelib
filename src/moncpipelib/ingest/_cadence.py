"""Shared cadence-boundary arithmetic for calendar-style release resolvers.

Extracted from ``resolvers/calendar.py`` (#463) so the timezone/DST-sensitive
"round an instant down to a cadence boundary" arithmetic exists once and is
extended (``daily``, ``lag_hours``) without duplicating it in the
forthcoming ``browser_export`` pattern.  ``CalendarReleaseResolver.
current_release`` delegates to :func:`cadence_boundary`; see that module for
the delegation.

Audit / compliance:

- Performs no I/O.  Deterministic given an explicit ``now``;
  ``datetime.now(tz=...)`` is the only nondeterminism, matching the
  resolver this was extracted from.
- Every ``date`` is a valid cadence boundary, so there is no return value
  that can encode failure -- see :func:`cadence_boundary`'s docstring for
  the failure-value analysis. Invalid input always raises.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

KNOWN_CADENCES: frozenset[str] = frozenset({"daily", "weekly", "monthly", "quarterly"})
"""Every cadence :func:`cadence_boundary` can compute.

Deliberately NOT the same set as ``resolvers/calendar.py::_KNOWN_CADENCES``,
which stays ``{weekly, monthly, quarterly}``: per #463 D4, ``daily`` is owned
by ``browser_export`` and widening the resolver's validated surface would
expose it to ``api_resolver`` contracts no consumer wants. Nor is it the same
set as ``KNOWN_BROWSER_EXPORT_CADENCES``, which excludes ``weekly`` for want
of an ``anchor_dow`` field. Three sets, three different jobs -- do not
"fix" the divergence.
"""

WEEKDAY_BY_NAME: dict[str, int] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}
"""Python's ``datetime.weekday()`` numbering. Moved here from
``resolvers/calendar.py`` so the resolver's ``validate_config`` and this
helper share one map."""


def cadence_boundary(
    *,
    cadence: str,
    anchor_tz: str,
    anchor_dow: str | None = None,
    lag_hours: int = 0,
    now: datetime | None = None,
) -> date:
    """Round ``now`` down to the start of its enclosing cadence period.

    Args:
        cadence: One of :data:`KNOWN_CADENCES`.
        anchor_tz: IANA timezone name (a ``str``, not a
            :class:`~zoneinfo.ZoneInfo`) in which the boundary is computed.
            Both call sites read this out of YAML as a string; taking a
            ``str`` here keeps ``ZoneInfo`` construction, and its failure
            mode, in exactly one place.
        anchor_dow: Required iff ``cadence == "weekly"``; must be a key of
            :data:`WEEKDAY_BY_NAME`. Ignored (not rejected) for every other
            cadence -- a stray ``anchor_dow`` alongside e.g. ``monthly`` is
            accepted for back-compat with the pre-extraction resolver, which
            only read it inside the weekly branch. ``validate_config``
            already forbids the stray key at load.
        lag_hours: Hours to subtract from ``now`` *before* truncating to a
            date. Must be ``>= 0``. Applied to the aware datetime, not the
            truncated date -- subtracting a ``timedelta`` from a ``date`` is
            a no-op (``timedelta(hours=3).days == 0``), which would make a
            configured lag silently do nothing.
        now: An aware datetime, injected for testing. ``None`` uses
            ``datetime.now(tz=ZoneInfo(anchor_tz))``. A naive ``now`` is
            rejected rather than silently interpreted in an unstated zone.

    Returns:
        The cadence boundary as a :class:`~datetime.date`.

    Raises:
        ValueError: ``lag_hours < 0``; ``now`` is naive; ``cadence`` is not
            in :data:`KNOWN_CADENCES`; ``cadence == "weekly"`` and
            ``anchor_dow`` is missing or not a key of :data:`WEEKDAY_BY_NAME`.
        ZoneInfoNotFoundError: ``anchor_tz`` is not a known IANA timezone.
            Propagates uncaught from ``ZoneInfo(anchor_tz)`` -- not caught
            and re-wrapped, so callers see the same exception type the
            pre-extraction resolver raised directly.

    Every ``date`` is a valid cadence boundary -- there is no sentinel this
    function could return to signal failure (e.g. a ``date.min`` sentinel
    would flow straight into a partition key and land bytes under
    ``0001-01-01`` with an ``action="uploaded"`` manifest). Failure is
    therefore always an exception, never a return value.
    """
    if lag_hours < 0:
        raise ValueError(f"cadence_boundary: 'lag_hours' must be >= 0; got {lag_hours}")

    tz = ZoneInfo(anchor_tz)

    if now is None:
        tz_now = datetime.now(tz=tz)
    else:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "cadence_boundary: 'now' must be timezone-aware; a naive "
                "datetime would be silently interpreted in an unstated zone"
            )
        tz_now = now.astimezone(tz)

    anchored = tz_now - timedelta(hours=lag_hours)
    today = anchored.date()

    if cadence == "daily":
        return today
    if cadence == "weekly":
        if anchor_dow is None:
            raise ValueError("cadence_boundary: 'anchor_dow' is required when cadence is 'weekly'")
        if anchor_dow not in WEEKDAY_BY_NAME:
            raise ValueError(
                f"cadence_boundary: unknown anchor_dow {anchor_dow!r}; "
                f"known: {sorted(WEEKDAY_BY_NAME)}"
            )
        target = WEEKDAY_BY_NAME[anchor_dow]
        delta = (today.weekday() - target) % 7
        return today - timedelta(days=delta)
    if cadence == "monthly":
        return today.replace(day=1)
    if cadence == "quarterly":
        quarter_start_month = ((today.month - 1) // 3) * 3 + 1
        return today.replace(month=quarter_start_month, day=1)

    raise ValueError(
        f"cadence_boundary: unknown cadence {cadence!r}; known: {sorted(KNOWN_CADENCES)}"
    )
