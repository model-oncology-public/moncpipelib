"""Calendar-cadence release resolver (synthetic, forward-only).

Synthesizes partition keys from a calendar cadence (weekly / monthly /
quarterly) without making any upstream API call.  Pairs with
:class:`~moncpipelib.ingest.patterns.api_resolver.ApiResolverPattern` to
support sources whose download URL is stable across snapshots and whose
partition cadence is consumer-defined (e.g. FDA NDC, FDA Drug Labels SPL
bulk).

Forward-only by design: past calendar boundaries cannot be backfilled
because upstream URLs in this pattern serve only the *current* snapshot
-- synthesizing a 2024-01-07 partition today would point it at today's
bytes, producing a misleading audit trail.  See #218 Non-goals.

Audit / compliance:

- Performs no I/O at any lifecycle phase (``validate_config``,
  ``current_release``, ``resolve_url``).  Deterministic given a frozen
  clock; ``datetime.now(tz=anchor_tz)`` is the only nondeterminism.
- Accepts no api_key (Protocol uniformity); contracts using this
  resolver omit the ``credential`` block per #218.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from moncpipelib.ingest.resolvers import ResolvedDownload

if TYPE_CHECKING:
    from moncpipelib.ingest.types import IngestContext


_KNOWN_CADENCES: frozenset[str] = frozenset({"weekly", "monthly", "quarterly"})

# Python's datetime.weekday(): Monday=0 ... Sunday=6.
_WEEKDAY_BY_NAME: dict[str, int] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


class CalendarReleaseResolver:
    """Synthesize a partition key from a calendar cadence.

    Stateless singleton -- ``__init__`` is parameterless.  Implements
    :class:`~moncpipelib.ingest.resolvers.ReleaseResolver`.

    Config (under ``ingest.api_resolver.resolver_config``):

    - ``start_date``: ISO date string (``YYYY-MM-DD``) or ``date``.
      Earliest valid partition; informational for now (forward-only
      means past partitions are not synthesized regardless).
    - ``cadence``: one of ``"weekly"``, ``"monthly"``, ``"quarterly"``.
    - ``anchor_dow``: required when ``cadence == "weekly"``; one of
      ``"Monday"``..``"Sunday"``.  Forbidden for monthly / quarterly
      (silent acceptance would mask operator misconfiguration).
    - ``anchor_tz``: IANA timezone name (default ``"UTC"``).  Validated
      via :mod:`zoneinfo`.  Sets the timezone in which cadence
      boundaries are computed.
    - ``url``: download URL.  Returned verbatim by :meth:`resolve_url`;
      consumers fetch this via the redacting client per the Phase 2
      audit posture.

    Output of :meth:`current_release`:

    - ``partition_key``: ISO date string of the cadence boundary
      (``YYYY-MM-DD`` for all cadences, per #218 Q3 -- uniform format
      simplifies glob-matching and ``effective_from_field`` hydration).
    - ``snapshot_date``: same value as ``partition_key``.
    - ``url``: the configured URL (so dispatchers / sensors can pick it
      up via ``key_from`` if they want).
    """

    name: ClassVar[str] = "calendar"

    discovery_requires_auth: ClassVar[bool] = False
    """Calendar partitions are synthesized from config alone; no upstream
    API call participates in discovery.  Declared explicitly for
    Protocol uniformity even though the conventional contract has no
    ``credential`` block (which would make the flag a no-op).
    """

    KNOWN_CALENDAR_CONFIG_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"start_date", "cadence", "anchor_dow", "anchor_tz", "url"}
    )

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate ``ingest.api_resolver.resolver_config`` for the
        calendar resolver.

        Per ADR-2: deterministic; no network; no filesystem I/O;
        rejects unknown keys.
        """
        errors: list[str] = []

        # start_date: required; accept ISO string or date (per #218 Q1).
        if "start_date" not in config:
            errors.append("start_date: required")
        else:
            value = config["start_date"]
            if isinstance(value, date) and not isinstance(value, datetime):
                pass  # already a date
            elif isinstance(value, str):
                try:
                    date.fromisoformat(value)
                except ValueError:
                    errors.append("start_date: must be an ISO date string (YYYY-MM-DD)")
            else:
                errors.append("start_date: must be an ISO date string or date")

        # cadence: required; bounded.
        cadence: str | None = None
        if "cadence" not in config:
            errors.append("cadence: required")
        else:
            value = config["cadence"]
            if not isinstance(value, str):
                errors.append("cadence: must be a string")
            elif value not in _KNOWN_CADENCES:
                known = sorted(_KNOWN_CADENCES)
                errors.append(f"cadence: must be one of {known}")
            else:
                cadence = value

        # anchor_dow: required iff cadence == "weekly"; forbidden otherwise
        # (per #218 Q2 -- error rather than silent ignore).
        anchor_dow_present = "anchor_dow" in config
        if cadence == "weekly":
            if not anchor_dow_present:
                errors.append("anchor_dow: required when cadence is 'weekly'")
            else:
                value = config["anchor_dow"]
                if not isinstance(value, str):
                    errors.append("anchor_dow: must be a string")
                elif value not in _WEEKDAY_BY_NAME:
                    known = sorted(_WEEKDAY_BY_NAME)
                    errors.append(f"anchor_dow: must be one of {known}")
        elif cadence in {"monthly", "quarterly"} and anchor_dow_present:
            errors.append(
                f"anchor_dow: forbidden when cadence is {cadence!r} "
                "(only valid with cadence='weekly')"
            )

        # anchor_tz: optional; default "UTC"; validated via zoneinfo.
        if "anchor_tz" in config:
            value = config["anchor_tz"]
            if not isinstance(value, str):
                errors.append("anchor_tz: must be a string")
            else:
                try:
                    ZoneInfo(value)
                except ZoneInfoNotFoundError:
                    errors.append(
                        f"anchor_tz: unknown timezone {value!r} (must be an IANA timezone name)"
                    )

        # url: required; non-empty string.
        if "url" not in config:
            errors.append("url: required")
        else:
            value = config["url"]
            if not isinstance(value, str) or not value:
                errors.append("url: must be a non-empty string")

        # Reject unknowns.
        for key in config:
            if key not in self.KNOWN_CALENDAR_CONFIG_FIELDS:
                errors.append(f"{key}: unknown field")

        return errors

    def current_release(
        self,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> dict[str, Any]:
        """Return the partition key for the current calendar boundary.

        ``api_key`` is ignored -- contracts using this resolver omit the
        ``credential`` block per #218.  Accepted for Protocol uniformity.
        """
        del api_key, ctx
        anchor_tz = ZoneInfo(str(config.get("anchor_tz", "UTC")))
        now = datetime.now(tz=anchor_tz).date()
        cadence = str(config["cadence"])

        if cadence == "weekly":
            target = _WEEKDAY_BY_NAME[str(config["anchor_dow"])]
            delta = (now.weekday() - target) % 7
            boundary = now - timedelta(days=delta)
        elif cadence == "monthly":
            boundary = now.replace(day=1)
        elif cadence == "quarterly":
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            boundary = now.replace(month=quarter_start_month, day=1)
        else:  # pragma: no cover -- guarded by validate_config
            raise ValueError(f"Unknown cadence {cadence!r}")

        partition_key = boundary.isoformat()
        return {
            "partition_key": partition_key,
            "snapshot_date": partition_key,
            "url": str(config["url"]),
        }

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> ResolvedDownload:
        """Return the configured URL verbatim with no filename hint.

        ``api_key`` and ``partition_key`` are accepted for Protocol
        uniformity but ignored -- the URL is static, and the partition
        identity is computed at ``current_release`` time.

        Per #270, ``filename`` is :data:`None`: the calendar resolver
        has no upstream knowledge of a semantic filename. The
        non-archive payload chain falls through to the URL basename
        for naming.
        """
        del api_key, partition_key, ctx
        return ResolvedDownload(url=str(config["url"]), filename=None)

    def historical_release(
        self,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> list[dict[str, Any]]:
        """Opt out of historical discovery -- return ``[]`` always.

        Per #218 Non-goals: the calendar resolver is forward-only.
        Past calendar boundaries cannot be backfilled because upstream
        URLs in this pattern serve only the current snapshot.
        Synthesizing past keys would produce a misleading audit trail.
        :class:`ApiResolverPattern` falls back to :meth:`current_release`
        when this returns ``[]``.
        """
        del api_key, config, ctx
        return []
