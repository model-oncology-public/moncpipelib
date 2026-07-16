"""Post-write ANALYZE maintenance.

Tracks public mirror issue model-oncology-public/moncpipelib#1.

PostgreSQL's autovacuum autoanalyzes ordinary tables and leaf partitions,
but never a partitioned parent: relkind ``p`` relations have no storage of
their own, so the per-table autoanalyze thresholds can never fire on them
(current behavior through PG18). The parent's aggregate ("inheritance")
statistics are therefore maintained only by explicit ``ANALYZE``; a freshly
loaded partitioned target queried through the parent plans against
``pg_class.reltuples = -1``.

This module owns the opt-in/opt-out post-write ``ANALYZE`` step:

- Runs AFTER the write transaction commits, in its own short transaction,
  so it reflects committed data and never extends the write txn's locks.
- Gated on change: a write that touched no rows performs no ``ANALYZE``.
- SCD2 targets are skipped entirely: ``scd2_finalize`` already issues a
  trailing in-txn ``ANALYZE <target>`` (#312/#319/#361) that recurses to
  the parent, so a second pass here would sample the tree twice.
- On PG >= 18, partitioned parents use ``ANALYZE ONLY``, which refreshes
  the parent's inheritance stats without rewriting per-leaf statistics
  (leaves stay owned by autovacuum). Older servers fall back to a plain
  recursive ``ANALYZE``.
- Failures log a warning but never fail the write (same pattern as
  ``_sync_pii_metadata`` / ``_auto_register_period``): the data is already
  committed, and ``ANALYZE`` requires table ownership or the ``MAINTAIN``
  privilege (PG17+), which some deployments may not grant the write role.
- ``VACUUM`` / ``FREEZE`` deliberately stay out of the write path; they are
  threshold-driven and belong to autovacuum / maintenance jobs.

Security note (HIPAA/PHI context): ``ANALYZE`` samples row values into
``pg_statistic``, exactly as autovacuum's autoanalyze already does for
every ordinary table and leaf partition. This introduces no new data
exposure; access to ``pg_statistic`` / ``pg_stats`` remains governed by
the cluster's existing controls. See SECURITY.md.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from psycopg import sql

if TYPE_CHECKING:
    import psycopg

    from moncpipelib.io_managers.enums import WriteMode
    from moncpipelib.resources.types import LoggingContext

VALID_ANALYZE_AFTER_WRITE: frozenset[str] = frozenset({"never", "partitioned", "always"})

# Writer-stats counters that indicate the write changed the target. When a
# stats dict carries none of these (e.g. the batched non-SCD2 path), the
# gate falls back to the row count.
_CHANGE_STAT_KEYS: tuple[str, ...] = (
    "rows_inserted",
    "rows_deleted",
    "rows_upserted",
    "rows_new",
    "rows_expired",
)

# First server_version_num supporting ``ANALYZE ONLY`` on partitioned tables.
_PG18_VERSION_NUM = 180000


def resolve_analyze_after_write(
    resource_default: str,
    write_config: dict[str, Any],
) -> str:
    """Resolve the effective ``analyze_after_write`` mode, validating it.

    Per-write overrides (asset metadata / ``write()`` parameter) win over the
    resource default. Raises ``ValueError`` on an unrecognized value so a typo
    fails before any write SQL runs rather than silently disabling the hook.
    """
    override = write_config.get("analyze_after_write")
    value = resource_default if override is None else str(override)
    if value not in VALID_ANALYZE_AFTER_WRITE:
        raise ValueError(
            f"Invalid analyze_after_write value {value!r}; "
            f"expected one of {sorted(VALID_ANALYZE_AFTER_WRITE)}"
        )
    return value


def _write_changed(stats: dict[str, Any], row_count: int) -> bool:
    """Return True when the write changed the target (or plausibly did)."""
    counters = [stats[k] for k in _CHANGE_STAT_KEYS if isinstance(stats.get(k), int)]
    if counters:
        return any(v > 0 for v in counters)
    return row_count > 0


def analyze_after_write(
    conn: psycopg.Connection,
    *,
    schema: str,
    bare_table: str,
    mode: str,
    write_mode: WriteMode,
    stats: dict[str, Any],
    row_count: int,
    context: LoggingContext,
) -> str | None:
    """Run the post-commit ``ANALYZE`` step against the write target.

    Must be called after the data transaction has committed; runs in its own
    short transaction on ``conn`` and commits it. Never raises: any failure
    is logged as a warning and the committed write stands.

    Returns the action taken, surfaced in the write stats for observability:

    - ``"parent"``: PG >= 18 ``ANALYZE ONLY`` on a partitioned parent.
    - ``"recursive"``: pre-18 fallback, plain ``ANALYZE`` on a partitioned
      parent (recurses into every leaf).
    - ``"table"``: plain ``ANALYZE`` on an ordinary table (``mode="always"``).
    - ``None``: skipped (mode ``"never"``, SCD2 target, unchanged write,
      ordinary table under mode ``"partitioned"``, or failure).
    """
    if mode == "never":
        return None
    # StrEnum: compare against the value to avoid importing WriteMode here.
    if write_mode == "scd2":
        # scd2_finalize already ANALYZEs the target in-txn (recursing to the
        # parent) -- see #312/#319/#361. A second pass would double-sample.
        context.log.debug("analyze_after_write: SCD2 writer maintains target stats; skipping")
        return None
    if not _write_changed(stats, row_count):
        context.log.debug("analyze_after_write: write changed no rows; skipping")
        return None

    table_name = f"{schema}.{bare_table}"
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT c.relkind FROM pg_class c"
                " JOIN pg_namespace n ON n.oid = c.relnamespace"
                " WHERE n.nspname = %s AND c.relname = %s",
                (schema, bare_table),
            )
            row = cursor.fetchone()
            if row is None:
                context.log.warning(
                    f"analyze_after_write: {table_name} not found in pg_class; skipping"
                )
                conn.rollback()
                return None

            is_partitioned_parent = row[0] == "p"
            if not is_partitioned_parent and mode == "partitioned":
                # Ordinary tables and leaves are autovacuum's job.
                conn.rollback()
                return None

            target = sql.Identifier(schema, bare_table)
            if is_partitioned_parent and conn.info.server_version >= _PG18_VERSION_NUM:
                statement = sql.SQL("ANALYZE ONLY {}").format(target)
                action = "parent"
            else:
                statement = sql.SQL("ANALYZE {}").format(target)
                action = "recursive" if is_partitioned_parent else "table"

            t0 = time.perf_counter()
            cursor.execute(statement)
            conn.commit()
            context.log.info(
                f"analyze_after_write: refreshed planner stats for {table_name} "
                f"({action}) in {time.perf_counter() - t0:.2f}s"
            )
            return action
    except Exception as e:
        # Post-commit maintenance must never fail the committed write.
        with contextlib.suppress(Exception):
            conn.rollback()
        context.log.warning(
            f"analyze_after_write: ANALYZE {table_name} failed "
            f"(write already committed, stats not refreshed): {e}"
        )
        return None
