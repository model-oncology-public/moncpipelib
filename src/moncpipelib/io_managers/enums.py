"""Enums and constants for PostgresIOManager write modes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from moncpipelib.config import SCD2_DEFAULTS

# Re-export for backwards compatibility
_SCD2_DEFAULTS = SCD2_DEFAULTS


@dataclass(frozen=True)
class ResolvedTarget:
    """Result of resolving an asset's target table, schema, and layer.

    Returned by ``PostgresIOManager._resolve_target()`` and used throughout
    the write/read paths to avoid re-deriving these values.
    """

    table_name: str
    """Fully-qualified table name (``schema.table``)."""

    schema: str
    """Resolved PostgreSQL schema name."""

    bare_table: str
    """Physical table name without schema (may include ``table_prefix``). Used for SQL."""

    layer: str | None
    """Derived layer (``bronze``/``silver``/``gold``), ``layer_override``, or ``None``."""

    canonical_table: str
    """Table name after suffix stripping but before prefix addition.

    Used for contract sink matching. Equals ``bare_table`` when no prefix is set.
    """


class WriteMode(StrEnum):
    """Write strategy for PostgresIOManager.

    Attributes:
        FULL_REFRESH: DELETE all rows then INSERT (default). Idempotent.
            When a Dagster partition context is active and partition_column
            is configured, automatically scopes DELETE to active partitions.
        UPSERT: INSERT ON CONFLICT UPDATE. Requires primary_key. Idempotent.
        APPEND: INSERT only, no deletion. Not idempotent without deduplication.
        SCD2: Slowly Changing Dimension Type 2. Expires old versions and inserts
            new versions atomically using a Postgres CTE. Requires business_key.
            When partition_column is configured, change detection is scoped to
            the active partition.
    """

    FULL_REFRESH = "full_refresh"
    UPSERT = "upsert"
    APPEND = "append"
    SCD2 = "scd2"


class FullRefreshMethod(StrEnum):
    """Method for clearing table in full_refresh mode.

    Attributes:
        AUTO: Choose based on DataFrame size (default). Uses TRUNCATE for large
            DataFrames (>= threshold), DELETE for smaller ones.
        DELETE: Always use DELETE. Safer locking (ROW EXCLUSIVE), allows
            concurrent reads, but slower for large tables.
        TRUNCATE: Always use TRUNCATE. Faster (O(1) vs O(n)), but acquires
            ACCESS EXCLUSIVE lock blocking all reads during the transaction.
    """

    AUTO = "auto"
    DELETE = "delete"
    TRUNCATE = "truncate"


class BulkInsertMethod(StrEnum):
    """Method for bulk INSERT operations.

    Attributes:
        AUTO: Choose based on DataFrame size (default). Uses COPY for large
            DataFrames (>= threshold), the executemany-based path for smaller
            ones.
        EXECUTE_VALUES: Always use the executemany-based path. Compatible with
            all write modes including upsert. Migration 014 routed this through
            the driver seam: under ``MONC_PG_DRIVER=psycopg2`` it dispatches to
            ``psycopg2.extras.execute_values``; under psycopg3 it dispatches to
            ``cursor.executemany`` (rewritten on the extended-query protocol
            with parameter pipelining and now competitive with execute_values).
            The string value ``"execute_values"`` is preserved for back-compat;
            it names the *intent*, not the underlying psycopg2 function.
        COPY: Always use PostgreSQL COPY protocol. Faster (4-5x), but only
            compatible with append and full_refresh modes (no ON CONFLICT).
    """

    AUTO = "auto"
    EXECUTE_VALUES = "execute_values"
    COPY = "copy"
