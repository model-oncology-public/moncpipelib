"""Central configuration for moncpipelib.

This module provides configuration defaults that can be overridden via
environment variables or explicitly in code.

Environment Variables:
    MONCPIPELIB_DEFAULT_DATABASE: Default database name for contract sources/sinks
        that omit an explicit ``database`` field
    MONCPIPELIB_OPENLINEAGE_SCHEMA_URL: Base URL for OpenLineage custom facet schemas
    MONCPIPELIB_OPENLINEAGE_NAMESPACE: Default namespace for OpenLineage events
    MONCPIPELIB_LINEAGE_TABLE: Name of the lineage tracking table
    MONCPIPELIB_LINEAGE_SCHEMA: Schema for the lineage tracking table
    MONCPIPELIB_PERIOD_REGISTRY_TABLE: Name of the period registry table
    MONCPIPELIB_PERIOD_REGISTRY_SCHEMA: Schema for the period registry table
    MONCPIPELIB_VERBOSE_METADATA: When truthy, write extra diagnostic metadata
        on Dagster outputs (e.g. per-batch timing breakdowns).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

DEFAULT_SCHEMA: str = "public"
"""Default PostgreSQL schema when none is specified."""

DEFAULT_DATABASE: str = os.environ.get("MONCPIPELIB_DEFAULT_DATABASE", "analytics")
"""Default PostgreSQL database name when a contract source/sink omits one.

Used by :meth:`DataContract.get_source_tables` /
:meth:`~DataContract.get_sink_tables` when an entry has no explicit
``database`` field.  Override via the ``MONCPIPELIB_DEFAULT_DATABASE``
environment variable.

NOTE: Deployments that relied on the historical built-in default must now
set ``MONCPIPELIB_DEFAULT_DATABASE`` explicitly to preserve prior behavior;
the fallback here is a generic placeholder, not a real database name.
"""

CONTRACT_FILE_PATTERN: str = "*.contract.yaml"
"""Glob pattern for discovering data contract files."""

SOURCE_FILE_PATTERN: str = "*.source.yaml"
"""Glob pattern for discovering data source files."""

INGEST_FILE_PATTERN: str = "*.ingest.yaml"
"""Glob pattern for discovering ingest contract files."""

AUTO_BATCH_THRESHOLD: int = 50_000
"""Row count threshold for automatic batch/COPY mode selection in writers."""


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable.

    Truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Anything else (including unset) returns *default*.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SKIP_LINEAGE_WRITES_ENV: str = "MONCPIPELIB_SKIP_LINEAGE_WRITES"
"""Environment variable enabling test-mode lineage isolation (#420)."""


def skip_lineage_writes() -> bool:
    """True when lineage / period-registry writes are disabled for this process.

    Integration-test and ephemeral harnesses redirect the *sink* table to an
    isolated schema, but historically the write path's lineage side-effects
    (``lineage.data_lineage``, ``lineage.contract_validation_runs``,
    ``lineage.column_metadata`` PII sync, ``lineage.period_registry``
    stamping, OpenLineage emission) still targeted the shared ``lineage``
    schema.  A CI run stamping ``silver_materialized`` on a real
    ``period_registry`` row makes the environment's sensor silently skip the
    first real materialization (#420).

    Set ``MONCPIPELIB_SKIP_LINEAGE_WRITES=1`` (or ``true`` / ``yes`` / ``on``)
    in the test harness to make every lineage and period-registry write a
    logged no-op.  The data write itself keeps byte-for-byte production
    shape (#424, #426): the managed ``_lineage_id`` / ``_lineage_key``
    columns are attached with real generated values, so NOT NULL sink
    constraints hold.  The id references no ``data_lineage`` row -- fine for
    ephemeral test sinks (dropped after the run; harnesses clone target
    tables with FKs stripped), and a skip-mode write against a REAL table
    that enforces the ``data_lineage`` FK fails loudly on that FK, which
    blocks test-isolated writes from landing in production tables.

    This is a test-isolation switch, NOT an operational toggle: production
    deployments must never set it, since it disables the HIPAA/SOC2 lineage
    audit trail.  Every skip is logged at WARNING level for exactly that
    reason.

    Read dynamically (not cached at import) so harnesses that configure the
    environment after import are honored.
    """
    return _parse_bool_env(SKIP_LINEAGE_WRITES_ENV)


VERBOSE_METADATA: bool = _parse_bool_env("MONCPIPELIB_VERBOSE_METADATA")
"""Emit extra diagnostic metadata on Dagster outputs.

When ``True``, write paths attach additional metadata fields useful for
performance investigations (e.g. per-batch ``t_iter_seconds`` /
``t_prep_seconds`` / ``t_copy_seconds`` breakdowns from
``PostgresResource._write_batched``) that are noise during normal
operation.

Override via ``MONCPIPELIB_VERBOSE_METADATA`` environment variable
(``1`` / ``true`` / ``yes`` / ``on``), or at runtime via
:func:`set_verbose_metadata` / :func:`verbose_metadata`.

Read lazily inside writer functions (local imports re-read on every
call) so runtime mutation takes effect on the next write.  Module-
level captures via ``from moncpipelib.config import VERBOSE_METADATA``
freeze the value at import time and will NOT see runtime toggles --
prefer :func:`set_verbose_metadata` or read ``moncpipelib.config.
VERBOSE_METADATA`` lazily.
"""


def set_verbose_metadata(enabled: bool = True) -> None:
    """Toggle verbose diagnostic metadata at runtime.

    Equivalent to setting ``MONCPIPELIB_VERBOSE_METADATA=true`` before
    process start, but callable from pipeline code.  Affects every
    subsequent write in this process until called again.

    Use the :func:`verbose_metadata` context manager instead when the
    scope is bounded to a single block (e.g. a single asset).

    Example::

        from moncpipelib import set_verbose_metadata

        set_verbose_metadata(True)
        # ... runs writes with timing breakdowns attached ...
        set_verbose_metadata(False)

    Args:
        enabled: ``True`` to turn verbose metadata on, ``False`` to
            turn it off.  Defaults to ``True`` for the common
            "flip on" case.
    """
    global VERBOSE_METADATA
    VERBOSE_METADATA = enabled


@contextmanager
def verbose_metadata(enabled: bool = True) -> Iterator[None]:
    """Context manager that scopes verbose metadata to a block.

    Restores the previous value on exit (including when an exception
    propagates out of the block), so it is safe to nest and to use
    around code that mutates the flag itself.

    Example::

        from moncpipelib import verbose_metadata

        with verbose_metadata():
            database.write(big_df, target="...", context=context)
        # flag is back to whatever it was before the block

    Args:
        enabled: Value to set inside the block.  Defaults to ``True``.
    """
    global VERBOSE_METADATA
    previous = VERBOSE_METADATA
    VERBOSE_METADATA = enabled
    try:
        yield
    finally:
        VERBOSE_METADATA = previous


REGISTRY_STATUS_REGISTERED: str = "registered"
"""Default status value for newly registered periods in the period registry."""


# ------------------------------------------------------------------
# Metadata keys (coordination contract between writers/sensors/jobs)
# ------------------------------------------------------------------


class MetadataKeys:
    """JSONB metadata keys used in the period registry.

    These form the coordination contract between writers, sensors,
    and reconciliation jobs.  Centralised here so a typo cannot
    silently break cross-code-location communication.
    """

    SILVER_MATERIALIZED_AT: str = "silver_materialized_at"
    SILVER_MATERIALIZED_BY: str = "silver_materialized_by"
    SILVER_RUN_ID: str = "silver_run_id"
    RECONCILED_AT: str = "reconciled_at"
    RECONCILED_BY: str = "reconciled_by"
    ROWS_TIMELINE_UPDATED: str = "rows_timeline_updated"
    ROWS_COLLAPSED: str = "rows_collapsed"
    ROWS_RENUMBERED: str = "rows_renumbered"
    REGISTERED_AT: str = "registered_at"


# ------------------------------------------------------------------
# Utility: schema.table parsing
# ------------------------------------------------------------------


def parse_schema_table(table_name: str, *, strict: bool = False) -> tuple[str, str]:
    """Parse a table name into ``(schema, table)``.

    Args:
        table_name: Either ``"schema.table"`` or just ``"table"``.
        strict: If *True*, raise :class:`ValueError` when the schema
            part is missing.  If *False* (default), fall back to
            :data:`DEFAULT_SCHEMA`.

    Returns:
        Tuple of *(schema, bare_table)*.

    Raises:
        ValueError: When *strict* is True and no dot-separator is
            found, or when the input is otherwise malformed.
    """
    parts = table_name.split(".")
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    if len(parts) == 1 and parts[0]:
        if strict:
            raise ValueError(
                f"target must be 'schema.table', got '{table_name}'. Example: 'silver.dim_provider'"
            )
        return DEFAULT_SCHEMA, parts[0]
    raise ValueError(
        f"Invalid table name format: '{table_name}'. Expected 'schema.table' or 'table'."
    )


@dataclass(frozen=True)
class OpenLineageDefaults:
    """Default configuration for OpenLineage integration."""

    schema_url_base: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_OPENLINEAGE_SCHEMA_URL",
            "https://github.com/model-oncology-public/moncpipelib/tree/main/schemas/openlineage/",
        )
    )
    """Base URL for custom facet JSON schemas.

    Override via MONCPIPELIB_OPENLINEAGE_SCHEMA_URL environment variable.
    Points to the schemas directory in the moncpipelib repository.
    """

    namespace: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_OPENLINEAGE_NAMESPACE",
            "moncpipelib",
        )
    )
    """Default namespace for OpenLineage jobs and datasets.

    Override via MONCPIPELIB_OPENLINEAGE_NAMESPACE environment variable.
    """


@dataclass(frozen=True)
class LineageDefaults:
    """Default configuration for lineage tracking."""

    # Column naming constants (structural, not environment-overridable)
    COLUMN_PREFIX: str = "_lineage_"
    ID_COLUMN: str = "_lineage_id"
    KEY_COLUMN: str = "_lineage_key"

    table_name: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_LINEAGE_TABLE",
            "data_lineage",
        )
    )
    """Name of the lineage tracking table.

    Override via MONCPIPELIB_LINEAGE_TABLE environment variable.
    """

    schema_name: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_LINEAGE_SCHEMA",
            "lineage",
        )
    )
    """Schema containing the lineage tracking table.

    Override via MONCPIPELIB_LINEAGE_SCHEMA environment variable.
    """


@dataclass(frozen=True)
class PeriodRegistryDefaults:
    """Default table/schema names for the period registry."""

    table_name: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_PERIOD_REGISTRY_TABLE", "period_registry"
        )
    )
    """Name of the period registry table.

    Override via MONCPIPELIB_PERIOD_REGISTRY_TABLE environment variable.
    """

    schema_name: str = field(
        default_factory=lambda: os.environ.get("MONCPIPELIB_PERIOD_REGISTRY_SCHEMA", "lineage")
    )
    """Schema containing the period registry table.

    Override via MONCPIPELIB_PERIOD_REGISTRY_SCHEMA environment variable.
    """


@dataclass(frozen=True)
class Scd2ReconciliationsDefaults:
    """Default table/schema names for the SCD2 reconciliations audit table.

    Migration 019 (#308) Phase 6.
    """

    table_name: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_SCD2_RECONCILIATIONS_TABLE", "scd2_reconciliations"
        )
    )
    """Name of the SCD2 reconciliations audit table.

    Override via MONCPIPELIB_SCD2_RECONCILIATIONS_TABLE environment variable.
    """

    schema_name: str = field(
        default_factory=lambda: os.environ.get("MONCPIPELIB_SCD2_RECONCILIATIONS_SCHEMA", "lineage")
    )
    """Schema containing the SCD2 reconciliations audit table.

    Override via MONCPIPELIB_SCD2_RECONCILIATIONS_SCHEMA environment variable.
    """


@dataclass(frozen=True)
class ContractValidationRunsDefaults:
    """Default table/schema names for the contract validation runs table.

    Migration 019 (#308) Phase 5.
    """

    table_name: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_CONTRACT_VALIDATION_RUNS_TABLE", "contract_validation_runs"
        )
    )
    """Name of the contract validation runs table.

    Override via MONCPIPELIB_CONTRACT_VALIDATION_RUNS_TABLE environment variable.
    """

    schema_name: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_CONTRACT_VALIDATION_RUNS_SCHEMA", "lineage"
        )
    )
    """Schema containing the contract validation runs table.

    Override via MONCPIPELIB_CONTRACT_VALIDATION_RUNS_SCHEMA environment variable.
    """


@dataclass(frozen=True)
class PipelineRegistryDefaults:
    """Default table/schema names for the pipeline registry.

    Migration 019 (#308) Phase 2.
    """

    table_name: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_PIPELINE_REGISTRY_TABLE", "pipeline_registry"
        )
    )
    """Name of the pipeline registry table.

    Override via MONCPIPELIB_PIPELINE_REGISTRY_TABLE environment variable.
    """

    schema_name: str = field(
        default_factory=lambda: os.environ.get("MONCPIPELIB_PIPELINE_REGISTRY_SCHEMA", "lineage")
    )
    """Schema containing the pipeline registry table.

    Override via MONCPIPELIB_PIPELINE_REGISTRY_SCHEMA environment variable.
    """


@dataclass(frozen=True)
class PoolDefaults:
    """Default SQLAlchemy connection pool settings.

    These control how ``PostgresResource.get_engine()`` manages database
    connections.  Azure PostgreSQL Flexible Server enforces a per-server
    connection limit (default 100 for most SKUs), so tuning these values
    is important in multi-pipeline deployments.
    """

    pool_size: int = field(
        default_factory=lambda: int(os.environ.get("MONCPIPELIB_POOL_SIZE", "5"))
    )
    """Number of persistent connections kept open in the pool.

    Override via MONCPIPELIB_POOL_SIZE environment variable.
    """

    max_overflow: int = field(
        default_factory=lambda: int(os.environ.get("MONCPIPELIB_POOL_MAX_OVERFLOW", "10"))
    )
    """Maximum additional connections allowed above ``pool_size``.

    Overflow connections are closed when returned to the pool.
    Override via MONCPIPELIB_POOL_MAX_OVERFLOW environment variable.
    """

    pool_timeout: int = field(
        default_factory=lambda: int(os.environ.get("MONCPIPELIB_POOL_TIMEOUT", "30"))
    )
    """Seconds to wait for a connection from the pool before raising an error.

    Override via MONCPIPELIB_POOL_TIMEOUT environment variable.
    """

    pool_recycle: int = field(
        default_factory=lambda: int(os.environ.get("MONCPIPELIB_POOL_RECYCLE", "1800"))
    )
    """Seconds after which a connection is recycled (closed and replaced).

    Prevents stale connections on Azure PostgreSQL where idle connections
    may be terminated by the server or load balancer after ~5 minutes.
    Default 1800 (30 minutes).
    Override via MONCPIPELIB_POOL_RECYCLE environment variable.
    """


@dataclass(frozen=True)
class MoncpipelibConfig:
    """Central configuration container for moncpipelib.

    Configuration values can be overridden in three ways (in order of precedence):
    1. Explicitly when creating resources/IO managers
    2. Environment variables
    3. Defaults defined here

    Example:
        ```python
        from moncpipelib.config import config

        # Access defaults
        print(config.openlineage.namespace)  # "moncpipelib" or env var

        # Or override via environment before import:
        # export MONCPIPELIB_OPENLINEAGE_NAMESPACE=my-pipeline
        ```
    """

    openlineage: OpenLineageDefaults = field(default_factory=OpenLineageDefaults)
    lineage: LineageDefaults = field(default_factory=LineageDefaults)
    period_registry: PeriodRegistryDefaults = field(default_factory=PeriodRegistryDefaults)
    pipeline_registry: PipelineRegistryDefaults = field(default_factory=PipelineRegistryDefaults)
    contract_validation_runs: ContractValidationRunsDefaults = field(
        default_factory=ContractValidationRunsDefaults
    )
    scd2_reconciliations: Scd2ReconciliationsDefaults = field(
        default_factory=Scd2ReconciliationsDefaults
    )
    pool: PoolDefaults = field(default_factory=PoolDefaults)


# Singleton instance for easy access
config = MoncpipelibConfig()

# ------------------------------------------------------------------
# Shared constants used by both io_managers and resources modules.
# Defined here (pure-data module) to avoid circular imports between
# io_managers/__init__.py and resources/postgres.py.
# ------------------------------------------------------------------

VALID_LAYERS: frozenset[str] = frozenset({"bronze", "silver", "gold"})
"""Valid data layer names for metadata column prefixes."""

PolarsEngineType = Literal["auto", "in-memory", "streaming", "gpu"]

POLARS_ENGINE: PolarsEngineType = os.environ.get(  # type: ignore[assignment]
    "MONCPIPELIB_POLARS_ENGINE", "streaming"
)
"""Default Polars engine for .collect() and collect_all() calls.

The streaming engine (Polars v1.31+) uses morsel-driven parallelism that
caps intermediate memory usage during joins, filters, and group-bys.
Falls back to in-memory silently for unsupported operations.

Override via MONCPIPELIB_POLARS_ENGINE environment variable.
"""


@dataclass(frozen=True)
class SCD2Config:
    """Column names and sentinel values for SCD2 bookkeeping.

    All writer and resource functions accept this as a single parameter
    instead of threading 5-6 individual column name arguments.

    ``sequence_col`` is populated per-business-key (1, 2, 3, ...) on each
    new SCD2 version.  Set to ``None`` to opt out.  If the target table
    does not contain a column with this name, the writer silently skips it.

    ``end_of_time`` is used as the ``effective_to`` value for currently
    active records instead of NULL.  This enables ``BETWEEN`` queries on
    the temporal range without special-casing NULLs.
    """

    effective_from_col: str = "effective_from"
    effective_to_col: str = "effective_to"
    is_current_col: str = "is_current"
    hash_col: str = "row_hash"
    sequence_col: str | None = "seq_id"
    end_of_time: str = "9999-12-31"

    @property
    def managed_columns(self) -> frozenset[str]:
        """Column names managed by the SCD2 writer (not user-supplied)."""
        cols: set[str] = {
            self.effective_from_col,
            self.effective_to_col,
            self.is_current_col,
            self.hash_col,
        }
        if self.sequence_col is not None:
            cols.add(self.sequence_col)
        return frozenset(cols)


# Backwards-compatible dict alias — prefer SCD2Config for new code.
_scd2 = SCD2Config()
SCD2_DEFAULTS: dict[str, str | None] = {
    "effective_from_col": _scd2.effective_from_col,
    "effective_to_col": _scd2.effective_to_col,
    "is_current_col": _scd2.is_current_col,
    "hash_col": _scd2.hash_col,
    "sequence_col": _scd2.sequence_col,
    "end_of_time": _scd2.end_of_time,
}
