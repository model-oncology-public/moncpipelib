"""PostgreSQL database resource for Dagster pipelines."""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl
import psycopg
import sqlalchemy as sa
from dagster import ConfigurableResource
from pydantic import PrivateAttr

# These imports are safe (no circular dependency) -- they are pure-data modules
# with no transitive import of moncpipelib.resources.
from moncpipelib.config import (
    CONTRACT_FILE_PATTERN,
    LineageDefaults,
    PoolDefaults,
    SCD2Config,
    parse_schema_table,
)
from moncpipelib.config import (
    VALID_LAYERS as _VALID_LAYERS,
)

# Re-export the Polars schema-override builder and the psycopg loader-restore
# helper from their dedicated module so existing imports of
# ``moncpipelib.resources.postgres.PostgresPolarsSchema`` / ``restore_default_handlers``
# continue to resolve. The redundant ``as`` aliasing signals an intentional
# re-export to mypy strict-mode and ruff F401. New code may import from
# ``moncpipelib.resources._schema`` directly.
from moncpipelib.resources._app_name import (
    bind_run_id as bind_run_id,
)
from moncpipelib.resources._app_name import (
    resolve_application_name,
)
from moncpipelib.resources._schema import (
    PostgresPolarsSchema as PostgresPolarsSchema,
)
from moncpipelib.resources._schema import (
    restore_default_handlers as restore_default_handlers,
)
from moncpipelib.resources.types import (
    SENTINEL,
    WriteContext,
    WriteResult,
    _Sentinel,
)

_WORK_MEM_LITERAL_RE = re.compile(r"\d+\s*(kB|MB|GB)")
"""Postgres ``work_mem`` literal: integer + unit (``kB``, ``MB``, or ``GB``).

Format-level validation only.  Postgres still enforces range (e.g. minimum
64 kB) and any out-of-range literal will be rejected server-side at
``set_config`` time.  Whitespace stripping is the caller's responsibility --
:meth:`PostgresResource._resolve_work_mem` strips before matching so env-var
values do not need to be pre-trimmed.
"""

_STATEMENT_TIMEOUT_LITERAL_RE = re.compile(r"\d+\s*(us|ms|s|min|h|d)?")
"""Postgres ``statement_timeout`` literal: integer with an optional time unit.

A bare integer is interpreted by Postgres as milliseconds; a unit suffix
(``us``, ``ms``, ``s``, ``min``, ``h``, ``d``) is accepted as written.  Format-
level validation only -- Postgres enforces range server-side at ``set_config``
time.  Whitespace stripping is the caller's responsibility;
:meth:`PostgresResource._resolve_statement_timeout` strips before matching.
"""

_WORK_MEM_DISABLE_TOKENS = frozenset({"none", "off", "disabled"})
"""Case-insensitive sentinels that resolve to ``None`` (skip the override).

Lets ``EnvVar('PG_RECONCILE_WORK_MEM')='off'`` disable the bump per
environment without code changes -- Dagster's ``EnvVar`` only resolves to
``str``, so a string sentinel is the only path to a disable from the env.
"""


if TYPE_CHECKING:
    from dagster import AssetChecksDefinition, AssetExecutionContext, OpExecutionContext

    from moncpipelib.contracts.models import (
        ContractValidationSummary,
        DataContract,
        Period,
        Severity,
        ValidationResult,
    )
    from moncpipelib.io_managers.enums import WriteMode
    from moncpipelib.io_managers.writers import WriterConfig
    from moncpipelib.lineage import LineageTracker
    from moncpipelib.resources.types import LoggingContext
    from moncpipelib.streaming import BatchedDataFrame


class PostgresResource(ConfigurableResource):
    """Configurable Dagster resource for PostgreSQL database connections.

    Provides connection management for PostgreSQL databases with SSL support.
    Designed for use with Azure Database for PostgreSQL.

    Example:
        ```python
        from dagster import asset, Definitions, EnvVar
        from moncpipelib import PostgresResource

        @asset
        def my_asset(database: PostgresResource):
            with database.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM my_table")
                    return cursor.fetchall()

        defs = Definitions(
            assets=[my_asset],
            resources={
                "database": PostgresResource(
                    host=EnvVar("DB_HOST"),
                    port=EnvVar.int("DB_PORT"),
                    user=EnvVar("DB_USER"),
                    password=EnvVar("DB_PASSWORD"),
                    database=EnvVar("DB_NAME"),
                ),
            },
        )
        ```
    """

    host: str
    """Database server hostname."""

    port: int = 5432
    """Database server port. Defaults to 5432."""

    user: str
    """Database username."""

    password: str
    """Database password."""

    database: str
    """Database name."""

    sslmode: str = "require"
    """SSL mode for connection. Defaults to 'require' for Azure PostgreSQL."""

    connect_timeout: int = 30
    """Connection timeout in seconds. Defaults to 30."""

    client_connection_check_interval: str | None = "10s"
    """Per-session ``client_connection_check_interval`` applied at connect time
    via the libpq ``options`` string on every connection the resource opens.

    Primary containment for the zombie-backend failure mode (#365): when a
    Dagster run-worker pod is torn down, the server-side query keeps executing
    until PostgreSQL next performs socket I/O -- backends have been observed
    running 18h and 68h past their run's death, pinning the cluster-wide vacuum
    xmin horizon and stalling reconciles.  This GUC makes the backend poll the
    client socket *during* query execution and self-abort ~10s after the client
    disappears, independent of any external reaper.  It is ``USERSET`` in
    PostgreSQL (settable per session) -- Azure Flexible Server declares it
    read-only at the *server-parameter* API, so it cannot be set in Bicep and
    must be applied per-connection here.  ``application_name`` (see
    :mod:`moncpipelib.resources._app_name`) remains the correlation key the
    data-platform reaper uses for terminal-run cleanup; the two are
    complementary.

    - ``"10s"`` (default): self-abort roughly 10s after client disappearance.
    - ``None`` or ``"none"`` / ``"off"`` / ``"disabled"`` (case-insensitive):
      no per-session check; backends rely on the next socket I/O as before.
      The string sentinels exist because ``EnvVar`` only resolves to ``str``.
    - Any PostgreSQL time literal (``"10s"``, ``"5s"``, ``"10000"`` ms):
      format-validated before being applied.

    Tunable per-environment via ``EnvVar('PG_CLIENT_CONNECTION_CHECK_INTERVAL')``.
    Resolved by :meth:`_resolve_statement_timeout` (identical literal grammar)
    and rendered into the connect ``options`` by :meth:`_connection_options`."""

    # ---- Write configuration fields ----

    contract_search_paths: list[str] | None = None
    """Paths to search for contract YAML files. If None, auto-discovers from asset location."""

    enable_row_lineage: bool = True
    """Enable row-level lineage tracking. Defaults to True (opt-out). Set False to disable."""

    add_metadata_columns: bool = True
    """Whether to add metadata columns. Defaults to True."""

    enforce_contracts: str = "error"
    """How to handle contract validation at write time.
    - error: Raise ContractViolationError on validation failure (default)
    - warn: Log warnings but continue write
    - silent: Skip validation entirely
    """

    analyze_after_write: str = "partitioned"
    """Post-commit ``ANALYZE`` of the write target (public mirror issue
    model-oncology-public/moncpipelib#1).

    - partitioned: ANALYZE only when the target is a partitioned parent --
      the one relation autovacuum never autoanalyzes (default)
    - always: ANALYZE the target after every changed write
    - never: no post-write ANALYZE

    Applies to all write modes except ``scd2``, whose writer already
    maintains target statistics in-transaction (#312/#319/#361). Skipped
    when a write changes no rows. On PG >= 18 a partitioned parent uses
    ``ANALYZE ONLY`` so per-leaf statistics stay owned by autovacuum.
    Failures warn but never fail the committed write. Overridable per-asset
    via ``@asset(metadata={"analyze_after_write": ...})`` or per-call via
    ``database.write(..., analyze_after_write=...)``.
    """

    bulk_insert_method: str = "auto"
    """Method for bulk INSERT operations.
    - auto: Use COPY for DataFrames >= threshold, execute_values otherwise (default)
    - execute_values: Always use execute_values (compatible with all write modes)
    - copy: Always use COPY protocol (faster, but only for append/full_refresh)
    """

    bulk_insert_threshold: int = 10_000
    """Row count threshold for auto bulk_insert_method. DataFrames at or above
    this size use COPY protocol; smaller use execute_values. Defaults to 10,000."""

    full_refresh_method: str = "auto"
    """Method for clearing tables in full_refresh mode.
    - auto: Use TRUNCATE for DataFrames >= threshold, DELETE otherwise (default)
    - delete: Always use DELETE (safer locking, slower for large tables)
    - truncate: Always use TRUNCATE (faster, but holds exclusive lock)
    """

    full_refresh_threshold: int = 10_000
    """Row count threshold for auto full_refresh_method. DataFrames at or above
    this size use TRUNCATE; smaller DataFrames use DELETE. Defaults to 10,000."""

    insert_chunk_size: int | None = None
    """Process DataFrames in chunks of this size during INSERT.
    - None (default): Auto-select (no chunking for < 50k rows, 50k chunks otherwise)
    - 0: Disable chunking (process all rows at once)
    - N > 0: Always use chunks of N rows
    """

    reconcile_work_mem: str | None = "256MB"
    """Per-transaction ``work_mem`` applied at the start of ``reconcile_scd2``.

    The reconcile statements run window-function CTEs (``LAG``, ``SUM OVER``,
    ``ROW_NUMBER OVER``) over the full target table; on multi-million-row
    tables the implicit sort can spill to temp files at the cluster default
    (typically 32 MB) via multi-pass external merge.  A per-tx bump via
    ``set_config('work_mem', value, true)`` removes the spill component
    without affecting concurrent sessions or other queries.  Whether the
    bump translates to a measurable wall-time reduction depends on the
    fraction of total cost the sort represents: validated against a 58 M /
    23 GB reconcile of ``reference_silver.npi_address`` on
    ``pg-nonprod``, where the 256 MB run eliminated sustained
    ``BuffileWrite`` spill seen at 32 MB.  At that scale, DELETE / index
    maintenance still dominates total wall time; the spill premium is the
    component this bump removes.  See migration 017
    (``docs/migrations/20260510_294-reconcile-work-mem.md``) for the
    bench appendix.

    - ``"256MB"`` (default): conservative bump that fits comfortably on the
      D4ds_v5 SKU (16 GB RAM, ``shared_buffers=4 GB``) even with parallel
      workers.  No-op for sorts that already fit in the cluster default.
    - ``None`` or one of ``"none"`` / ``"off"`` / ``"disabled"`` (case-
      insensitive): skip the override entirely; reconcile runs at the cluster
      default.  The string sentinels exist because ``EnvVar`` only resolves to
      ``str`` -- setting ``PG_RECONCILE_WORK_MEM='off'`` in npe is the only
      path to a disable from the env.
    - Any Postgres ``work_mem`` literal (``"512MB"``, ``"1GB"``, ``"32kB"``):
      format-validated against ``\\d+\\s*(kB|MB|GB)`` before being applied.
      Postgres enforces the actual range server-side (minimum 64 kB), so
      sub-minimum values pass the regex but are rejected at ``set_config``.

    Tunable per-environment via ``EnvVar('PG_RECONCILE_WORK_MEM')`` so npe and
    prd can carry different headroom without code changes.  Per-call override
    available on ``reconcile_scd2(work_mem=...)``.

    The applied value is logged at INFO on each ``reconcile_scd2`` call
    (logger ``moncpipelib.resources``) and surfaced both in the
    ``reconcile_scd2`` return dict under ``work_mem`` and on the Dagster
    ``MaterializeResult`` metadata produced by ``make_reconciliation_asset``
    (#306).
    """

    scd2_change_detection_work_mem: str | None = None
    """Per-transaction ``work_mem`` applied to the SCD2 *writer's*
    change-detection statements (the count LEFT JOIN, expire UPDATE, Stage-1
    anti-join CTAS, and ``detect_deletes`` UPDATE in ``scd2_finalize``).

    Distinct from :attr:`reconcile_work_mem`, which covers only
    ``reconcile_scd2``.  The writer path previously ran these anti-joins at the
    cluster default ``work_mem`` (~32 MB); on large reference tables a larger
    ``work_mem`` lowers the planner's hash-build cost estimate, which can tip
    plan choice away from a full-table sequential scan and removes hash/sort
    spill (#361).

    Defaults to ``None`` (cluster default).  The bump's benefit on the writer
    path is plan-dependent and should be confirmed with ``EXPLAIN`` against the
    live table before being enabled fleet-wide; once validated, set e.g.
    ``"256MB"`` here or via ``EnvVar``.  Accepts the same literals and disable
    sentinels as :attr:`reconcile_work_mem`; format-validated by
    :meth:`_resolve_work_mem`."""

    scd2_change_detection_statement_timeout: str | None = "30min"
    """Per-statement ``statement_timeout`` bounding the SCD2 writer's
    *target-reading* anti-join statements (count, expire, Stage-1 CTAS, and
    ``detect_deletes``).  The Stage-2 bulk INSERT is left unbounded.

    Containment for the #361 failure mode: a single ``detect_deletes`` UPDATE
    ran read-bound for ~68h on ``npi_address``, holding one transaction open
    and pinning the cluster-wide vacuum xmin horizon the entire time.  A bound
    aborts a degenerate plan in minutes and releases its snapshot instead of
    grinding for days.

    - ``"30min"`` (default): clears any realistic healthy anti-join (the same
      statement completes in ~5s on a single cached snapshot) while capping the
      blast radius far below the observed 68h.
    - ``None`` or ``"none"`` / ``"off"`` / ``"disabled"`` (case-insensitive):
      no bound; the statements run unbounded as before.  The string sentinels
      exist because ``EnvVar`` only resolves to ``str``.
    - Any Postgres ``statement_timeout`` literal (``"10min"``, ``"600s"``,
      ``"900000"`` ms): format-validated before being applied.

    Tunable per-environment via
    ``EnvVar('PG_SCD2_CHANGE_DETECTION_STATEMENT_TIMEOUT')``.  Resolved by
    :meth:`_resolve_statement_timeout`."""

    # ---- Test isolation fields ----

    schema_override: str | None = None
    """When set, all writes are redirected to this schema instead of the
    schema parsed from the ``target`` parameter.  Used by the integration
    test runner to isolate writes to a test schema."""

    table_prefix: str = ""
    """Prefix prepended to the bare table name for write isolation.  Used
    together with ``schema_override`` for integration test table naming."""

    # OpenLineage configuration (optional)
    openlineage_url: str | None = None
    """OpenLineage API endpoint URL. If set, enables OpenLineage event emission."""

    openlineage_namespace: str = "moncpipelib"
    """OpenLineage namespace for jobs and datasets. Defaults to 'moncpipelib'."""

    openlineage_api_key: str | None = None
    """Optional API key for OpenLineage authentication."""

    # ---- Private attributes ----

    _engine: sa.engine.Engine | None = PrivateAttr(default=None)
    _lineage_tracker: LineageTracker | None = PrivateAttr(default=None)
    _openlineage_emitter: Any = PrivateAttr(default=None)
    _period_registry_available: bool | None = PrivateAttr(default=None)
    _pipeline_registry_available: bool | None = PrivateAttr(default=None)
    _contract_validation_runs_available: bool | None = PrivateAttr(default=None)
    _scd2_reconciliations_available: bool | None = PrivateAttr(default=None)

    def _connection_options(self, raw_interval: str | None | _Sentinel = SENTINEL) -> str:
        """Build the libpq ``options`` string applied at connect time (#365).

        Currently carries ``client_connection_check_interval`` so every backend
        the resource opens self-aborts ~10s after its client disappears (see the
        field docstring).  Returns ``""`` when the interval is disabled so the
        result can be passed unconditionally to ``psycopg.connect(options=...)``
        (libpq treats an empty options string as a no-op).

        Args:
            raw_interval: Override for the raw interval value.  Defaults to the
                resource field; the asset-check connection factory passes the
                ``EnvVar``-resolved value since it runs outside Dagster's
                resource-resolution path.
        """
        interval_in = (
            self.client_connection_check_interval
            if isinstance(raw_interval, _Sentinel)
            else raw_interval
        )
        # ``client_connection_check_interval`` shares the time-literal grammar of
        # ``statement_timeout``, so validation is reused -- but re-raise with the
        # correct field name so a misconfigured PG_CLIENT_CONNECTION_CHECK_INTERVAL
        # does not surface a confusing "statement_timeout" error.
        try:
            interval = self._resolve_statement_timeout(interval_in)
        except ValueError:
            raise ValueError(
                f"invalid client_connection_check_interval: {interval_in!r} "
                "(expected an integer in milliseconds or an integer with a unit "
                "-- e.g. '10s', '5000', '1min' -- or 'none' / 'off' / 'disabled' "
                "to disable the per-session check)"
            ) from None
        if interval is None:
            return ""
        return f"-c client_connection_check_interval={interval}"

    @contextmanager
    def get_connection(self) -> Iterator[psycopg.Connection]:
        """Get a database connection as a context manager.

        The connection is automatically closed when the context exits.
        Use this for most database operations.

        Yields:
            A psycopg2 connection object.

        Example:
            ```python
            with database.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            ```
        """
        conn: psycopg.Connection = psycopg.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
            sslmode=self.sslmode,
            connect_timeout=self.connect_timeout,
            # #365: tag the backend with the owning Dagster run_id so a live
            # ``pg_stat_activity`` row can be correlated to its run (enables
            # zombie-backend reaping). Falls back to a stable identifier.
            application_name=resolve_application_name(),
            # #365: per-session client_connection_check_interval so the backend
            # self-aborts ~10s after its client disappears (primary fix).
            options=self._connection_options(),
        )
        PostgresPolarsSchema.register_uuid_adapter(conn)
        PostgresPolarsSchema.register_json_adapters(conn)
        try:
            yield conn
        finally:
            conn.close()

    def get_connection_raw(self) -> psycopg.Connection:
        """Get a database connection without context management.

        The caller is responsible for closing the connection.
        Prefer `get_connection()` context manager for most use cases.

        Returns:
            A psycopg2 connection object.

        Warning:
            You must call `conn.close()` when done to avoid connection leaks.
        """
        conn: psycopg.Connection = psycopg.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
            sslmode=self.sslmode,
            connect_timeout=self.connect_timeout,
            # #365: tag the backend with the owning Dagster run_id (see
            # ``get_connection``).
            application_name=resolve_application_name(),
            # #365: per-session client_connection_check_interval (see
            # ``get_connection``).
            options=self._connection_options(),
        )
        PostgresPolarsSchema.register_uuid_adapter(conn)
        PostgresPolarsSchema.register_json_adapters(conn)
        return conn

    def get_engine(self) -> sa.engine.Engine:
        """Get a SQLAlchemy engine for advanced connection options.

        The engine is cached and reused across calls. Use this when you need
        SQLAlchemy-specific features like server-side cursor streaming.

        Returns:
            A SQLAlchemy Engine connected to the configured database.

        Example:
            ```python
            @asset
            def my_asset(database: PostgresResource) -> pl.DataFrame:
                engine = database.get_engine()
                with engine.connect() as conn:
                    streaming_conn = conn.execution_options(stream_results=True)
                    for batch in pl.read_database(
                        query="SELECT * FROM large_table",
                        connection=streaming_conn,
                        iter_batches=True,
                        batch_size=50_000,
                    ):
                        process(batch)
            ```
        """
        if self._engine is None:
            url = sa.engine.URL.create(
                drivername="postgresql+psycopg",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
                query={"sslmode": self.sslmode},
            )
            try:
                _pool = PoolDefaults()
                engine = sa.create_engine(
                    url,
                    connect_args={"connect_timeout": self.connect_timeout},
                    pool_pre_ping=True,
                    hide_parameters=True,
                    pool_size=_pool.pool_size,
                    max_overflow=_pool.max_overflow,
                    pool_timeout=_pool.pool_timeout,
                    pool_recycle=_pool.pool_recycle,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create database engine for {self.host}:{self.port}/{self.database}"
                ) from exc

            # #365: resolve application_name / client_connection_check_interval
            # on every *physical* connect rather than freezing them in
            # connect_args at engine-creation time. The engine is cached for the
            # resource's lifetime, so a creation-time value would stick even if
            # ``bind_run_id`` runs afterwards (e.g. a pure-read asset that hits
            # ``read_batched`` before any write/reconcile binds the run_id).
            # ``do_connect`` fires for each new pooled DBAPI connection, so each
            # one carries the run_id current at connect time. Guarded on a real
            # Engine: events can only be registered on genuine SQLAlchemy
            # targets (tests that mock ``create_engine`` get a non-Engine here).
            if isinstance(engine, sa.engine.Engine):

                @sa.event.listens_for(engine, "do_connect")
                def _apply_connection_identity(
                    _dialect: Any,
                    _conn_rec: Any,
                    _cargs: Any,
                    cparams: dict[str, Any],
                ) -> None:
                    cparams["application_name"] = resolve_application_name()
                    cparams["options"] = self._connection_options()

            self._engine = engine
        return self._engine

    @contextmanager
    def get_streaming_connection(self) -> Iterator[sa.engine.Connection]:
        """Get a SQLAlchemy connection with server-side cursor streaming enabled.

        This is a convenience method that combines get_engine() with
        execution_options(stream_results=True).

        Yields:
            A SQLAlchemy Connection with stream_results=True.

        Example:
            ```python
            with database.get_streaming_connection() as conn:
                for batch in pl.read_database(
                    "SELECT * FROM large_table",
                    conn,
                    iter_batches=True,
                    batch_size=50_000,
                ):
                    chunks.append(batch)
            ```
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            yield conn.execution_options(stream_results=True)

    def read_batched(
        self,
        query: str,
        *,
        batch_size: int = 50_000,
        order_by: str | list[str] | None = None,
        method: Literal["streaming", "offset"] = "streaming",
        context: OpExecutionContext | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Read query results in memory-efficient batches.

        See module-level read_batched() for full documentation.

        Example:
            ```python
            @asset
            def my_asset(context, database: PostgresResource) -> pl.DataFrame:
                chunks = []
                for batch in database.read_batched(
                    "SELECT * FROM large_table",
                    context=context,
                ):
                    chunks.append(batch)
                return pl.concat(chunks)
            ```
        """
        yield from read_batched(
            query,
            self,
            batch_size=batch_size,
            order_by=order_by,
            method=method,
            context=context,
        )

    def read_batched_to_dataframe(
        self,
        query: str,
        *,
        batch_size: int = 50_000,
        order_by: str | list[str] | None = None,
        method: Literal["streaming", "offset"] = "streaming",
        context: OpExecutionContext | None = None,
    ) -> pl.DataFrame:
        """Read query and return as single DataFrame.

        Convenience method that concatenates all batches.

        Example:
            ```python
            @asset
            def my_asset(context, database: PostgresResource) -> pl.DataFrame:
                return database.read_batched_to_dataframe(
                    "SELECT * FROM large_table",
                    context=context,
                )
            ```
        """
        return read_batched_to_dataframe(
            query,
            self,
            batch_size=batch_size,
            order_by=order_by,
            method=method,
            context=context,
        )

    # ===================================================================
    # Write support
    # ===================================================================

    def write(
        self,
        data: pl.DataFrame | BatchedDataFrame,
        *,
        target: str,
        context: AssetExecutionContext | WriteContext,
        write_mode: WriteMode | str | _Sentinel = SENTINEL,
        primary_key: list[str] | None = None,
        update_columns: list[str] | None = None,
        skip_unchanged: bool = False,
        partition_column: str | None = None,
        business_key: list[str] | None = None,
        tracked_columns: list[str] | None = None,
        detect_deletes: bool = False,
        sequence_column: str | None | _Sentinel = SENTINEL,
        contract: DataContract | None | _Sentinel = SENTINEL,
        source_file: str | None = None,
        pipeline_id: str | None = None,
        effective_date: date | None = None,
        source_id: str | None = None,
        source_uri: str | None = None,
        analyze_after_write: str | None = None,
    ) -> WriteResult:
        """Write a DataFrame or BatchedDataFrame to a PostgreSQL table.

        This is the primary write method for direct resource usage (as opposed
        to the IO manager path). The ``target`` parameter is always an explicit
        ``"schema.table"`` string.

        Args:
            data: Polars DataFrame or BatchedDataFrame to write.
            target: Fully-qualified target table as ``"schema.table"``.
            context: Dagster ``AssetExecutionContext`` or pre-built ``WriteContext``.
            write_mode: Write strategy. When omitted (``SENTINEL``), the
                contract's sink ``mode`` is authoritative; if no contract is
                found the effective default is ``full_refresh``.
            primary_key: Column(s) for upsert conflict detection.
            update_columns: Column(s) to update on upsert conflict. None = all non-key.
            skip_unchanged: Upsert only. When True, conflicting rows whose
                update columns are all unchanged (NULL-safe comparison) are
                not rewritten -- no dead tuple, index churn, or WAL for no-op
                updates. Opt-in because ``ON UPDATE`` triggers no longer fire
                for unchanged rows. Rejected with ``ValueError`` on non-upsert
                modes. Contract sinks may declare ``skip_unchanged`` instead
                (four-way reconciliation applies).
            partition_column: Column for partition-scoped writes.
            business_key: Column(s) for SCD2 business entity identification.
            tracked_columns: Column(s) to hash for SCD2 change detection.
                ``None`` (default) auto-derives all non-key, non-lineage
                DataFrame columns. An explicit empty list -- or ``None``
                when the business key covers every data column -- selects
                presence-only SCD2 (#432): the hash is computed over the
                business key, attribute change detection never fires, and
                versioning is driven purely by key presence (pair with
                ``detect_deletes=True`` so vanished keys close their spans).
            detect_deletes: SCD2 flag to expire missing business keys.
            sequence_column: Column name for per-business-key version sequence.
                ``SENTINEL`` (default) = use ``SCD2_DEFAULTS["sequence_col"]``.
                ``None`` = explicitly opt out.  ``str`` = use that column name.
                If the target table lacks this column, it is silently skipped.
            contract: Contract to validate against. ``SENTINEL`` (default) = auto-discover,
                ``None`` = skip, ``DataContract`` instance = use it.
            source_file: Source file path for lineage tracking.
            pipeline_id: Stable pipeline UUID from contract.
            effective_date: Override for SCD2 effective timestamps. When set,
                this date is used instead of ``now()`` for ``effective_from``
                on new inserts and ``effective_to`` on expired rows. Ignored
                for non-SCD2 write modes.
            source_id: Registry source_id UUID for auto-stamping
                ``silver_materialized_at`` in the period registry after a
                successful write. When set with a partition context, moncpipelib
                stamps the registry automatically without manual
                ``update_period_metadata()`` calls.
            source_uri: Resolved source URI for the period registry row.
                **Required for bronze writes against a contract whose
                ``data_source.periods`` is a ``FromIngestTemplate``** -- pass
                the resolved blob path obtained from
                ``resolve_source_for_partition(...)``. For enumerated-period
                bronze writes the registry row's ``source_uri`` is taken from
                the matched ``Period.source`` and this argument is ignored.
                For non-bronze writes this argument is ignored.

                Period registry registration uses ``partition_keys[0]`` as the
                partition key. When a from_ingest write batches multiple
                partitions into a single ``database.write(...)`` call, only
                the first partition is registered; callers needing
                per-partition registration must split into per-partition
                writes.
            analyze_after_write: Post-commit ``ANALYZE`` behavior for this
                write: ``"partitioned"`` / ``"always"`` / ``"never"``.
                ``None`` (default) = use the resource-level
                ``analyze_after_write`` setting. See the field docstring for
                semantics.

        Returns:
            ``WriteResult`` with statistics and metadata from the write operation.

        Raises:
            ValueError: If ``target`` is not in ``"schema.table"`` format,
                if write configuration is invalid, or if a from_ingest
                bronze write omits ``source_uri`` or lacks a Dagster
                partition context (validated before any write SQL runs).
            ContractViolationError: If contract validation fails and enforcement
                is ``ERROR``.
        """
        from moncpipelib.streaming import BatchedDataFrame as _BatchedDataFrame

        # Capture wall-clock start so ``WriteResult.duration_seconds``
        # reflects the full operator-visible cost of the write
        # (contract loading + the write itself + lineage / period-registry
        # stamping). ``time.perf_counter`` is the monotonic clock and is
        # safe for short and long durations alike. The reconcile path
        # already uses this pattern -- see ``reconcile_scd2``.
        _t0 = time.perf_counter()

        # Parse target and apply test-isolation overrides. The declared
        # (pre-override) schema is kept for contract sink matching (#405):
        # sink comparison must see the schema the caller named, not the
        # test-isolation override.
        schema, bare_table = self._parse_target(target)
        declared_schema = schema
        layer = schema if schema in _VALID_LAYERS else None
        if self.schema_override:
            schema = self.schema_override
        if self.table_prefix:
            bare_table = f"{self.table_prefix}{bare_table}"
        table_name = f"{schema}.{bare_table}"

        # Normalize context to WriteContext
        wctx = self._normalize_context(context)

        # #365: bind the run_id so connections opened below (and in step-executor
        # pods, whose hostname does not encode it) carry it as application_name.
        bind_run_id(wctx.run_id)

        # Resolve write_mode sentinel: caller omitted → not explicit
        write_mode_explicit = not isinstance(write_mode, _Sentinel)
        resolved_write_mode = "full_refresh" if isinstance(write_mode, _Sentinel) else write_mode

        # Resolve sequence_column sentinel
        resolved_seq_col: str | None = (
            SCD2Config().sequence_col if isinstance(sequence_column, _Sentinel) else sequence_column
        )

        # Build write config dict from explicit params
        write_config = self._build_write_config(
            write_mode=resolved_write_mode,
            write_mode_explicit=write_mode_explicit,
            primary_key=primary_key,
            update_columns=update_columns,
            skip_unchanged=skip_unchanged,
            partition_column=partition_column,
            business_key=business_key,
            tracked_columns=tracked_columns,
            detect_deletes=detect_deletes,
            sequence_column=resolved_seq_col,
            sequence_column_explicit=not isinstance(sequence_column, _Sentinel),
            analyze_after_write=analyze_after_write,
        )

        # Contract resolution
        loaded_contract = self._load_contract_for_write(
            contract_param=contract,
            asset_name=wctx.asset_name,
            layer=layer,
        )
        if pipeline_id is None and loaded_contract is not None:
            pipeline_id = loaded_contract.pipeline_id
        # Fall back to contract layer when schema doesn't match VALID_LAYERS
        # (e.g., target="reference_bronze.table" with contract layer="bronze")
        if layer is None and loaded_contract is not None and loaded_contract.layer in _VALID_LAYERS:
            layer = loaded_contract.layer

        # Hard-cutover invariant for from_ingest bronze writes: source_uri and
        # a Dagster partition context are required so that
        # lineage.period_registry can record the resolved blob path the data
        # was loaded from. Validate after contract resolution but before any
        # write SQL runs so caller bugs cannot leave partial state.
        if loaded_contract is not None and loaded_contract.data_source is not None:
            from moncpipelib.contracts.models import FromIngestTemplate

            if isinstance(loaded_contract.data_source.periods, FromIngestTemplate):
                source_name = loaded_contract.data_source.source_name
                if source_uri is None:
                    raise ValueError(
                        f"database.write(...) requires source_uri for from_ingest "
                        f"source {source_name!r}: pass the resolved blob path "
                        f"obtained from resolve_source_for_partition(...)"
                    )
                if not wctx.has_partition_key or not wctx.partition_keys:
                    raise ValueError(
                        f"database.write(...) requires a Dagster partition context "
                        f"for from_ingest source {source_name!r}"
                    )

        # Route to single or batched path
        if isinstance(data, _BatchedDataFrame):
            result = self._write_batched(
                batched=data,
                table_name=table_name,
                schema=schema,
                bare_table=bare_table,
                layer=layer,
                wctx=wctx,
                write_config=write_config,
                loaded_contract=loaded_contract,
                source_file=source_file,
                pipeline_id=pipeline_id,
                effective_date=effective_date,
                source_id=source_id,
                source_uri=source_uri,
                target_schema=declared_schema,
            )
        elif not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"PostgresResource.write() expected pl.DataFrame or BatchedDataFrame, "
                f"got {type(data).__name__}."
            )
        else:
            result = self._write_single(
                df=data,
                table_name=table_name,
                schema=schema,
                bare_table=bare_table,
                layer=layer,
                wctx=wctx,
                write_config=write_config,
                loaded_contract=loaded_contract,
                source_file=source_file,
                pipeline_id=pipeline_id,
                effective_date=effective_date,
                source_id=source_id,
                source_uri=source_uri,
                target_schema=declared_schema,
            )

        # Attach wall-clock duration via ``dataclasses.replace`` since
        # ``WriteResult`` is ``frozen=True``. Done here so neither write
        # branch needs to thread ``_t0`` through its internals; the
        # branches already populate every other field.
        return dataclasses.replace(
            result,
            duration_seconds=round(time.perf_counter() - _t0, 3),
        )

    def _make_check_connection_factory(self) -> Callable[[], psycopg.Connection]:
        """Build the psycopg connection factory used by asset-check execution.

        Asset check ops bypass Dagster's resource lifecycle, so ``EnvVar`` fields
        are resolved at call time via ``_resolve_envvar``.  Connections carry the
        same #365 identity as every other site: ``application_name`` (run
        correlation) and the ``client_connection_check_interval`` ``options``
        (self-abort containment).  Shared by both ``make_contract_checks``
        implementations (resource and IO manager) so the two cannot drift.
        """
        _self = self

        def _connection_factory() -> psycopg.Connection:
            resolve = _resolve_envvar
            return psycopg.connect(
                host=resolve(_self.host),
                port=int(resolve(_self.port)),
                user=resolve(_self.user),
                password=resolve(_self.password),
                dbname=resolve(_self.database),
                sslmode=resolve(_self.sslmode),
                # Match get_connection / get_connection_raw so a check against an
                # unreachable host fails fast instead of hanging indefinitely.
                connect_timeout=int(resolve(_self.connect_timeout)),
                # #365: run-id correlation + per-session self-abort. ``resolve``
                # bridges EnvVar since checks run outside resource resolution.
                application_name=resolve_application_name(),
                options=_self._connection_options(resolve(_self.client_connection_check_interval)),
            )

        return _connection_factory

    def make_contract_checks(
        self,
        contracts_dir: str | Path,
        asset_key_prefix: Sequence[str] | None = None,
        *,
        batched: bool = True,
        op_tags: dict[str, Any] | None = None,
    ) -> list[AssetChecksDefinition]:
        """Generate Dagster asset checks for all contracts in a directory.

        Recursively scans ``contracts_dir`` for ``*.contract.yaml`` files and
        generates Dagster asset checks for each contract found. Connection
        credentials are resolved at check execution time (not at definition
        time), so this method works correctly with Dagster ``EnvVar``.

        When ``batched=True`` (default), checks run as SQL queries directly
        in PostgreSQL -- no data is loaded into Python. Use ``op_tags`` to
        configure k8s pod resources if needed.

        Args:
            contracts_dir: Directory to recursively scan for contract files.
            asset_key_prefix: Optional prefix for asset keys (e.g., ``["bronze"]``).
                When set, checks attach to ``[*prefix, asset]``; otherwise they
                attach to ``[schema, table]`` derived from the resolved sink
                table, matching the IO manager's asset key convention.
            batched: Bundle all checks per contract into a single op (default True).
            op_tags: Dagster op tags (e.g., ``dagster-k8s/config`` for pod resources).

        Returns:
            List of ``AssetChecksDefinition`` ready for Dagster ``Definitions``.
        """
        from moncpipelib.contracts.checks import (
            _derive_check_asset_key,
            _make_deferred_df_loader,
            _resolve_check_table,
            generate_asset_checks_from_contract,
        )
        from moncpipelib.contracts.loader import load_contract

        contracts_path = Path(contracts_dir)

        # Auto-wire contract_search_paths so write-time contract discovery
        # reuses the same directory.
        if not self.contract_search_paths:
            resolved_dir = str(contracts_path.resolve())
            object.__setattr__(self, "contract_search_paths", [resolved_dir])

        _connection_factory = self._make_check_connection_factory()

        all_checks: list[AssetChecksDefinition] = []

        for contract_file in contracts_path.rglob(CONTRACT_FILE_PATTERN):
            contract_obj = load_contract(contract_file)

            fq_table = _resolve_check_table(
                contract_obj,
                schema_override=None,
                default_schema=None,
                db_schema="",
                table_suffix_to_strip="",
                table_prefix=None,
            )

            df_loader = _make_deferred_df_loader(
                connection_factory=_connection_factory,
                fq_table=fq_table,
            )

            asset_key = _derive_check_asset_key(
                contract_obj,
                fq_table=fq_table,
                asset_key_prefix=asset_key_prefix,
            )

            checks = generate_asset_checks_from_contract(
                contract_obj,
                asset_key,
                df_loader,
                batched=batched,
                connection_factory=_connection_factory,
                fq_table=fq_table,
                op_tags=op_tags,
            )
            all_checks.extend(checks)

        return all_checks

    # ------------------------------------------------------------------
    # Private: lazy initialization
    # ------------------------------------------------------------------

    def _get_lineage_tracker(self) -> LineageTracker | None:
        """Get the lineage tracker, initializing lazily if needed.

        Returns:
            LineageTracker instance if row lineage is enabled, None otherwise.

        Note on the cache shape: we key on ``self._lineage_tracker``
        identity rather than a separate ``_initialized`` flag. The flag
        approach has a latent bug under ``model_copy(update={...})``:
        Pydantic ``PrivateAttr`` values are copied by reference, so a
        resource ``model_copy``'d from a parent that already
        cache-missed (``_initialized=True`` + ``_lineage_tracker=None``
        because ``enable_row_lineage=False``) inherits the stale flag.
        When the child sets ``enable_row_lineage=True`` it still sees
        ``_initialized=True`` and skips the init -- lineage is silently
        disabled. Caching on the tracker itself avoids that trap: the
        ``None`` case re-attempts init every call, but the call is
        cheap (just an attribute read) when ``enable_row_lineage`` is
        False and the engine is only constructed once when it's True.
        """
        if self._lineage_tracker is not None:
            return self._lineage_tracker
        if not self.enable_row_lineage:
            return None

        from moncpipelib.lineage import LineageTracker as _LineageTracker

        lineage_url = sa.engine.URL.create(
            drivername="postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query={"sslmode": self.sslmode},
        )
        try:
            engine = sa.create_engine(lineage_url)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create lineage engine for {self.host}:{self.port}/{self.database}"
            ) from exc
        self._lineage_tracker = _LineageTracker(engine)
        return self._lineage_tracker

    def _get_openlineage_emitter(self) -> Any | None:
        """Get the OpenLineage emitter, initializing lazily if needed.

        Returns:
            OpenLineageEmitter instance if configured, None otherwise.
            Always None in test-mode lineage isolation (#420), so no
            START/COMPLETE/FAIL events are emitted for ephemeral runs;
            the per-write WARNING in ``_resolve_skip_lineage`` covers
            this skip.

        See ``_get_lineage_tracker`` for the rationale on caching by
        emitter identity instead of a separate ``_initialized`` flag --
        same ``model_copy``-preserves-private-attrs trap applies here.
        """
        from moncpipelib.config import skip_lineage_writes

        if skip_lineage_writes():
            return None
        if self._openlineage_emitter is not None:
            return self._openlineage_emitter
        if not self.openlineage_url:
            return None

        try:
            from moncpipelib.lineage.openlineage import (
                OpenLineageConfig,
                OpenLineageEmitter,
            )

            config = OpenLineageConfig(
                url=self.openlineage_url,
                namespace=self.openlineage_namespace,
                api_key=self.openlineage_api_key,
            )
            self._openlineage_emitter = OpenLineageEmitter(config)
        except ImportError:
            pass
        return self._openlineage_emitter

    # ------------------------------------------------------------------
    # Private: target parsing and context normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_target(target: str) -> tuple[str, str]:
        """Parse a ``"schema.table"`` string into (schema, bare_table).

        Delegates to :func:`~moncpipelib.config.parse_schema_table` with
        ``strict=True`` so that an unqualified table name is rejected.

        Raises:
            ValueError: If *target* does not contain exactly one dot.
        """
        return parse_schema_table(target, strict=True)

    @staticmethod
    def _normalize_context(
        context: AssetExecutionContext | WriteContext,
    ) -> WriteContext:
        """Convert *context* to ``WriteContext`` if it is not already one."""
        if isinstance(context, WriteContext):
            return context
        return WriteContext.from_asset_context(context)

    # ------------------------------------------------------------------
    # Private: write configuration building
    # ------------------------------------------------------------------

    def _get_writer_config(self) -> WriterConfig:
        """Build a ``WriterConfig`` from this resource's attributes."""
        from moncpipelib.io_managers.enums import BulkInsertMethod, FullRefreshMethod
        from moncpipelib.io_managers.writers import WriterConfig as _WriterConfig

        return _WriterConfig(
            bulk_insert_method=BulkInsertMethod(self.bulk_insert_method),
            bulk_insert_threshold=self.bulk_insert_threshold,
            full_refresh_method=FullRefreshMethod(self.full_refresh_method),
            full_refresh_threshold=self.full_refresh_threshold,
            insert_chunk_size=self.insert_chunk_size,
        )

    def _scd2_change_detection_settings(self) -> tuple[str | None, str | None]:
        """Resolve the SCD2 change-detection ``work_mem`` / timeout knobs (#361).

        Resolves both per-tx settings once for forwarding to ``execute_scd2`` /
        ``scd2_finalize``.  Resolution happens here (not in the writer) so
        malformed config fails fast and the writer stays free of validation.

        Returns:
            ``(work_mem, statement_timeout)`` -- each a canonical literal or
            ``None`` when disabled.
        """
        return (
            self._resolve_work_mem(self.scd2_change_detection_work_mem),
            self._resolve_statement_timeout(self.scd2_change_detection_statement_timeout),
        )

    @staticmethod
    def _build_write_config(
        *,
        write_mode: WriteMode | str,
        write_mode_explicit: bool = True,
        primary_key: list[str] | None,
        update_columns: list[str] | None,
        skip_unchanged: bool = False,
        partition_column: str | None,
        business_key: list[str] | None,
        tracked_columns: list[str] | None,
        detect_deletes: bool,
        sequence_column: str | None = None,
        sequence_column_explicit: bool = False,
        analyze_after_write: str | None = None,
    ) -> dict[str, Any]:
        """Build the internal write_config dict from explicit parameters.

        When ``write_mode_explicit`` is ``False`` (caller omitted the
        parameter), the contract's sink ``mode`` is authoritative during
        reconciliation.  When ``True`` (caller passed a value), the
        reconciler enforces consistency between the two sources.
        """
        from moncpipelib.io_managers.enums import WriteMode as _WriteMode

        resolved_mode = _WriteMode(write_mode) if isinstance(write_mode, str) else write_mode
        return {
            "write_mode": resolved_mode,
            "write_mode_explicit": write_mode_explicit,
            "primary_key": primary_key,
            "primary_key_explicit": primary_key is not None,
            "update_columns": update_columns,
            "skip_unchanged": skip_unchanged,
            "skip_unchanged_explicit": skip_unchanged is not False,
            "partition_column": partition_column,
            "partition_column_explicit": partition_column is not None,
            "business_key": business_key,
            "business_key_explicit": business_key is not None,
            "tracked_columns": tracked_columns,
            "tracked_columns_explicit": tracked_columns is not None,
            "detect_deletes": detect_deletes,
            "detect_deletes_explicit": detect_deletes is not False,
            "sequence_col": sequence_column,
            "sequence_col_explicit": sequence_column_explicit,
            "analyze_after_write": analyze_after_write,
            # SCD2 defaults
            "scd2": SCD2Config(),
            "effective_from_col": SCD2Config().effective_from_col,
            "effective_to_col": SCD2Config().effective_to_col,
            "is_current_col": SCD2Config().is_current_col,
            "hash_col": SCD2Config().hash_col,
        }

    # ------------------------------------------------------------------
    # Private: contract discovery and enforcement
    # ------------------------------------------------------------------

    def _get_contract_search_paths(self) -> list[Path | str] | None:
        """Return contract search paths for write-time contract discovery."""
        if self.contract_search_paths:
            return [Path(p) for p in self.contract_search_paths]
        return None

    def _load_contract_for_write(
        self,
        *,
        contract_param: DataContract | None | _Sentinel,
        asset_name: str,
        layer: str | None,
    ) -> DataContract | None:
        """Resolve the contract to use for a write call (thin wrapper).

        Delegates to :func:`_contract_helpers.load_contract_for_write`.  The
        resource supplies the ``enforce_mode`` and ``contract_search_paths``
        values derived from ``self`` so the helper does not depend on the
        resource class.
        """
        from moncpipelib.resources._contract_helpers import load_contract_for_write

        return load_contract_for_write(
            contract_param=contract_param,
            asset_name=asset_name,
            layer=layer,
            enforce_mode=self.enforce_contracts,
            contract_search_paths=self._get_contract_search_paths(),
        )

    def _enforce_contract(
        self,
        df: pl.DataFrame,
        wctx: WriteContext,
        preloaded_contract: DataContract | None = None,
        *,
        layer: str | None = None,
        skip_table_expectations: bool = False,
    ) -> tuple[DataContract | None, ContractValidationSummary | None]:
        """Validate DataFrame against contract if one exists (thin wrapper).

        Delegates to :func:`_contract_helpers.enforce_contract`.  The bound
        :meth:`_log_validation_result` is passed through so test patches on
        ``PostgresResource._log_validation_result`` still take effect.

        Raises:
            ContractViolationError: If validation fails and enforcement is ERROR.
        """
        from moncpipelib.resources._contract_helpers import enforce_contract

        return enforce_contract(
            df,
            wctx,
            preloaded_contract,
            layer=layer,
            skip_table_expectations=skip_table_expectations,
            enforce_mode=self.enforce_contracts,
            contract_search_paths=self._get_contract_search_paths(),
            log_validation_result=self._log_validation_result,
        )

    @staticmethod
    def _log_validation_result(
        check_name: str,
        result: ValidationResult,
        wctx: LoggingContext,
        severity: Severity | None = None,
    ) -> None:
        """Log a validation result with appropriate severity (thin wrapper)."""
        from moncpipelib.resources._contract_helpers import log_validation_result

        log_validation_result(check_name, result, wctx, severity)

    # ------------------------------------------------------------------
    # Private: write config validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_write_config(
        write_config: dict[str, Any],
        df_columns: list[str],
        asset_name: str,
    ) -> None:
        """Validate write configuration is consistent (thin wrapper).

        Delegates to :func:`_contract_helpers.validate_write_config`.

        Raises:
            ValueError: If configuration is invalid or references missing columns.
        """
        from moncpipelib.resources._contract_helpers import validate_write_config

        validate_write_config(write_config, df_columns, asset_name)

    def _validate_columns(
        self,
        cursor: psycopg.Cursor,
        table_name: str,
        df_columns: list[str],
        asset_name: str,
        exclude_from_table: set[str] | None = None,
    ) -> None:
        """Validate DataFrame columns match target table schema (thin wrapper).

        Delegates to :func:`_contract_helpers.validate_columns`.

        Raises:
            ValueError: If columns don't match.
        """
        from moncpipelib.resources._contract_helpers import validate_columns

        validate_columns(cursor, table_name, df_columns, asset_name, exclude_from_table)

    def _validate_partition_safety(
        self,
        wctx: WriteContext,
        write_config: dict[str, Any],
        asset_name: str,
    ) -> None:
        """Validate partition context + write mode combinations (thin wrapper).

        Delegates to :func:`_contract_helpers.validate_partition_safety`.

        Raises:
            ContractViolationError: For unsafe combinations.
        """
        from moncpipelib.resources._contract_helpers import validate_partition_safety

        validate_partition_safety(wctx, write_config, asset_name)

    # ------------------------------------------------------------------
    # Private: period partition injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_period_partition_column(
        df: pl.DataFrame,
        write_config: dict[str, Any],
        loaded_contract: DataContract | None,
        effective_date: date | None,
        wctx: WriteContext | None = None,
    ) -> pl.DataFrame:
        """Inject partition column if applicable.

        Resolves the partition value from two sources (in priority order):

        1. **Period manifest** -- when ``effective_date`` matches a contract
           period that has ``partition_key``, uses that value. This is the
           bronze path where periods come from ``*.source.yaml``.
        2. **Dagster partition context** -- when the asset has a Dagster
           partition key and no period match was found, uses the partition
           key directly. This is the silver path where partitions come
           from the registry or other Dagster partition definitions.

        In both cases, the column is only injected if ``partition_column``
        is declared in the write config and the column is not already
        present in the DataFrame.
        """
        partition_column = write_config.get("partition_column")
        if not partition_column:
            return df

        # Don't override if column already exists in DataFrame
        if partition_column in df.columns:
            return df

        # Path 1: match against data_source periods (bronze with *.source.yaml)
        if (
            effective_date is not None
            and loaded_contract is not None
            and loaded_contract.data_source is not None
            and loaded_contract.data_source.periods
        ):
            # FromIngestTemplate sources resolve partition keys through the
            # ingest boundary rather than an enumerated period list; skip
            # this path and let path 2 (Dagster partition key) handle it.
            ds_periods = loaded_contract.data_source.periods
            if isinstance(ds_periods, list):
                for period in ds_periods:
                    if period.effective_from == effective_date and period.partition_key:
                        return df.with_columns(pl.lit(period.partition_key).alias(partition_column))

        # Path 2: use Dagster partition key directly (silver / registry-backed)
        if wctx is not None and wctx.has_partition_key and wctx.partition_keys:
            return df.with_columns(pl.lit(wctx.partition_keys[0]).alias(partition_column))

        return df

    # ------------------------------------------------------------------
    # Private: SCD2 preparation
    # ------------------------------------------------------------------

    def _prepare_scd2(
        self,
        df: pl.DataFrame,
        write_config: dict[str, Any],
        wctx: WriteContext,
    ) -> tuple[pl.DataFrame, list[str], set[str]]:
        """Compute SCD2 row hash and determine columns to exclude from validation.

        Presence-only mode (#432): when no change-detection columns resolve
        -- ``tracked_columns`` is an explicit empty list, or it is omitted
        and every DataFrame column is part of the business key / lineage --
        the row hash is computed over the (sorted) business key itself.
        Within a matched key the hash is then constant, so the writer's
        change predicate (``t.hash <> s.hash``) can never fire: versioning
        reduces to key presence, with ``detect_deletes`` closing spans for
        vanished keys. Junction/reference tables whose business key is the
        full source tuple use this instead of nominating a formal tracked
        column. The under-specified-key case is still rejected downstream
        by the per-partition staging uniqueness guard (#419).

        Returns:
            Tuple of (df_with_hash_column, hash_columns, scd2_exclude_columns).
        """
        from moncpipelib.transforms.hashing import compute_row_hash as _compute_row_hash

        _scd2_cfg: SCD2Config = write_config["scd2"]
        hash_col = _scd2_cfg.hash_col
        tracked_cols: list[str] | None = write_config.get("tracked_columns")
        bk: list[str] = write_config["business_key"] or []

        if tracked_cols is None:
            lineage_cols = {LineageDefaults.ID_COLUMN, LineageDefaults.KEY_COLUMN}
            hash_columns = sorted(
                c for c in df.columns if c not in set(bk) and c not in lineage_cols
            )
        else:
            hash_columns = tracked_cols
            # Warn about DataFrame columns excluded from change detection
            lineage_cols = {LineageDefaults.ID_COLUMN, LineageDefaults.KEY_COLUMN}
            all_excluded = set(bk) | lineage_cols | {hash_col}
            uncovered = sorted(
                c for c in df.columns if c not in set(tracked_cols) and c not in all_excluded
            )
            if uncovered:
                wctx.log.warning(
                    f"SCD2 tracked_columns does not include DataFrame columns: "
                    f"{uncovered}. Changes to these columns will NOT trigger "
                    f"row versioning. Add them to tracked_columns in the "
                    f"contract if they should be tracked."
                )

        if not hash_columns:
            if not bk:
                raise ValueError(
                    "SCD2 requires at least one column for change-detection "
                    "hashing and business_key is empty. Set business_key, and "
                    "either add data columns to the DataFrame or specify "
                    "tracked_columns explicitly."
                )
            # Presence-only SCD2 (#432): hash the business key so row_hash
            # stays NOT NULL and constant within a key -- change detection
            # structurally never fires; versioning is driven by key
            # presence/absence alone.
            hash_columns = sorted(bk)
            if write_config.get("detect_deletes", False):
                wctx.log.info(
                    f"SCD2 presence-only mode: no tracked_columns resolve, so "
                    f"row_hash is computed over the business key {hash_columns}. "
                    f"Versions open on key appearance and close via "
                    f"detect_deletes on key absence (#432)."
                )
            else:
                wctx.log.warning(
                    f"SCD2 presence-only mode without detect_deletes: no "
                    f"tracked_columns resolve, so rows version on business-key "
                    f"presence only ({hash_columns}) -- but detect_deletes is "
                    f"False, so spans will never close. New keys insert; "
                    f"nothing ever expires. Enable detect_deletes if vanished "
                    f"keys should close their spans (#432)."
                )

        df = df.with_columns(_compute_row_hash(hash_columns, alias=hash_col))

        # Exclude only temporal/bookkeeping columns that the SCD2 writer adds
        # to the target table but are NOT in the incoming DataFrame.
        # hash_col IS in the DataFrame (added above) so must NOT be excluded.
        scd2_exclude: set[str] = {
            _scd2_cfg.effective_from_col,
            _scd2_cfg.effective_to_col,
            _scd2_cfg.is_current_col,
        }
        if _scd2_cfg.sequence_col is not None:
            scd2_exclude.add(_scd2_cfg.sequence_col)

        return df, hash_columns, scd2_exclude

    # ------------------------------------------------------------------
    # Private: lineage and metadata
    # ------------------------------------------------------------------

    def _resolve_skip_lineage(self, wctx: WriteContext) -> bool:
        """Resolve #420 test-mode lineage isolation for this write.

        Returns True when ``MONCPIPELIB_SKIP_LINEAGE_WRITES`` is set,
        logging one WARNING per write so run logs record that the audit
        trail was intentionally skipped.  See
        :func:`moncpipelib.config.skip_lineage_writes` for the full
        rationale and scope.
        """
        from moncpipelib.config import SKIP_LINEAGE_WRITES_ENV, skip_lineage_writes

        if not skip_lineage_writes():
            return False
        wctx.log.warning(
            f"{SKIP_LINEAGE_WRITES_ENV} is set: skipping data_lineage record, "
            f"contract_validation_runs, PII metadata sync, period_registry "
            f"stamping, pipeline_registry upsert, and OpenLineage emission "
            f"for this write (asset={wctx.asset_name}). Managed lineage "
            f"columns are attached with real generated values; the id "
            f"references no data_lineage row (#424, #426). Test/ephemeral "
            f"isolation only -- never set this in production (#420)."
        )
        return True

    # Cap on the number of partition keys serialised into the
    # ``data_lineage.metadata`` payload.  Today's partition definitions in
    # moncpipelib produce ≤ ~30 keys per run, so this is purely defensive
    # against a future Dagster partition definition yielding hundreds.
    _LINEAGE_METADATA_PARTITION_KEY_CAP: int = 50

    def _build_lineage_metadata_payload(
        self,
        *,
        write_config: dict[str, Any],
        wctx: WriteContext,
        loaded_contract: DataContract | None,
        contract_summary: ContractValidationSummary | None,
    ) -> dict[str, Any]:
        """Build the ``data_lineage.metadata`` JSONB payload.

        Captures the per-write observability surface the row's typed
        columns don't carry (write mode, partition shape, contract
        enforcement outcome) without duplicating columns already on
        ``data_lineage``.  Caller-defined metadata (e.g. the
        ``pgaudit_enriched`` ``{"pipeline": ..., "uploaded": N}`` shape)
        is out of scope here — this path is the resource write path only.

        Two single-path-only keys (asymmetry with the batched path):

        - ``contract_status``: ``contract_summary`` is unavailable on
          the batched path until *after* the lineage INSERT has fired
          (the first batch's ``_enforce_contract`` runs inside the
          cursor block, after the INSERT — see :py:meth:`_write_batched`).
        - On the batched path, ``write_mode`` / ``partition_column`` may
          also reflect the caller-supplied ``write_config`` rather than
          the contract-reconciled values, because
          ``ContractReconciler.reconcile_write_config`` also runs inside
          the cursor block after the INSERT.  In practice callers pass
          ``write_mode`` that matches the contract, but be aware when
          querying batched-path rows.

        ``contract_validation_runs`` still carries the full per-check
        result set on both paths, so no audit data is lost — only the
        lightweight per-write summary surface diverges.

        Returns a dict the caller threads into ``lineage_insert_kwargs``.
        Returns an empty dict (not ``None``) so callers can always insert
        a payload — even on no-contract writes — and ``write_mode`` plus
        partition shape are observable.
        """
        from moncpipelib.io_managers.enums import WriteMode as _WriteMode

        payload: dict[str, Any] = {}

        raw_write_mode = write_config.get("write_mode")
        if isinstance(raw_write_mode, _WriteMode):
            payload["write_mode"] = raw_write_mode.value
        elif isinstance(raw_write_mode, str):
            payload["write_mode"] = raw_write_mode

        partition_column = write_config.get("partition_column")
        if isinstance(partition_column, str) and partition_column:
            payload["partition_column"] = partition_column

        if wctx.has_partition_key and wctx.partition_keys:
            keys = list(wctx.partition_keys)
            cap = self._LINEAGE_METADATA_PARTITION_KEY_CAP
            if len(keys) > cap:
                overflow = len(keys) - cap
                keys = keys[:cap] + [f"... +{overflow} more"]
            payload["partition_keys"] = keys

        if loaded_contract is not None:
            payload["contract_enforcement"] = self.enforce_contracts

        if contract_summary is not None:
            payload["contract_status"] = contract_summary.status

        return payload

    def _prepare_lineage(
        self,
        df: pl.DataFrame,
        wctx: WriteContext,
        source_file: str | None = None,
        pipeline_id: str | None = None,
        *,
        layer: str | None = None,
        source_system: str | None = None,
        transformation_type: str | None = None,
        row_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[pl.DataFrame, str, str, dict[str, Any]]:
        """Prepare lineage tracking without touching the database.

        Migration 018 Phase 3: lineage row creation is split from id
        generation so the INSERT can run on the same psycopg cursor (and
        therefore in the same transaction) as the subsequent data DML.
        This method does the side-effect-free work -- generate ids,
        extract upstream parents, attach lineage columns to the DataFrame
        -- and returns the kwargs the caller will later pass to
        ``LineageTracker.write_lineage_record(cursor, ...)``.

        ``row_count`` is the count to record on the lineage row. For
        single-DataFrame writes pass ``len(df)``; for batched writes pass
        the size hint (``batched.total_rows_hint``) since the actual
        total is only known after the DML loop completes.
        """
        lineage_tracker = self._get_lineage_tracker()
        assert lineage_tracker is not None
        assert layer is not None

        parent_lineage_ids: list[str] | None = None
        if LineageDefaults.ID_COLUMN in df.columns:
            try:
                parent_lineage_ids = lineage_tracker.get_parent_lineage_ids(df)
                wctx.log.info(f"Extracted {len(parent_lineage_ids)} parent lineage IDs")
            except Exception as e:
                wctx.log.warning(f"Failed to extract parent lineage IDs: {e}")

        lineage_id, lineage_key = lineage_tracker.generate_lineage_ids(
            asset_name=wctx.asset_name,
            layer=layer,
            run_id=wctx.run_id,
            source_file=source_file,
        )

        df = lineage_tracker.attach_lineage_to_dataframe(df, lineage_id, lineage_key)

        insert_kwargs: dict[str, Any] = {
            "lineage_id": lineage_id,
            "lineage_key": lineage_key,
            "run_id": wctx.run_id,
            "asset_name": wctx.asset_name,
            "layer": layer,
            "source_file": source_file,
            "row_count": row_count if row_count is not None else len(df),
            "parent_lineage_ids": parent_lineage_ids,
            "pipeline_id": pipeline_id,
            "source_system": source_system,
            "transformation_type": transformation_type,
            "is_backfill": wctx.is_backfill,
            "backfill_id": wctx.backfill_id,
            "metadata": metadata,
        }

        return df, lineage_id, lineage_key, insert_kwargs

    def _add_metadata_columns(
        self,
        df: pl.DataFrame,
        wctx: WriteContext,
        source_file: str | None = None,
        *,
        layer: str | None = None,
    ) -> pl.DataFrame:
        """Add layer-specific metadata columns to the DataFrame."""
        if not self.add_metadata_columns or layer is None:
            return df

        if layer not in _VALID_LAYERS:
            raise ValueError(
                f"Invalid layer '{layer}'. Must be one of: {', '.join(sorted(_VALID_LAYERS))}"
            )

        run_id = wctx.run_id
        processed_at = int(time.time())

        run_id_col = f"_{layer}_run_id"
        processed_at_col = f"_{layer}_processed_at"

        df = df.with_columns(
            pl.lit(run_id).alias(run_id_col),
            pl.lit(processed_at).alias(processed_at_col),
        )

        if source_file is not None:
            df = df.with_columns(pl.lit(source_file).alias("_source_file"))
        elif "_source_file" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.String).alias("_source_file"))

        return df

    @staticmethod
    def _attach_skip_mode_lineage_columns(
        df: pl.DataFrame,
        lineage_id: str,
        lineage_key: str,
    ) -> pl.DataFrame:
        """Attach managed lineage columns in #420 test-mode isolation (#424, #426).

        Byte-for-byte production shape: both ``_lineage_id`` and
        ``_lineage_key`` carry real generated values, so NOT NULL sink
        constraints hold (#426 -- consumer models may declare either or both
        NOT NULL, and UPSERT staging tables LIKE-clone those constraints).
        The id references no ``data_lineage`` row: ephemeral sinks are
        dropped after the run, and test harnesses clone target tables with
        FKs stripped. A skip-mode write against a REAL table that enforces
        the ``data_lineage`` FK fails loudly on that FK -- intentional, it
        blocks test-isolated writes from landing in production tables.
        """
        return df.with_columns(
            pl.lit(lineage_id).alias(LineageDefaults.ID_COLUMN),
            pl.lit(lineage_key).alias(LineageDefaults.KEY_COLUMN),
        )

    # ------------------------------------------------------------------
    # Lineage-table availability checks
    # ------------------------------------------------------------------

    def _check_table_exists(
        self,
        cursor: psycopg.Cursor,
        *,
        schema: str,
        table: str,
        cache_attr: str,
        wctx: WriteContext | None = None,
        logger: logging.Logger | None = None,
        log_level: Literal["warning", "debug"] = "debug",
        log_message: str,
    ) -> bool:
        """Backing helper for the four ``_check_*`` registry methods.

        Caches the ``information_schema.tables`` probe on ``self.<cache_attr>``
        so each table is queried at most once per resource lifetime.  ``wctx``
        is preferred for first-miss logging; ``logger`` is the fallback when
        the caller has no write context (read path).
        """
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cast(bool, cached)

        cursor.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
            (schema, table),
        )
        available = cursor.fetchone() is not None
        setattr(self, cache_attr, available)
        if not available:
            if wctx is not None:
                getattr(wctx.log, log_level)(log_message)
            elif logger is not None:
                getattr(logger, log_level)(log_message)
        return available

    # ------------------------------------------------------------------
    # Period registry
    # ------------------------------------------------------------------

    def _check_period_registry(
        self,
        cursor: psycopg.Cursor,
        wctx: WriteContext | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> bool:
        """Check if the period registry table exists (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import check_period_registry

        return check_period_registry(self, cursor, wctx, logger=logger)

    def get_registry_periods(
        self,
        source_id: str,
        status: str | None = "materialized",
    ) -> list[dict[str, Any]]:
        """Query the period registry for a source's periods (thin wrapper).

        Args:
            source_id: Data source identifier.
            status: Filter by status. ``None`` returns all statuses.
                Defaults to ``"materialized"``.

        Returns:
            List of dicts with ``partition_key``, ``effective_from``,
            ``effective_to``, ``source_uri``, ``status``, ``registered_by``,
            and ``registered_at`` keys. Empty list if the registry table
            does not exist.
        """
        from moncpipelib.resources._registry_helpers import get_registry_periods

        return get_registry_periods(self, source_id, status)

    def register_period(
        self,
        source_id: str,
        partition_key: str,
        effective_from: date,
        effective_to: date | None = None,
        source_uri: str | None = None,
        status: str = "materialized",
        source_name: str | None = None,
        registered_by: str | None = None,
        run_id: str | None = None,
        pipeline_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a period in the period registry (thin wrapper).

        Uses ``INSERT ... ON CONFLICT (source_id, partition_key) DO UPDATE``
        to upsert the period.  The caller is responsible for ensuring the
        registry table exists.  Use this for explicit, standalone
        registration outside of a write path.
        """
        from moncpipelib.resources._registry_helpers import register_period

        register_period(
            self,
            source_id,
            partition_key,
            effective_from,
            effective_to=effective_to,
            source_uri=source_uri,
            status=status,
            source_name=source_name,
            registered_by=registered_by,
            run_id=run_id,
            pipeline_id=pipeline_id,
            metadata=metadata,
        )

    def _upsert_registry_row(
        self,
        cursor: Any,
        *,
        source_id: str,
        source_name: str | None,
        partition_key: str,
        effective_from: date,
        effective_to: date | None,
        source_uri: str | None,
        status: str,
        registered_by: str | None,
        run_id: str | None,
        pipeline_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Execute the period registry upsert on the given cursor (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import upsert_registry_row

        upsert_registry_row(
            cursor,
            source_id=source_id,
            source_name=source_name,
            partition_key=partition_key,
            effective_from=effective_from,
            effective_to=effective_to,
            source_uri=source_uri,
            status=status,
            registered_by=registered_by,
            run_id=run_id,
            pipeline_id=pipeline_id,
            metadata=metadata,
        )

    def update_period_metadata(
        self,
        source_id: str,
        partition_key: str,
        metadata_updates: dict[str, Any],
    ) -> None:
        """Merge keys into an existing period registry row's metadata JSONB (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import update_period_metadata

        update_period_metadata(self, source_id, partition_key, metadata_updates)

    @staticmethod
    def _resolve_work_mem(value: str | None) -> str | None:
        """Normalize and validate a ``work_mem`` config value.

        Single resolution surface for both the resource field
        (``reconcile_work_mem``) and the per-call override
        (``reconcile_scd2(work_mem=...)``).  Strips surrounding whitespace,
        recognizes the disable sentinels (``None``, ``""``, or ``"none"`` /
        ``"off"`` / ``"disabled"`` case-insensitive), and format-validates
        anything else.  Called from ``reconcile_scd2`` *before* the
        connection is opened so malformed input fails fast without
        consuming a Postgres backend or holding an advisory lock.

        Args:
            value: Raw config value -- ``None`` or a string.

        Returns:
            ``None`` if the input represents a disable, otherwise the
            stripped literal (e.g. ``"  256MB  "`` -> ``"256MB"``).  The
            returned value is in the shape ``_apply_work_mem_local``
            expects.

        Raises:
            ValueError: If ``value`` is a non-empty, non-disable string
                that does not match ``\\d+\\s*(kB|MB|GB)``.
        """
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or stripped.lower() in _WORK_MEM_DISABLE_TOKENS:
            return None
        if not _WORK_MEM_LITERAL_RE.fullmatch(stripped):
            raise ValueError(
                f"invalid work_mem literal: {value!r} "
                "(expected an integer followed by kB, MB, or GB -- e.g. '256MB', '1GB' "
                "-- or 'none' / 'off' / 'disabled' to skip the override)"
            )
        return stripped

    @staticmethod
    def _resolve_statement_timeout(value: str | None) -> str | None:
        """Normalize and validate a ``statement_timeout`` config value.

        Resolution surface for :attr:`scd2_change_detection_statement_timeout`.
        Strips surrounding whitespace, recognizes the disable sentinels
        (``None``, ``""``, or ``"none"`` / ``"off"`` / ``"disabled"`` case-
        insensitive), and format-validates anything else.  Mirrors
        :meth:`_resolve_work_mem` so env-var and config inputs are handled
        identically across the two knobs.

        Args:
            value: Raw config value -- ``None`` or a string.

        Returns:
            ``None`` if the input represents a disable, otherwise the stripped
            literal (e.g. ``"  30min "`` -> ``"30min"``) ready for ``set_config``.

        Raises:
            ValueError: If ``value`` is a non-empty, non-disable string that
                does not match an integer with an optional time unit.
        """
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or stripped.lower() in _WORK_MEM_DISABLE_TOKENS:
            return None
        if not _STATEMENT_TIMEOUT_LITERAL_RE.fullmatch(stripped):
            raise ValueError(
                f"invalid statement_timeout literal: {value!r} "
                "(expected an integer in milliseconds or an integer with a unit "
                "-- e.g. '30min', '600s', '900000' -- or 'none' / 'off' / "
                "'disabled' to disable the bound)"
            )
        return stripped

    @staticmethod
    def _apply_work_mem_local(cursor: psycopg.Cursor, value: str) -> str:
        """Apply a per-transaction ``work_mem`` setting via ``set_config``.

        ``SET`` is a Postgres utility statement and does not accept bind
        parameters, so the parameterized equivalent is
        ``set_config(name, value, is_local)``.  ``set_config`` returns the
        post-canonicalization value of the parameter (e.g. ``"262144kB"``
        becomes ``"256MB"``), so a separate ``SHOW work_mem`` is not
        required to read back the normalized form.

        Pre-validation lives in :meth:`_resolve_work_mem`; this method
        re-validates as defense-in-depth so any caller bypassing the
        resolver still cannot inject arbitrary input through the literal
        path.

        Args:
            cursor: Open cursor inside the transaction whose ``work_mem``
                should be bumped.  The setting reverts on commit or
                rollback.
            value: Pre-stripped ``work_mem`` literal -- a positive integer
                followed by ``kB``, ``MB``, or ``GB``.

        Returns:
            Canonical ``work_mem`` string as returned by ``set_config``.
            Suitable for direct equality comparison against another
            canonical form, or numeric comparison via ``pg_size_bytes()``.

        Raises:
            ValueError: If ``value`` does not match the expected literal
                shape.  The error message includes the offending input and
                examples of valid forms.
        """
        if not _WORK_MEM_LITERAL_RE.fullmatch(value):
            raise ValueError(
                f"invalid work_mem literal: {value!r} "
                "(expected an integer followed by kB, MB, or GB -- e.g. '256MB', '1GB')"
            )
        cursor.execute("SELECT set_config('work_mem', %s, true)", (value,))
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("set_config('work_mem', ...) returned no row")
        return str(row[0])

    @staticmethod
    def _extract_reconcile_context_signals(
        context: Any,
    ) -> tuple[str | None, str | None]:
        """Return ``(run_id, asset_name)`` from a Dagster context (thin wrapper)."""
        from moncpipelib.resources._scd2_helpers import extract_reconcile_context_signals

        return extract_reconcile_context_signals(context)

    @staticmethod
    def _build_scd2_reconciliation_metadata_payload(
        *,
        business_key: list[str],
        collapse_duplicates: bool,
        contract: DataContract | None,
    ) -> dict[str, Any]:
        """Build the ``scd2_reconciliations.metadata`` JSONB payload (thin wrapper)."""
        from moncpipelib.resources._scd2_helpers import (
            build_scd2_reconciliation_metadata_payload,
        )

        return build_scd2_reconciliation_metadata_payload(
            business_key=business_key,
            collapse_duplicates=collapse_duplicates,
            contract=contract,
        )

    @staticmethod
    def _resolve_scd2_sink(contract: DataContract) -> dict[str, Any]:
        """Find the SCD2 sink from a contract (thin wrapper).

        Raises:
            ValueError: If no SCD2 sink, multiple SCD2 sinks, or missing
                business_key.
        """
        from moncpipelib.resources._scd2_helpers import resolve_scd2_sink

        return resolve_scd2_sink(contract)

    def reconcile_scd2(
        self,
        target: str | None = None,
        business_key: list[str] | None = None,
        *,
        contract: DataContract | None = None,
        scd2: SCD2Config | None = None,
        collapse_duplicates: bool = True,
        work_mem: str | None | _Sentinel = SENTINEL,
        context: AssetExecutionContext | OpExecutionContext | WriteContext | None = None,
        run_id: str | None = None,
        asset_name: str | None = None,
        pipeline_id: str | None = None,
    ) -> dict[str, int | str | float | None]:
        """Atomically reconcile SCD2 timeline across partitions.

        Partition-scoped SCD2 writes leave every version with
        ``is_current=True`` and ``effective_to=NULL`` because each partition
        only sees its own rows. This method stitches the full timeline using
        a ``LEAD()`` window function and optionally collapses consecutive
        identical versions.

        When ``contract`` is provided, ``target`` and ``business_key`` are
        derived from the contract's SCD2 sink configuration. Explicit values
        override the contract.

        **Concurrency:** acquires a transaction-scoped advisory lock keyed on
        ``hashtext(target)`` before any DML, so concurrent ``reconcile_scd2``
        invocations against the same target serialize. The lock is auto-released
        on commit or rollback. Reconciles against different targets do not
        block each other. The lock does NOT serialize against ongoing
        ``database.write()`` SCD2 writes -- a write committed during a reconcile
        may leave new versions with ``is_current=true`` / ``effective_to=NULL``
        that won't be stitched until the next reconcile. This is intentional
        (writes should not block on reconciliation) and matches pre-#278
        behavior. See #278; see #279 for the orchestration-side fix that
        eliminates the per-partition triggering pattern this lock insures
        against.

        **Collapse SQL form:** the consecutive-duplicate DELETE uses a
        ``ROW_NUMBER`` + ``USING ranked WHERE rn > 1`` rewrite rather than
        ``DELETE WHERE id NOT IN (SELECT id FROM keepers)``. Postgres cannot
        transform ``NOT IN (subquery)`` to anti-join semantics through a CTE
        projection chain even when the underlying column is ``NOT NULL``,
        producing a ``Filter: NOT (ANY (SubPlan))`` plan that is O(N x M) at
        scale. Do not "simplify" back to ``NOT IN``. See #277.

        **Memory tuning:** ``collapse_sql`` and ``timeline_sql`` both contain
        window functions over the full target table; their sorts can spill
        to temp files at the cluster default ``work_mem`` (~32 MB) once the
        sort domain grows past what the planner can keep in memory.  See
        ``reconcile_work_mem`` on the resource (primary surface) and the
        ``work_mem`` argument below (per-call override).  The bump removes
        the spill component; DELETE and index maintenance still dominate
        total wall time at production scale.  See #294 and migration 017.

        Args:
            target: Fully-qualified ``"schema.table"`` target. Derived from
                contract if not provided.
            business_key: Column(s) that identify a logical entity. Derived
                from contract if not provided.
            contract: Optional contract to derive target and business_key from.
            scd2: SCD2 column configuration. Defaults to ``SCD2Config()``.
            collapse_duplicates: If ``True``, remove consecutive versions
                with identical ``hash_col`` values before timeline stitching.
            work_mem: Per-call override for the per-transaction ``work_mem``
                applied before reconcile DML.  Resolution order:

                - ``SENTINEL`` (default, caller did not pass): use the
                  resource field ``reconcile_work_mem``.
                - ``None`` or one of ``"none"`` / ``"off"`` / ``"disabled"``
                  (case-insensitive): skip the override, run at the cluster
                  default ``work_mem``.
                - Postgres ``work_mem`` literal (``"512MB"``, ``"1GB"``,
                  ...): apply this value for this call only; resource
                  field is ignored.

                Format-validated against ``\\d+\\s*(kB|MB|GB)`` *before* the
                connection is opened so malformed input fails fast.  Range
                (e.g. minimum 64 kB) is enforced server-side at
                ``set_config`` time.  Reverts on commit or rollback.
            context: Dagster ``AssetExecutionContext`` /
                ``OpExecutionContext``, or a pre-built ``WriteContext``.
                When supplied, ``run_id`` and (if not otherwise resolved)
                ``asset_name`` are extracted from it -- the idiomatic
                shape for Dagster-orchestrated callers including custom
                multi-contract reconcile loops.  Mirrors the
                ``database.write(context=...)`` convention on this
                resource.  Explicit ``run_id`` / ``asset_name`` kwargs
                still win over context-derived values.
            run_id: Caller-supplied run identifier.  Required if
                ``context`` is not passed.  Takes precedence over
                ``context.run_id`` when both are supplied.
            asset_name: Caller-supplied asset name override.  Takes
                precedence over both ``contract.asset`` and
                ``context.asset_key``.
            pipeline_id: Caller-supplied pipeline_registry FK.  Falls
                back to ``contract.pipeline_id`` when a contract is
                passed.

        Returns:
            Dict with ``rows_timeline_updated``, ``rows_collapsed``,
            ``rows_renumbered`` counts, ``work_mem`` (resolved per-tx
            ``work_mem`` literal applied to the reconcile transaction
            -- e.g. ``"256MB"`` -- or ``None`` when no override was
            applied), and ``duration_seconds`` (migration 019 Phase 6:
            wall-clock duration of the reconcile transaction measured by
            ``time.perf_counter`` for the new ``scd2_reconciliations``
            audit row).

        Raises:
            ValueError: If target or business_key cannot be resolved, or if
                the resolved ``work_mem`` value fails format validation.
        """
        from moncpipelib.resources._scd2_helpers import reconcile_scd2

        return reconcile_scd2(
            self,
            target,
            business_key,
            contract=contract,
            scd2=scd2,
            collapse_duplicates=collapse_duplicates,
            work_mem=work_mem,
            context=context,
            run_id=run_id,
            asset_name=asset_name,
            pipeline_id=pipeline_id,
        )

    def _reconcile_scd2_with_cursor(
        self,
        cursor: psycopg.Cursor,
        *,
        target: str,
        business_key: list[str],
        scd2: SCD2Config,
        collapse_duplicates: bool,
        work_mem: str | None = None,
    ) -> tuple[int, int, int]:
        """Execute SCD2 reconciliation statements against an open cursor (thin wrapper).

        Caller owns the connection and transaction lifecycle (commit / rollback /
        close). This split exists so integration tests can inject a wrapped
        cursor (see ``ExplainCapturingCursor``) to assert plan shape on the
        collapse DELETE -- the load-bearing regression guard against any future
        "simplify back to NOT IN" PR (#277).

        Returns:
            Tuple ``(rows_collapsed, rows_timeline_updated, rows_renumbered)``.
        """
        from moncpipelib.resources._scd2_helpers import reconcile_scd2_with_cursor

        return reconcile_scd2_with_cursor(
            self,
            cursor,
            target=target,
            business_key=business_key,
            scd2=scd2,
            collapse_duplicates=collapse_duplicates,
            work_mem=work_mem,
        )

    def _auto_register_period(
        self,
        conn: psycopg.Connection,
        loaded_contract: DataContract | None,
        effective_date: date | None,
        wctx: WriteContext,
        source_id: str | None = None,
        source_uri: str | None = None,
    ) -> None:
        """Auto-register the current period after a successful write (thin wrapper).

        Three paths covered:

        - **Bronze, enumerated periods:** ``data_source.periods`` is a
          ``list[Period]`` and ``effective_date`` matches one.
        - **Bronze, from_ingest periods:** ``data_source.periods`` is a
          ``FromIngestTemplate``; ``source_uri`` + ``partition_keys[0]`` come
          from the caller / write context (validated upstream).
        - **Silver path:** stamps ``silver_materialized`` metadata when the
          write has partition context.

        Failures log a warning but do not fail the write (same try/except/warn
        pattern as ``_sync_pii_metadata``).
        """
        from moncpipelib.resources._scd2_helpers import auto_register_period

        auto_register_period(
            self,
            conn,
            loaded_contract,
            effective_date,
            wctx,
            source_id=source_id,
            source_uri=source_uri,
        )

    # ------------------------------------------------------------------
    # Post-write ANALYZE (public mirror issue model-oncology-public/moncpipelib#1)
    # ------------------------------------------------------------------

    def _resolve_analyze_after_write(self, write_config: dict[str, Any]) -> str:
        """Resolve and validate the effective analyze_after_write mode (thin wrapper)."""
        from moncpipelib.resources._analyze_helpers import resolve_analyze_after_write

        return resolve_analyze_after_write(self.analyze_after_write, write_config)

    def _analyze_after_write(
        self,
        conn: psycopg.Connection,
        *,
        schema: str,
        bare_table: str,
        mode: str,
        write_mode: WriteMode,
        stats: dict[str, Any],
        row_count: int,
        wctx: WriteContext,
    ) -> str | None:
        """Post-commit ANALYZE of the write target (thin wrapper).

        Failures log a warning but do not fail the write (same pattern as
        ``_sync_pii_metadata`` / ``_auto_register_period``).
        """
        from moncpipelib.resources._analyze_helpers import analyze_after_write

        return analyze_after_write(
            conn,
            schema=schema,
            bare_table=bare_table,
            mode=mode,
            write_mode=write_mode,
            stats=stats,
            row_count=row_count,
            context=wctx,
        )

    # ------------------------------------------------------------------
    # Pipeline registry (migration 019 / #308 Phase 2)
    # ------------------------------------------------------------------

    def _check_pipeline_registry(
        self,
        cursor: psycopg.Cursor,
        wctx: WriteContext | None = None,
    ) -> bool:
        """Check if ``lineage.pipeline_registry`` exists (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import check_pipeline_registry

        return check_pipeline_registry(self, cursor, wctx)

    def _check_contract_validation_runs(
        self,
        cursor: psycopg.Cursor,
        wctx: WriteContext | None = None,
    ) -> bool:
        """Check if ``lineage.contract_validation_runs`` exists (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import check_contract_validation_runs

        return check_contract_validation_runs(self, cursor, wctx)

    def _check_scd2_reconciliations(
        self,
        cursor: psycopg.Cursor,
        logger: logging.Logger | None = None,
    ) -> bool:
        """Check if ``lineage.scd2_reconciliations`` exists (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import check_scd2_reconciliations

        return check_scd2_reconciliations(self, cursor, logger)

    def _pipeline_registry_upsert(
        self,
        cursor: psycopg.Cursor,
        *,
        loaded_contract: DataContract | None,
        wctx: WriteContext,
        layer: str | None,
    ) -> None:
        """Upsert ``lineage.pipeline_registry`` from contract + Dagster context (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import pipeline_registry_upsert

        pipeline_registry_upsert(
            self,
            cursor,
            loaded_contract=loaded_contract,
            wctx=wctx,
            layer=layer,
        )

    def _pipeline_registry_row_matches(
        self,
        cursor: psycopg.Cursor,
        loaded_contract: DataContract,
        wctx: WriteContext,
    ) -> bool:
        """Return True when the registry row already matches contract identity (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import pipeline_registry_row_matches

        return pipeline_registry_row_matches(cursor, loaded_contract, wctx)

    def _pipeline_registry_upsert_committed(
        self,
        *,
        loaded_contract: DataContract | None,
        wctx: WriteContext,
        layer: str | None,
    ) -> None:
        """Register the pipeline in its own short-lived autocommit connection (thin wrapper)."""
        from moncpipelib.resources._registry_helpers import pipeline_registry_upsert_committed

        pipeline_registry_upsert_committed(
            self,
            loaded_contract=loaded_contract,
            wctx=wctx,
            layer=layer,
        )

    def _sync_pii_metadata(
        self,
        cursor: psycopg.Cursor,
        table_name: str,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Sync PII / PHI classification from contract to lineage.column_metadata (SCD2).

        For each non-managed column in the contract, performs a close-and-insert
        into ``lineage.column_metadata``:

        1. If the current open record has different tags, close it (set valid_to).
        2. If no open record with the same tags exists, insert a new one.

        This is idempotent: calling with the same contract and tags produces no
        new records.  If ``lineage.column_metadata`` does not exist (e.g. the
        lineage schema has not been deployed), this method is a no-op.
        """
        import json

        schema_name, bare_table = parse_schema_table(table_name)

        # Skip if the lineage metadata table hasn't been deployed yet
        cursor.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'lineage' AND table_name = 'column_metadata'
            """,
        )
        if cursor.fetchone() is None:
            return

        updated_count = 0
        for col in contract.get_non_managed_columns():
            tags = json.dumps({"pii": col.pii, "phi": col.phi})

            # Close current record if tags changed
            cursor.execute(
                """
                UPDATE lineage.column_metadata
                SET valid_to = NOW()
                WHERE schema_name = %s AND table_name = %s AND column_name = %s
                  AND valid_to IS NULL AND tags IS DISTINCT FROM %s::jsonb
                """,
                (schema_name, bare_table, col.name, tags),
            )

            # Insert new record only if no matching open record exists
            cursor.execute(
                """
                INSERT INTO lineage.column_metadata
                    (schema_name, table_name, column_name, tags, updated_by, contract_name)
                SELECT %s, %s, %s, %s::jsonb, %s, %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM lineage.column_metadata
                    WHERE schema_name = %s AND table_name = %s AND column_name = %s
                      AND valid_to IS NULL AND tags = %s::jsonb
                )
                """,
                (
                    schema_name,
                    bare_table,
                    col.name,
                    tags,
                    wctx.run_id,
                    contract.asset,
                    schema_name,
                    bare_table,
                    col.name,
                    tags,
                ),
            )

            if cursor.rowcount > 0:
                updated_count += 1

        if updated_count:
            wctx.log.info(f"Synced PII metadata for {updated_count} column(s) on {table_name}")

    # ------------------------------------------------------------------
    # Private: single DataFrame write orchestration
    # ------------------------------------------------------------------

    def _write_single(
        self,
        *,
        df: pl.DataFrame,
        table_name: str,
        schema: str,
        bare_table: str,
        layer: str | None,
        wctx: WriteContext,
        write_config: dict[str, Any],
        loaded_contract: DataContract | None,
        source_file: str | None,
        pipeline_id: str | None,
        effective_date: date | None = None,
        source_id: str | None = None,
        source_uri: str | None = None,
        target_schema: str | None = None,
    ) -> WriteResult:
        """Orchestrate a single-DataFrame write to PostgreSQL.

        This mirrors the logic in ``PostgresIOManager.handle_output`` for
        single DataFrames, but operates on explicit parameters instead of
        Dagster ``OutputContext`` metadata.
        """
        from moncpipelib.contracts.reconciliation import ContractReconciler
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.io_managers.writers import (
            execute_append,
            execute_full_refresh,
            execute_partition_scoped,
            execute_scd2,
            execute_upsert,
        )

        write_mode: WriteMode = write_config["write_mode"]
        primary_key: list[str] | None = write_config["primary_key"]
        update_columns: list[str] | None = write_config["update_columns"]

        # Fail fast on an invalid analyze_after_write value before any
        # write SQL runs (mirror issue model-oncology-public/moncpipelib#1).
        analyze_mode = self._resolve_analyze_after_write(write_config)

        # Enforce contract
        _contract, contract_summary = self._enforce_contract(
            df,
            wctx,
            preloaded_contract=loaded_contract,
            layer=layer,
        )
        if _contract is not None and loaded_contract is None:
            loaded_contract = _contract

        # Reconcile contract sink fields
        ContractReconciler.reconcile_write_config(
            loaded_contract,
            bare_table,
            write_config,
            wctx,
            target_schema=target_schema,
        )
        write_mode = write_config["write_mode"]
        primary_key = write_config["primary_key"]
        update_columns = write_config["update_columns"]
        partition_column: str | None = write_config["partition_column"]

        # Partition safety
        self._validate_partition_safety(wctx, write_config, wctx.asset_name)

        # Inject partition column if applicable (before validation)
        df = self._inject_period_partition_column(
            df, write_config, loaded_contract, effective_date, wctx
        )

        # Resolve contract lineage config
        lineage_cfg = loaded_contract.lineage if loaded_contract is not None else None
        lineage_enabled = lineage_cfg.enabled if lineage_cfg is not None else True

        # #420: test-mode lineage isolation. The write keeps byte-for-byte
        # production shape -- managed lineage columns are attached in the
        # branch below with real generated values (#424, #426) -- but
        # nothing is written to the shared ``lineage`` schema: the LOCAL
        # ``lineage_id`` / ``lineage_insert_kwargs`` stay None so the
        # lineage-row INSERT and contract_validation_runs gates no-op, and
        # the PII sync and period_registry stamping below are gated off
        # explicitly.
        skip_lineage = self._resolve_skip_lineage(wctx)

        # Lineage or metadata columns. Phase 3 of migration 018: do the
        # side-effect-free prep here (id generation, parent extraction,
        # column attachment) and defer the lineage-row INSERT to the
        # cursor block below so it shares a transaction with the data DML.
        lineage_id: str | None = None
        lineage_key: str | None = None
        lineage_insert_kwargs: dict[str, Any] | None = None
        lineage_tracker = self._get_lineage_tracker()
        if lineage_tracker and layer and lineage_enabled:
            if skip_lineage:
                # #420/#424/#426: production shape without the lineage row.
                # ``generate_lineage_ids`` is pure/client-side. The real
                # generated pair is attached to the DataFrame (NOT NULL
                # sinks hold), but the LOCAL ``lineage_id`` stays None on
                # purpose: every side-effect gate below (lineage-row
                # INSERT, contract_validation_runs) keys on it, and
                # ``WriteResult.lineage_id`` must stay None because no
                # ``data_lineage`` row exists.
                skip_id, skip_key = lineage_tracker.generate_lineage_ids(
                    asset_name=wctx.asset_name,
                    layer=layer,
                    run_id=wctx.run_id,
                    source_file=source_file,
                )
                df = self._attach_skip_mode_lineage_columns(df, skip_id, skip_key)
                lineage_key = skip_key
            else:
                # Single-path: contract enforce + reconcile have already run
                # above, so ``contract_summary`` reflects the validation
                # outcome and ``write_config`` reflects contract-reconciled
                # values.  Both flow into the metadata payload.
                lineage_metadata = self._build_lineage_metadata_payload(
                    write_config=write_config,
                    wctx=wctx,
                    loaded_contract=loaded_contract,
                    contract_summary=contract_summary,
                )
                df, lineage_id, lineage_key, lineage_insert_kwargs = self._prepare_lineage(
                    df,
                    wctx,
                    source_file,
                    pipeline_id=pipeline_id,
                    layer=layer,
                    source_system=lineage_cfg.source_system if lineage_cfg else None,
                    transformation_type=(lineage_cfg.transformation_type if lineage_cfg else None),
                    metadata=lineage_metadata,
                )
        else:
            df = self._add_metadata_columns(df, wctx, source_file, layer=layer)

        # SCD2 hash
        scd2_exclude: set[str] | None = None
        if write_mode == WriteMode.SCD2:
            df, _, scd2_exclude = self._prepare_scd2(df, write_config, wctx)

        # Validate write config against DataFrame columns
        self._validate_write_config(write_config, df.columns, wctx.asset_name)

        columns = df.columns

        # OpenLineage START
        openlineage_run_id: str | None = None
        openlineage_emitter = self._get_openlineage_emitter()
        if openlineage_emitter:
            openlineage_run_id = openlineage_emitter.emit_start(
                job_name=wctx.asset_name,
                run_id=wctx.run_id,
            )

        # Extract partition values
        partition_values = wctx.partition_keys

        # Phase 4: ``replaces_lineage_id`` is set inside the cursor block
        # below when ``write_mode`` is FULL_REFRESH. Hoist to outer scope
        # so ``WriteResult`` construction below the try block sees it
        # regardless of whether the lineage INSERT actually fires.
        replaces_lineage_id: str | None = None

        # Post-commit ANALYZE action, set inside the cursor block after the
        # data txn commits; hoisted so the stats surfacing below sees it.
        analyze_action: str | None = None

        # Migration 019 (#308) Phase 2 + issue #332: register the pipeline
        # in its own short-lived autocommit transaction BEFORE the data
        # write transaction opens. The Phase 4 FK from
        # ``data_lineage.pipeline_id`` resolves against the committed
        # registry row (no requirement that it sit inside the same txn —
        # FK checks see committed rows just as well, and a committed row
        # does not hold the registry row's exclusive lock for the
        # duration of the silver write). See issue #332 for the
        # serialization bug this avoids.
        self._pipeline_registry_upsert_committed(
            loaded_contract=loaded_contract,
            wctx=wctx,
            layer=layer,
        )

        conn = self.get_connection_raw()
        try:
            with conn.cursor() as cursor:
                # Validate columns match table schema
                self._validate_columns(
                    cursor,
                    table_name,
                    columns,
                    wctx.asset_name,
                    exclude_from_table=scd2_exclude,
                )

                # Phase 3 of migration 018: insert the lineage row on the
                # same cursor BEFORE the data DML, so the row is visible
                # to the FK check on subsequent ``_lineage_id`` writes
                # without any production DDL change (the 793 enforced
                # NOT DEFERRABLE FKs against ``data_lineage(lineage_id)``
                # are satisfied by same-txn uncommitted reads).
                #
                # Phase 4: before the lineage INSERT, resolve the
                # partition date (so ``data_date`` / ``data_date_range``
                # populate on the row) and look up the prior row that
                # this write replaces (so ``replaces_lineage_id`` chains
                # successive FULL_REFRESH runs). Both lookups run on
                # the same cursor inside the same transaction.
                if lineage_tracker is not None and lineage_insert_kwargs is not None:
                    data_date, data_date_range = wctx.resolve_partition_dates(write_config)
                    lineage_insert_kwargs["data_date"] = data_date
                    lineage_insert_kwargs["data_date_range"] = data_date_range
                    if write_mode == WriteMode.FULL_REFRESH:
                        replaces_lineage_id = lineage_tracker.find_prior_lineage_id(
                            cursor,
                            asset_name=wctx.asset_name,
                            layer=layer or "",
                            data_date=data_date,
                            data_date_range=data_date_range,
                        )
                        lineage_insert_kwargs["replaces_lineage_id"] = replaces_lineage_id
                    lineage_tracker.write_lineage_record(cursor, **lineage_insert_kwargs)
                    wctx.log.info(f"Created lineage record: {lineage_id} ({lineage_key})")

                # Execute write
                config = self._get_writer_config()
                if write_mode == WriteMode.FULL_REFRESH:
                    if partition_values is not None and partition_column is not None:
                        stats = execute_partition_scoped(
                            config,
                            cursor,
                            table_name,
                            df,
                            partition_column,
                            wctx,
                            partition_values=partition_values,
                        )
                    else:
                        stats = execute_full_refresh(config, cursor, table_name, df, wctx)
                elif write_mode == WriteMode.UPSERT:
                    assert primary_key is not None
                    stats = execute_upsert(
                        config,
                        cursor,
                        table_name,
                        df,
                        primary_key,
                        update_columns,
                        wctx,
                        skip_unchanged=write_config.get("skip_unchanged", False),
                    )
                elif write_mode == WriteMode.APPEND:
                    stats = execute_append(config, cursor, table_name, df, wctx)
                elif write_mode == WriteMode.SCD2:
                    scd2_bk: list[str] = write_config["business_key"]
                    assert scd2_bk is not None
                    cd_work_mem, cd_timeout = self._scd2_change_detection_settings()
                    stats = execute_scd2(
                        config,
                        cursor,
                        table_name,
                        df,
                        scd2_bk,
                        write_config["scd2"],
                        wctx,
                        detect_deletes=write_config["detect_deletes"],
                        partition_column=partition_column if partition_values else None,
                        partition_values=partition_values,
                        effective_date=effective_date,
                        change_detection_work_mem=cd_work_mem,
                        change_detection_statement_timeout=cd_timeout,
                    )
                else:
                    raise ValueError(f"Unknown write_mode: {write_mode}")

                # Sync PII metadata before commit so it's atomic with the
                # write. Skipped in test-mode lineage isolation (#420):
                # the sync targets the shared lineage.column_metadata
                # table even when the sink is redirected.
                if loaded_contract is not None and not skip_lineage:
                    with conn.cursor() as pii_cursor:
                        self._sync_pii_metadata(
                            pii_cursor,
                            table_name,
                            loaded_contract,
                            wctx,
                        )

                # Migration 019 (#308) Phase 5: persist per-check contract
                # validation results on the same cursor, before commit, so
                # the audit trail is atomic with the data write. FK to
                # ``data_lineage.lineage_id`` resolves against the row
                # inserted earlier in this txn (Phase 3 of #309).
                if (
                    contract_summary is not None
                    and contract_summary.check_results
                    and lineage_id is not None
                    and self._check_contract_validation_runs(cursor, wctx)
                ):
                    n = self._get_lineage_tracker().write_validation_runs(  # type: ignore[union-attr]
                        cursor,
                        lineage_id=lineage_id,
                        check_results=contract_summary.check_results,
                    )
                    wctx.log.debug(
                        "Persisted %d contract_validation_runs for lineage_id=%s",
                        n,
                        lineage_id,
                    )

                conn.commit()
                wctx.log.info(f"Write complete for {table_name} (mode={write_mode.value})")

                # Auto-register period in the period registry. Skipped in
                # test-mode lineage isolation (#420): an ephemeral run
                # stamping silver_materialized on the real registry makes
                # the environment sensor silently skip the first real load.
                if not skip_lineage:
                    self._auto_register_period(
                        conn, loaded_contract, effective_date, wctx, source_id, source_uri
                    )

                # Post-commit ANALYZE: partitioned parents are never
                # autoanalyzed by autovacuum, so refresh their aggregate
                # stats here (mirror issue model-oncology-public/moncpipelib#1).
                analyze_action = self._analyze_after_write(
                    conn,
                    schema=schema,
                    bare_table=bare_table,
                    mode=analyze_mode,
                    write_mode=write_mode,
                    stats=dict(stats),
                    row_count=len(df),
                    wctx=wctx,
                )

            # OpenLineage COMPLETE
            if openlineage_emitter and openlineage_run_id:
                openlineage_emitter.emit_complete(
                    job_name=wctx.asset_name,
                    run_id=openlineage_run_id,
                    output_dataset=table_name,
                    row_count=len(df),
                    df=df,
                    lineage_id=lineage_id,
                    lineage_key=lineage_key,
                    layer=layer,
                    source_file=source_file,
                    pii_columns=(
                        loaded_contract.get_pii_column_names() if loaded_contract else None
                    ),
                    phi_columns=(
                        loaded_contract.get_phi_column_names() if loaded_contract else None
                    ),
                )

        except Exception as e:
            conn.rollback()
            wctx.log.error(f"Error writing to {table_name}: {e}")

            if openlineage_emitter and openlineage_run_id:
                openlineage_emitter.emit_fail(
                    job_name=wctx.asset_name,
                    run_id=openlineage_run_id,
                    error_message=str(e),
                )
            raise
        finally:
            conn.close()

        # Surface the post-commit ANALYZE action for observability (flows
        # into Dagster output metadata via ``to_dagster_metadata``).
        if analyze_action is not None:
            stats["analyze_after_write"] = analyze_action

        return WriteResult(
            table_name=table_name,
            schema=schema,
            layer=layer,
            write_mode=write_mode,
            stats=dict(stats),
            row_count=len(df),
            batch_count=1,
            contract_summary=contract_summary,
            contract=loaded_contract,
            lineage_id=lineage_id,
            lineage_key=lineage_key,
            columns=columns,
            source_file=source_file,
            primary_key=primary_key,
            partition_column=partition_column,
            business_key=write_config.get("business_key"),
            is_backfill=wctx.is_backfill,
            backfill_id=wctx.backfill_id,
            replaces_lineage_id=replaces_lineage_id,
            # Phase 5: single-write path already passes ``parent_lineage_ids``
            # through ``_prepare_lineage`` -> ``write_lineage_record``;
            # mirror the count on the WriteResult so the materialization
            # event view exposes it symmetrically with the batched path.
            parent_lineage_count=(
                len(lineage_insert_kwargs["parent_lineage_ids"])
                if (
                    lineage_insert_kwargs is not None
                    and lineage_insert_kwargs.get("parent_lineage_ids")
                )
                else 0
            ),
            partition_keys=list(wctx.partition_keys) if wctx.partition_keys else None,
            source_uri=source_uri,
            pipeline_id=pipeline_id,
            effective_date=effective_date,
        )

    # ------------------------------------------------------------------
    # Private: batched DataFrame write orchestration
    # ------------------------------------------------------------------

    def _write_batched(
        self,
        *,
        batched: BatchedDataFrame,
        table_name: str,
        schema: str,
        bare_table: str,
        layer: str | None,
        wctx: WriteContext,
        write_config: dict[str, Any],
        loaded_contract: DataContract | None,
        source_file: str | None,
        pipeline_id: str | None,
        effective_date: date | None = None,
        source_id: str | None = None,
        source_uri: str | None = None,
        target_schema: str | None = None,
    ) -> WriteResult:
        """Orchestrate a batched write to PostgreSQL.

        Processes each batch sequentially within a single transaction.
        Contract validation and reconciliation happen on the first batch.
        """
        from moncpipelib.contracts.reconciliation import ContractReconciler
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.io_managers.writers import (
            clear_table,
            execute_upsert,
            insert_rows,
            scd2_create_staging,
            scd2_finalize,
            scd2_insert_staging,
        )

        write_mode: WriteMode = write_config["write_mode"]
        primary_key: list[str] | None = write_config["primary_key"]
        update_columns: list[str] | None = write_config["update_columns"]
        partition_column: str | None = write_config["partition_column"]

        # Fail fast on an invalid analyze_after_write value before any
        # write SQL runs (mirror issue model-oncology-public/moncpipelib#1).
        analyze_mode = self._resolve_analyze_after_write(write_config)
        analyze_action: str | None = None

        # Resolve contract lineage config
        lineage_cfg = loaded_contract.lineage if loaded_contract is not None else None
        lineage_enabled = lineage_cfg.enabled if lineage_cfg is not None else True

        # #420: test-mode lineage isolation (mirrors _write_single; #424/#426:
        # lineage columns attached per batch with real generated values).
        skip_lineage = self._resolve_skip_lineage(wctx)

        # Generate lineage ids and stash INSERT kwargs (if enabled). The
        # actual lineage-row INSERT runs on the cursor below, INSIDE the
        # same transaction as the data DML -- see Phase 3 of migration
        # 018. ``write_lineage_record`` is called as the first statement
        # inside the cursor block to satisfy the production FK against
        # ``data_lineage(lineage_id)`` before any batch writes its
        # ``_lineage_id``.
        lineage_id: str | None = None
        lineage_key: str | None = None
        lineage_insert_kwargs: dict[str, Any] | None = None
        # #420/#424/#426 skip mode: one real (id, key) pair per write,
        # attached to every batch in the DML loop below. The LOCAL
        # ``lineage_id`` stays None so the side-effect gates no-op.
        skip_attach_ids: tuple[str, str] | None = None
        lineage_tracker = self._get_lineage_tracker()

        if lineage_tracker and layer and lineage_enabled:
            if skip_lineage:
                # #420/#424/#426: production shape without the lineage row.
                # ``lineage_id`` / ``lineage_insert_kwargs`` stay None so
                # the lineage-row INSERT, parent-id UPDATE, and
                # contract_validation_runs gates below no-op.
                skip_attach_ids = lineage_tracker.generate_lineage_ids(
                    asset_name=wctx.asset_name,
                    layer=layer,
                    run_id=wctx.run_id,
                    source_file=source_file,
                )
                lineage_key = skip_attach_ids[1]
            else:
                estimated_rows = batched.total_rows_hint if batched.total_rows_hint else 0
                lineage_id, lineage_key = lineage_tracker.generate_lineage_ids(
                    asset_name=wctx.asset_name,
                    layer=layer,
                    run_id=wctx.run_id,
                    source_file=source_file,
                )
                # Batched-path asymmetry vs. single-path: ``_enforce_contract``
                # and ``ContractReconciler.reconcile_write_config`` both run
                # inside the cursor block below (on batch 0) AFTER the lineage
                # INSERT fires, so ``contract_summary`` is unavailable here
                # and ``write_config`` reflects caller input rather than the
                # contract-reconciled shape.  Documented on
                # ``_build_lineage_metadata_payload``.
                lineage_metadata = self._build_lineage_metadata_payload(
                    write_config=write_config,
                    wctx=wctx,
                    loaded_contract=loaded_contract,
                    contract_summary=None,
                )
                lineage_insert_kwargs = {
                    "lineage_id": lineage_id,
                    "lineage_key": lineage_key,
                    "run_id": wctx.run_id,
                    "asset_name": wctx.asset_name,
                    "layer": layer,
                    "source_file": source_file,
                    "row_count": estimated_rows,
                    "pipeline_id": pipeline_id,
                    "source_system": lineage_cfg.source_system if lineage_cfg else None,
                    "transformation_type": (
                        lineage_cfg.transformation_type if lineage_cfg else None
                    ),
                    "is_backfill": wctx.is_backfill,
                    "backfill_id": wctx.backfill_id,
                    "metadata": lineage_metadata,
                }

        # OpenLineage START
        openlineage_run_id: str | None = None
        openlineage_emitter = self._get_openlineage_emitter()
        if openlineage_emitter:
            openlineage_run_id = openlineage_emitter.emit_start(
                job_name=wctx.asset_name,
                run_id=wctx.run_id,
            )

        partition_values = wctx.partition_keys

        conn = self.get_connection_raw()
        total_rows = 0
        total_batches = 0
        contract_summary: ContractValidationSummary | None = None
        last_batch_columns: list[str] = []

        # SCD2 batched state
        scd2_stage_table: str | None = None
        scd2_hash_columns: list[str] | None = None
        scd2_exclude: set[str] | None = None
        scd2_stats: dict[str, int | str] | None = None

        # Per-phase timers for ClientRead / idle-in-txn diagnostics (#260).
        # ``time.perf_counter()`` is sub-microsecond, so we always collect;
        # emission is gated on ``MONCPIPELIB_VERBOSE_METADATA``.  Buckets:
        #   t_iter -- wall time inside ``next(batched.batches)`` (upstream
        #             batch production: blob read, parse, transform).  The
        #             DB sees this as ``idle in transaction`` between
        #             COPYs.
        #   t_prep -- contract enforce, partition inject, lineage attach,
        #             SCD2 hash, validation, table prep.
        #   t_copy -- the actual ``insert_rows`` / ``execute_upsert`` /
        #             ``scd2_insert_staging`` call (CSV serialize + COPY).
        t_iter_total = 0.0
        t_prep_total = 0.0
        t_copy_total = 0.0

        # Phase 4: ``replaces_lineage_id`` is set inside the cursor block
        # below when ``write_mode`` is FULL_REFRESH. Hoist to outer scope
        # so ``WriteResult`` sees it regardless of execution path.
        replaces_lineage_id: str | None = None

        # Phase 5: accumulate upstream ``_lineage_id`` values across every
        # batch (multi-source iterators are valid -- ``BatchedDataFrame``
        # does not enforce single-source semantics, so first-batch peek
        # would silently drop later batches' parents). After the DML loop
        # we UPDATE the lineage row's ``parent_lineage_ids`` column on
        # the same cursor / transaction. Hoist to outer scope so
        # ``WriteResult.parent_lineage_count`` sees the final count even
        # if the write rolls back.
        batched_parent_ids: set[str] = set()

        # Migration 019 (#308) Phase 2 + issue #332: register the pipeline
        # in a short-lived autocommit transaction BEFORE the batched data
        # write transaction opens. See ``_write_single`` for the
        # rationale and ``_pipeline_registry_upsert_committed`` for the
        # lock-contention bug this avoids.
        self._pipeline_registry_upsert_committed(
            loaded_contract=loaded_contract,
            wctx=wctx,
            layer=layer,
        )

        try:
            with conn.cursor() as cursor:
                # Phase 3 of migration 018: write the lineage row on the
                # same cursor BEFORE any batch DML so the FK on each
                # batch's ``_lineage_id`` resolves against the in-flight
                # same-txn row. Commit is deferred to ``conn.commit()``
                # at the end of the cursor block.
                #
                # Phase 4: resolve ``data_date`` / ``data_date_range`` and
                # look up ``replaces_lineage_id`` (for FULL_REFRESH only)
                # before the lineage INSERT, all on the same cursor.
                if lineage_tracker is not None and lineage_insert_kwargs is not None:
                    data_date, data_date_range = wctx.resolve_partition_dates(write_config)
                    lineage_insert_kwargs["data_date"] = data_date
                    lineage_insert_kwargs["data_date_range"] = data_date_range
                    if write_mode == WriteMode.FULL_REFRESH:
                        replaces_lineage_id = lineage_tracker.find_prior_lineage_id(
                            cursor,
                            asset_name=wctx.asset_name,
                            layer=layer or "",
                            data_date=data_date,
                            data_date_range=data_date_range,
                        )
                        lineage_insert_kwargs["replaces_lineage_id"] = replaces_lineage_id
                    lineage_tracker.write_lineage_record(cursor, **lineage_insert_kwargs)
                    wctx.log.info(f"Created lineage record: {lineage_id} ({lineage_key})")

                last_iter_end = time.perf_counter()
                for i, batch_df in enumerate(batched.batches):
                    t_batch_start = time.perf_counter()
                    t_iter_total += t_batch_start - last_iter_end
                    # First batch -- pre-inject phase: enforce contract,
                    # reconcile sink fields, then validate partition safety.
                    # MUST run before ``_inject_period_partition_column`` so
                    # the contract-derived ``partition_column`` is in
                    # ``write_config`` when inject reads it (#258).  The
                    # reconciled value persists in ``write_config`` across
                    # subsequent batches, so ``i > 0`` iterations inject
                    # correctly without re-running reconcile.
                    if i == 0:
                        _contract, contract_summary = self._enforce_contract(
                            batch_df,
                            wctx,
                            preloaded_contract=loaded_contract,
                            layer=layer,
                            skip_table_expectations=True,
                        )
                        if _contract is not None and loaded_contract is None:
                            loaded_contract = _contract

                        ContractReconciler.reconcile_write_config(
                            loaded_contract,
                            bare_table,
                            write_config,
                            wctx,
                            target_schema=target_schema,
                        )
                        write_mode = write_config["write_mode"]
                        primary_key = write_config["primary_key"]
                        update_columns = write_config["update_columns"]
                        partition_column = write_config["partition_column"]

                        self._validate_partition_safety(wctx, write_config, wctx.asset_name)

                    # Inject period partition column if applicable.  Runs
                    # AFTER the i==0 reconcile above so ``write_config
                    # ["partition_column"]`` reflects the contract's sink
                    # value (#258).  Pre-#258, inject ran first and bailed
                    # because the kwarg-derived value was None.
                    batch_df = self._inject_period_partition_column(
                        batch_df, write_config, loaded_contract, effective_date, wctx
                    )

                    # Phase 5 of migration 018: accumulate upstream
                    # ``_lineage_id`` values from this batch BEFORE the
                    # next ``attach_lineage_to_dataframe`` overwrites the
                    # column with the new in-flight lineage id. The post-
                    # DML UPDATE below threads the union of every batch's
                    # ids onto ``data_lineage.parent_lineage_ids``.
                    # ``df.select(col).unique()`` is sub-millisecond for
                    # 10k--100k-row batches; cost is negligible vs the
                    # DML it sits beside.
                    if lineage_id and lineage_key and LineageDefaults.ID_COLUMN in batch_df.columns:
                        batched_parent_ids.update(
                            batch_df.select(LineageDefaults.ID_COLUMN)
                            .unique()
                            .to_series()
                            .to_list()
                        )

                    # Apply lineage or metadata to this batch
                    if lineage_id and lineage_key:
                        lt = self._get_lineage_tracker()
                        if lt:
                            batch_df = lt.attach_lineage_to_dataframe(
                                batch_df,
                                lineage_id,
                                lineage_key,
                            )
                    elif skip_attach_ids is not None:
                        # #420/#424/#426: production shape without the
                        # lineage row -- real generated id + key, no
                        # data_lineage row.
                        batch_df = self._attach_skip_mode_lineage_columns(
                            batch_df,
                            skip_attach_ids[0],
                            skip_attach_ids[1],
                        )
                    else:
                        batch_df = self._add_metadata_columns(
                            batch_df,
                            wctx,
                            source_file,
                            layer=layer,
                        )

                    # First batch -- post-inject phase: SCD2 prep + column
                    # validation + table preparation.  These need
                    # ``batch_df.columns`` to already include the injected
                    # partition column (it may be referenced in primary_key
                    # / update_columns) and to match the on-disk schema.
                    # Table expectations (row_count, etc.) are deferred to
                    # post-write SQL validation since they need aggregate
                    # data across all batches.
                    if i == 0:
                        # SCD2: compute hash and create staging table on first batch
                        if write_mode == WriteMode.SCD2:
                            batch_df, scd2_hash_columns, scd2_exclude = self._prepare_scd2(
                                batch_df,
                                write_config,
                                wctx,
                            )
                            scd2_stage_table = scd2_create_staging(
                                cursor,
                                table_name,
                                write_config["scd2"],
                            )

                        self._validate_write_config(
                            write_config,
                            batch_df.columns,
                            wctx.asset_name,
                        )
                        self._validate_columns(
                            cursor,
                            table_name,
                            batch_df.columns,
                            wctx.asset_name,
                            exclude_from_table=scd2_exclude,
                        )

                        # Prepare table based on write mode
                        if write_mode == WriteMode.FULL_REFRESH:
                            if partition_values is not None and partition_column is not None:
                                placeholders = ", ".join(["%s"] * len(partition_values))
                                delete_sql = (
                                    f"DELETE FROM {table_name} WHERE "
                                    f'"{partition_column}" IN ({placeholders})'
                                )  # noqa: S608
                                cursor.execute(delete_sql, partition_values)
                                deleted_count = cursor.rowcount
                                wctx.log.info(
                                    f"Deleted {deleted_count} rows from {table_name} "
                                    f"for {len(partition_values)} partition value(s)"
                                )
                            else:
                                config = self._get_writer_config()
                                clear_table(
                                    config,
                                    cursor,
                                    table_name,
                                    batched.total_rows_hint or 0,
                                    wctx,
                                )

                    # Write this batch
                    batch_rows = len(batch_df)
                    t_copy_start = time.perf_counter()
                    t_prep_total += t_copy_start - t_batch_start
                    if write_mode == WriteMode.SCD2:
                        assert scd2_stage_table is not None
                        assert scd2_hash_columns is not None
                        if i > 0:
                            from moncpipelib.transforms.hashing import (
                                compute_row_hash as _compute_row_hash,
                            )

                            batch_df = batch_df.with_columns(
                                _compute_row_hash(
                                    scd2_hash_columns,
                                    alias=write_config["scd2"].hash_col,
                                )
                            )
                        scd2_insert_staging(
                            self._get_writer_config(), cursor, scd2_stage_table, batch_df, wctx
                        )
                    elif write_mode == WriteMode.UPSERT:
                        assert primary_key is not None
                        config = self._get_writer_config()
                        execute_upsert(
                            config,
                            cursor,
                            table_name,
                            batch_df,
                            primary_key,
                            update_columns,
                            wctx,
                            skip_unchanged=write_config.get("skip_unchanged", False),
                        )
                    else:
                        config = self._get_writer_config()
                        insert_rows(config, cursor, table_name, batch_df, write_mode, wctx)

                    last_iter_end = time.perf_counter()
                    t_copy_total += last_iter_end - t_copy_start

                    total_rows += batch_rows
                    total_batches += 1
                    last_batch_columns = list(batch_df.columns)

                    if batched.total_rows_hint:
                        pct = min(100, total_rows * 100 // batched.total_rows_hint)
                        wctx.log.info(
                            f"Write batch {total_batches}: wrote {batch_rows:,} rows "
                            f"({total_rows:,}/{batched.total_rows_hint:,} = {pct}%)"
                        )
                    else:
                        wctx.log.info(
                            f"Write batch {total_batches}: wrote {batch_rows:,} rows "
                            f"({total_rows:,} total)"
                        )

                # Capture the trailing ``next()`` call that raised
                # StopIteration -- still upstream-iterator time, not
                # post-write work like SCD2 finalize / PII sync below.
                t_iter_total += time.perf_counter() - last_iter_end

                # SCD2: finalize after all batches
                if write_mode == WriteMode.SCD2:
                    if total_rows == 0:
                        if write_config["detect_deletes"]:
                            raise ValueError(
                                f"detect_deletes=True with an empty BatchedDataFrame for table "
                                f"'{table_name}'. This would expire ALL current records."
                            )
                        scd2_stats = {
                            "rows_new": 0,
                            "rows_expired": 0,
                            "rows_inserted": 0,
                            "rows_unchanged": 0,
                            "rows_deleted": 0,
                        }
                    else:
                        assert scd2_stage_table is not None
                        cd_work_mem, cd_timeout = self._scd2_change_detection_settings()
                        scd2_stats = scd2_finalize(
                            cursor,
                            table_name,
                            scd2_stage_table,
                            total_rows,
                            last_batch_columns,
                            write_config["business_key"],
                            write_config["scd2"],
                            wctx,
                            detect_deletes=write_config["detect_deletes"],
                            partition_column=partition_column if partition_values else None,
                            partition_values=partition_values,
                            effective_date=effective_date,
                            change_detection_work_mem=cd_work_mem,
                            change_detection_statement_timeout=cd_timeout,
                        )

                # Post-write validation: run table expectations against
                # the actual written data (within the open transaction).
                if (
                    loaded_contract is not None
                    and loaded_contract.expectations
                    and self.enforce_contracts != "silent"
                ):
                    from moncpipelib.contracts.exceptions import (
                        ContractViolationError as _ContractViolationError,
                    )
                    from moncpipelib.contracts.models import (
                        Severity as _Severity,
                    )
                    from moncpipelib.contracts.validators import (
                        run_post_write_expectations,
                    )

                    # run_post_write_expectations only understands enumerated
                    # periods; FromIngestTemplate sources get None here and
                    # gain manifest-based expectation hydration in Phase 2.
                    ds = loaded_contract.data_source if loaded_contract else None
                    period_list: list[Period] | None = (
                        ds.periods if ds is not None and isinstance(ds.periods, list) else None
                    )
                    post_write_results = run_post_write_expectations(
                        cursor=cursor,
                        table_name=table_name,
                        expectations=loaded_contract.expectations,
                        lineage_id=lineage_id,
                        total_rows=total_rows,
                        is_scd2=(write_mode == WriteMode.SCD2),
                        periods=period_list,
                        effective_date=effective_date,
                        effective_from_col=write_config.get("effective_from_col", "effective_from"),
                    )

                    # Migration 019 (#308) Phase 5: append each post-write
                    # expectation result to the summary's ``check_results``
                    # so the same audit row is persisted into
                    # ``contract_validation_runs`` alongside the in-DataFrame
                    # checks.
                    from moncpipelib.contracts.models import (
                        CheckResultRow as _CheckResultRow,
                    )

                    for exp, result in post_write_results:
                        if not result.passed:
                            self._log_validation_result(
                                exp.expectation_type,
                                result,
                                wctx,
                                severity=exp.severity,
                            )
                            if (
                                exp.severity == _Severity.ERROR
                                and self.enforce_contracts == "error"
                            ):
                                raise _ContractViolationError(
                                    f"Contract validation failed for "
                                    f"{wctx.asset_name}:\n"
                                    f"  - {result.message}",
                                    asset_name=wctx.asset_name,
                                    violations=[result],
                                )
                            if contract_summary is not None:
                                contract_summary.total_checks += 1
                                if exp.severity == _Severity.ERROR:
                                    contract_summary.failed_checks += 1
                                    contract_summary.violations.append(result.message)
                                    contract_summary.status = "failed"
                                else:
                                    contract_summary.warned_checks += 1
                                    contract_summary.warnings.append(result.message)
                                contract_summary.check_results.append(
                                    _CheckResultRow(
                                        check_name=exp.expectation_type,
                                        severity=exp.severity.value,
                                        passed=False,
                                        failed_count=result.failed_count,
                                        total_count=result.total_count,
                                        sample_failures=result.sample_failures,
                                    )
                                )
                        else:
                            self._log_validation_result(exp.expectation_type, result, wctx)
                            if contract_summary is not None:
                                contract_summary.total_checks += 1
                                contract_summary.passed_checks += 1
                                contract_summary.check_results.append(
                                    _CheckResultRow(
                                        check_name=exp.expectation_type,
                                        severity=exp.severity.value,
                                        passed=True,
                                        failed_count=result.failed_count,
                                        total_count=result.total_count,
                                        sample_failures=None,
                                    )
                                )

                # Phase 5 of migration 018: now that every batch has been
                # iterated, set ``parent_lineage_ids`` on the in-flight
                # lineage row to the union of upstream ids seen across
                # all batches. Same cursor / transaction; if it fails,
                # the data DML rolls back too. Skipped when empty -- the
                # row already carries NULL.
                if lineage_tracker is not None and lineage_id is not None and batched_parent_ids:
                    lineage_tracker.update_parent_lineage_ids(
                        cursor,
                        lineage_id=lineage_id,
                        parent_lineage_ids=sorted(batched_parent_ids),
                    )

                # Sync PII metadata before commit so it's atomic with the
                # write. Skipped in test-mode lineage isolation (#420):
                # the sync targets the shared lineage.column_metadata
                # table even when the sink is redirected.
                if loaded_contract is not None and not skip_lineage:
                    with conn.cursor() as pii_cursor:
                        self._sync_pii_metadata(
                            pii_cursor,
                            table_name,
                            loaded_contract,
                            wctx,
                        )

                # Migration 019 (#308) Phase 5: persist per-check contract
                # validation results on the same cursor, before commit, so
                # the audit trail is atomic with the data write. Mirrors
                # the _write_single placement so both paths emit identical
                # contract_validation_runs rows.
                if (
                    contract_summary is not None
                    and contract_summary.check_results
                    and lineage_id is not None
                    and self._check_contract_validation_runs(cursor, wctx)
                ):
                    n = self._get_lineage_tracker().write_validation_runs(  # type: ignore[union-attr]
                        cursor,
                        lineage_id=lineage_id,
                        check_results=contract_summary.check_results,
                    )
                    wctx.log.debug(
                        "Persisted %d contract_validation_runs for lineage_id=%s",
                        n,
                        lineage_id,
                    )

                conn.commit()
                wctx.log.info(
                    f"Batched write complete for {table_name} "
                    f"(mode={write_mode.value}, {total_batches} batches, {total_rows:,} rows)"
                )

                # Auto-register period in the period registry. Skipped in
                # test-mode lineage isolation (#420): an ephemeral run
                # stamping silver_materialized on the real registry makes
                # the environment sensor silently skip the first real load.
                if not skip_lineage:
                    self._auto_register_period(
                        conn, loaded_contract, effective_date, wctx, source_id, source_uri
                    )

                # Post-commit ANALYZE: partitioned parents are never
                # autoanalyzed by autovacuum, so refresh their aggregate
                # stats here (mirror issue model-oncology-public/moncpipelib#1).
                # The batched non-SCD2 path has no per-mode counters, so the
                # change gate falls back to total_rows.
                analyze_action = self._analyze_after_write(
                    conn,
                    schema=schema,
                    bare_table=bare_table,
                    mode=analyze_mode,
                    write_mode=write_mode,
                    stats=dict(scd2_stats) if scd2_stats is not None else {},
                    row_count=total_rows,
                    wctx=wctx,
                )

            # Update stats on the BatchedDataFrame instance
            batched.rows_written = total_rows
            batched.batches_written = total_batches

            # OpenLineage COMPLETE
            if openlineage_emitter and openlineage_run_id:
                openlineage_emitter.emit_complete(
                    job_name=wctx.asset_name,
                    run_id=openlineage_run_id,
                    output_dataset=table_name,
                    row_count=total_rows,
                    df=None,
                    lineage_id=lineage_id,
                    lineage_key=lineage_key,
                    layer=layer,
                    source_file=source_file,
                    pii_columns=(
                        loaded_contract.get_pii_column_names() if loaded_contract else None
                    ),
                    phi_columns=(
                        loaded_contract.get_phi_column_names() if loaded_contract else None
                    ),
                )

        except Exception as e:
            conn.rollback()
            wctx.log.error(f"Error in batched write to {table_name}: {e}")

            if openlineage_emitter and openlineage_run_id:
                openlineage_emitter.emit_fail(
                    job_name=wctx.asset_name,
                    run_id=openlineage_run_id,
                    error_message=str(e),
                )
            raise
        finally:
            conn.close()

        final_stats: dict[str, Any] = {"rows_written": total_rows, "batches_written": total_batches}
        if scd2_stats is not None:
            final_stats.update(scd2_stats)
        # Surface the post-commit ANALYZE action for observability (flows
        # into Dagster output metadata via ``to_dagster_metadata``).
        if analyze_action is not None:
            final_stats["analyze_after_write"] = analyze_action

        # Verbose metadata: per-phase timings for ClientRead diagnostics
        # (#260).  Always logged at INFO when enabled so an operator
        # tailing logs sees the breakdown; surfaced as Dagster output
        # metadata so it is queryable run-over-run.
        from moncpipelib.config import VERBOSE_METADATA

        if VERBOSE_METADATA:
            final_stats["t_iter_seconds"] = round(t_iter_total, 3)
            final_stats["t_prep_seconds"] = round(t_prep_total, 3)
            final_stats["t_copy_seconds"] = round(t_copy_total, 3)
            wctx.log.info(
                f"Batched write timings for {table_name}: "
                f"iter={t_iter_total:.2f}s prep={t_prep_total:.2f}s "
                f"copy={t_copy_total:.2f}s ({total_batches} batches)"
            )

        return WriteResult(
            table_name=table_name,
            schema=schema,
            layer=layer,
            write_mode=write_mode,
            stats=final_stats,
            row_count=total_rows,
            batch_count=total_batches,
            contract_summary=contract_summary,
            contract=loaded_contract,
            lineage_id=lineage_id,
            lineage_key=lineage_key,
            columns=last_batch_columns,
            source_file=source_file,
            primary_key=primary_key,
            partition_column=partition_column,
            business_key=write_config.get("business_key"),
            is_backfill=wctx.is_backfill,
            backfill_id=wctx.backfill_id,
            replaces_lineage_id=replaces_lineage_id,
            parent_lineage_count=len(batched_parent_ids),
            partition_keys=list(wctx.partition_keys) if wctx.partition_keys else None,
            source_uri=source_uri,
            pipeline_id=pipeline_id,
            effective_date=effective_date,
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _resolve_envvar(value: Any) -> Any:
    """Resolve a Dagster EnvVar to its environment variable value.

    Asset check functions are not Dagster resources, so Dagster's resource
    system never resolves EnvVar fields for them.  This helper bridges the
    gap by calling ``get_value()`` at check execution time.
    """
    if hasattr(value, "get_value") and callable(value.get_value):
        return value.get_value()
    return value


# ---------------------------------------------------------------------------
# Module-level batched read utilities
# ---------------------------------------------------------------------------


def _read_batched_streaming(
    query: str,
    connection: sa.engine.Engine | sa.engine.Connection,
    batch_size: int,
    context: OpExecutionContext | None,
) -> Iterator[pl.DataFrame]:
    """Internal: streaming method using server-side cursors."""

    def _log(msg: str) -> None:
        if context:
            context.log.info(msg)

    if isinstance(connection, sa.engine.Engine):
        conn_ctx: Any = connection.connect()
    else:
        conn_ctx = nullcontext(connection)

    try:
        with conn_ctx as conn:
            # Register type adapters so psycopg2 returns UUIDs and JSON as strings
            PostgresPolarsSchema.register_uuid_adapter_sa(conn)
            PostgresPolarsSchema.register_json_adapters_sa(conn)

            # Probe cursor description for deterministic column types
            schema_overrides = PostgresPolarsSchema.from_sa_connection(conn, query)
            if schema_overrides:
                _log(f"Schema overrides: {len(schema_overrides)} columns from cursor description")

            streaming_conn = conn.execution_options(stream_results=True)

            _log(f"Starting streaming read (batch_size={batch_size:,})")

            batch_num = 0
            total_rows = 0

            for batch in pl.read_database(
                query=query,
                connection=streaming_conn,
                iter_batches=True,
                batch_size=batch_size,
                schema_overrides=schema_overrides,
                infer_schema_length=0,
            ):
                batch_num += 1
                total_rows += len(batch)
                _log(f"Read batch {batch_num}: {len(batch):,} rows ({total_rows:,} total)")
                yield batch

            _log(f"Read completed: {total_rows:,} rows in {batch_num} batches")
    finally:
        if isinstance(connection, sa.engine.Engine) and hasattr(conn_ctx, "close"):
            conn_ctx.close()


def _read_batched_offset(
    query: str,
    connection: psycopg.Connection,
    batch_size: int,
    order_by: str | list[str],
    context: OpExecutionContext | None,
) -> Iterator[pl.DataFrame]:
    """Internal: offset method using LIMIT/OFFSET pagination."""

    def _log(msg: str) -> None:
        if context:
            context.log.info(msg)

    if isinstance(order_by, str):
        order_by = [order_by]

    order_clause = ", ".join(order_by)

    # Register type adapters so psycopg2 returns UUIDs and JSON as strings
    PostgresPolarsSchema.register_uuid_adapter(connection)
    PostgresPolarsSchema.register_json_adapters(connection)

    # Probe cursor description for deterministic column types
    schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(connection, query)
    if schema_overrides:
        _log(f"Schema overrides: {len(schema_overrides)} columns from cursor description")

    count_query = f"SELECT COUNT(*) as cnt FROM ({query}) AS subq"  # noqa: S608
    total_rows: int = pl.read_database(count_query, connection)["cnt"][0]

    _log(f"Starting offset read: {total_rows:,} rows (batch_size={batch_size:,})")

    batch_num = 0
    rows_fetched = 0

    for offset in range(0, total_rows, batch_size):
        batch_num += 1

        paginated_query = (  # noqa: S608
            f"SELECT * FROM ({query}) AS subq"
            f" ORDER BY {order_clause}"
            f" LIMIT {batch_size} OFFSET {offset}"
        )

        batch = pl.read_database(
            paginated_query,
            connection,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )
        rows_fetched += len(batch)

        _log(
            f"Read batch {batch_num}: rows {offset:,}-{offset + len(batch):,}"
            f" ({rows_fetched:,}/{total_rows:,})"
        )

        yield batch

        if len(batch) < batch_size:
            break

    _log(f"Read completed: {rows_fetched:,} rows in {batch_num} batches")


def read_batched(
    query: str,
    connection: PostgresResource | sa.engine.Engine | sa.engine.Connection | psycopg.Connection,
    *,
    batch_size: int = 50_000,
    order_by: str | list[str] | None = None,
    method: Literal["streaming", "offset"] = "streaming",
    context: OpExecutionContext | None = None,
) -> Iterator[pl.DataFrame]:
    """Read large database tables in memory-efficient batches.

    Provides two methods for batched reading:

    - **streaming** (default): Uses server-side cursors via SQLAlchemy for true
      streaming. Most memory-efficient, single query, consistent snapshot.

    - **offset**: Uses LIMIT/OFFSET pagination. Works with raw psycopg2 connections
      but requires ORDER BY for consistency, executes multiple queries, and has
      O(n) performance degradation for large offsets.

    Args:
        query: SQL SELECT query to execute. For offset method, should NOT include
            LIMIT/OFFSET clauses (they will be added automatically).
        connection: Database connection. Can be:
            - PostgresResource: Will use appropriate method automatically
            - SQLAlchemy Engine or Connection: Uses streaming method
            - psycopg2 connection: Falls back to offset method
        batch_size: Number of rows per batch. Default 50,000. Reduce for wide
            tables (many columns) or increase for narrow tables.
        order_by: Column(s) to order by. Required for offset method to ensure
            consistent ordering across batches. Should be indexed columns.
            Ignored for streaming method.
        method: Batching method to use. "streaming" (default) or "offset".
        context: Optional Dagster context for progress logging.

    Yields:
        pl.DataFrame for each batch of rows.

    Raises:
        ValueError: If offset method is used without order_by.

    Example:
        ```python
        @asset
        def my_asset(context, database: PostgresResource) -> pl.DataFrame:
            chunks = []
            for batch in read_batched(
                "SELECT * FROM large_table",
                database,
                batch_size=50_000,
                context=context,
            ):
                chunks.append(batch)
            return pl.concat(chunks)
        ```
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # Handle PostgresResource
    if isinstance(connection, PostgresResource):
        if method == "streaming":
            engine = connection.get_engine()
            yield from _read_batched_streaming(query, engine, batch_size, context)
        else:
            if order_by is None:
                raise ValueError("order_by is required for offset method")
            with connection.get_connection() as conn:
                yield from _read_batched_offset(query, conn, batch_size, order_by, context)
        return

    # Handle SQLAlchemy Engine/Connection
    if isinstance(connection, (sa.engine.Engine, sa.engine.Connection)):
        if method == "offset":
            raise ValueError("offset method requires psycopg connection, not SQLAlchemy")
        yield from _read_batched_streaming(query, connection, batch_size, context)
        return

    # Handle raw psycopg connection
    if method == "streaming":
        raise ValueError(
            "streaming method requires SQLAlchemy connection. "
            "Use database.get_engine() or method='offset'"
        )
    if order_by is None:
        raise ValueError("order_by is required for offset method")
    yield from _read_batched_offset(query, connection, batch_size, order_by, context)


def _empty_frame_with_schema_sa(
    probe: str, original_query: str, conn: sa.engine.Connection
) -> pl.DataFrame:
    """Read a zero-row frame over a SQLAlchemy connection, schema preserved.

    ``probe`` (a ``LIMIT 0`` wrap) is what executes; ``original_query`` is only
    used to derive ``schema_overrides`` from the cursor description.
    """
    PostgresPolarsSchema.register_uuid_adapter_sa(conn)
    PostgresPolarsSchema.register_json_adapters_sa(conn)
    schema_overrides = PostgresPolarsSchema.from_sa_connection(conn, original_query)
    return pl.read_database(
        query=probe,
        connection=conn,
        schema_overrides=schema_overrides,
        infer_schema_length=0,
    )


def _empty_frame_with_schema(
    query: str,
    connection: PostgresResource | sa.engine.Engine | sa.engine.Connection | psycopg.Connection,
) -> pl.DataFrame:
    """Build a schema-aware empty DataFrame for *query* (#358).

    When a batched read returns no rows, ``read_batched`` yields no DataFrames
    at all, so a naive ``pl.DataFrame()`` would drop the column set the cursor
    description still carries -- breaking downstream ``.select`` / ``.join`` on
    named columns with ``ColumnNotFoundError`` even though the schema was
    available at read time.

    This probes the result schema with a ``LIMIT 0`` read, reusing the same
    type adapters and ``schema_overrides`` as the warm path so the empty frame's
    columns and dtypes match what a populated read would have produced.

    Falls back to a bare ``pl.DataFrame()`` if the probe fails, preserving the
    pre-#358 behavior rather than raising on the empty path.
    """
    probe = f"SELECT * FROM ({query}) AS _empty_probe LIMIT 0"  # noqa: S608
    try:
        if isinstance(connection, PostgresResource):
            engine = connection.get_engine()
            with engine.connect() as conn:
                return _empty_frame_with_schema_sa(probe, query, conn)
        if isinstance(connection, sa.engine.Engine):
            with connection.connect() as conn:
                return _empty_frame_with_schema_sa(probe, query, conn)
        if isinstance(connection, sa.engine.Connection):
            # Reuses the caller's connection rather than opening a fresh one.
            # By the time we get here the streaming read has fully drained its
            # (zero-row) cursor, and pl.read_database resets cursor state
            # between reads, so the probe runs cleanly even if the connection
            # was opened with stream_results=True (get_streaming_connection()).
            return _empty_frame_with_schema_sa(probe, query, connection)
        # Raw psycopg connection (offset path).
        PostgresPolarsSchema.register_uuid_adapter(connection)
        PostgresPolarsSchema.register_json_adapters(connection)
        schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(connection, query)
        return pl.read_database(
            query=probe,
            connection=connection,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )
    except Exception as exc:
        # Intentional broad catch: a probe failure must not turn an empty-result
        # read into an error (that would regress the pre-#358 contract). But a
        # silent column-less frame at a DB boundary is an observability gap, so
        # surface it as a warning before falling back.
        logging.getLogger("moncpipelib.resources").warning(
            "Empty-read schema probe failed for query %r: %s; returning column-less DataFrame",
            query[:200],
            exc,
        )
        return pl.DataFrame()


def read_batched_to_dataframe(
    query: str,
    connection: PostgresResource | sa.engine.Engine | sa.engine.Connection | psycopg.Connection,
    *,
    batch_size: int = 50_000,
    order_by: str | list[str] | None = None,
    method: Literal["streaming", "offset"] = "streaming",
    context: OpExecutionContext | None = None,
) -> pl.DataFrame:
    """Read large database table and return as single DataFrame.

    Convenience wrapper around read_batched() that concatenates all batches.

    See read_batched() for full documentation.

    When the query returns no rows, the result is a zero-row frame that still
    carries the query's column schema (probed from the cursor description),
    not a column-less ``pl.DataFrame()`` -- so downstream ``.select`` / ``.join``
    on named columns work against an unpopulated source (#358).

    Returns:
        pl.DataFrame containing all rows from the query.

    Example:
        ```python
        @asset
        def my_asset(context, database: PostgresResource) -> pl.DataFrame:
            return read_batched_to_dataframe(
                "SELECT * FROM large_table",
                database,
                batch_size=50_000,
                context=context,
            )
        ```
    """
    chunks = list(
        read_batched(
            query,
            connection,
            batch_size=batch_size,
            order_by=order_by,
            method=method,
            context=context,
        )
    )
    if not chunks:
        return _empty_frame_with_schema(query, connection)
    return pl.concat(chunks)
