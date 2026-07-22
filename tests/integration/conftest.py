"""Integration test fixtures for moncpipelib.

Provides session-scoped PostgreSQL and Azurite testcontainers and reusable
fixtures for creating SCD2 target tables, verifying database state,
constructing PostgresIOManager instances, and exercising
BlobStorageResource against a real (emulated) Azure surface.

Requires Docker. Install deps: uv sync --extra dev --extra integration
Run tests: uv run pytest -m integration -v
"""

from __future__ import annotations

import contextlib
import re
import uuid
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest
from azure.storage.blob import BlobServiceClient
from testcontainers.azurite import AzuriteContainer  # type: ignore[import-untyped]
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from moncpipelib.io_managers.postgres import PostgresIOManager
from moncpipelib.resources.blob import BlobStorageResource
from moncpipelib.resources.postgres import PostgresResource

# ---------------------------------------------------------------------------
# Session-scoped: one container per test session (~2-3s startup)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Start a PostgreSQL 16 container for the entire test session."""
    with PostgresContainer(
        image="postgres:16-alpine",
        username="test_user",
        password="test_password",
        dbname="test_moncpipelib",
    ) as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_connection_params(
    postgres_container: PostgresContainer,
) -> dict[str, Any]:
    """Extract connection parameters from the running container."""
    return {
        "host": postgres_container.get_container_host_ip(),
        "port": int(postgres_container.get_exposed_port(5432)),
        "user": "test_user",
        "password": "test_password",
        "dbname": "test_moncpipelib",
        "sslmode": "disable",
    }


# ---------------------------------------------------------------------------
# Session-scoped: create the test schema once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _create_test_schemas(pg_connection_params: dict[str, Any]) -> None:
    """Create schemas used by integration tests."""
    conn = psycopg.connect(**pg_connection_params)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS test_scd2")
            cur.execute("CREATE SCHEMA IF NOT EXISTS test_write")
            cur.execute("CREATE SCHEMA IF NOT EXISTS test_lineage")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Function-scoped: fresh connection per test
# ---------------------------------------------------------------------------


@pytest.fixture()
def pg_connection(
    pg_connection_params: dict[str, Any],
) -> Generator[psycopg.Connection, None, None]:
    """Provide a fresh database connection for each test. Auto-closed after."""
    conn = psycopg.connect(**pg_connection_params)
    conn.autocommit = False
    yield conn
    conn.close()


@pytest.fixture()
def pg_cursor(
    pg_connection: psycopg.Connection,
) -> Generator[psycopg.Cursor, None, None]:
    """Provide a cursor that commits on teardown."""
    cursor = pg_connection.cursor()
    yield cursor
    pg_connection.commit()
    cursor.close()


# ---------------------------------------------------------------------------
# SCD2 table creation helper
# ---------------------------------------------------------------------------


class SCD2TableBuilder:
    """Helper for creating SCD2-compatible target tables."""

    def __init__(
        self,
        conn: psycopg.Connection,
        schema: str = "test_scd2",
    ) -> None:
        self._conn = conn
        self._schema = schema

    def create_table(
        self,
        table_name: str,
        business_key_columns: dict[str, str],
        tracked_columns: dict[str, str],
        *,
        effective_from_col: str = "effective_from",
        effective_to_col: str = "effective_to",
        is_current_col: str = "is_current",
        hash_col: str = "row_hash",
        include_identity: bool = True,
        include_unique_index: bool = True,
        extra_columns: dict[str, str] | None = None,
    ) -> str:
        """Create an SCD2 target table and return its fully-qualified name.

        Args:
            table_name: Table name (without schema).
            business_key_columns: {col_name: pg_type} for business key cols.
            tracked_columns: {col_name: pg_type} for tracked data cols.
            effective_from_col: Temporal column name.
            effective_to_col: Temporal column name.
            is_current_col: Current flag column name.
            hash_col: Hash column name.
            include_identity: Add BIGINT GENERATED ALWAYS AS IDENTITY pk.
            include_unique_index: Create partial unique index on business key.
            extra_columns: Additional columns to add.

        Returns:
            Fully-qualified table name: schema.table_name
        """
        fqn = f"{self._schema}.{table_name}"

        col_defs: list[str] = []
        if include_identity:
            col_defs.append("id BIGINT GENERATED ALWAYS AS IDENTITY")

        for col, dtype in business_key_columns.items():
            col_defs.append(f'"{col}" {dtype} NOT NULL')

        for col, dtype in tracked_columns.items():
            col_defs.append(f'"{col}" {dtype}')

        if extra_columns:
            for col, dtype in extra_columns.items():
                col_defs.append(f'"{col}" {dtype}')

        col_defs.append(f'"{hash_col}" TEXT NOT NULL')
        col_defs.append(f'"{effective_from_col}" TIMESTAMPTZ NOT NULL DEFAULT now()')
        col_defs.append(f'"{effective_to_col}" TIMESTAMPTZ')
        col_defs.append(f'"{is_current_col}" BOOLEAN NOT NULL DEFAULT true')

        ddl = f"CREATE TABLE IF NOT EXISTS {fqn} ({', '.join(col_defs)})"

        with self._conn.cursor() as cur:
            cur.execute(ddl)

            if include_unique_index:
                bk_cols = ", ".join(f'"{c}"' for c in business_key_columns)
                idx_name = f"uq_{table_name}_current"
                cur.execute(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {idx_name} "
                    f'ON {fqn} ({bk_cols}) WHERE ("{is_current_col}")'
                )

        self._conn.commit()
        return fqn

    def truncate(self, fqn: str) -> None:
        """TRUNCATE a table. Rolls back any failed transaction first."""
        if self._conn.closed:
            return
        self._conn.rollback()
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE {fqn} RESTART IDENTITY CASCADE")
        self._conn.commit()

    def drop(self, fqn: str) -> None:
        """DROP a table. Rolls back any failed transaction first."""
        if self._conn.closed:
            return
        self._conn.rollback()
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {fqn} CASCADE")
        self._conn.commit()


@pytest.fixture()
def scd2_table_builder(
    pg_connection: psycopg.Connection,
) -> SCD2TableBuilder:
    """Provide an SCD2TableBuilder bound to the current test connection."""
    return SCD2TableBuilder(pg_connection, schema="test_scd2")


# ---------------------------------------------------------------------------
# SCD2 database state verifier
# ---------------------------------------------------------------------------


class SCD2Verifier:
    """Helper for asserting SCD2 table state after writes."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def count_current(self, fqn: str, is_current_col: str = "is_current") -> int:
        """Count rows where is_current = true."""
        with self._conn.cursor() as cur:
            cur.execute(f'SELECT count(*) FROM {fqn} WHERE "{is_current_col}" = true')
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    def count_expired(self, fqn: str, is_current_col: str = "is_current") -> int:
        """Count rows where is_current = false."""
        with self._conn.cursor() as cur:
            cur.execute(f'SELECT count(*) FROM {fqn} WHERE "{is_current_col}" = false')
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    def count_total(self, fqn: str) -> int:
        """Count all rows."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {fqn}")
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    def get_current_row(
        self,
        fqn: str,
        business_key_col: str,
        business_key_val: str,
        is_current_col: str = "is_current",
    ) -> dict[str, Any] | None:
        """Fetch the current version of a business entity."""
        with self._conn.cursor() as cur:
            cur.execute(
                f'SELECT * FROM {fqn} WHERE "{business_key_col}" = %s '
                f'AND "{is_current_col}" = true',
                (business_key_val,),
            )
            cols = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            return dict(zip(cols, row, strict=True)) if row else None

    def get_history(
        self,
        fqn: str,
        business_key_col: str,
        business_key_val: str,
        effective_from_col: str = "effective_from",
    ) -> list[dict[str, Any]]:
        """Fetch all versions of a business entity, ordered by effective_from."""
        with self._conn.cursor() as cur:
            cur.execute(
                f'SELECT * FROM {fqn} WHERE "{business_key_col}" = %s '
                f'ORDER BY "{effective_from_col}"',
                (business_key_val,),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def assert_temporal_chain(
        self,
        fqn: str,
        business_key_col: str,
        business_key_val: str,
        *,
        effective_from_col: str = "effective_from",
        effective_to_col: str = "effective_to",
        is_current_col: str = "is_current",
    ) -> None:
        """Assert that the temporal chain for a business key is valid.

        Validates:
        - Exactly one row has is_current=true and effective_to IS NULL
        - All expired rows have effective_to IS NOT NULL
        - effective_to[i] == effective_from[i+1] for consecutive versions
        """
        history = self.get_history(fqn, business_key_col, business_key_val, effective_from_col)
        assert len(history) > 0, f"No rows found for {business_key_val}"

        current_rows = [r for r in history if r[is_current_col]]
        assert len(current_rows) == 1, (
            f"Expected exactly 1 current row for {business_key_val}, found {len(current_rows)}"
        )

        from datetime import date

        from moncpipelib.config import SCD2_DEFAULTS

        end_of_time = date.fromisoformat(SCD2_DEFAULTS["end_of_time"])

        current = current_rows[0]
        eff_to = current[effective_to_col]
        # Current row should have the end-of-time sentinel (or None for legacy data)
        is_sentinel = eff_to is not None and (
            (hasattr(eff_to, "date") and eff_to.date() == end_of_time) or eff_to == end_of_time
        )
        assert eff_to is None or is_sentinel, (
            f"Current row effective_to should be {end_of_time} or None, got {eff_to}"
        )

        expired = [r for r in history if not r[is_current_col]]
        for r in expired:
            assert r[effective_to_col] is not None

        # Verify consecutive version timestamps match
        for i in range(len(history) - 1):
            if history[i][effective_to_col] is not None:
                assert history[i][effective_to_col] == history[i + 1][effective_from_col], (
                    f"Temporal gap for {business_key_val}: "
                    f"row {i} {effective_to_col}={history[i][effective_to_col]} != "
                    f"row {i + 1} {effective_from_col}={history[i + 1][effective_from_col]}"
                )


@pytest.fixture()
def scd2_verifier(
    pg_connection: psycopg.Connection,
) -> SCD2Verifier:
    """Provide an SCD2Verifier bound to the current test connection."""
    return SCD2Verifier(pg_connection)


# ---------------------------------------------------------------------------
# IO Manager factory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres_resource(
    pg_connection_params: dict[str, Any],
) -> PostgresResource:
    """Create a PostgresResource pointed at the test container."""
    return PostgresResource(
        host=pg_connection_params["host"],
        port=pg_connection_params["port"],
        user=pg_connection_params["user"],
        password=pg_connection_params["password"],
        database=pg_connection_params["dbname"],
        sslmode="disable",
        enable_row_lineage=False,
        add_metadata_columns=False,
    )


@pytest.fixture()
def io_manager_factory(
    postgres_resource: PostgresResource,
) -> Callable[..., PostgresIOManager]:
    """Factory that creates PostgresIOManager instances pointed at the test container.

    Returns a callable accepting keyword overrides for IO manager fields.
    Connection, lineage, and metadata column settings come from the shared
    ``postgres_resource`` fixture.
    """

    def _factory(**kwargs: Any) -> PostgresIOManager:
        # Allow overriding the resource itself for tests that need different settings
        resource = kwargs.pop("postgres_resource", None) or postgres_resource

        # Allow overriding resource-level fields by creating a new resource
        resource_overrides = {}
        for key in (
            "enable_row_lineage",
            "add_metadata_columns",
            "full_refresh_method",
            "full_refresh_threshold",
            "bulk_insert_method",
            "bulk_insert_threshold",
            "insert_chunk_size",
        ):
            if key in kwargs:
                resource_overrides[key] = kwargs.pop(key)

        if resource_overrides:
            resource = resource.model_copy(update=resource_overrides)

        defaults: dict[str, Any] = {
            "postgres_resource": resource,
            "db_schema": "test_scd2",
        }
        defaults.update(kwargs)
        return PostgresIOManager(**defaults)

    return _factory


# ---------------------------------------------------------------------------
# Mock Dagster OutputContext helper
# ---------------------------------------------------------------------------


def make_mock_output_context(
    asset_name: str = "dim_product",
    run_id: str = "integration-test-run",
    metadata: dict[str, Any] | None = None,
    partition_key: str | None = None,
) -> MagicMock:
    """Create a mock Dagster OutputContext for integration tests.

    Only the minimal interface used by handle_output is mocked:
    - context.asset_key.to_user_string() -> asset_name
    - context.asset_key.path -> [asset_name]
    - context.run_id -> run_id
    - context.metadata -> metadata dict
    - context.log -> MagicMock
    - context.add_output_metadata -> MagicMock
    - context.has_partition_key / partition_key / asset_partition_keys:
      driven by ``partition_key`` (defaults to non-partitioned).
    """
    context = MagicMock()
    context.asset_key.to_user_string.return_value = asset_name
    context.asset_key.path = [asset_name]
    context.run_id = run_id
    context.metadata = metadata or {}
    context.log = MagicMock()
    context.add_output_metadata = MagicMock()
    if partition_key is None:
        context.has_partition_key = False
    else:
        context.has_partition_key = True
        context.partition_key = partition_key
        context.asset_partition_keys = [partition_key]
    # Migration 018 Phase 1 reads ``context.run.backfill_id`` and
    # ``context.run.tags``. Pin them to real ``None`` / ``{}`` so the
    # backfill-signal extractor sees deterministic values instead of
    # auto-generated child mocks (which would otherwise surface as
    # MagicMock-typed metadata payloads downstream).
    context.run.backfill_id = None
    context.run.tags = {}
    return context


# ---------------------------------------------------------------------------
# Generic table builder (non-SCD2)
# ---------------------------------------------------------------------------


class TableBuilder:
    """Helper for creating generic test tables."""

    def __init__(
        self,
        conn: psycopg.Connection,
        schema: str = "test_write",
    ) -> None:
        self._conn = conn
        self._schema = schema

    def create_table(
        self,
        name: str,
        columns: dict[str, str],
        *,
        primary_key: list[str] | None = None,
    ) -> str:
        """Create a table and return its fully-qualified name.

        Args:
            name: Table name (without schema).
            columns: {col_name: pg_type} mapping.
            primary_key: Columns for PRIMARY KEY constraint.

        Returns:
            Fully-qualified table name: schema.name
        """
        fqn = f"{self._schema}.{name}"
        col_defs = [f'"{col}" {dtype}' for col, dtype in columns.items()]

        if primary_key:
            pk_cols = ", ".join(f'"{c}"' for c in primary_key)
            col_defs.append(f"PRIMARY KEY ({pk_cols})")

        ddl = f"CREATE TABLE IF NOT EXISTS {fqn} ({', '.join(col_defs)})"
        with self._conn.cursor() as cur:
            cur.execute(ddl)
        self._conn.commit()
        return fqn

    def insert_rows(
        self,
        fqn: str,
        columns: list[str],
        rows: list[tuple[Any, ...]],
    ) -> None:
        """Insert rows into a table for pre-populating test state."""
        if not rows:
            return
        cols_str = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f"INSERT INTO {fqn} ({cols_str}) VALUES ({placeholders})"
        with self._conn.cursor() as cur:
            for row in rows:
                cur.execute(sql, row)
        self._conn.commit()

    def count(self, fqn: str) -> int:
        """Count all rows in a table."""
        self._conn.rollback()
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {fqn}")
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    def read_all(self, fqn: str, order_by: str | None = None) -> list[dict[str, Any]]:
        """Read all rows from a table as dicts."""
        self._conn.rollback()
        query = f"SELECT * FROM {fqn}"
        if order_by:
            query += f' ORDER BY "{order_by}"'
        with self._conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def drop(self, fqn: str) -> None:
        """DROP a table."""
        if self._conn.closed:
            return
        self._conn.rollback()
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {fqn} CASCADE")
        self._conn.commit()

    def truncate(self, fqn: str) -> None:
        """TRUNCATE a table."""
        if self._conn.closed:
            return
        self._conn.rollback()
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE {fqn} RESTART IDENTITY CASCADE")
        self._conn.commit()

    def analyze(self, fqn: str) -> None:
        """ANALYZE a table so ``pg_class.reltuples`` reflects its contents.

        Freshly created tables report ``reltuples = -1`` ("never analyzed"),
        which the AUTO full-refresh fallback deliberately treats as "no
        estimate available" rather than "empty" (#4). Seeding rows is
        therefore not enough on its own to exercise that path.
        """
        if self._conn.closed:
            return
        self._conn.rollback()
        with self._conn.cursor() as cur:
            cur.execute(f"ANALYZE {fqn}")
        self._conn.commit()


@pytest.fixture()
def table_builder(
    pg_connection: psycopg.Connection,
) -> TableBuilder:
    """Provide a TableBuilder bound to the current test connection."""
    return TableBuilder(pg_connection, schema="test_write")


# ---------------------------------------------------------------------------
# Azurite (Azure Blob Storage emulator)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def azurite_container() -> Generator[AzuriteContainer, None, None]:
    """Start an Azurite container for the entire test session.

    Azurite is the Microsoft-supported local emulator for Azure Storage.
    It uses the well-known development account credentials -- these are
    hard-coded into the emulator and have no production analog, so the
    connection string never leaves the test process.

    ``--skipApiVersionCheck`` is required because the Azure SDK ships
    bleeding-edge ``x-ms-version`` headers that frequently outpace the
    Azurite ``latest`` image. Version-skew compatibility is not what
    these tests cover -- they exercise the resource's read/write logic.
    """
    azurite = AzuriteContainer().with_command(
        "azurite --blobHost 0.0.0.0 --queueHost 0.0.0.0 --tableHost 0.0.0.0 --skipApiVersionCheck"
    )
    with azurite as started:
        yield started


@pytest.fixture(scope="session")
def azurite_connection_string(azurite_container: AzuriteContainer) -> str:
    """Return the LOCALHOST connection string for the running Azurite."""
    return azurite_container.get_connection_string()


@pytest.fixture()
def blob_container_name(
    azurite_connection_string: str,
) -> Generator[str, None, None]:
    """Create a fresh blob container for the test and tear it down after.

    Yields a unique container name so tests are fully isolated even when
    they run in parallel against the same Azurite instance.
    """
    name = f"test-{uuid.uuid4().hex[:12]}"
    service = BlobServiceClient.from_connection_string(azurite_connection_string)
    service.create_container(name)
    try:
        yield name
    finally:
        with contextlib.suppress(Exception):
            service.delete_container(name)


@pytest.fixture()
def blob_resource(
    azurite_connection_string: str,
    blob_container_name: str,
) -> BlobStorageResource:
    """Provide a BlobStorageResource wired to the per-test Azurite container.

    The resource is constructed with the production code path
    (``BlobStorageResource(...)``), then ``_service_client`` is replaced
    with one built from the Azurite connection string. ``_credential`` is
    set to a sentinel because Azurite uses shared-key auth, not AAD --
    bypassing ``setup_for_execution`` keeps the production code path
    unchanged (workload identity only) while still exercising every
    other method on the resource end-to-end.
    """
    resource = BlobStorageResource(
        storage_account="devstoreaccount1",
        container_public=blob_container_name,
    )
    resource._credential = MagicMock(name="UnusedCredential")  # noqa: SLF001
    resource._service_client = BlobServiceClient.from_connection_string(  # noqa: SLF001
        azurite_connection_string
    )
    return resource


# ---------------------------------------------------------------------------
# EXPLAIN plan-shape capture (Phase 2 of migration 015 / issue #274)
# ---------------------------------------------------------------------------

# PostgreSQL's EXPLAIN supports only a fixed set of statements: SELECT,
# INSERT, UPDATE, DELETE, MERGE, VALUES, EXECUTE, DECLARE, CREATE TABLE AS,
# and CREATE MATERIALIZED VIEW AS.  Other CREATE TABLE variants (e.g.
# ``CREATE TEMP TABLE foo (LIKE bar INCLUDING DEFAULTS)`` used by the
# staging-table builder) are not EXPLAIN-able and produce a syntax error
# if prefixed.  This regex matches the CTAS form -- ``CREATE [TEMP] TABLE
# ... AS SELECT ...`` -- and lets the wrapper skip the LIKE form.
_CTAS_RE = re.compile(r"CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\b.*\bAS\b", re.IGNORECASE | re.DOTALL)
# Matches both bare ``DELETE FROM`` and CTE-prefixed ``WITH ... DELETE FROM``
# forms. Anchored at start (after lstrip) so it does not match DELETEs that
# appear inside string literals or comments. Used to capture the plan shape
# of ``reconcile_scd2``'s collapse DELETE (#277).
_DELETE_RE = re.compile(
    r"^(?:WITH\b.*?\bDELETE\s+FROM\b|DELETE\s+FROM\b)",
    re.IGNORECASE | re.DOTALL,
)


class ExplainCapturingCursor:
    """Cursor wrapper that captures ``EXPLAIN (FORMAT JSON)`` plan trees.

    For each ``execute()`` call whose SQL is a CTAS, ``INSERT INTO``, or
    ``DELETE FROM`` (including ``WITH ... DELETE FROM`` CTE-prefixed forms),
    the wrapper first runs an ``EXPLAIN (FORMAT JSON)`` of the same
    SQL+params to capture the plan tree, then proceeds with the real
    execute.  ``EXPLAIN`` without ``ANALYZE`` is non-destructive on
    PostgreSQL (it returns the plan but does not execute the statement),
    so wrapping does not double-write.

    Used by SCD2 self-modification tests (#274) to assert Stage 2 INSERTs
    have no JOIN node, and by SCD2 reconcile tests (#277) to assert the
    collapse DELETE has no ``SubPlan`` node.
    """

    def __init__(self, real_cursor: psycopg.Cursor) -> None:
        self._cursor = real_cursor
        self.plans: list[tuple[str, list[dict[str, Any]]]] = []

    def execute(self, sql: Any, params: Any = None) -> Any:
        # Pre-#312 the writer code emitted only str SQL; #312 introduced
        # ``psycopg.sql.Composed`` for ANALYZE statements (safe identifier
        # quoting). The wrapper's job is to capture CTAS / INSERT / DELETE
        # plans -- ANALYZE is not in that set, so we just pass non-str
        # SQL straight through without inspecting it.
        if not isinstance(sql, str):
            return self._cursor.execute(sql, params)

        stripped = sql.lstrip()
        is_ctas = _CTAS_RE.search(stripped) is not None
        is_insert = stripped.upper().startswith("INSERT INTO")
        is_delete = _DELETE_RE.search(stripped) is not None
        if is_ctas or is_insert or is_delete:
            self._cursor.execute("EXPLAIN (FORMAT JSON) " + sql, params)
            row = self._cursor.fetchone()
            if row is not None:
                self.plans.append((sql, row[0]))
        return self._cursor.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


def collect_node_types(plan_node: dict[str, Any]) -> list[str]:
    """Walk a Postgres EXPLAIN (FORMAT JSON) plan tree, returning every
    ``Node Type`` value (root + descendants) in document order."""
    types: list[str] = [plan_node["Node Type"]]
    for child in plan_node.get("Plans", []):
        types.extend(collect_node_types(child))
    return types
