"""Integration tests for UUID-to-Polars-String type inference.

Validates that PostgreSQL UUID columns are correctly read as ``pl.String``
(not ``pl.Object``) across all read paths:
- Direct ``pl.read_database`` with the UUID adapter registered
- ``PostgresIOManager.load_input``
- ``read_batched`` (streaming and offset methods)
- Lineage ID extraction via ``get_parent_lineage_ids``

Without the psycopg2 UUID-to-string adapter, psycopg2 returns ``uuid.UUID``
Python objects that Polars stores as ``pl.Object`` -- a dtype that cannot be
cast to ``pl.String`` and that ``schema_overrides`` alone cannot fix.

Requires Docker.  Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import psycopg
import pytest

from moncpipelib.config import LineageDefaults
from moncpipelib.lineage.tracker import LineageTracker
from moncpipelib.resources.postgres import (
    PostgresPolarsSchema,
    PostgresResource,
    read_batched,
    read_batched_to_dataframe,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TEST_SCHEMA = "test_uuid"
TABLE_NAME = f"{TEST_SCHEMA}.uuid_test"
LINEAGE_TABLE = f"{TEST_SCHEMA}.uuid_lineage_test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _create_uuid_schema(pg_connection_params: dict[str, Any]) -> None:
    """Create the test schema and seed table with UUID data."""
    conn = psycopg.connect(**pg_connection_params)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA}")

            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
            cur.execute(
                f"""
                CREATE TABLE {TABLE_NAME} (
                    id UUID DEFAULT gen_random_uuid(),
                    name TEXT NOT NULL,
                    value INT NOT NULL
                )
                """
            )

            # Insert rows with explicit UUIDs so we can assert on them
            cur.execute(
                f"""
                INSERT INTO {TABLE_NAME} (id, name, value) VALUES
                    ('11111111-1111-1111-1111-111111111111', 'alpha', 10),
                    ('22222222-2222-2222-2222-222222222222', 'bravo', 20),
                    ('33333333-3333-3333-3333-333333333333', 'charlie', 30)
                """
            )

            # Table with lineage columns (simulates a table written by the IO manager)
            cur.execute(f"DROP TABLE IF EXISTS {LINEAGE_TABLE}")
            cur.execute(
                f"""
                CREATE TABLE {LINEAGE_TABLE} (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    "{LineageDefaults.ID_COLUMN}" UUID NOT NULL,
                    "{LineageDefaults.KEY_COLUMN}" TEXT NOT NULL
                )
                """
            )

            lineage_id = str(uuid.uuid4())
            cur.execute(
                f"""
                INSERT INTO {LINEAGE_TABLE} (name, "{LineageDefaults.ID_COLUMN}", "{LineageDefaults.KEY_COLUMN}") VALUES
                    ('row1', '{lineage_id}', 'key-1'),
                    ('row2', '{lineage_id}', 'key-1'),
                    ('row3', '{lineage_id}', 'key-1')
                """
            )
    finally:
        conn.close()


@pytest.fixture()
def raw_conn(pg_connection_params: dict[str, Any]) -> psycopg.Connection:
    """Provide a fresh driver-native connection (no UUID adapter registered).

    Routes through the driver seam so the same fixture works under both
    ``MONC_PG_DRIVER=psycopg2`` and ``MONC_PG_DRIVER=psycopg3``.
    """
    conn = psycopg.connect(**pg_connection_params)
    yield conn  # type: ignore[misc]
    conn.close()


@pytest.fixture()
def pg_resource(pg_connection_params: dict[str, Any]) -> PostgresResource:
    """Create a PostgresResource pointed at the test container."""
    return PostgresResource(
        host=pg_connection_params["host"],
        port=pg_connection_params["port"],
        user=pg_connection_params["user"],
        password=pg_connection_params["password"],
        database=pg_connection_params["dbname"],
        sslmode="disable",
    )


# ---------------------------------------------------------------------------
# Test: Baseline -- uuid.UUID Python objects in plain DataFrames
# ---------------------------------------------------------------------------


class TestUUIDBehaviorPythonObjects:
    """Verify that uuid.UUID Python objects become pl.Object in manual DataFrames.

    This demonstrates the core problem: when code constructs a DataFrame from
    Python objects that include uuid.UUID (as psycopg2 returns by default in
    some environments or when using cursor.fetchall() directly), Polars stores
    them as pl.Object -- a dtype that cannot be cast to pl.String.

    Polars' ``read_database`` may handle UUIDs correctly via OID-based
    inference in some versions, but the adapter provides a guaranteed fix at
    the driver level regardless of the Polars version or backend used.
    """

    def test_uuid_objects_become_pl_object(self) -> None:
        """uuid.UUID Python objects are stored as pl.Object."""
        uuids = [uuid.uuid4(), uuid.uuid4()]
        df = pl.DataFrame({"id": uuids})
        assert df.schema["id"] == pl.Object

    def test_pl_object_cannot_be_cast_to_string(self) -> None:
        """pl.Object columns containing uuid.UUID cannot be cast to pl.String."""
        uuids = [uuid.uuid4(), uuid.uuid4()]
        df = pl.DataFrame({"id": uuids})
        with pytest.raises(
            (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError),
            match="cannot cast",
        ):
            df.with_columns(pl.col("id").cast(pl.String))

    def test_string_uuids_are_pl_string(self) -> None:
        """String UUIDs (the adapter's output) are correctly pl.String."""
        str_uuids = [str(uuid.uuid4()), str(uuid.uuid4())]
        df = pl.DataFrame({"id": str_uuids})
        assert df.schema["id"] == pl.String


# ---------------------------------------------------------------------------
# Test: UUID adapter -- the fix
# ---------------------------------------------------------------------------


class TestUUIDAdapterRegistration:
    """Verify that register_uuid_adapter makes UUIDs arrive as pl.String."""

    def test_uuid_column_is_string_with_adapter(self, raw_conn: psycopg.Connection) -> None:
        """After registering the adapter, UUID columns are pl.String."""
        PostgresPolarsSchema.register_uuid_adapter(raw_conn)
        df = pl.read_database(f"SELECT * FROM {TABLE_NAME}", raw_conn)

        assert df.schema["id"] == pl.String

    def test_uuid_values_are_correct_strings(self, raw_conn: psycopg.Connection) -> None:
        """UUID values should be standard hyphenated lowercase strings."""
        PostgresPolarsSchema.register_uuid_adapter(raw_conn)
        df = pl.read_database(f"SELECT * FROM {TABLE_NAME} ORDER BY name", raw_conn)

        ids = df["id"].to_list()
        assert ids == [
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]

    def test_null_uuid_is_none(self, raw_conn: psycopg.Connection) -> None:
        """NULL UUID values should be None, not the string 'None'."""
        PostgresPolarsSchema.register_uuid_adapter(raw_conn)
        # Mix NULL and non-NULL so polars can infer the column type
        df = pl.read_database(
            f"SELECT id FROM {TABLE_NAME} UNION ALL SELECT NULL::uuid",
            raw_conn,
        )

        assert df.schema["id"] == pl.String
        null_rows = df.filter(pl.col("id").is_null())
        assert len(null_rows) == 1

    def test_adapter_idempotent(self, raw_conn: psycopg.Connection) -> None:
        """Calling register_uuid_adapter multiple times is safe."""
        PostgresPolarsSchema.register_uuid_adapter(raw_conn)
        PostgresPolarsSchema.register_uuid_adapter(raw_conn)

        df = pl.read_database(f"SELECT * FROM {TABLE_NAME}", raw_conn)
        assert df.schema["id"] == pl.String


# ---------------------------------------------------------------------------
# Test: SQLAlchemy adapter path
# ---------------------------------------------------------------------------


class TestUUIDAdapterSQLAlchemy:
    """Verify the SQLAlchemy adapter path (used by streaming reads)."""

    def test_register_uuid_adapter_sa(self, pg_resource: PostgresResource) -> None:
        """register_uuid_adapter_sa should work on SQLAlchemy connections."""
        engine = pg_resource.get_engine()
        with engine.connect() as sa_conn:
            PostgresPolarsSchema.register_uuid_adapter_sa(sa_conn)

            df = pl.read_database(f"SELECT * FROM {TABLE_NAME}", sa_conn)
            assert df.schema["id"] == pl.String


# ---------------------------------------------------------------------------
# Test: PostgresResource integration
# ---------------------------------------------------------------------------


class TestPostgresResourceUUID:
    """Verify PostgresResource methods automatically handle UUIDs."""

    def test_get_connection_registers_adapter(self, pg_resource: PostgresResource) -> None:
        """get_connection() should auto-register the UUID adapter."""
        with pg_resource.get_connection() as conn:
            df = pl.read_database(f"SELECT * FROM {TABLE_NAME}", conn)
            assert df.schema["id"] == pl.String

    def test_get_connection_raw_registers_adapter(self, pg_resource: PostgresResource) -> None:
        """get_connection_raw() should auto-register the UUID adapter."""
        conn = pg_resource.get_connection_raw()
        try:
            df = pl.read_database(f"SELECT * FROM {TABLE_NAME}", conn)
            assert df.schema["id"] == pl.String
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Test: Batched reads
# ---------------------------------------------------------------------------


class TestBatchedReadUUID:
    """Verify batched read utilities produce pl.String for UUID columns."""

    def test_streaming_read_uuid_is_string(self, pg_resource: PostgresResource) -> None:
        """Streaming batched read should produce pl.String for UUID columns."""
        batches = list(
            pg_resource.read_batched(
                f"SELECT * FROM {TABLE_NAME}",
                method="streaming",
                batch_size=2,
            )
        )
        assert len(batches) > 0

        for batch in batches:
            assert batch.schema["id"] == pl.String, f"Expected pl.String, got {batch.schema['id']}"
            # Values should be strings, not uuid.UUID
            for val in batch["id"].to_list():
                assert isinstance(val, str), f"Expected str, got {type(val)}"

    def test_offset_read_uuid_is_string(self, pg_resource: PostgresResource) -> None:
        """Offset batched read should produce pl.String for UUID columns."""
        batches = list(
            pg_resource.read_batched(
                f"SELECT * FROM {TABLE_NAME}",
                method="offset",
                order_by="name",
                batch_size=2,
            )
        )
        assert len(batches) > 0

        for batch in batches:
            assert batch.schema["id"] == pl.String

    def test_read_batched_to_dataframe_uuid_is_string(self, pg_resource: PostgresResource) -> None:
        """read_batched_to_dataframe should produce pl.String for UUID columns."""
        df = pg_resource.read_batched_to_dataframe(
            f"SELECT * FROM {TABLE_NAME}",
        )
        assert df.schema["id"] == pl.String
        assert len(df) == 3

    def test_module_level_read_batched_uuid_is_string(self, pg_resource: PostgresResource) -> None:
        """Module-level read_batched with engine should produce pl.String."""
        engine = pg_resource.get_engine()
        batches = list(
            read_batched(
                f"SELECT * FROM {TABLE_NAME}",
                engine,
                batch_size=10,
            )
        )
        assert len(batches) > 0
        assert batches[0].schema["id"] == pl.String

    def test_module_level_read_batched_to_dataframe(self, pg_resource: PostgresResource) -> None:
        """Module-level read_batched_to_dataframe should produce pl.String."""
        engine = pg_resource.get_engine()
        df = read_batched_to_dataframe(
            f"SELECT * FROM {TABLE_NAME}",
            engine,
        )
        assert df.schema["id"] == pl.String
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Test: IO Manager load_input
# ---------------------------------------------------------------------------


class TestIOManagerLoadInputUUID:
    """Verify PostgresIOManager.load_input reads UUIDs as pl.String."""

    def test_load_input_uuid_is_string(
        self,
        pg_connection_params: dict[str, Any],
    ) -> None:
        """load_input should produce pl.String for UUID columns."""
        from moncpipelib.io_managers.postgres import PostgresIOManager

        io_mgr = PostgresIOManager(
            postgres_resource=PostgresResource(
                host=pg_connection_params["host"],
                port=pg_connection_params["port"],
                user=pg_connection_params["user"],
                password=pg_connection_params["password"],
                database=pg_connection_params["dbname"],
                sslmode="disable",
                enable_row_lineage=False,
                add_metadata_columns=False,
            ),
            db_schema=TEST_SCHEMA,
        )

        # Build a mock InputContext that points at the test table
        context = MagicMock()
        context.asset_key.path = ["uuid_test"]
        context.upstream_output = None
        context.log = MagicMock()

        df = io_mgr.load_input(context)

        assert df.schema["id"] == pl.String
        assert len(df) == 3

        # Values should be proper UUID strings
        for val in df["id"].to_list():
            assert isinstance(val, str)
            uuid.UUID(val)  # Validates format

    def test_load_input_with_column_projection(
        self,
        pg_connection_params: dict[str, Any],
    ) -> None:
        """load_input with column projection should still produce pl.String for UUIDs."""
        from moncpipelib.io_managers.postgres import PostgresIOManager

        io_mgr = PostgresIOManager(
            postgres_resource=PostgresResource(
                host=pg_connection_params["host"],
                port=pg_connection_params["port"],
                user=pg_connection_params["user"],
                password=pg_connection_params["password"],
                database=pg_connection_params["dbname"],
                sslmode="disable",
                enable_row_lineage=False,
                add_metadata_columns=False,
            ),
            db_schema=TEST_SCHEMA,
        )

        context = MagicMock()
        context.asset_key.path = ["uuid_test"]
        context.upstream_output = MagicMock()
        context.upstream_output.metadata = {"columns": ["id", "name"]}
        context.log = MagicMock()

        df = io_mgr.load_input(context)

        assert df.schema["id"] == pl.String
        assert list(df.columns) == ["id", "name"]


# ---------------------------------------------------------------------------
# Test: Lineage ID extraction (the downstream consumer)
# ---------------------------------------------------------------------------


class TestLineageExtractionUUID:
    """Verify that lineage IDs can be extracted from UUID columns loaded by the IO manager."""

    def test_get_parent_lineage_ids_returns_strings(
        self,
        pg_connection_params: dict[str, Any],
    ) -> None:
        """get_parent_lineage_ids should return list[str] for UUID lineage columns."""
        from moncpipelib.io_managers.postgres import PostgresIOManager

        io_mgr = PostgresIOManager(
            postgres_resource=PostgresResource(
                host=pg_connection_params["host"],
                port=pg_connection_params["port"],
                user=pg_connection_params["user"],
                password=pg_connection_params["password"],
                database=pg_connection_params["dbname"],
                sslmode="disable",
                enable_row_lineage=False,
                add_metadata_columns=False,
            ),
            db_schema=TEST_SCHEMA,
        )

        # Load the table with lineage columns
        context = MagicMock()
        context.asset_key.path = ["uuid_lineage_test"]
        context.upstream_output = None
        context.log = MagicMock()

        df = io_mgr.load_input(context)

        # Lineage column should be pl.String, not pl.Object
        assert df.schema[LineageDefaults.ID_COLUMN] == pl.String

        # Extract parent lineage IDs (this is what fails without the fix)
        import sqlalchemy as sa

        engine = sa.create_engine(
            f"postgresql+psycopg://{pg_connection_params['user']}:{pg_connection_params['password']}"
            f"@{pg_connection_params['host']}:{pg_connection_params['port']}"
            f"/{pg_connection_params['dbname']}?sslmode=disable"
        )
        tracker = LineageTracker(engine)
        parent_ids = tracker.get_parent_lineage_ids(df)

        assert len(parent_ids) == 1  # All 3 rows share the same lineage_id
        assert isinstance(parent_ids[0], str)
        uuid.UUID(parent_ids[0])  # Validates format

    def test_lineage_id_unique_extraction(
        self,
        raw_conn: psycopg.Connection,
        pg_connection_params: dict[str, Any],
    ) -> None:
        """Multiple distinct lineage IDs should all be extracted as strings."""
        PostgresPolarsSchema.register_uuid_adapter(raw_conn)

        # Insert rows with different lineage IDs
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())

        raw_conn.autocommit = True
        with raw_conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TEST_SCHEMA}.uuid_multi_lineage (
                    name TEXT,
                    "{LineageDefaults.ID_COLUMN}" UUID
                )
                """
            )
            cur.execute(f"TRUNCATE {TEST_SCHEMA}.uuid_multi_lineage")
            cur.execute(
                f"""
                INSERT INTO {TEST_SCHEMA}.uuid_multi_lineage (name, "{LineageDefaults.ID_COLUMN}")
                VALUES ('a', '{id_a}'), ('b', '{id_b}'), ('c', '{id_a}')
                """
            )

        df = pl.read_database(
            f"SELECT * FROM {TEST_SCHEMA}.uuid_multi_lineage",
            raw_conn,
        )

        assert df.schema[LineageDefaults.ID_COLUMN] == pl.String

        import sqlalchemy as sa

        engine = sa.create_engine(
            f"postgresql+psycopg://{pg_connection_params['user']}:{pg_connection_params['password']}"
            f"@{pg_connection_params['host']}:{pg_connection_params['port']}"
            f"/{pg_connection_params['dbname']}?sslmode=disable"
        )
        tracker = LineageTracker(engine)
        parent_ids = tracker.get_parent_lineage_ids(df)

        assert len(parent_ids) == 2
        assert all(isinstance(pid, str) for pid in parent_ids)
        assert set(parent_ids) == {id_a, id_b}
