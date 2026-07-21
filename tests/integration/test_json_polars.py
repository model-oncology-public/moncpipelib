"""Integration tests for JSON/JSONB-to-Polars-String type inference.

Validates that PostgreSQL JSON and JSONB columns are correctly read as
``pl.String`` (not deserialized into Python dicts/lists that cause
``pl.Object`` or ``ComputeError``) across all read paths:
- Direct ``pl.read_database`` with the JSON adapter registered
- ``read_batched_to_dataframe`` (streaming and offset methods)
- ``PostgresResource.get_connection()`` and ``get_connection_raw()``

Without the psycopg2 JSON string adapters, psycopg2 calls ``json.loads()``
on JSON/JSONB wire values, producing Python dicts and lists.  Polars cannot
infer a consistent schema from heterogeneous objects (especially JSONB arrays
with varying lengths) and raises ``ComputeError``.

Requires Docker.  Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl
import psycopg
import pytest

from moncpipelib.resources.postgres import (
    PostgresPolarsSchema,
    PostgresResource,
    read_batched_to_dataframe,
    restore_default_handlers,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TEST_SCHEMA = "test_json"
TABLE_NAME = f"{TEST_SCHEMA}.json_test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _create_json_schema(pg_connection_params: dict[str, Any]) -> None:
    """Create the test schema and seed table with JSON/JSONB data.

    Inserts rows with varying-length JSONB arrays to reproduce the exact
    ComputeError scenario from issue #118.
    """
    conn = psycopg.connect(**pg_connection_params)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA}")

            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
            cur.execute(
                f"""
                CREATE TABLE {TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    metadata_json JSON,
                    metadata_jsonb JSONB,
                    tags JSONB,
                    config JSONB
                )
                """
            )

            # Insert rows with VARYING array lengths and nested structures --
            # this is the exact pattern that breaks Polars schema inference
            # when psycopg2 deserializes to Python objects.
            cur.execute(
                f"""
                INSERT INTO {TABLE_NAME} (name, metadata_json, metadata_jsonb, tags, config) VALUES
                    ('alpha',
                     '{{"key": "val1", "count": 1}}'::json,
                     '{{"key": "val1", "count": 1}}'::jsonb,
                     '["ORAL"]'::jsonb,
                     '{{"nested": {{"a": 1}}}}'::jsonb),
                    ('bravo',
                     '{{"key": "val2", "count": 2, "extra": true}}'::json,
                     '{{"key": "val2", "count": 2, "extra": true}}'::jsonb,
                     '["TOPICAL", "ORAL"]'::jsonb,
                     '{{"nested": {{"a": 2, "b": 3}}}}'::jsonb),
                    ('charlie',
                     '{{"key": "val3"}}'::json,
                     '{{"key": "val3"}}'::jsonb,
                     '["IV", "TOPICAL", "ORAL"]'::jsonb,
                     'null'::jsonb),
                    ('delta',
                     NULL,
                     NULL,
                     NULL,
                     NULL)
                """
            )
    finally:
        conn.close()


@pytest.fixture()
def raw_conn(pg_connection_params: dict[str, Any]) -> psycopg.Connection:
    """Provide a fresh driver-native connection (no adapters registered).

    Routes through the driver seam so the same fixture works under both
    ``MONC_PG_DRIVER=psycopg2`` and ``MONC_PG_DRIVER=psycopg3``.  The
    annotation reflects the psycopg2 default; under psycopg3 the runtime
    object is structurally compatible for the methods these tests use.
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
# Test: JSON adapter -- the fix
# ---------------------------------------------------------------------------


class TestJSONAdapterRegistration:
    """Verify that register_json_adapters makes JSON/JSONB arrive as pl.String."""

    def test_jsonb_with_adapter_is_string(self, raw_conn: psycopg.Connection) -> None:
        """JSONB columns are pl.String when the adapter is registered."""
        PostgresPolarsSchema.register_json_adapters(raw_conn)
        schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(
            raw_conn, f"SELECT * FROM {TABLE_NAME}"
        )
        df = pl.read_database(
            f"SELECT * FROM {TABLE_NAME}",
            raw_conn,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )
        assert df.schema["metadata_jsonb"] == pl.String
        assert df.schema["tags"] == pl.String
        assert df.schema["config"] == pl.String

    def test_json_with_adapter_is_string(self, raw_conn: psycopg.Connection) -> None:
        """JSON columns are pl.String when the adapter is registered."""
        PostgresPolarsSchema.register_json_adapters(raw_conn)
        schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(
            raw_conn, f"SELECT * FROM {TABLE_NAME}"
        )
        df = pl.read_database(
            f"SELECT * FROM {TABLE_NAME}",
            raw_conn,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )
        assert df.schema["metadata_json"] == pl.String

    def test_json_values_are_valid_json_strings(self, raw_conn: psycopg.Connection) -> None:
        """String values returned by the adapter are parseable JSON."""
        PostgresPolarsSchema.register_json_adapters(raw_conn)
        schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(
            raw_conn, f"SELECT * FROM {TABLE_NAME}"
        )
        df = pl.read_database(
            f"SELECT * FROM {TABLE_NAME} WHERE metadata_jsonb IS NOT NULL",
            raw_conn,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )

        for value in df["metadata_jsonb"].to_list():
            parsed = json.loads(value)
            assert isinstance(parsed, dict)

        for value in df["tags"].to_list():
            parsed = json.loads(value)
            assert isinstance(parsed, list)

    def test_null_json_is_none(self, raw_conn: psycopg.Connection) -> None:
        """NULL JSON/JSONB values come through as Python None."""
        PostgresPolarsSchema.register_json_adapters(raw_conn)
        schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(
            raw_conn, f"SELECT * FROM {TABLE_NAME}"
        )
        df = pl.read_database(
            f"SELECT * FROM {TABLE_NAME} WHERE name = 'delta'",
            raw_conn,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )
        assert df["metadata_json"][0] is None
        assert df["metadata_jsonb"][0] is None
        assert df["tags"][0] is None
        assert df["config"][0] is None

    def test_varying_array_lengths_no_error(self, raw_conn: psycopg.Connection) -> None:
        """JSONB arrays with varying lengths don't cause ComputeError.

        This is the exact scenario from issue #118: the ``tags`` column has
        arrays of length 1, 2, and 3. Without the adapter, psycopg2 returns
        Python lists of varying lengths, and Polars fails to infer a
        consistent schema.
        """
        PostgresPolarsSchema.register_json_adapters(raw_conn)
        schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(
            raw_conn, f"SELECT * FROM {TABLE_NAME}"
        )

        # This would raise ComputeError without the adapter
        df = pl.read_database(
            f"SELECT * FROM {TABLE_NAME}",
            raw_conn,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )

        assert len(df) == 4
        assert df.schema["tags"] == pl.String

    def test_adapter_is_idempotent(self, raw_conn: psycopg.Connection) -> None:
        """Registering the adapter multiple times is safe."""
        PostgresPolarsSchema.register_json_adapters(raw_conn)
        PostgresPolarsSchema.register_json_adapters(raw_conn)

        schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(
            raw_conn, f"SELECT * FROM {TABLE_NAME}"
        )
        df = pl.read_database(
            f"SELECT * FROM {TABLE_NAME} WHERE name = 'alpha'",
            raw_conn,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )
        assert df.schema["metadata_jsonb"] == pl.String


# ---------------------------------------------------------------------------
# Test: PostgresResource connections auto-register the adapter
# ---------------------------------------------------------------------------


class TestPostgresResourceJSON:
    """Verify that PostgresResource connections have JSON adapters pre-registered."""

    def test_get_connection_returns_json_as_string(self, pg_resource: PostgresResource) -> None:
        """get_connection() auto-registers JSON adapters."""
        with pg_resource.get_connection() as conn:
            schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(
                conn, f"SELECT * FROM {TABLE_NAME}"
            )
            df = pl.read_database(
                f"SELECT * FROM {TABLE_NAME}",
                conn,
                schema_overrides=schema_overrides,
                infer_schema_length=0,
            )

        assert df.schema["metadata_jsonb"] == pl.String
        assert df.schema["tags"] == pl.String
        # Verify the varying-length arrays survived without error
        assert len(df) == 4

    def test_get_connection_raw_returns_json_as_string(self, pg_resource: PostgresResource) -> None:
        """get_connection_raw() auto-registers JSON adapters."""
        conn = pg_resource.get_connection_raw()
        try:
            schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(
                conn, f"SELECT * FROM {TABLE_NAME}"
            )
            df = pl.read_database(
                f"SELECT * FROM {TABLE_NAME}",
                conn,
                schema_overrides=schema_overrides,
                infer_schema_length=0,
            )
        finally:
            conn.close()

        assert df.schema["metadata_jsonb"] == pl.String
        assert df.schema["tags"] == pl.String


# ---------------------------------------------------------------------------
# Test: Batched reads
# ---------------------------------------------------------------------------


class TestBatchedReadJSON:
    """Verify JSON string behavior through the batched read path."""

    def test_batched_read_json_is_string(self, pg_resource: PostgresResource) -> None:
        """read_batched_to_dataframe returns JSON/JSONB as pl.String."""
        df = read_batched_to_dataframe(
            f"SELECT * FROM {TABLE_NAME}",
            pg_resource,
            batch_size=100,
        )

        assert df.schema["metadata_json"] == pl.String
        assert df.schema["metadata_jsonb"] == pl.String
        assert df.schema["tags"] == pl.String
        assert df.schema["config"] == pl.String
        assert len(df) == 4

    def test_batched_read_varying_arrays(self, pg_resource: PostgresResource) -> None:
        """Batched read handles varying-length JSONB arrays without error."""
        # Use small batch size to force multiple batches
        df = read_batched_to_dataframe(
            f"SELECT * FROM {TABLE_NAME} ORDER BY id",
            pg_resource,
            batch_size=2,
        )

        assert len(df) == 4
        # All tags values should be valid JSON strings (or null)
        non_null_tags = df.filter(pl.col("tags").is_not_null())["tags"].to_list()
        for tag_str in non_null_tags:
            parsed = json.loads(tag_str)
            assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Test: Opt-out mechanism
# ---------------------------------------------------------------------------


class TestJSONAdapterOptOut:
    """Verify callers can restore psycopg2's default JSON deserialization."""

    def test_opt_out_restores_parsed_objects(self, raw_conn: psycopg.Connection) -> None:
        """Restoring default handlers makes the driver return parsed Python objects.

        Migration 014 routed both registration and restore through the
        driver seam; the seam's ``restore_default_handlers`` re-registers
        psycopg2's stock JSON / JSONB handlers (or psycopg3's stock
        ``JsonbLoader`` / ``JsonLoader``) so the next read returns dicts
        / lists rather than raw strings.
        """
        # First register our adapter through the seam.
        PostgresPolarsSchema.register_json_adapters(raw_conn)

        # Then opt out by restoring the driver's default handlers.
        restore_default_handlers(raw_conn)

        # Now the driver should return Python objects again.
        with raw_conn.cursor() as cur:
            cur.execute(f"SELECT metadata_jsonb FROM {TABLE_NAME} WHERE name = 'alpha'")
            row = cur.fetchone()
            assert row is not None
            # With default handlers, JSONB should be a Python dict, not a string.
            assert isinstance(row[0], dict)
