"""Integration tests for schema-preserving empty reads (#358).

``read_batched_to_dataframe`` used to return a column-less ``pl.DataFrame()``
when the underlying query produced no rows, which broke any downstream code
that referenced named columns (``.select`` / ``.join``) with
``ColumnNotFoundError`` -- even though the cursor description carried the
schema the whole time. These tests exercise the fix against a real PostgreSQL
container across the streaming (resource / engine) and offset (raw psycopg)
read paths.

Requires Docker. Run with: ``uv run pytest -m integration -v``.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import psycopg
import pytest

from moncpipelib.resources.postgres import (
    PostgresResource,
    read_batched_to_dataframe,
)

pytestmark = pytest.mark.integration

TEST_SCHEMA = "test_empty_read"
EMPTY_TABLE = f"{TEST_SCHEMA}.always_empty"
FILTERED_TABLE = f"{TEST_SCHEMA}.has_rows"

# Columns chosen to exercise the OID -> Polars dtype overrides (uuid -> String,
# timestamptz -> Datetime, numeric -> Float64) so the empty frame is asserted
# to carry typed columns, not just names.
_EMPTY_COLUMNS = ["alias_normalized", "canonical_facility_name", "is_active", "facility_id"]


@pytest.fixture(scope="session", autouse=True)
def _create_empty_read_schema(pg_connection_params: dict[str, Any]) -> None:
    """Create the schema plus a permanently-empty table and a populated table."""
    conn = psycopg.connect(**pg_connection_params)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA}")

            cur.execute(f"DROP TABLE IF EXISTS {EMPTY_TABLE}")
            cur.execute(
                f"""
                CREATE TABLE {EMPTY_TABLE} (
                    alias_normalized TEXT NOT NULL,
                    canonical_facility_name TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    facility_id UUID
                )
                """
            )

            cur.execute(f"DROP TABLE IF EXISTS {FILTERED_TABLE}")
            cur.execute(
                f"""
                CREATE TABLE {FILTERED_TABLE} (
                    alias_normalized TEXT NOT NULL,
                    canonical_facility_name TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    facility_id UUID
                )
                """
            )
            cur.execute(
                f"""
                INSERT INTO {FILTERED_TABLE}
                    (alias_normalized, canonical_facility_name, facility_id)
                VALUES
                    ('main st', 'Main Street Hospital', gen_random_uuid())
                """
            )
    finally:
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


class TestEmptyReadPreservesSchema:
    """The empty-result frame keeps the query's columns and dtypes."""

    def test_empty_table_preserves_schema(self, pg_resource: PostgresResource) -> None:
        """Reading an empty source table yields a zero-row frame with columns."""
        df = pg_resource.read_batched_to_dataframe(
            f"SELECT alias_normalized, canonical_facility_name, is_active, facility_id "
            f"FROM {EMPTY_TABLE}",
        )
        assert df.height == 0
        assert df.columns == _EMPTY_COLUMNS

    def test_empty_filter_preserves_schema(self, pg_resource: PostgresResource) -> None:
        """A non-empty table with a WHERE that excludes every row keeps the schema."""
        df = pg_resource.read_batched_to_dataframe(
            f"SELECT alias_normalized, canonical_facility_name, is_active, facility_id "
            f"FROM {FILTERED_TABLE} WHERE 1 = 0",
        )
        assert df.height == 0
        assert df.columns == _EMPTY_COLUMNS

    def test_consumer_can_select_named_column_on_empty_read(
        self, pg_resource: PostgresResource
    ) -> None:
        """The reported failure mode: ``.select`` on a named column must not raise."""
        df = pg_resource.read_batched_to_dataframe(
            f"SELECT alias_normalized, canonical_facility_name FROM {EMPTY_TABLE}",
        )
        projected = df.select(
            "alias_normalized",
            pl.col("canonical_facility_name").alias("canonical"),
        )
        assert projected.height == 0
        assert projected.columns == ["alias_normalized", "canonical"]

    def test_empty_read_join_against_populated_frame(self, pg_resource: PostgresResource) -> None:
        """A left-join against an empty alias frame -- the #358 real-world case."""
        left = pl.DataFrame({"alias_normalized": ["main st", "other"]})
        aliases = pg_resource.read_batched_to_dataframe(
            f"SELECT alias_normalized, canonical_facility_name FROM {EMPTY_TABLE}",
        )
        joined = left.join(aliases, on="alias_normalized", how="left")
        assert joined.height == 2
        assert "canonical_facility_name" in joined.columns
        assert joined["canonical_facility_name"].null_count() == 2

    def test_empty_read_applies_schema_overrides(self, pg_resource: PostgresResource) -> None:
        """UUID columns come back as ``pl.String`` even on the empty path."""
        df = pg_resource.read_batched_to_dataframe(
            f"SELECT facility_id, is_active FROM {EMPTY_TABLE}",
        )
        assert df.schema["facility_id"] == pl.String
        assert df.schema["is_active"] == pl.Boolean

    def test_module_level_engine_path_preserves_schema(self, pg_resource: PostgresResource) -> None:
        """The module-level function with a SQLAlchemy engine keeps the schema."""
        df = read_batched_to_dataframe(
            f"SELECT alias_normalized, canonical_facility_name FROM {EMPTY_TABLE}",
            pg_resource.get_engine(),
        )
        assert df.height == 0
        assert df.columns == ["alias_normalized", "canonical_facility_name"]

    def test_offset_path_preserves_schema(self, pg_connection: psycopg.Connection) -> None:
        """The offset method (raw psycopg connection) also keeps the schema."""
        df = read_batched_to_dataframe(
            f"SELECT alias_normalized, canonical_facility_name FROM {EMPTY_TABLE}",
            pg_connection,
            method="offset",
            order_by="alias_normalized",
        )
        assert df.height == 0
        assert df.columns == ["alias_normalized", "canonical_facility_name"]

    def test_populated_read_is_unchanged(self, pg_resource: PostgresResource) -> None:
        """The non-empty path is unaffected -- rows and columns flow through."""
        df = pg_resource.read_batched_to_dataframe(
            f"SELECT alias_normalized, canonical_facility_name FROM {FILTERED_TABLE}",
        )
        assert df.height == 1
        assert df.columns == ["alias_normalized", "canonical_facility_name"]
        assert df["alias_normalized"][0] == "main st"
