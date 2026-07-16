"""Integration tests for migration 018 Phase 5 ``parent_lineage_ids``.

Verifies that batched writes accumulate upstream ``_lineage_id`` values
across every batch and persist the union onto
``data_lineage.parent_lineage_ids``. Requires Docker.

Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any
from uuid import UUID

import polars as pl
import psycopg
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager
from moncpipelib.streaming import BatchedDataFrame

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


@pytest.fixture()
def lineage_table(pg_connection: psycopg.Connection) -> str:
    """Same minimal lineage table as the other Phase 3/4 integration tests."""
    fqn = "lineage.data_lineage"
    with pg_connection.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS lineage")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lineage.data_lineage (
                lineage_id          uuid PRIMARY KEY,
                lineage_key         text NOT NULL,
                run_id              text NOT NULL,
                asset_name          text NOT NULL,
                pipeline_id         uuid,
                layer               text NOT NULL,
                source_file         text,
                source_system       text,
                data_date           date,
                data_date_range     daterange,
                row_count           integer,
                is_backfill         boolean NOT NULL DEFAULT FALSE,
                backfill_reason     text,
                backfill_id         text,
                replaces_lineage_id uuid,
                parent_lineage_ids  uuid[],
                transformation_type text,
                metadata            jsonb,
                processed_at        timestamptz NOT NULL DEFAULT NOW(),
                created_by          text DEFAULT CURRENT_USER
            )
            """
        )
        cur.execute("DELETE FROM lineage.data_lineage WHERE asset_name LIKE 'parents_test_%'")
    pg_connection.commit()
    return fqn


def _seed_upstream_lineage_rows(
    conn: psycopg.Connection, _asset_name: str, count: int
) -> list[str]:
    """Seed ``count`` upstream lineage rows; return their UUIDs.

    ``_asset_name`` is currently unused (seeded rows belong to the
    synthetic ``upstream_asset``) but accepted so callers can document
    the downstream asset the parents will be referenced from.
    """
    ids = [str(uuid.uuid4()) for _ in range(count)]
    with conn.cursor() as cur:
        for i, lid in enumerate(ids):
            cur.execute(
                """
                INSERT INTO lineage.data_lineage (
                    lineage_id, lineage_key, run_id, asset_name, layer
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (lid, f"v1:up:bronze:seed:{i}", "seed-run", "upstream_asset", "bronze"),
            )
    conn.commit()
    return ids


@pytest.mark.integration
class TestBatchedParentLineageAccumulation:
    """Batched writes record ``parent_lineage_ids`` as the union of
    upstream ``_lineage_id`` values seen across every batch.

    Uses the IO-manager path (not direct ``resource.write()``) because
    the IO manager's metadata-driven ``layer_override`` is the simplest
    way to enable lineage when the target schema (``test_write``) isn't
    a recognised layer name.
    """

    TABLE_NAME: str = f"parents_test_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        pg_connection: psycopg.Connection,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        lineage_table: str,  # noqa: ARG002 -- fixture used for setup side-effect
    ) -> Any:
        # Minimal table: ``id`` for primary key, plus the two managed
        # lineage columns. No ``name`` -- keeps every batch DataFrame in
        # this test family schema-equivalent to the target.
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "_lineage_id": "uuid REFERENCES lineage.data_lineage(lineage_id)",
                "_lineage_key": "text",
            },
            primary_key=["id"],
        )
        self.builder = table_builder
        self.pg_conn = pg_connection
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=True,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def _read_parent_lineage_ids(self) -> list[UUID] | None:
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT parent_lineage_ids
                FROM lineage.data_lineage
                WHERE asset_name = %s
                ORDER BY processed_at DESC
                LIMIT 1
                """,
                (self.TABLE_NAME,),
            )
            row = cur.fetchone()
            assert row is not None
            return row[0]

    def test_multi_batch_unions_parents(self) -> None:
        """Three batches, three different upstream IDs → union of all
        three appears on the lineage row."""
        upstream_ids = _seed_upstream_lineage_rows(self.pg_conn, self.TABLE_NAME, count=3)

        df1 = pl.DataFrame({"id": [1], "_lineage_id": [upstream_ids[0]]}).with_columns(
            pl.col("_lineage_id").cast(pl.String)
        )
        df2 = pl.DataFrame({"id": [2], "_lineage_id": [upstream_ids[1]]}).with_columns(
            pl.col("_lineage_id").cast(pl.String)
        )
        df3 = pl.DataFrame({"id": [3], "_lineage_id": [upstream_ids[2]]}).with_columns(
            pl.col("_lineage_id").cast(pl.String)
        )
        batched = BatchedDataFrame(batches=iter([df1, df2, df3]), total_rows_hint=3)

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            run_id="parents-run-multi",
            metadata={"write_mode": "append", "layer_override": "bronze"},
        )

        self.io_mgr.handle_output(ctx, batched)

        parents = self._read_parent_lineage_ids()
        assert parents is not None
        # ``sorted`` because Phase 5 sorts for determinism.
        assert sorted(str(u) for u in parents) == sorted(upstream_ids)

    def test_no_lineage_id_column_leaves_parents_null(self) -> None:
        """Without ``_lineage_id`` on any batch, the lineage row's
        ``parent_lineage_ids`` column stays NULL (not empty array)."""
        df1 = pl.DataFrame({"id": [1]})
        df2 = pl.DataFrame({"id": [2]})
        batched = BatchedDataFrame(batches=iter([df1, df2]), total_rows_hint=2)

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            run_id="parents-run-no-id",
            metadata={"write_mode": "append", "layer_override": "bronze"},
        )

        self.io_mgr.handle_output(ctx, batched)

        assert self._read_parent_lineage_ids() is None
