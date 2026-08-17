"""Integration tests for migration 018 Phase 4 ``replaces_lineage_id``.

Verifies that sequential ``FULL_REFRESH`` writes of the same asset
produce a linked chain via ``replaces_lineage_id``, and that
partition-scoped writes only chain within the same partition.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import polars as pl
import psycopg
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


@pytest.fixture()
def lineage_table(pg_connection: psycopg.Connection) -> str:
    """Create ``lineage.data_lineage`` for replaces-chain tests.

    Same minimal shape as ``test_lineage_atomicity``'s fixture.
    Production schema is broader; Phase 7 runbook governs it.
    """
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
        # Broader pattern covers both TestReplacesLineageIdChain
        # (``replaces_test_*``) and TestReplacesLineageIdPartitionScoped
        # (``replaces_partition_test_*``). Without the wider prefix,
        # rows from the partition class accumulate across tests because
        # the class shares a single ``TABLE_NAME``.
        cur.execute("DELETE FROM lineage.data_lineage WHERE asset_name LIKE 'replaces_%'")
    pg_connection.commit()
    return fqn


@pytest.mark.integration
class TestReplacesLineageIdChain:
    """``FULL_REFRESH`` writes of the same asset should chain via
    ``replaces_lineage_id``, while ``UPSERT`` / ``APPEND`` / ``SCD2``
    writes never set it.
    """

    TABLE_NAME: str = f"replaces_test_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        pg_connection: psycopg.Connection,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        lineage_table: str,  # noqa: ARG002 -- fixture used for setup side-effect
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "name": "TEXT",
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

    def _lineage_rows_for_asset(self) -> list[dict[str, Any]]:
        """Return all lineage rows for the current test asset, sorted by
        ``processed_at``."""
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT lineage_id, replaces_lineage_id, data_date, processed_at
                FROM lineage.data_lineage
                WHERE asset_name = %s
                ORDER BY processed_at
                """,
                (self.TABLE_NAME,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def test_first_full_refresh_has_null_replaces(self) -> None:
        """The first ``FULL_REFRESH`` for an asset has no prior row, so
        ``replaces_lineage_id`` must be NULL."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh", "layer_override": "bronze"},
        )
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})

        self.io_mgr.handle_output(ctx, df)

        rows = self._lineage_rows_for_asset()
        assert len(rows) == 1
        assert rows[0]["replaces_lineage_id"] is None

    def test_sequential_full_refresh_writes_chain(self) -> None:
        """Two ``FULL_REFRESH`` writes of the same whole-table asset
        produce a linked chain: row 2 links back to row 1, row 1 has
        no predecessor."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh", "layer_override": "bronze"},
        )
        self.io_mgr.handle_output(ctx, pl.DataFrame({"id": [1], "name": ["a"]}))
        self.io_mgr.handle_output(ctx, pl.DataFrame({"id": [2], "name": ["b"]}))
        self.io_mgr.handle_output(ctx, pl.DataFrame({"id": [3], "name": ["c"]}))

        rows = self._lineage_rows_for_asset()
        assert len(rows) == 3
        assert rows[0]["replaces_lineage_id"] is None
        assert rows[1]["replaces_lineage_id"] == rows[0]["lineage_id"]
        assert rows[2]["replaces_lineage_id"] == rows[1]["lineage_id"]

    def test_upsert_never_sets_replaces(self) -> None:
        """``UPSERT`` is accumulative, not replacement; ``replaces_lineage_id``
        must always be NULL even on the second write."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"], "layer_override": "bronze"},
        )
        self.io_mgr.handle_output(ctx, pl.DataFrame({"id": [1], "name": ["a"]}))
        self.io_mgr.handle_output(ctx, pl.DataFrame({"id": [2], "name": ["b"]}))

        rows = self._lineage_rows_for_asset()
        assert len(rows) == 2
        assert rows[0]["replaces_lineage_id"] is None
        assert rows[1]["replaces_lineage_id"] is None


@pytest.mark.integration
class TestReplacesLineageIdPartitionScoped:
    """Partition-scoped ``FULL_REFRESH`` writes chain within the same
    partition only.
    """

    TABLE_NAME: str = f"replaces_partition_test_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        pg_connection: psycopg.Connection,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        lineage_table: str,  # noqa: ARG002 -- fixture used for setup side-effect
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "date_col": "DATE NOT NULL",
                "name": "TEXT",
                "_lineage_id": "uuid REFERENCES lineage.data_lineage(lineage_id)",
                "_lineage_key": "text",
            },
            primary_key=["id", "date_col"],
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

    def _lineage_rows_for_partition(self, data_date: str) -> list[dict[str, Any]]:
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT lineage_id, replaces_lineage_id, data_date
                FROM lineage.data_lineage
                WHERE asset_name = %s AND data_date = %s
                ORDER BY processed_at
                """,
                (self.TABLE_NAME, data_date),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def test_same_partition_writes_chain(self) -> None:
        """Two partition-scoped FULL_REFRESH writes of the same date
        produce a linked chain on that partition."""
        from datetime import date

        ctx_1 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            partition_key="2026-05-15",
            metadata={
                "write_mode": "full_refresh",
                "partition_column": "date_col",
                "layer_override": "bronze",
            },
        )
        ctx_2 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            partition_key="2026-05-15",
            metadata={
                "write_mode": "full_refresh",
                "partition_column": "date_col",
                "layer_override": "bronze",
            },
        )
        self.io_mgr.handle_output(
            ctx_1, pl.DataFrame({"id": [1], "date_col": [date(2026, 5, 15)], "name": ["a"]})
        )
        self.io_mgr.handle_output(
            ctx_2, pl.DataFrame({"id": [2], "date_col": [date(2026, 5, 15)], "name": ["b"]})
        )

        rows = self._lineage_rows_for_partition("2026-05-15")
        assert len(rows) == 2
        assert rows[0]["replaces_lineage_id"] is None
        assert rows[1]["replaces_lineage_id"] == rows[0]["lineage_id"]

    def test_different_partitions_do_not_chain(self) -> None:
        """A FULL_REFRESH of partition A then partition B must NOT link
        B back to A -- they describe different data slices."""
        from datetime import date

        for partition in ("2026-05-15", "2026-05-16"):
            ctx = make_mock_output_context(
                asset_name=self.TABLE_NAME,
                partition_key=partition,
                metadata={
                    "write_mode": "full_refresh",
                    "partition_column": "date_col",
                    "layer_override": "bronze",
                },
            )
            self.io_mgr.handle_output(
                ctx,
                pl.DataFrame(
                    {
                        "id": [1],
                        "date_col": [date.fromisoformat(partition)],
                        "name": ["x"],
                    }
                ),
            )

        rows_a = self._lineage_rows_for_partition("2026-05-15")
        rows_b = self._lineage_rows_for_partition("2026-05-16")
        assert len(rows_a) == 1
        assert len(rows_b) == 1
        assert rows_a[0]["replaces_lineage_id"] is None
        # Different partition; no chain to A.
        assert rows_b[0]["replaces_lineage_id"] is None
