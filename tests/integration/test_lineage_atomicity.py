"""Integration tests for migration 018 Phase 3 same-transaction lineage.

Verifies that ``PostgresResource.write()`` runs the lineage-row INSERT
and the data DML inside a single PostgreSQL transaction. A failure on
either side leaves no orphan rows in either table.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import polars as pl
import psycopg
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


@pytest.fixture()
def lineage_table(pg_connection: psycopg.Connection) -> str:
    """Create a minimal ``lineage.data_lineage`` table for atomicity tests.

    Mirrors the production columns this test class touches; not exhaustive.
    The Phase 7 production runbook ensures the real schema includes every
    column the tracker INSERTs.
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
        # Wipe any state from prior runs of this test module.
        cur.execute("DELETE FROM lineage.data_lineage WHERE asset_name LIKE 'atomicity_test_%'")
    pg_connection.commit()
    return fqn


@pytest.mark.integration
class TestLineageWriteAtomicity:
    """Phase 3 invariant: lineage row + data DML commit (or roll back)
    atomically. The plan's primary correctness goal.
    """

    TABLE_NAME: str = f"atomicity_test_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        pg_connection: psycopg.Connection,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        lineage_table: str,  # noqa: ARG002 -- fixture used for setup side-effect (creates table)
    ) -> Any:
        # Data table with FK on _lineage_id (mirrors production shape).
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "name": "TEXT",
                "_lineage_id": ("uuid REFERENCES lineage.data_lineage(lineage_id)"),
                "_lineage_key": "text",
            },
            primary_key=["id"],
        )
        self.builder = table_builder
        self.pg_conn = pg_connection
        # IO manager with lineage enabled so the resource's write path runs
        # the Phase 3 same-txn flow.
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=True,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def _count_lineage_for_asset(self) -> int:
        """Count lineage rows belonging to the current test asset."""
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM lineage.data_lineage WHERE asset_name = %s",
                (self.TABLE_NAME,),
            )
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    def test_successful_write_inserts_both_rows_atomically(self) -> None:
        """Happy path: a successful ``write()`` must produce both a
        lineage row and the data rows referencing it."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh", "layer_override": "bronze"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        self.io_mgr.handle_output(ctx, df)

        # Data rows present.
        assert self.builder.count(self.fqn) == 3
        # Exactly one lineage row for this asset.
        assert self._count_lineage_for_asset() == 1
        # The FK actually resolves -- every data row's _lineage_id
        # points to a real lineage row.
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) FROM {self.fqn} d JOIN lineage.data_lineage l "  # noqa: S608
                "ON d._lineage_id = l.lineage_id"
            )
            row = cur.fetchone()
            assert row is not None
            assert int(row[0]) == 3

    def test_data_dml_failure_rolls_back_lineage_row(self) -> None:
        """Simulate a data-DML failure mid-write: the lineage row must
        roll back too, leaving zero rows in both tables for this asset.
        """
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            run_id="atomicity_run_dml_fail",
            metadata={"write_mode": "full_refresh", "layer_override": "bronze"},
        )
        df = pl.DataFrame({"id": [10, 20], "name": ["x", "y"]})

        baseline = self._count_lineage_for_asset()

        with (
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                side_effect=RuntimeError("simulated DML failure"),
            ),
            pytest.raises(RuntimeError, match="simulated DML failure"),
        ):
            self.io_mgr.handle_output(ctx, df)

        # Neither lineage nor data should have been committed.
        assert self._count_lineage_for_asset() == baseline
        assert self.builder.count(self.fqn) == 0

    def test_lineage_insert_failure_rolls_back_and_runs_no_dml(self) -> None:
        """Simulate a unique-violation on the lineage INSERT (e.g., the
        unlikely lineage_key collision). The data DML must never run."""
        # Pre-seed a lineage row whose key the write would otherwise
        # collide with. Phase 3's INSERT uses a unique lineage_key per
        # row, so we inject a conflicting key via a patched
        # ``generate_lineage_ids``.
        seeded_id = str(uuid.uuid4())
        seeded_key = f"v1:{self.TABLE_NAME}:bronze:dup-key:atomic"
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lineage.data_lineage (
                    lineage_id, lineage_key, run_id, asset_name, layer
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (seeded_id, seeded_key, "seed", self.TABLE_NAME, "bronze"),
            )
        self.pg_conn.commit()

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            run_id="atomicity_run_lineage_fail",
            metadata={"write_mode": "full_refresh", "layer_override": "bronze"},
        )
        df = pl.DataFrame({"id": [30, 40], "name": ["m", "n"]})

        # Force the tracker to return a duplicate (lineage_id, lineage_key)
        # so the same-txn INSERT raises UniqueViolation.
        from moncpipelib.lineage.tracker import LineageTracker

        with (
            patch.object(
                LineageTracker,
                "generate_lineage_ids",
                return_value=(seeded_id, seeded_key),
            ),
            pytest.raises(psycopg.errors.UniqueViolation),
        ):
            self.io_mgr.handle_output(ctx, df)

        # Only the pre-seeded row remains.
        assert self._count_lineage_for_asset() == 1
        # No data rows from the failed write.
        assert self.builder.count(self.fqn) == 0
