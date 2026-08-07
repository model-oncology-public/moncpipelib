"""Integration tests for BatchedDataFrame handle_output against a real PostgreSQL database.

Tests the batched write path: full_refresh with multiple batches, append mode,
metadata reporting, and single-batch edge case.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import psycopg
import pytest

from moncpipelib.streaming import BatchedDataFrame

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


class TestBatchedOutput:
    """Verify PostgresIOManager handles BatchedDataFrame writes correctly."""

    TABLE_COLUMNS: dict[str, str] = {
        "id": "INTEGER",
        "name": "TEXT",
        "value": "DOUBLE PRECISION",
    }

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        pg_connection: psycopg.Connection,
        io_manager_factory: Any,
    ) -> Any:
        self.suffix = uuid.uuid4().hex[:8]
        self.table_name = f"batched_{self.suffix}"
        self.fqn = table_builder.create_table(
            self.table_name,
            columns=self.TABLE_COLUMNS,
            primary_key=["id"],
        )
        self.builder = table_builder
        self.conn = pg_connection
        self.io_mgr = io_manager_factory(db_schema="test_write")
        yield
        self.builder.drop(self.fqn)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_batched_full_refresh(self) -> None:
        """Create BatchedDataFrame from 2 DataFrames, handle_output with full_refresh.

        Verify all rows from both batches are present in the table.
        """
        df1 = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["alpha", "bravo", "charlie"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        df2 = pl.DataFrame(
            {
                "id": [4, 5],
                "name": ["delta", "echo"],
                "value": [4.0, 5.0],
            }
        )

        total = len(df1) + len(df2)
        batched = BatchedDataFrame(batches=iter([df1, df2]), total_rows_hint=total)

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "full_refresh"},
        )
        self.io_mgr.handle_output(ctx, batched)

        # Verify all rows landed
        row_count = self.builder.count(self.fqn)
        assert row_count == 5

        # Verify data integrity
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert [r["name"] for r in rows] == ["alpha", "bravo", "charlie", "delta", "echo"]

    def test_batched_append(self) -> None:
        """Append mode with batches; verify cumulative rows across multiple writes."""
        # First write: 3 rows
        df_batch1 = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["one", "two", "three"],
                "value": [10.0, 20.0, 30.0],
            }
        )
        batched1 = BatchedDataFrame(batches=iter([df_batch1]), total_rows_hint=3)

        ctx1 = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "append"},
        )
        self.io_mgr.handle_output(ctx1, batched1)
        assert self.builder.count(self.fqn) == 3

        # Second write: 2 more rows (different IDs to avoid PK conflict)
        df_batch2 = pl.DataFrame(
            {
                "id": [4, 5],
                "name": ["four", "five"],
                "value": [40.0, 50.0],
            }
        )
        batched2 = BatchedDataFrame(batches=iter([df_batch2]), total_rows_hint=2)

        ctx2 = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "append"},
        )
        self.io_mgr.handle_output(ctx2, batched2)

        # Cumulative: 3 + 2 = 5 rows
        assert self.builder.count(self.fqn) == 5

    def test_batched_metadata(self) -> None:
        """Verify output metadata includes batches_written and rows_written."""
        df1 = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["foo", "bar"],
                "value": [1.5, 2.5],
            }
        )
        df2 = pl.DataFrame(
            {
                "id": [3],
                "name": ["baz"],
                "value": [3.5],
            }
        )

        total = len(df1) + len(df2)
        batched = BatchedDataFrame(batches=iter([df1, df2]), total_rows_hint=total)

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "full_refresh"},
        )
        self.io_mgr.handle_output(ctx, batched)

        # Verify add_output_metadata was called
        ctx.add_output_metadata.assert_called_once()
        metadata_call: dict[str, Any] = ctx.add_output_metadata.call_args[0][0]

        assert "rows_written" in metadata_call
        assert "batches_written" in metadata_call

        # Verify the BatchedDataFrame instance was updated
        assert batched.rows_written == 3
        assert batched.batches_written == 2

    def test_single_batch(self) -> None:
        """BatchedDataFrame with 1 batch; verify it writes correctly."""
        df = pl.DataFrame(
            {
                "id": [10, 20],
                "name": ["single-a", "single-b"],
                "value": [100.0, 200.0],
            }
        )

        batched = BatchedDataFrame(batches=iter([df]), total_rows_hint=len(df))

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "full_refresh"},
        )
        self.io_mgr.handle_output(ctx, batched)

        row_count = self.builder.count(self.fqn)
        assert row_count == 2

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert rows[0]["name"] == "single-a"
        assert rows[1]["name"] == "single-b"

        # Verify batched metadata
        assert batched.rows_written == 2
        assert batched.batches_written == 1


class TestBatchedPartitionColumnFromContract:
    """End-to-end coverage for #258.  Drives ``PostgresResource.write()``
    with a ``BatchedDataFrame`` against a partitioned bronze table, with
    ``partition_column`` declared in the contract sink and NOT passed
    by the caller -- the exact RxNorm Phase 2a shape from
    data-platform#613.

    Pre-#258, ``_write_batched`` ran ``_inject_period_partition_column``
    BEFORE ``ContractReconciler.reconcile_write_config``, so inject
    bailed (``write_config["partition_column"]`` was ``None``) and
    ``_validate_write_config`` raised ``partition_column 'load_period'
    not found in DataFrame``.
    """

    PARTITION_TABLE_COLUMNS: dict[str, str] = {
        "id": "TEXT",
        "name": "TEXT",
        "load_period": "TEXT",
    }

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        pg_connection: psycopg.Connection,
        postgres_resource: Any,
    ) -> Any:
        self.suffix = uuid.uuid4().hex[:8]
        self.table_name = f"partitioned_{self.suffix}"
        self.fqn = table_builder.create_table(
            self.table_name,
            columns=self.PARTITION_TABLE_COLUMNS,
        )
        self.builder = table_builder
        self.conn = pg_connection
        self.resource = postgres_resource
        yield
        self.builder.drop(self.fqn)

    @staticmethod
    def _partition_contract(table: str) -> Any:  # type: ignore[no-untyped-def]
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset=table,
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.STRING, nullable=False),
                    Column(name="name", type=ColumnType.STRING, nullable=True),
                    # ``managed=True`` excludes the column from
                    # ``validate_schema``'s "Missing columns" check --
                    # the canonical way production contracts declare
                    # columns that ``_inject_period_partition_column``
                    # populates.  Without it, ``_enforce_contract``
                    # raises before inject runs (enforce -> reconcile
                    # -> inject is the established order).
                    Column(
                        name="load_period",
                        type=ColumnType.STRING,
                        nullable=False,
                        managed=True,
                    ),
                ]
            ),
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": table,
                    "partition_column": "load_period",
                }
            ],
        )

    def test_batched_append_with_contract_partition_column(self) -> None:
        """Caller passes a partitioned ``BatchedDataFrame`` to
        ``database.write()`` without ``partition_column``; the contract
        declares ``partition_column: load_period``.  The write must
        succeed and every landed row must carry the partition key."""
        from moncpipelib.resources.types import WriteContext

        df1 = pl.DataFrame({"id": ["a", "b"], "name": ["alpha", "bravo"]})
        df2 = pl.DataFrame({"id": ["c"], "name": ["charlie"]})
        batched = BatchedDataFrame(batches=iter([df1, df2]), total_rows_hint=3)

        wctx = WriteContext(
            asset_name=f"reference_bronze/{self.table_name}",
            run_id=f"run-{self.suffix}",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-01-04"],
        )

        self.resource.write(
            batched,
            target=f"test_write.{self.table_name}",
            context=wctx,
            write_mode="append",
            contract=self._partition_contract(self.table_name),
            # partition_column intentionally omitted -- this is the #258 shape.
        )

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3
        # Every row across both batches carries the partition key from
        # ``wctx.partition_keys[0]`` -- proves inject ran post-reconcile.
        assert all(r["load_period"] == "2024-01-04" for r in rows), (
            f"expected all rows to have load_period='2024-01-04', got {rows}"
        )

    def test_batched_full_refresh_partition_scoped(self) -> None:
        """Partition-scoped FULL_REFRESH: write partition A, then partition B,
        then re-write partition B.  Partition A rows must remain intact
        (the partition-scoped DELETE in ``_write_batched`` only removes
        rows for the active partition).  This exercises the
        ``partition_column``-driven DELETE branch that ALSO depends on
        reconcile having run before any of the i==0 SQL fires."""
        from moncpipelib.resources.types import WriteContext

        contract = self._partition_contract(self.table_name)

        # Partition A
        df_a = pl.DataFrame({"id": ["a1", "a2"], "name": ["alpha-1", "alpha-2"]})
        batched_a = BatchedDataFrame(batches=iter([df_a]), total_rows_hint=2)
        wctx_a = WriteContext(
            asset_name=f"reference_bronze/{self.table_name}",
            run_id=f"run-{self.suffix}-a",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-01-01"],
        )
        self.resource.write(
            batched_a,
            target=f"test_write.{self.table_name}",
            context=wctx_a,
            write_mode="full_refresh",
            contract=contract,
        )

        # Partition B
        df_b = pl.DataFrame({"id": ["b1"], "name": ["bravo-1"]})
        batched_b = BatchedDataFrame(batches=iter([df_b]), total_rows_hint=1)
        wctx_b = WriteContext(
            asset_name=f"reference_bronze/{self.table_name}",
            run_id=f"run-{self.suffix}-b",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-02-01"],
        )
        self.resource.write(
            batched_b,
            target=f"test_write.{self.table_name}",
            context=wctx_b,
            write_mode="full_refresh",
            contract=contract,
        )

        # Re-write Partition B (delete-then-insert scoped to load_period='2024-02-01')
        df_b2 = pl.DataFrame({"id": ["b2"], "name": ["bravo-2"]})
        batched_b2 = BatchedDataFrame(batches=iter([df_b2]), total_rows_hint=1)
        wctx_b2 = WriteContext(
            asset_name=f"reference_bronze/{self.table_name}",
            run_id=f"run-{self.suffix}-b2",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-02-01"],
        )
        self.resource.write(
            batched_b2,
            target=f"test_write.{self.table_name}",
            context=wctx_b2,
            write_mode="full_refresh",
            contract=contract,
        )

        rows = self.builder.read_all(self.fqn, order_by="id")
        # Partition A rows must remain (scoped delete of partition B
        # didn't touch them).
        partitions_present = {(r["id"], r["load_period"]) for r in rows}
        assert ("a1", "2024-01-01") in partitions_present
        assert ("a2", "2024-01-01") in partitions_present
        # Partition B's first write was overwritten by the second.
        assert ("b1", "2024-02-01") not in partitions_present
        assert ("b2", "2024-02-01") in partitions_present
