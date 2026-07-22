"""Integration tests for partition-aware write modes.

Validates that full_refresh and upsert write modes correctly scope operations
to active Dagster partitions when partition_column is configured. Also tests
partition-scoped load_input and guard rail enforcement against a real PostgreSQL
testcontainer.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from moncpipelib.contracts.exceptions import ContractViolationError
from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_partitioned_output_context(
    asset_name: str,
    partition_keys: list[str],
    metadata: dict[str, Any] | None = None,
    run_id: str = "integration-test-run",
) -> MagicMock:
    """Create a mock Dagster OutputContext with partition awareness.

    Simulates a Dagster partitioned asset run where context.has_partition_key
    is True and context.asset_partition_keys returns the given keys.
    """
    ctx = make_mock_output_context(
        asset_name=asset_name,
        run_id=run_id,
        metadata=metadata,
    )
    ctx.has_partition_key = True
    ctx.asset_partition_keys = partition_keys
    ctx.partition_key = partition_keys[0]
    return ctx


def make_partitioned_input_context(
    asset_name: str,
    partition_keys: list[str],
    upstream_metadata: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock Dagster InputContext with partition awareness."""
    context = MagicMock()
    # InputContext uses upstream_output for metadata, not direct metadata
    upstream = MagicMock()
    upstream.metadata = upstream_metadata or {}
    context.upstream_output = upstream
    context.asset_key.to_user_string.return_value = asset_name
    context.asset_key.path = [asset_name]
    context.log = MagicMock()
    context.has_partition_key = True
    context.asset_partition_keys = partition_keys
    context.partition_key = partition_keys[0]
    return context


# ---------------------------------------------------------------------------
# TestPartitionScopedFullRefresh
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPartitionScopedFullRefresh:
    """Full refresh with partition scoping replaces only active partitions."""

    TABLE_NAME: str = f"pa_fr_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "region": "TEXT NOT NULL",
                "value": "NUMERIC",
            },
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def test_single_partition_scoped_full_refresh(self) -> None:
        """Only the active partition's rows are replaced; other partitions are untouched."""
        # Pre-populate with data from two partitions
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "region", "value"],
            rows=[
                (1, "east", 10.0),
                (2, "east", 20.0),
                (3, "west", 30.0),
                (4, "west", 40.0),
            ],
        )
        assert self.builder.count(self.fqn) == 4

        # Write only the "east" partition with new data
        ctx = make_partitioned_output_context(
            asset_name=self.TABLE_NAME,
            partition_keys=["east"],
            metadata={
                "write_mode": "full_refresh",
                "partition_column": "region",
            },
        )
        df = pl.DataFrame(
            {
                "id": [10, 20],
                "region": ["east", "east"],
                "value": [100.0, 200.0],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        # "west" rows should be untouched, "east" rows should be replaced
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 4  # 2 west (untouched) + 2 east (new)

        west_rows = [r for r in rows if r["region"] == "west"]
        east_rows = [r for r in rows if r["region"] == "east"]
        assert len(west_rows) == 2
        assert {r["id"] for r in west_rows} == {3, 4}  # untouched
        assert len(east_rows) == 2
        assert {r["id"] for r in east_rows} == {10, 20}  # new data

    def test_multi_partition_backfill(self) -> None:
        """Multiple partitions specified in a backfill replace all active partitions."""
        # Pre-populate with three partitions
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "region", "value"],
            rows=[
                (1, "east", 10.0),
                (2, "west", 20.0),
                (3, "north", 30.0),
            ],
        )

        # Backfill east + west (north untouched)
        ctx = make_partitioned_output_context(
            asset_name=self.TABLE_NAME,
            partition_keys=["east", "west"],
            metadata={
                "write_mode": "full_refresh",
                "partition_column": "region",
            },
        )
        df = pl.DataFrame(
            {
                "id": [10, 20],
                "region": ["east", "west"],
                "value": [100.0, 200.0],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3  # 1 north + 2 new

        north_rows = [r for r in rows if r["region"] == "north"]
        assert len(north_rows) == 1
        assert north_rows[0]["id"] == 3  # untouched

        replaced = [r for r in rows if r["region"] in ("east", "west")]
        assert len(replaced) == 2
        assert {r["id"] for r in replaced} == {10, 20}

    def test_guard_rail_no_partition_column_raises(self) -> None:
        """Partitioned full_refresh without partition_column raises ContractViolationError."""
        ctx = make_partitioned_output_context(
            asset_name=self.TABLE_NAME,
            partition_keys=["east"],
            metadata={"write_mode": "full_refresh"},
            # NOTE: no partition_column
        )
        df = pl.DataFrame(
            {
                "id": [1],
                "region": ["east"],
                "value": [10.0],
            }
        )
        with pytest.raises(ContractViolationError, match="partition_column"):
            self.io_mgr.handle_output(ctx, df)


# ---------------------------------------------------------------------------
# TestPartitionScopedUpsert
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPartitionScopedUpsert:
    """Upsert with partition-aware context (PK includes partition_column)."""

    TABLE_NAME: str = f"pa_ups_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "region": "TEXT NOT NULL",
                "value": "NUMERIC",
            },
            primary_key=["id", "region"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def test_upsert_with_partition_context(self) -> None:
        """Upsert works normally when partition_column is part of the primary key."""
        # Pre-populate
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "region", "value"],
            rows=[
                (1, "east", 10.0),
                (2, "west", 20.0),
            ],
        )

        # Upsert "east" partition: update id=1, insert id=3
        ctx = make_partitioned_output_context(
            asset_name=self.TABLE_NAME,
            partition_keys=["east"],
            metadata={
                "write_mode": "upsert",
                "primary_key": ["id", "region"],
                "partition_column": "region",
            },
        )
        df = pl.DataFrame(
            {
                "id": [1, 3],
                "region": ["east", "east"],
                "value": [100.0, 300.0],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3

        # id=1 east was updated
        east_1 = next(r for r in rows if r["id"] == 1)
        assert float(east_1["value"]) == 100.0

        # id=2 west untouched
        west_2 = next(r for r in rows if r["id"] == 2)
        assert float(west_2["value"]) == 20.0

        # id=3 east was inserted
        east_3 = next(r for r in rows if r["id"] == 3)
        assert float(east_3["value"]) == 300.0

    def test_upsert_partition_col_not_in_pk_raises(self) -> None:
        """Upsert with partition_column not in primary_key raises ContractViolationError."""
        ctx = make_partitioned_output_context(
            asset_name=self.TABLE_NAME,
            partition_keys=["east"],
            metadata={
                "write_mode": "upsert",
                "primary_key": ["id"],  # region NOT included
                "partition_column": "region",
            },
        )
        df = pl.DataFrame(
            {
                "id": [1],
                "region": ["east"],
                "value": [10.0],
            }
        )
        with pytest.raises(ContractViolationError, match="primary_key"):
            self.io_mgr.handle_output(ctx, df)


# ---------------------------------------------------------------------------
# TestPartitionAwareLoadInput
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPartitionAwareLoadInput:
    """Partition-filtered load_input returns only matching partition rows."""

    TABLE_NAME: str = f"pa_load_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "region": "TEXT NOT NULL",
                "value": "NUMERIC",
            },
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        # Pre-populate with multi-partition data
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "region", "value"],
            rows=[
                (1, "east", 10.0),
                (2, "east", 20.0),
                (3, "west", 30.0),
                (4, "north", 40.0),
            ],
        )
        yield
        self.builder.drop(self.fqn)

    def test_single_partition_load(self) -> None:
        """load_input with partition context returns only matching rows."""
        ctx = make_partitioned_input_context(
            asset_name=self.TABLE_NAME,
            partition_keys=["east"],
            upstream_metadata={"partition_column": "region"},
        )
        df = self.io_mgr.load_input(ctx)

        assert len(df) == 2
        assert set(df["region"].to_list()) == {"east"}
        assert set(df["id"].to_list()) == {1, 2}

    def test_multi_partition_load(self) -> None:
        """load_input with multiple partition keys returns all matching rows."""
        ctx = make_partitioned_input_context(
            asset_name=self.TABLE_NAME,
            partition_keys=["east", "west"],
            upstream_metadata={"partition_column": "region"},
        )
        df = self.io_mgr.load_input(ctx)

        assert len(df) == 3
        assert set(df["region"].to_list()) == {"east", "west"}

    def test_non_partitioned_load_returns_all(self) -> None:
        """load_input without partition context returns all rows."""
        context = MagicMock()
        context.asset_key.to_user_string.return_value = self.TABLE_NAME
        context.asset_key.path = [self.TABLE_NAME]
        context.log = MagicMock()
        context.has_partition_key = False
        context.upstream_output = None

        df = self.io_mgr.load_input(context)

        assert len(df) == 4
        assert set(df["region"].to_list()) == {"east", "west", "north"}


# ---------------------------------------------------------------------------
# Streaming-memory acceptance test (Migration 012 Phase C / #244)
#
# Pre-fix the parameterized ``load_input`` branch did:
#
#     rows = cursor.fetchall()                      # 1st full materialization
#     df = pl.DataFrame(
#         {col: [row[i] for row in rows] for ...},  # 2nd full materialization
#     )
#
# Both ``rows`` (tuple-of-tuples) and the dict-of-lists were alive
# simultaneously during DataFrame construction, peaking at ~2x the
# result-set size.  Post-fix, ``pl.read_database`` builds the
# DataFrame directly via Arrow buffers (allocated outside the Python
# heap).  This test pins the streaming bound: peak Python heap during
# a 100k-row partition-filtered load stays well under what the pre-fix
# dict-of-lists approach would consume.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestPartitionedLoadInputStreamingMemory:
    """100k-row partition-scoped ``load_input`` is memory-bounded."""

    TABLE_NAME: str = f"pa_load_mem_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "region": "TEXT NOT NULL",
                "v_int": "INTEGER",
                "v_text": "TEXT",
                "v_numeric": "NUMERIC",
            },
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )

        # 100k rows split across 2 partitions; the partition-filtered
        # load returns 50k of them.  Big enough that pre-fix's 2x peak
        # would be clearly distinguishable from post-fix's single
        # Arrow-backed DataFrame.
        rows = [
            (i, "east" if i % 2 == 0 else "west", i, f"v_{i:08d}", float(i) / 100.0)
            for i in range(100_000)
        ]
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "region", "v_int", "v_text", "v_numeric"],
            rows=rows,
        )
        yield
        self.builder.drop(self.fqn)

    def test_partitioned_load_100k_rows_is_memory_bounded(self) -> None:
        """50k-row partition-scoped load stays under 100 MiB peak heap.

        Pre-fix the parameterized branch held ``cursor.fetchall()``'s
        tuple-of-tuples AND a dict-of-lists rebuilt from those tuples
        in memory simultaneously, peaking at ~2x the result-set size.
        Post-fix uses ``pl.read_database(execute_options=...)`` which
        builds the DataFrame from Arrow buffers (allocated outside
        Python heap) -- peak Python heap drops accordingly.

        Threshold: 100 MiB.  Generous enough to absorb Polars'
        DataFrame baseline + tracemalloc's own overhead, while still
        failing loud if a future change re-introduces the dict-of-lists
        materialization.
        """
        import tracemalloc

        threshold_bytes = 100 * 1024 * 1024

        ctx = make_partitioned_input_context(
            asset_name=self.TABLE_NAME,
            partition_keys=["east"],
            upstream_metadata={"partition_column": "region"},
        )

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            df = self.io_mgr.load_input(ctx)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # Sanity: we actually loaded 50k rows (the partition slice).
        assert len(df) == 50_000
        assert set(df["region"].to_list()) == {"east"}

        assert peak <= threshold_bytes, (
            f"peak Python heap was {peak / 1024 / 1024:.1f} MiB during a "
            f"50k-row partition-scoped load -- streaming regression?  "
            f"Threshold: {threshold_bytes / 1024 / 1024:.0f} MiB."
        )
