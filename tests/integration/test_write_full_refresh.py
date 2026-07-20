"""Integration tests for PostgresIOManager full_refresh write mode.

Validates DELETE/TRUNCATE clearing strategies, bulk insert methods
(execute_values / COPY), type preservation, and output metadata reporting
against a real PostgreSQL testcontainer.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager
from moncpipelib.streaming import BatchedDataFrame

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# TestFullRefreshDelete
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullRefreshDelete:
    """Full refresh using DELETE (the default for small DataFrames)."""

    TABLE_NAME: str = f"fr_delete_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT", "value": "NUMERIC"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def test_initial_load_into_empty_table(self) -> None:
        """Full refresh into an empty table inserts all rows."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 3

    def test_replaces_all_rows(self) -> None:
        """Full refresh replaces pre-existing rows with new data."""
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "value"],
            rows=[(10, "old_a", 10.0), (20, "old_b", 20.0)],
        )
        assert self.builder.count(self.fqn) == 2

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame(
            {"id": [100, 200, 300], "name": ["x", "y", "z"], "value": [100.0, 200.0, 300.0]}
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3
        assert {r["id"] for r in rows} == {100, 200, 300}

    def test_empty_dataframe_clears_table(self) -> None:
        """Full refresh with an empty DataFrame removes all existing rows."""
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "value"],
            rows=[(1, "a", 1.0), (2, "b", 2.0)],
        )
        assert self.builder.count(self.fqn) == 2

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int32),
                "name": pl.Series([], dtype=pl.Utf8),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )
        self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 0

    def test_delete_method_explicit(self) -> None:
        """full_refresh_method='delete' uses DELETE even for large DataFrames."""
        resource = self.io_mgr.postgres_resource.model_copy(
            update={"full_refresh_method": "delete"},
        )
        io_mgr = PostgresIOManager(
            postgres_resource=resource,
            db_schema="test_write",
        )
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "value"],
            rows=[(1, "a", 1.0)],
        )

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame({"id": [10, 20], "name": ["x", "y"], "value": [10.0, 20.0]})
        io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 2

        # Verify the metadata reports clear_method=delete
        ctx.add_output_metadata.assert_called_once()
        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["clear_method"].value == "delete"


# ---------------------------------------------------------------------------
# TestFullRefreshTruncate
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullRefreshTruncate:
    """Full refresh with TRUNCATE clearing strategy."""

    TABLE_NAME: str = f"fr_truncate_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr_factory = io_manager_factory
        yield
        self.builder.drop(self.fqn)

    def test_truncate_method_explicit(self) -> None:
        """full_refresh_method='truncate' always uses TRUNCATE."""
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            full_refresh_method="truncate",
        )
        self.builder.insert_rows(self.fqn, columns=["id", "name"], rows=[(1, "a"), (2, "b")])

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame({"id": [10], "name": ["x"]})
        io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 1
        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["clear_method"].value == "truncate"

    def test_auto_uses_delete_for_small(self) -> None:
        """AUTO mode with a small DataFrame (<threshold) uses DELETE."""
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            full_refresh_method="auto",
            full_refresh_threshold=100,
        )
        self.builder.insert_rows(self.fqn, columns=["id", "name"], rows=[(1, "a")])

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        # 3 rows is well below the threshold of 100
        df = pl.DataFrame({"id": [10, 20, 30], "name": ["x", "y", "z"]})
        io_mgr.handle_output(ctx, df)

        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["clear_method"].value == "delete"

    def test_auto_uses_truncate_for_large(self) -> None:
        """AUTO mode with a DataFrame >= threshold uses TRUNCATE."""
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            full_refresh_method="auto",
            full_refresh_threshold=2,  # very low threshold
        )
        self.builder.insert_rows(self.fqn, columns=["id", "name"], rows=[(1, "a")])

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        # 5 rows >= threshold of 2
        df = pl.DataFrame({"id": [10, 20, 30, 40, 50], "name": ["a", "b", "c", "d", "e"]})
        io_mgr.handle_output(ctx, df)

        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["clear_method"].value == "truncate"

    def test_auto_batched_without_hint_uses_existing_row_count(self) -> None:
        """Streamed AUTO sizes the clear on the target, not the incoming frame.

        The batched path has no incoming row count -- ``total_rows_hint`` is
        optional progress metadata -- so before #4 it passed 0 and AUTO could
        never reach TRUNCATE at any volume. The decision now falls back to the
        target's ``pg_class.reltuples``.

        The seeded rows must be ANALYZEd: ``reltuples`` is -1 on a table that
        has never been analyzed, which is deliberately read as "no estimate"
        and would assert the DELETE fallback rather than the fix.
        """
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            full_refresh_method="auto",
            full_refresh_threshold=3,
        )
        # 5 existing rows >= threshold of 3, so the clear should TRUNCATE.
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name"],
            rows=[(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")],
        )
        self.builder.analyze(self.fqn)

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        # One small batch, and no total_rows_hint -- the pre-#4 shape.
        batched = BatchedDataFrame(batches=iter([pl.DataFrame({"id": [10], "name": ["x"]})]))
        io_mgr.handle_output(ctx, batched)

        assert self.builder.count(self.fqn) == 1
        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["clear_method"].value == "truncate"

    def test_auto_batched_small_existing_table_still_deletes(self) -> None:
        """The fallback is a real decision, not a blanket switch to TRUNCATE."""
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            full_refresh_method="auto",
            full_refresh_threshold=100,
        )
        self.builder.insert_rows(self.fqn, columns=["id", "name"], rows=[(1, "a"), (2, "b")])
        self.builder.analyze(self.fqn)

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        batched = BatchedDataFrame(batches=iter([pl.DataFrame({"id": [10], "name": ["x"]})]))
        io_mgr.handle_output(ctx, batched)

        assert self.builder.count(self.fqn) == 1
        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["clear_method"].value == "delete"

    def test_auto_batched_never_analyzed_target_deletes(self) -> None:
        """``reltuples = -1`` is unknown, not zero -- stay on the safer path."""
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            full_refresh_method="auto",
            full_refresh_threshold=3,
        )
        # Seeded well above the threshold, but deliberately NOT analyzed.
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name"],
            rows=[(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")],
        )

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        batched = BatchedDataFrame(batches=iter([pl.DataFrame({"id": [10], "name": ["x"]})]))
        io_mgr.handle_output(ctx, batched)

        assert self.builder.count(self.fqn) == 1
        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["clear_method"].value == "delete"


# ---------------------------------------------------------------------------
# TestFullRefreshInsertMethods
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullRefreshInsertMethods:
    """Validate execute_values and COPY insert methods."""

    TABLE_NAME: str = f"fr_insert_{uuid.uuid4().hex[:8]}"
    TYPED_TABLE_NAME: str = f"fr_typed_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT"},
            primary_key=["id"],
        )
        self.typed_fqn = table_builder.create_table(
            self.TYPED_TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "score": "DOUBLE PRECISION",
                "label": "TEXT",
                "event_date": "DATE",
                "active": "BOOLEAN",
                "notes": "TEXT",
            },
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr_factory = io_manager_factory
        yield
        self.builder.drop(self.fqn)
        self.builder.drop(self.typed_fqn)

    def test_execute_values_explicit(self) -> None:
        """bulk_insert_method='execute_values' forces execute_values path."""
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            bulk_insert_method="execute_values",
        )
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 2
        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["insert_method"].value == "execute_values"

    def test_copy_explicit(self) -> None:
        """bulk_insert_method='copy' forces COPY protocol path."""
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            bulk_insert_method="copy",
        )
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 3
        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["insert_method"].value == "copy"

    def test_copy_preserves_types(self) -> None:
        """COPY inserts preserve int, float, text, date, bool, and nullable columns."""
        io_mgr = self.io_mgr_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            bulk_insert_method="copy",
        )
        ctx = make_mock_output_context(
            asset_name=self.TYPED_TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame(
            {
                "id": [1, 2],
                "score": [3.14, 2.72],
                "label": ["alpha", "beta"],
                "event_date": [date(2025, 1, 15), date(2025, 6, 30)],
                "active": [True, False],
                "notes": ["some text", None],
            }
        )
        io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.typed_fqn, order_by="id")
        assert len(rows) == 2

        # Row 1: all values present
        r1 = rows[0]
        assert r1["id"] == 1
        assert abs(r1["score"] - 3.14) < 1e-9
        assert r1["label"] == "alpha"
        assert r1["event_date"] == date(2025, 1, 15)
        assert r1["active"] is True
        assert r1["notes"] == "some text"

        # Row 2: nullable column is NULL
        r2 = rows[1]
        assert r2["id"] == 2
        assert r2["active"] is False
        assert r2["notes"] is None


# ---------------------------------------------------------------------------
# TestFullRefreshMetadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullRefreshMetadata:
    """Verify output metadata is emitted with expected keys."""

    TABLE_NAME: str = f"fr_meta_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def test_output_metadata_emitted(self) -> None:
        """add_output_metadata is called with write_mode, target_table, and stat keys."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        self.io_mgr.handle_output(ctx, df)

        ctx.add_output_metadata.assert_called_once()
        meta = ctx.add_output_metadata.call_args[0][0]

        # Core keys that must always be present
        expected_keys = {
            "write_mode",
            "target_table",
            "column_count",
            "columns",
            "rows_deleted",
            "rows_inserted",
            "clear_method",
            "insert_method",
        }
        assert expected_keys.issubset(set(meta.keys())), (
            f"Missing metadata keys: {expected_keys - set(meta.keys())}"
        )

        # Validate values
        assert meta["write_mode"].value == "full_refresh"
        assert meta["rows_inserted"].value == 2
        assert meta["column_count"].value == 2
