"""Integration tests for PostgresIOManager.load_input against a real PostgreSQL database.

Tests the complete input path: table read, column projection, empty tables,
and data type round-trip fidelity with only the Dagster context mocked.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import polars as pl
import psycopg
import pytest

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


class TestLoadInput:
    """Verify PostgresIOManager.load_input reads data correctly from PostgreSQL."""

    TABLE_COLUMNS: dict[str, str] = {
        "id": "INTEGER",
        "name": "TEXT",
        "value": "DOUBLE PRECISION",
        "active": "BOOLEAN",
        "created_at": "DATE",
        "notes": "TEXT",
    }

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        pg_connection: psycopg.Connection,
        io_manager_factory: Any,
    ) -> Any:
        self.suffix = uuid.uuid4().hex[:8]
        self.table_name = f"load_input_{self.suffix}"
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
    # Helpers
    # ------------------------------------------------------------------

    def _write_via_handle_output(self, df: pl.DataFrame) -> None:
        """Write data through handle_output so the round-trip path is exercised."""
        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "full_refresh"},
        )
        self.io_mgr.handle_output(ctx, df)

    def _load_input(
        self,
        *,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Build a mock InputContext and call load_input."""
        from unittest.mock import MagicMock

        context = MagicMock()
        context.asset_key.path = [self.table_name]
        context.log = MagicMock()

        if columns is not None:
            context.upstream_output = MagicMock()
            context.upstream_output.metadata = {"columns": columns}
        else:
            context.upstream_output = None

        return self.io_mgr.load_input(context)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_read_full_table(self) -> None:
        """Write data via handle_output, then load_input; verify DataFrame matches."""
        source_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["alpha", "bravo", "charlie"],
                "value": [1.1, 2.2, 3.3],
                "active": [True, False, True],
                "created_at": [date(2025, 1, 1), date(2025, 6, 15), date(2025, 12, 31)],
                "notes": ["first", "second", "third"],
            }
        )
        self._write_via_handle_output(source_df)

        result = self._load_input()

        assert len(result) == 3
        assert set(result.columns) == set(source_df.columns)

        # Sort by id for deterministic comparison
        result_sorted = result.sort("id")
        assert result_sorted["name"].to_list() == ["alpha", "bravo", "charlie"]
        assert result_sorted["value"].to_list() == [1.1, 2.2, 3.3]

    def test_column_projection(self) -> None:
        """Upstream metadata specifies columns; verify only those are returned."""
        source_df = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["alice", "bob"],
                "value": [10.0, 20.0],
                "active": [True, False],
                "created_at": [date(2025, 3, 1), date(2025, 3, 2)],
                "notes": ["note-a", "note-b"],
            }
        )
        self._write_via_handle_output(source_df)

        result = self._load_input(columns=["id", "name"])

        assert list(result.columns) == ["id", "name"]
        assert len(result) == 2

    def test_empty_table(self) -> None:
        """Load from an empty table; verify an empty DataFrame is returned."""
        result = self._load_input()

        assert len(result) == 0
        assert isinstance(result, pl.DataFrame)

    def test_data_types_round_trip(self) -> None:
        """Write int/float/text/date/bool/null, load, verify types preserved."""
        source_df = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["present", None],
                "value": [99.9, None],
                "active": [True, None],
                "created_at": [date(2025, 7, 4), None],
                "notes": [None, "has-notes"],
            }
        )
        self._write_via_handle_output(source_df)

        result = self._load_input()
        result_sorted = result.sort("id")

        assert len(result_sorted) == 2

        # Row 1: all values present
        row1 = result_sorted.row(0, named=True)
        assert row1["id"] == 1
        assert row1["name"] == "present"
        assert row1["value"] == pytest.approx(99.9)
        assert row1["active"] is True
        assert row1["created_at"] == date(2025, 7, 4)
        assert row1["notes"] is None

        # Row 2: nulls where expected
        row2 = result_sorted.row(1, named=True)
        assert row2["id"] == 2
        assert row2["name"] is None
        assert row2["value"] is None
        assert row2["active"] is None
        assert row2["created_at"] is None
        assert row2["notes"] == "has-notes"
