"""Integration tests for PostgresIOManager append write mode.

Tests the append write path (INSERT only, no deletion) against a real PostgreSQL
testcontainer. Validates basic append semantics, bulk insert method selection,
and data type handling.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Generator
from datetime import date

import polars as pl
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Append basic semantics
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAppendBasic:
    """Verify core append behaviour: insert-only, no deletion."""

    TABLE = f"append_basic_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Generator[None, None, None]:
        self.builder = table_builder
        self.fqn = table_builder.create_table(
            self.TABLE,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT NOT NULL", "value": "NUMERIC"},
            primary_key=["id"],
        )
        self.io_mgr = io_manager_factory(db_schema="test_write")
        yield
        self.builder.drop(self.fqn)

    def test_append_to_empty_table(self) -> None:
        """Appending to an empty table inserts all rows."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE,
            metadata={"write_mode": "append"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 3

    def test_append_preserves_existing(self) -> None:
        """Appending does not remove pre-existing rows."""
        # Pre-populate
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "value"],
            rows=[(10, "existing", 99.0)],
        )
        assert self.builder.count(self.fqn) == 1

        ctx = make_mock_output_context(
            asset_name=self.TABLE,
            metadata={"write_mode": "append"},
        )
        df = pl.DataFrame({"id": [20], "name": ["new"], "value": [42.0]})
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 2
        assert rows[0]["id"] == 10
        assert rows[1]["id"] == 20

    def test_append_empty_dataframe(self) -> None:
        """Appending an empty DataFrame is a no-op."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE,
            metadata={"write_mode": "append"},
        )
        df = pl.DataFrame({"id": [], "name": [], "value": []}).cast(
            {"id": pl.Int64, "name": pl.String, "value": pl.Float64}
        )
        self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 0

    def test_multiple_appends(self) -> None:
        """Successive appends accumulate rows cumulatively."""
        for batch_idx in range(3):
            ctx = make_mock_output_context(
                asset_name=self.TABLE,
                metadata={"write_mode": "append"},
            )
            start = batch_idx * 10
            df = pl.DataFrame(
                {
                    "id": [start + 1, start + 2],
                    "name": [f"b{batch_idx}_r1", f"b{batch_idx}_r2"],
                    "value": [float(start + 1), float(start + 2)],
                }
            )
            self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 6


# ---------------------------------------------------------------------------
# Append with explicit insert methods
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAppendInsertMethods:
    """Verify that both execute_values and COPY insert methods work for append."""

    TABLE_EV = f"append_ev_{uuid.uuid4().hex[:8]}"
    TABLE_COPY = f"append_copy_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Generator[None, None, None]:
        self.builder = table_builder
        self.fqn_ev = table_builder.create_table(
            self.TABLE_EV,
            columns={"id": "INTEGER NOT NULL", "label": "TEXT NOT NULL"},
        )
        self.fqn_copy = table_builder.create_table(
            self.TABLE_COPY,
            columns={"id": "INTEGER NOT NULL", "label": "TEXT NOT NULL"},
        )
        self.io_mgr_ev = io_manager_factory(
            db_schema="test_write",
            bulk_insert_method="execute_values",
        )
        self.io_mgr_copy = io_manager_factory(
            db_schema="test_write",
            bulk_insert_method="copy",
        )
        yield
        self.builder.drop(self.fqn_ev)
        self.builder.drop(self.fqn_copy)

    def test_execute_values(self) -> None:
        """Append via execute_values inserts all rows correctly."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_EV,
            metadata={"write_mode": "append"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "label": ["x", "y", "z"]})
        self.io_mgr_ev.handle_output(ctx, df)

        assert self.builder.count(self.fqn_ev) == 3
        rows = self.builder.read_all(self.fqn_ev, order_by="id")
        assert [r["label"] for r in rows] == ["x", "y", "z"]

    def test_copy_protocol(self) -> None:
        """Append via COPY protocol inserts all rows correctly."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_COPY,
            metadata={"write_mode": "append"},
        )
        df = pl.DataFrame({"id": [4, 5, 6], "label": ["a", "b", "c"]})
        self.io_mgr_copy.handle_output(ctx, df)

        assert self.builder.count(self.fqn_copy) == 3
        rows = self.builder.read_all(self.fqn_copy, order_by="id")
        assert [r["label"] for r in rows] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Append with various data types
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAppendDataTypes:
    """Verify append handles a variety of PostgreSQL column types."""

    TABLE = f"append_dtypes_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Generator[None, None, None]:
        self.builder = table_builder
        self.fqn = table_builder.create_table(
            self.TABLE,
            columns={
                "int_col": "INTEGER NOT NULL",
                "float_col": "DOUBLE PRECISION",
                "text_col": "TEXT",
                "date_col": "DATE",
                "bool_col": "BOOLEAN",
                "nullable_col": "TEXT",
            },
        )
        self.io_mgr = io_manager_factory(db_schema="test_write")
        yield
        self.builder.drop(self.fqn)

    def test_various_column_types(self) -> None:
        """Append correctly writes int, float, text, date, bool, and nullable columns."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE,
            metadata={"write_mode": "append"},
        )
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, None],
                "text_col": ["hello", "world", "!"],
                "date_col": [date(2024, 1, 1), date(2024, 6, 15), date(2024, 12, 31)],
                "bool_col": [True, False, True],
                "nullable_col": ["val", None, "other"],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="int_col")
        assert len(rows) == 3

        # Verify types and values round-tripped correctly
        assert rows[0]["int_col"] == 1
        assert rows[0]["float_col"] == pytest.approx(1.1)
        assert rows[0]["text_col"] == "hello"
        assert rows[0]["date_col"] == date(2024, 1, 1)
        assert rows[0]["bool_col"] is True
        assert rows[0]["nullable_col"] == "val"

        # Nullable round-trip
        assert rows[1]["float_col"] == pytest.approx(2.2)
        assert rows[1]["nullable_col"] is None

        assert rows[2]["float_col"] is None
        assert rows[2]["nullable_col"] == "other"


# ---------------------------------------------------------------------------
# Streaming-memory acceptance test (Migration 012 Phase D / #245)
#
# Pre-fix ``insert_with_copy`` serialized the entire DataFrame into a
# single ``BytesIO`` before handing it to ``cursor.copy_expert`` -- peak
# Python heap during the COPY scaled with the DataFrame's serialized
# size (~50 bytes/row * row_count for typical schemas).  Post-fix the
# CSV serialization is sliced into ``insert_chunk_size`` chunks (default
# 50k rows = ~2.5 MiB CSV per chunk) so peak heap during the COPY tracks
# the chunk size rather than the full DataFrame.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestAppendCopyStreamingMemory:
    """1M-row append-via-COPY stays within bounded peak heap delta."""

    TABLE = f"append_mem_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Generator[None, None, None]:
        self.builder = table_builder
        self.fqn = table_builder.create_table(
            self.TABLE,
            columns={
                "id": "INTEGER NOT NULL",
                "k1": "TEXT NOT NULL",
                "k2": "TEXT NOT NULL",
                "v_int": "INTEGER",
                "v_numeric": "NUMERIC(12, 4)",
            },
        )
        # Force the COPY path regardless of row count, and use the
        # default 50k-row chunk size (auto-mode kicks in at >= 50k).
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            bulk_insert_method="copy",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def test_one_million_row_append_is_memory_bounded(self) -> None:
        """1M-row append-via-COPY peaks at <= ~50 MiB (psycopg2) / 25 MiB (psycopg3).

        Pre-fix the CSV serialization for 1M rows of this schema would
        produce a ~60 MiB ``BytesIO`` buffer, all alive on the Python
        heap during ``cursor.copy_expert``.  Migration 012 chunked the
        serialization at 50k rows = ~3 MiB per chunk; Migration 014
        Phase D ported the COPY path through the driver seam so under
        psycopg3 a single ``cursor.copy()`` context streams every
        chunk into one COPY invocation rather than N separate
        ``copy_expert`` calls.

        Thresholds:
        - psycopg2: 50 MiB delta -- generous enough to absorb
          COPY-protocol bookkeeping, psycopg2 buffers, and tracemalloc's
          own overhead, while still failing loud if a future change
          re-introduces the full-payload BytesIO.
        - psycopg3: 25 MiB delta -- the streaming win lets us halve the
          bound; one COPY invocation, one chunk in flight at a time, no
          per-chunk BytesIO reset overhead from the psycopg2 path's N
          separate copy_expert calls.
        """
        import os
        import tracemalloc

        n_rows = 1_000_000
        threshold_mib = 25 if os.environ.get("MONC_PG_DRIVER") == "psycopg3" else 50
        threshold_bytes = threshold_mib * 1024 * 1024

        # Build a wide-ish DataFrame (~60-80 MiB serialized as CSV) BEFORE
        # the tracemalloc baseline so the in-memory DataFrame does not
        # count against the peak measurement.
        df = pl.DataFrame(
            {
                "id": list(range(n_rows)),
                "k1": [f"k1_{i:08d}" for i in range(n_rows)],
                "k2": [f"k2_{i:08d}" for i in range(n_rows)],
                "v_int": list(range(n_rows)),
                "v_numeric": [float(i) / 100.0 for i in range(n_rows)],
            }
        )
        ctx = make_mock_output_context(
            asset_name=self.TABLE,
            metadata={"write_mode": "append"},
        )

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            self.io_mgr.handle_output(ctx, df)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # Sanity: row count round-tripped.
        assert self.builder.count(self.fqn) == n_rows

        assert peak <= threshold_bytes, (
            f"peak Python heap delta during a {n_rows:,}-row "
            f"COPY-append was {peak / 1024 / 1024:.1f} MiB -- "
            f"streaming regression?  Threshold: "
            f"{threshold_bytes / 1024 / 1024:.0f} MiB."
        )
