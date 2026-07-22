"""Integration tests for PostgresIOManager upsert write mode.

Validates INSERT ON CONFLICT behavior including initial inserts, row updates,
mixed insert/update, idempotency, primary key handling (single, composite,
missing), column control (update_columns, DO NOTHING), and data type support
against a real PostgreSQL testcontainer.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# TestUpsertBasic
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUpsertBasic:
    """Core upsert behavior: insert, update, mixed, and idempotency."""

    TABLE_NAME: str = f"ups_basic_{uuid.uuid4().hex[:8]}"

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

    def test_initial_insert(self) -> None:
        """Upsert into an empty table inserts all rows."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 3
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert rows[0]["name"] == "a"
        assert rows[2]["name"] == "c"

    def test_update_existing_rows(self) -> None:
        """Upsert with matching keys updates existing rows."""
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "value"],
            rows=[(1, "old_a", 10.0), (2, "old_b", 20.0)],
        )

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": [1, 2], "name": ["new_a", "new_b"], "value": [11.0, 22.0]})
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 2
        assert rows[0]["name"] == "new_a"
        assert float(rows[0]["value"]) == 11.0
        assert rows[1]["name"] == "new_b"
        assert float(rows[1]["value"]) == 22.0

    def test_mixed_insert_and_update(self) -> None:
        """Upsert handles a mix of new and existing rows."""
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "value"],
            rows=[(1, "existing", 10.0)],
        )

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["updated", "brand_new_b", "brand_new_c"],
                "value": [99.0, 2.0, 3.0],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3
        assert rows[0]["name"] == "updated"
        assert float(rows[0]["value"]) == 99.0
        assert rows[1]["name"] == "brand_new_b"
        assert rows[2]["name"] == "brand_new_c"

    def test_no_change_idempotent(self) -> None:
        """Upserting the same data twice leaves the table unchanged."""
        ctx1 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "value": [1.0, 2.0]})
        self.io_mgr.handle_output(ctx1, df)

        ctx2 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        self.io_mgr.handle_output(ctx2, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 2
        assert rows[0]["name"] == "a"
        assert float(rows[0]["value"]) == 1.0


# ---------------------------------------------------------------------------
# TestUpsertPrimaryKey
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUpsertPrimaryKey:
    """Primary key variations: single, composite, missing."""

    SINGLE_PK_TABLE: str = f"ups_pk_single_{uuid.uuid4().hex[:8]}"
    COMPOSITE_PK_TABLE: str = f"ups_pk_composite_{uuid.uuid4().hex[:8]}"
    NO_PK_TABLE: str = f"ups_pk_missing_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.single_fqn = table_builder.create_table(
            self.SINGLE_PK_TABLE,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT"},
            primary_key=["id"],
        )
        self.composite_fqn = table_builder.create_table(
            self.COMPOSITE_PK_TABLE,
            columns={
                "region": "TEXT NOT NULL",
                "product_id": "INTEGER NOT NULL",
                "quantity": "INTEGER",
            },
            primary_key=["region", "product_id"],
        )
        self.no_pk_fqn = table_builder.create_table(
            self.NO_PK_TABLE,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT"},
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.single_fqn)
        self.builder.drop(self.composite_fqn)
        self.builder.drop(self.no_pk_fqn)

    def test_single_column_primary_key(self) -> None:
        """Upsert with a single-column primary key works correctly."""
        self.builder.insert_rows(self.single_fqn, columns=["id", "name"], rows=[(1, "old")])

        ctx = make_mock_output_context(
            asset_name=self.SINGLE_PK_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": [1, 2], "name": ["updated", "new"]})
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.single_fqn, order_by="id")
        assert len(rows) == 2
        assert rows[0]["name"] == "updated"
        assert rows[1]["name"] == "new"

    def test_composite_primary_key(self) -> None:
        """Upsert with a composite primary key updates only on full key match."""
        self.builder.insert_rows(
            self.composite_fqn,
            columns=["region", "product_id", "quantity"],
            rows=[("east", 100, 5), ("west", 100, 10)],
        )

        ctx = make_mock_output_context(
            asset_name=self.COMPOSITE_PK_TABLE,
            metadata={
                "write_mode": "upsert",
                "primary_key": ["region", "product_id"],
            },
        )
        df = pl.DataFrame(
            {
                "region": ["east", "west", "south"],
                "product_id": [100, 100, 200],
                "quantity": [50, 10, 25],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.composite_fqn, order_by="region")
        assert len(rows) == 3
        # east/100 updated from 5 -> 50
        east_row = next(r for r in rows if r["region"] == "east" and r["product_id"] == 100)
        assert east_row["quantity"] == 50
        # west/100 unchanged at 10
        west_row = next(r for r in rows if r["region"] == "west")
        assert west_row["quantity"] == 10
        # south/200 is new
        south_row = next(r for r in rows if r["region"] == "south")
        assert south_row["quantity"] == 25

    def test_missing_primary_key_raises(self) -> None:
        """Upsert without primary_key raises ValueError."""
        ctx = make_mock_output_context(
            asset_name=self.NO_PK_TABLE,
            metadata={"write_mode": "upsert"},
        )
        df = pl.DataFrame({"id": [1], "name": ["a"]})

        with pytest.raises(ValueError, match="requires primary_key"):
            self.io_mgr.handle_output(ctx, df)


# ---------------------------------------------------------------------------
# TestUpsertColumnControl
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUpsertColumnControl:
    """Column update control: all non-key, specific, and DO NOTHING."""

    TABLE_NAME: str = f"ups_colctrl_{uuid.uuid4().hex[:8]}"

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
                "name": "TEXT",
                "category": "TEXT",
                "score": "NUMERIC",
            },
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

    def test_update_all_non_key_columns(self) -> None:
        """Default upsert (no update_columns) updates all non-PK columns."""
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "category", "score"],
            rows=[(1, "old_name", "old_cat", 1.0)],
        )

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame(
            {"id": [1], "name": ["new_name"], "category": ["new_cat"], "score": [99.0]}
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert rows[0]["name"] == "new_name"
        assert rows[0]["category"] == "new_cat"
        assert float(rows[0]["score"]) == 99.0

    def test_update_specific_columns(self) -> None:
        """update_columns restricts which columns are updated on conflict."""
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "category", "score"],
            rows=[(1, "original_name", "original_cat", 1.0)],
        )

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={
                "write_mode": "upsert",
                "primary_key": ["id"],
                "update_columns": ["score"],
            },
        )
        df = pl.DataFrame(
            {"id": [1], "name": ["changed_name"], "category": ["changed_cat"], "score": [42.0]}
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        # Only score should be updated; name and category remain original
        assert rows[0]["name"] == "original_name"
        assert rows[0]["category"] == "original_cat"
        assert float(rows[0]["score"]) == 42.0

    def test_do_nothing_on_conflict(self) -> None:
        """update_columns=[] (empty) yields ON CONFLICT DO NOTHING."""
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "category", "score"],
            rows=[(1, "keep_this", "keep_cat", 1.0)],
        )

        # Setting update_columns to the primary key columns only means the set
        # of update columns (non-key minus explicit) becomes empty, triggering
        # DO NOTHING. We achieve this by making update_columns equal to the PK.
        # Actually, the IO manager has special handling: if update_columns
        # resolves to an empty list, it uses DO NOTHING. We can trigger this
        # by setting update_columns to an empty list explicitly.
        io_mgr = PostgresIOManager(
            postgres_resource=self.io_mgr.postgres_resource,
            db_schema="test_write",
            update_columns=[],
        )

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["should_not_overwrite", "new_row"],
                "category": ["ignored", "new_cat"],
                "score": [999.0, 2.0],
            }
        )
        io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 2
        # Row 1: unchanged because DO NOTHING on conflict
        assert rows[0]["name"] == "keep_this"
        assert float(rows[0]["score"]) == 1.0
        # Row 2: inserted because no conflict
        assert rows[1]["name"] == "new_row"


# ---------------------------------------------------------------------------
# TestUpsertDataTypes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUpsertDataTypes:
    """Data type handling in upsert mode."""

    NULLABLE_TABLE: str = f"ups_nullable_{uuid.uuid4().hex[:8]}"
    TYPES_TABLE: str = f"ups_types_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.nullable_fqn = table_builder.create_table(
            self.NULLABLE_TABLE,
            columns={
                "id": "INTEGER NOT NULL",
                "optional_name": "TEXT",
                "optional_value": "NUMERIC",
            },
            primary_key=["id"],
        )
        self.types_fqn = table_builder.create_table(
            self.TYPES_TABLE,
            columns={
                "id": "INTEGER NOT NULL",
                "label": "TEXT",
                "amount": "DOUBLE PRECISION",
                "count": "BIGINT",
                "active": "BOOLEAN",
            },
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.nullable_fqn)
        self.builder.drop(self.types_fqn)

    def test_nullable_columns(self) -> None:
        """Upsert correctly handles NULL values in insert and update paths."""
        # Insert with a NULL value
        ctx1 = make_mock_output_context(
            asset_name=self.NULLABLE_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df1 = pl.DataFrame(
            {
                "id": [1, 2],
                "optional_name": ["has_name", None],
                "optional_value": [10.0, None],
            }
        )
        self.io_mgr.handle_output(ctx1, df1)

        rows = self.builder.read_all(self.nullable_fqn, order_by="id")
        assert rows[0]["optional_name"] == "has_name"
        assert rows[1]["optional_name"] is None
        assert rows[1]["optional_value"] is None

        # Update: set previously non-NULL to NULL and NULL to non-NULL
        ctx2 = make_mock_output_context(
            asset_name=self.NULLABLE_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df2 = pl.DataFrame(
            {
                "id": [1, 2],
                "optional_name": [None, "now_has_name"],
                "optional_value": [None, 20.0],
            }
        )
        self.io_mgr.handle_output(ctx2, df2)

        rows = self.builder.read_all(self.nullable_fqn, order_by="id")
        assert rows[0]["optional_name"] is None
        assert rows[0]["optional_value"] is None
        assert rows[1]["optional_name"] == "now_has_name"
        assert float(rows[1]["optional_value"]) == 20.0

    def test_text_and_numeric_types(self) -> None:
        """Upsert preserves TEXT, DOUBLE PRECISION, BIGINT, and BOOLEAN types."""
        ctx = make_mock_output_context(
            asset_name=self.TYPES_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame(
            {
                "id": [1, 2],
                "label": ["alpha", "beta"],
                "amount": [3.14159, -0.001],
                "count": [1_000_000, 0],
                "active": [True, False],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.types_fqn, order_by="id")
        assert len(rows) == 2

        r1 = rows[0]
        assert r1["label"] == "alpha"
        assert abs(r1["amount"] - 3.14159) < 1e-4
        assert r1["count"] == 1_000_000
        assert r1["active"] is True

        r2 = rows[1]
        assert r2["label"] == "beta"
        assert abs(r2["amount"] - (-0.001)) < 1e-6
        assert r2["count"] == 0
        assert r2["active"] is False

        # Upsert with updated values to confirm type preservation on UPDATE path
        ctx2 = make_mock_output_context(
            asset_name=self.TYPES_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df2 = pl.DataFrame(
            {
                "id": [1, 2],
                "label": ["gamma", "delta"],
                "amount": [2.71828, 100.5],
                "count": [42, 9_999_999],
                "active": [False, True],
            }
        )
        self.io_mgr.handle_output(ctx2, df2)

        rows_updated = self.builder.read_all(self.types_fqn, order_by="id")
        assert rows_updated[0]["label"] == "gamma"
        assert rows_updated[0]["active"] is False
        assert rows_updated[1]["count"] == 9_999_999
        assert rows_updated[1]["active"] is True
