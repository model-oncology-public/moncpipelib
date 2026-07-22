"""Integration tests for handle_output with SCD2 mode against a real PostgreSQL database.

Tests the complete output path: hash computation -> column validation ->
_execute_scd2 -> metadata reporting, with only the Dagster context mocked.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid

import polars as pl
import psycopg
import pytest

from .conftest import SCD2TableBuilder, SCD2Verifier, make_mock_output_context

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Scenario 14: Full handle_output end-to-end
# ---------------------------------------------------------------------------


class TestHandleOutputSCD2EndToEnd:
    """Full handle_output flow with real DB, mocking only Dagster context."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        scd2_table_builder: SCD2TableBuilder,
        scd2_verifier: SCD2Verifier,
        pg_connection: psycopg.Connection,
        io_manager_factory,
    ):
        self.table_name = f"dim_product_ho_{uuid.uuid4().hex[:8]}"
        self.table = scd2_table_builder.create_table(
            table_name=self.table_name,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"product_name": "TEXT", "price": "NUMERIC"},
        )
        self.conn = pg_connection
        self.verifier = scd2_verifier
        self.io_mgr_factory = io_manager_factory
        self.builder = scd2_table_builder
        yield
        self.builder.drop(self.table)

    def test_initial_load_via_handle_output(self):
        """handle_output computes hash, writes data, reports metadata."""
        io_mgr = self.io_mgr_factory()

        context = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )

        df = pl.DataFrame(
            {
                "product_id": ["PROD-001", "PROD-002"],
                "product_name": ["Widget", "Gadget"],
                "price": [9.99, 19.99],
            }
        )

        io_mgr.handle_output(context, df)

        # Verify database state
        assert self.verifier.count_current(self.table) == 2

        # Verify row_hash was computed and stored
        row = self.verifier.get_current_row(self.table, "product_id", "PROD-001")
        assert row is not None
        assert row["row_hash"] is not None
        assert len(row["row_hash"]) == 64

        # Verify Dagster metadata was reported
        context.add_output_metadata.assert_called_once()
        metadata_call = context.add_output_metadata.call_args[0][0]
        assert "write_mode" in metadata_call
        assert "business_key" in metadata_call

    def test_change_detection_via_handle_output(self):
        """handle_output correctly detects and applies SCD2 changes."""
        io_mgr = self.io_mgr_factory()

        context_v1 = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )
        df_v1 = pl.DataFrame(
            {
                "product_id": ["PROD-001", "PROD-002"],
                "product_name": ["Widget", "Gadget"],
                "price": [9.99, 19.99],
            }
        )
        io_mgr.handle_output(context_v1, df_v1)

        context_v2 = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )
        df_v2 = pl.DataFrame(
            {
                "product_id": ["PROD-001", "PROD-002"],
                "product_name": ["Widget Pro", "Gadget"],
                "price": [12.99, 19.99],
            }
        )
        io_mgr.handle_output(context_v2, df_v2)

        # PROD-001: 1 expired + 1 current; PROD-002: 1 current
        assert self.verifier.count_total(self.table) == 3
        assert self.verifier.count_current(self.table) == 2
        assert self.verifier.count_expired(self.table) == 1

        # Current PROD-001 has new values
        current = self.verifier.get_current_row(self.table, "product_id", "PROD-001")
        assert current is not None
        assert current["product_name"] == "Widget Pro"

    def test_tracked_columns_subset_via_handle_output(self):
        """Explicit tracked_columns limits which columns affect hash."""
        io_mgr = self.io_mgr_factory()

        # Add a 'category' column to the table
        table_name_cat = f"dim_product_ho_tc_{uuid.uuid4().hex[:8]}"
        fqn = self.builder.create_table(
            table_name=table_name_cat,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"product_name": "TEXT", "price": "NUMERIC", "category": "TEXT"},
        )

        context_v1 = make_mock_output_context(
            asset_name=table_name_cat,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
                "tracked_columns": ["product_name", "price"],
            },
        )
        df_v1 = pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "product_name": ["Widget"],
                "price": [9.99],
                "category": ["Tools"],
            }
        )
        io_mgr.handle_output(context_v1, df_v1)

        # Change only category (not tracked)
        context_v2 = make_mock_output_context(
            asset_name=table_name_cat,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
                "tracked_columns": ["product_name", "price"],
            },
        )
        df_v2 = pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "product_name": ["Widget"],
                "price": [9.99],
                "category": ["Hardware"],  # changed but not tracked
            }
        )
        io_mgr.handle_output(context_v2, df_v2)

        # Should still be 1 row total (no SCD2 change triggered)
        assert self.verifier.count_total(fqn) == 1

        self.builder.drop(fqn)

    def test_handle_output_with_metadata_columns(self):
        """When add_metadata_columns=True, layer columns are written."""
        table_name_meta = f"dim_product_ho_meta_{uuid.uuid4().hex[:8]}"
        fqn = self.builder.create_table(
            table_name=table_name_meta,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"product_name": "TEXT", "price": "NUMERIC"},
            extra_columns={
                "_silver_run_id": "TEXT",
                "_silver_processed_at": "DOUBLE PRECISION",
                "_source_file": "TEXT",
            },
        )

        io_mgr = self.io_mgr_factory(add_metadata_columns=True, layer="silver")

        context = make_mock_output_context(
            asset_name=table_name_meta,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )
        df = pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "product_name": ["Widget"],
                "price": [9.99],
            }
        )
        io_mgr.handle_output(context, df)

        row = self.verifier.get_current_row(fqn, "product_id", "PROD-001")
        assert row is not None
        assert row["_silver_run_id"] is not None
        assert row["_silver_processed_at"] is not None

        self.builder.drop(fqn)


# ---------------------------------------------------------------------------
# Scenario 15: Column validation against real schema
# ---------------------------------------------------------------------------


class TestHandleOutputColumnValidation:
    """_validate_columns works correctly against real information_schema."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        scd2_table_builder: SCD2TableBuilder,
        pg_connection: psycopg.Connection,
        io_manager_factory,
    ):
        self.table_name = f"dim_product_val_{uuid.uuid4().hex[:8]}"
        self.table = scd2_table_builder.create_table(
            table_name=self.table_name,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"product_name": "TEXT", "price": "NUMERIC"},
        )
        self.conn = pg_connection
        self.io_mgr_factory = io_manager_factory
        self.builder = scd2_table_builder
        yield
        self.builder.drop(self.table)

    def test_extra_column_in_dataframe_raises(self):
        """DataFrame has a column not in the table."""
        io_mgr = self.io_mgr_factory()

        context = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )
        df = pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "product_name": ["Widget"],
                "price": [9.99],
                "extra_col": ["nope"],
            }
        )

        with pytest.raises(ValueError, match="Column mismatch"):
            io_mgr.handle_output(context, df)

    def test_missing_column_in_dataframe_raises(self):
        """DataFrame is missing a column that exists in the table."""
        io_mgr = self.io_mgr_factory()

        context = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )
        # Missing 'price' column
        df = pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "product_name": ["Widget"],
            }
        )

        with pytest.raises(ValueError, match="Column mismatch"):
            io_mgr.handle_output(context, df)

    def test_scd2_temporal_columns_excluded_from_validation(self):
        """effective_from/to/is_current exist in table but not DataFrame -- should pass."""
        io_mgr = self.io_mgr_factory()

        context = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )
        df = pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "product_name": ["Widget"],
                "price": [9.99],
            }
        )

        # Should not raise (temporal columns are excluded from validation)
        io_mgr.handle_output(context, df)

    def test_identity_column_excluded_from_validation(self):
        """The 'id' identity column exists in the table but not DataFrame -- should pass."""
        io_mgr = self.io_mgr_factory()

        context = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )
        df = pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "product_name": ["Widget"],
                "price": [9.99],
            }
        )

        # The table has an 'id' IDENTITY column, but DataFrame doesn't include it.
        # _validate_columns should exclude identity columns automatically.
        io_mgr.handle_output(context, df)


# ---------------------------------------------------------------------------
# Scenario 16: detect_deletes via handle_output end-to-end
# ---------------------------------------------------------------------------


class TestHandleOutputSCD2DetectDeletes:
    """Full handle_output flow with detect_deletes=True."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        scd2_table_builder: SCD2TableBuilder,
        scd2_verifier: SCD2Verifier,
        pg_connection: psycopg.Connection,
        io_manager_factory,
    ):
        self.table_name = f"dim_product_ho_dd_{uuid.uuid4().hex[:8]}"
        self.table = scd2_table_builder.create_table(
            table_name=self.table_name,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"product_name": "TEXT", "price": "NUMERIC"},
        )
        self.conn = pg_connection
        self.verifier = scd2_verifier
        self.io_mgr_factory = io_manager_factory
        self.builder = scd2_table_builder
        yield
        self.builder.drop(self.table)

    def test_detect_deletes_via_handle_output(self):
        """handle_output with detect_deletes metadata expires absent records."""
        io_mgr = self.io_mgr_factory()

        # Initial load: two products
        context_v1 = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
            },
        )
        df_v1 = pl.DataFrame(
            {
                "product_id": ["PROD-001", "PROD-002"],
                "product_name": ["Widget", "Gadget"],
                "price": [9.99, 19.99],
            }
        )
        io_mgr.handle_output(context_v1, df_v1)
        assert self.verifier.count_current(self.table) == 2

        # Second load: only PROD-001, with detect_deletes=True
        context_v2 = make_mock_output_context(
            asset_name=self.table_name,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
                "detect_deletes": True,
            },
        )
        df_v2 = pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "product_name": ["Widget"],
                "price": [9.99],
            }
        )
        io_mgr.handle_output(context_v2, df_v2)

        # PROD-001 still current, PROD-002 expired
        assert self.verifier.count_current(self.table) == 1
        assert self.verifier.count_expired(self.table) == 1
        assert self.verifier.get_current_row(self.table, "product_id", "PROD-001") is not None
        assert self.verifier.get_current_row(self.table, "product_id", "PROD-002") is None

        # Metadata was reported
        context_v2.add_output_metadata.assert_called_once()
        metadata_call = context_v2.add_output_metadata.call_args[0][0]
        assert "write_mode" in metadata_call
