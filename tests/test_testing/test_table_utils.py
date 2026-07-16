"""Tests for table utilities module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa

from moncpipelib.testing.table_utils import create_test_table_from_model


class TestCreateTestTableFromModel:
    """Tests for create_test_table_from_model."""

    def test_executes_create_ddl(self):
        """Test that DDL is generated from model and executed on cursor."""
        # Create a mock SQLAlchemy column
        mock_column = MagicMock()
        mock_column.copy.return_value = mock_column
        mock_column.name = "id"

        # Create a mock SQLAlchemy table
        mock_table = MagicMock()
        mock_table.columns = [mock_column]

        # Create a mock SQLAlchemy model
        mock_model = MagicMock()
        mock_model.__table__ = mock_table

        # Create mock cursor and connection
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("moncpipelib.testing.table_utils.sa.Table"),
            patch("moncpipelib.testing.table_utils.CreateTable") as mock_create_table,
        ):
            mock_compiled = MagicMock()
            mock_compiled.__str__ = MagicMock(
                return_value="CREATE TABLE integration_tests.test_t (id INTEGER)"
            )
            mock_create_table.return_value.compile.return_value = mock_compiled

            create_test_table_from_model(
                conn=mock_conn,
                model=mock_model,
                target_schema="integration_tests",
                target_table="test_orders",
            )

        # Verify cursor.execute was called with DDL string
        mock_cursor.execute.assert_called_once()

    def test_model_without_table_raises(self):
        """Test that a model without __table__ raises AttributeError."""
        mock_model = type("MockModel", (), {})()
        mock_conn = MagicMock()

        with pytest.raises(AttributeError):
            create_test_table_from_model(
                conn=mock_conn,
                model=mock_model,
                target_schema="test",
                target_table="test_table",
            )

    def test_passes_correct_schema_and_table_name(self):
        """Test that target schema and table name are passed correctly."""
        mock_column = MagicMock()
        mock_column.copy.return_value = mock_column

        mock_table = MagicMock()
        mock_table.columns = [mock_column]

        mock_model = MagicMock()
        mock_model.__table__ = mock_table

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("moncpipelib.testing.table_utils.sa.Table") as mock_table_cls,
            patch("moncpipelib.testing.table_utils.CreateTable") as mock_create_table,
        ):
            mock_compiled = MagicMock()
            mock_compiled.__str__ = MagicMock(return_value="CREATE TABLE ...")
            mock_create_table.return_value.compile.return_value = mock_compiled

            create_test_table_from_model(
                conn=mock_conn,
                model=mock_model,
                target_schema="my_test_schema",
                target_table="my_test_table",
            )

            # Verify sa.Table was called with the right schema/table
            mock_table_cls.assert_called_once()
            call_args = mock_table_cls.call_args
            assert call_args[0][0] == "my_test_table"
            assert call_args[1]["schema"] == "my_test_schema"

    def test_index_metadata_stripped_from_copied_columns(self):
        """Test that index flag is cleared on columns copied to test tables.

        Regression test: col.copy() preserves index=True. When combined with
        long test table prefixes, SQLAlchemy generates index names exceeding
        PostgreSQL's 63-char identifier limit.
        """
        # Build a real SQLAlchemy model with indexed columns
        metadata = sa.MetaData()
        source_table = sa.Table(
            "dim_provider",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("_lineage_id", sa.Text, index=True, nullable=False),
            sa.Column("_lineage_key", sa.Text, index=True, nullable=False),
            schema="synthetic_gold",
        )

        # Sanity: confirm source columns have index=True
        assert source_table.columns["_lineage_id"].index is True
        assert source_table.columns["_lineage_key"].index is True

        # Create a mock model wrapping the real table
        mock_model = MagicMock()
        mock_model.__table__ = source_table

        # Mock the connection/cursor so we don't need a real database
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Call the real function (no mocking sa.Table -- we need real column copying)
        create_test_table_from_model(
            conn=mock_conn,
            model=mock_model,
            target_schema="integration_tests",
            target_table="amorgan_monc_f11823e_dim_provider",
        )

        # The DDL should have been executed without error
        mock_cursor.execute.assert_called_once()

        # Verify the test table has no indexes (only primary key constraint)
        # If index=True leaked through, the table would have auto-generated indexes
        ddl = mock_cursor.execute.call_args[0][0]
        assert "CREATE INDEX" not in ddl
