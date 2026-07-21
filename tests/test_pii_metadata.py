"""Tests for PostgresResource._sync_pii_metadata (SCD2 column metadata)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from moncpipelib.contracts import Column, ColumnType, DataContract, Schema
from moncpipelib.resources.postgres import PostgresResource
from moncpipelib.resources.types import WriteContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def resource() -> PostgresResource:
    """Minimal PostgresResource for testing."""
    return PostgresResource(
        host="localhost",
        port=5432,
        user="testuser",
        password="testpass",
        database="testdb",
    )


@pytest.fixture
def wctx() -> WriteContext:
    """WriteContext with a mock logger."""
    return WriteContext(
        asset_name="test_asset",
        run_id="run-abc-123",
        log=MagicMock(),
    )


@pytest.fixture
def contract() -> DataContract:
    """Contract with mixed PII columns."""
    return DataContract(
        version="1.0",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        asset="staging.patients",
        layer="bronze",
        schema=Schema(
            columns=[
                Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
                Column(name="ssn", type=ColumnType.STRING, nullable=True, pii=True),
                Column(name="claim_id", type=ColumnType.STRING, nullable=False, pii=False),
                Column(
                    name="_bronze_run_id",
                    type=ColumnType.STRING,
                    nullable=False,
                    managed=True,
                ),
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyncPiiMetadata:
    """Tests for _sync_pii_metadata."""

    def test_inserts_for_each_non_managed_column(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Should execute close + insert for each non-managed column."""
        cursor = MagicMock()
        # Simulate all inserts producing a new row
        cursor.rowcount = 1

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        # 1 table-existence check + 3 non-managed columns * 2 SQL statements = 7
        assert cursor.execute.call_count == 7

    def test_skips_managed_columns(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Managed columns (e.g., _bronze_run_id) should not be synced."""
        cursor = MagicMock()
        cursor.rowcount = 0

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        # _bronze_run_id is managed, so only 3 columns get processed
        # Skip the first call (table-existence check, no params)
        executed_sql_calls = cursor.execute.call_args_list[1:]
        for c in executed_sql_calls:
            params = c[0][1]
            # None of the param tuples should reference the managed column
            assert "_bronze_run_id" not in params

    def test_tags_contain_pii_flag(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Tags JSON should reflect the column's pii field."""
        cursor = MagicMock()
        cursor.rowcount = 0

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        # Collect all INSERT calls (every other execute)
        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT" in c[0][0]]
        tags_by_column: dict[str, dict] = {}
        for c in insert_calls:
            params = c[0][1]
            # params[2] = column_name, params[3] = tags json
            tags_by_column[params[2]] = json.loads(params[3])

        assert tags_by_column["patient_id"] == {"pii": True, "phi": True}
        assert tags_by_column["ssn"] == {"pii": True, "phi": True}
        assert tags_by_column["claim_id"] == {"pii": False, "phi": False}

    def test_tags_phi_diverges_from_pii(
        self,
        resource: PostgresResource,
        wctx: WriteContext,
    ) -> None:
        """An explicit phi annotation overrides the pii-mirroring default (#391)."""
        contract = DataContract(
            version="1.0",
            pipeline_id="550e8400-e29b-41d4-a716-446655440000",
            asset="staging.providers",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(
                        name="provider_npi",
                        type=ColumnType.STRING,
                        nullable=False,
                        pii=True,
                        phi=False,
                    ),
                    Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
                ]
            ),
        )
        cursor = MagicMock()
        cursor.rowcount = 0

        resource._sync_pii_metadata(cursor, "staging.providers", contract, wctx)

        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT" in c[0][0]]
        tags_by_column: dict[str, dict] = {}
        for c in insert_calls:
            params = c[0][1]
            tags_by_column[params[2]] = json.loads(params[3])

        assert tags_by_column["provider_npi"] == {"pii": True, "phi": False}
        assert tags_by_column["patient_id"] == {"pii": True, "phi": True}

    def test_parses_schema_qualified_table_name(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Schema-qualified table names should be split correctly."""
        cursor = MagicMock()
        cursor.rowcount = 0

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        # First column call is an UPDATE (index 1, after table-existence check)
        first_call_params = cursor.execute.call_args_list[1][0][1]
        assert first_call_params[0] == "staging"
        assert first_call_params[1] == "patients"

    def test_defaults_to_public_schema(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Unqualified table names should default to 'public' schema."""
        cursor = MagicMock()
        cursor.rowcount = 0

        resource._sync_pii_metadata(cursor, "patients", contract, wctx)

        first_call_params = cursor.execute.call_args_list[1][0][1]
        assert first_call_params[0] == "public"
        assert first_call_params[1] == "patients"

    def test_sets_updated_by_to_run_id(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """updated_by should be set to the WriteContext run_id."""
        cursor = MagicMock()
        cursor.rowcount = 0

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT" in c[0][0]]
        for c in insert_calls:
            params = c[0][1]
            # params[4] = updated_by
            assert params[4] == "run-abc-123"

    def test_sets_contract_name(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """contract_name should be set to the contract's asset field."""
        cursor = MagicMock()
        cursor.rowcount = 0

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT" in c[0][0]]
        for c in insert_calls:
            params = c[0][1]
            # params[5] = contract_name
            assert params[5] == "staging.patients"

    def test_logs_updated_count(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Should log a message when columns are updated."""
        cursor = MagicMock()
        cursor.rowcount = 1  # every insert produces a row

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        wctx.log.info.assert_called_once()
        msg = wctx.log.info.call_args[0][0]
        assert "3 column(s)" in msg
        assert "staging.patients" in msg

    def test_no_log_when_nothing_changed(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Should not log when no new records are inserted (idempotent)."""
        cursor = MagicMock()
        cursor.rowcount = 0  # no inserts (all records already current)

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        wctx.log.info.assert_not_called()

    def test_scd2_close_and_insert_sql_structure(
        self,
        resource: PostgresResource,
        contract: DataContract,
        wctx: WriteContext,
    ) -> None:
        """Verify SQL follows close-then-insert pattern for each column."""
        cursor = MagicMock()
        cursor.rowcount = 0

        resource._sync_pii_metadata(cursor, "staging.patients", contract, wctx)

        # Skip the first call (table-existence check)
        calls = cursor.execute.call_args_list[1:]
        # For each non-managed column: UPDATE (close) then INSERT (open)
        for i in range(0, len(calls), 2):
            update_sql = calls[i][0][0]
            insert_sql = calls[i + 1][0][0]
            assert "UPDATE lineage.column_metadata" in update_sql
            assert "SET valid_to = NOW()" in update_sql
            assert "tags IS DISTINCT FROM" in update_sql
            assert "INSERT INTO lineage.column_metadata" in insert_sql
            assert "WHERE NOT EXISTS" in insert_sql
