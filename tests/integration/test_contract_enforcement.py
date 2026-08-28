"""Integration tests for contract mode and primary key enforcement.

Validates that contract YAML files drive IO manager write behaviour when
loaded via ``contract_search_paths``. Covers mode reconciliation, primary key
reconciliation, conflict detection, and combined contract-driven upsert.

Requires Docker. Run with: uv run pytest -m integration tests/integration/test_contract_enforcement.py -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
import yaml

from moncpipelib.contracts import (
    ContractEnforcementMode,
    ContractViolationError,
)
from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import SCD2TableBuilder, SCD2Verifier, TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_contract(
    tmp_path: Any,
    asset_name: str,
    *,
    columns: list[dict[str, Any]],
    sinks: list[dict[str, Any]] | None = None,
    layer: str = "silver",
) -> None:
    """Write a contract YAML file to ``tmp_path`` for the given asset."""
    contract: dict[str, Any] = {
        "version": "1.0",
        "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "asset": asset_name,
        "layer": layer,
        "schema": {"columns": columns},
    }
    if sinks is not None:
        contract["sinks"] = sinks
    contract_file = tmp_path / f"{asset_name}.contract.yaml"
    contract_file.write_text(yaml.dump(contract, sort_keys=False))


# ---------------------------------------------------------------------------
# A. Mode Enforcement (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestContractModeEnforcement:
    """Contract sink ``mode`` drives write behaviour end-to-end."""

    TABLE_NAME: str = f"cme_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        tmp_path: Any,
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.tmp_path = tmp_path
        self._io_factory = io_manager_factory
        yield
        self.builder.drop(self.fqn)

    def _make_io_manager(self, tmp_path: Any) -> PostgresIOManager:
        return self._io_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts=ContractEnforcementMode.ERROR,
            contract_search_paths=[str(tmp_path)],
        )

    # ------------------------------------------------------------------
    # 1. Contract mode drives write behaviour
    # ------------------------------------------------------------------

    def test_contract_mode_drives_upsert(self) -> None:
        """Contract sink mode=upsert overrides IO manager default (full_refresh).

        Pre-populates the table, writes overlapping data, and asserts that
        existing rows were updated (not deleted+replaced) via upsert.
        """
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name"],
            rows=[(1, "original"), (2, "keep_me")],
        )

        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "name", "type": "string", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "upsert",
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={},  # No write_mode in metadata — contract is sole source
        )
        df = pl.DataFrame({"id": [1, 3], "name": ["updated", "new_row"]})
        io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3
        assert rows[0]["name"] == "updated"  # id=1 upserted
        assert rows[1]["name"] == "keep_me"  # id=2 untouched (not full_refresh)
        assert rows[2]["name"] == "new_row"  # id=3 inserted

    # ------------------------------------------------------------------
    # 2. Contract mode conflicts with asset metadata → error
    # ------------------------------------------------------------------

    def test_contract_mode_conflict_raises_error(self) -> None:
        """Contract mode=full_refresh + metadata mode=append → ContractViolationError."""
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "full_refresh",
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "append"},
        )
        df = pl.DataFrame({"id": [1], "name": ["should_not_write"]})

        with pytest.raises(ContractViolationError, match="Write mode conflict"):
            io_mgr.handle_output(ctx, df)

        # Table should be unchanged
        assert self.builder.count(self.fqn) == 0

    # ------------------------------------------------------------------
    # 3. Contract mode agrees with metadata → warning
    # ------------------------------------------------------------------

    def test_contract_mode_agreement_logs_warning(self) -> None:
        """Both contract and metadata say full_refresh → write succeeds with warning."""
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "full_refresh",
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh"},
        )
        df = pl.DataFrame({"id": [10, 20], "name": ["alpha", "beta"]})
        io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 2
        assert rows[0]["name"] == "alpha"
        assert rows[1]["name"] == "beta"

        # Verify redundancy warning was logged (bare table name, not schema-qualified)
        ctx.log.warning.assert_any_call(
            f"Write mode 'full_refresh' is declared in both the asset "
            f"metadata and the contract's sink definition for "
            f"'{self.TABLE_NAME}'. "
            f"Remove it from one location to avoid ambiguity."
        )


# ---------------------------------------------------------------------------
# B. Primary Key Enforcement (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestContractPrimaryKeyEnforcement:
    """Contract ``primary_key: true`` columns drive upsert key end-to-end."""

    TABLE_NAME: str = f"cpke_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        tmp_path: Any,
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.tmp_path = tmp_path
        self._io_factory = io_manager_factory
        yield
        self.builder.drop(self.fqn)

    def _make_io_manager(self, tmp_path: Any) -> PostgresIOManager:
        return self._io_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts=ContractEnforcementMode.ERROR,
            contract_search_paths=[str(tmp_path)],
        )

    # ------------------------------------------------------------------
    # 4. Contract PK drives upsert key
    # ------------------------------------------------------------------

    def test_contract_pk_drives_upsert(self) -> None:
        """Contract primary_key columns are used as the upsert key.

        No ``primary_key`` in asset metadata — contract is the sole source.
        """
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name"],
            rows=[(1, "original"), (2, "keep_me")],
        )

        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "name", "type": "string", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "upsert",
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={},  # No write_mode, no primary_key — contract drives both
        )
        df = pl.DataFrame({"id": [1, 3], "name": ["updated", "new_row"]})
        io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3
        assert rows[0]["name"] == "updated"
        assert rows[1]["name"] == "keep_me"
        assert rows[2]["name"] == "new_row"

    # ------------------------------------------------------------------
    # 5. Contract PK conflicts with metadata → error
    # ------------------------------------------------------------------

    def test_contract_pk_conflict_raises_error(self) -> None:
        """Contract says PK=id, metadata says PK=name → ContractViolationError."""
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "name", "type": "string", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "upsert",
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["name"]},
        )
        df = pl.DataFrame({"id": [1], "name": ["should_not_write"]})

        with pytest.raises(ContractViolationError, match="Primary key conflict"):
            io_mgr.handle_output(ctx, df)

    # ------------------------------------------------------------------
    # 6. Contract PK agrees with metadata → warning
    # ------------------------------------------------------------------

    def test_contract_pk_agreement_logs_warning(self) -> None:
        """Both contract and metadata say PK=id → write succeeds with warning."""
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name"],
            rows=[(1, "old")],
        )

        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "name", "type": "string", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "upsert",
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": [1], "name": ["updated"]})
        io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 1
        assert rows[0]["name"] == "updated"

        # Verify redundancy warning was logged
        ctx.log.warning.assert_any_call(
            "Primary key ['id'] is declared in both the asset metadata "
            "and the contract's column definitions. "
            "Remove it from one location to avoid ambiguity."
        )


# ---------------------------------------------------------------------------
# C. Combined Mode + PK from Contract
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestContractCombinedModePK:
    """Contract fully specifies upsert config — no metadata needed."""

    TABLE_NAME: str = f"ccmp_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        tmp_path: Any,
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT", "value": "NUMERIC"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.tmp_path = tmp_path
        self._io_factory = io_manager_factory
        yield
        self.builder.drop(self.fqn)

    # ------------------------------------------------------------------
    # 7. Contract fully specifies upsert config
    # ------------------------------------------------------------------

    def test_contract_drives_full_upsert_no_metadata(self) -> None:
        """Contract provides both mode=upsert and primary_key — empty metadata works.

        Pre-populates the table, writes overlapping + new data, and confirms
        the upsert used the contract-specified primary key correctly.
        """
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "value"],
            rows=[
                (1, "alpha", 10.0),
                (2, "beta", 20.0),
            ],
        )

        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "value", "type": "decimal", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "upsert",
                },
            ],
        )

        io_mgr = self._io_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts=ContractEnforcementMode.ERROR,
            contract_search_paths=[str(self.tmp_path)],
        )

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={},  # Completely empty — contract is the only config
        )
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["alpha_updated", "beta_updated", "gamma"],
                "value": [11.0, 22.0, 33.0],
            }
        )
        io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3
        assert rows[0]["name"] == "alpha_updated"
        assert float(rows[0]["value"]) == 11.0
        assert rows[1]["name"] == "beta_updated"
        assert float(rows[1]["value"]) == 22.0
        assert rows[2]["name"] == "gamma"
        assert float(rows[2]["value"]) == 33.0

    def test_contract_upsert_idempotent(self) -> None:
        """Writing the same data twice via contract-driven upsert is idempotent."""
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "value", "type": "decimal", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "upsert",
                },
            ],
        )

        io_mgr = self._io_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts=ContractEnforcementMode.ERROR,
            contract_search_paths=[str(self.tmp_path)],
        )

        df = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["a", "b"],
                "value": [1.0, 2.0],
            }
        )

        ctx1 = make_mock_output_context(asset_name=self.TABLE_NAME, metadata={})
        io_mgr.handle_output(ctx1, df)

        ctx2 = make_mock_output_context(asset_name=self.TABLE_NAME, metadata={})
        io_mgr.handle_output(ctx2, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 2
        assert rows[0]["name"] == "a"
        assert float(rows[0]["value"]) == 1.0

    def test_contract_mode_without_pk_for_full_refresh(self) -> None:
        """Contract specifying mode=full_refresh without PK works correctly.

        full_refresh does not require a primary key, so no PK in the contract
        or metadata is valid.
        """
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "value", "type": "decimal", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_write",
                    "table": self.TABLE_NAME,
                    "mode": "full_refresh",
                },
            ],
        )

        io_mgr = self._io_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts=ContractEnforcementMode.ERROR,
            contract_search_paths=[str(self.tmp_path)],
        )

        # Pre-populate
        self.builder.insert_rows(
            self.fqn,
            columns=["id", "name", "value"],
            rows=[(1, "old", 10.0), (2, "stale", 20.0)],
        )

        ctx = make_mock_output_context(asset_name=self.TABLE_NAME, metadata={})
        df = pl.DataFrame({"id": [99], "name": ["replacement"], "value": [99.0]})
        io_mgr.handle_output(ctx, df)

        # full_refresh should have replaced all rows
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 1
        assert rows[0]["id"] == 99
        assert rows[0]["name"] == "replacement"


# ---------------------------------------------------------------------------
# D. SCD2 Enforcement (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestContractSCD2Enforcement:
    """Contract SCD2 fields (business_key, tracked_columns, detect_deletes) enforced end-to-end."""

    TABLE_NAME: str = f"cscd2e_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        scd2_table_builder: SCD2TableBuilder,
        scd2_verifier: SCD2Verifier,
        io_manager_factory: Callable[..., PostgresIOManager],
        tmp_path: Any,
    ) -> Any:
        self.fqn = scd2_table_builder.create_table(
            self.TABLE_NAME,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"name": "TEXT", "price": "NUMERIC"},
        )
        self.scd2_builder = scd2_table_builder
        self.verifier = scd2_verifier
        self.tmp_path = tmp_path
        self._io_factory = io_manager_factory
        yield
        self.scd2_builder.drop(self.fqn)

    def _make_io_manager(self, tmp_path: Any) -> PostgresIOManager:
        return self._io_factory(
            db_schema="test_scd2",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts=ContractEnforcementMode.ERROR,
            contract_search_paths=[str(tmp_path)],
        )

    # ------------------------------------------------------------------
    # 8. Contract business_key drives SCD2 write (no metadata BK)
    # ------------------------------------------------------------------

    def test_contract_business_key_drives_scd2_write(self) -> None:
        """Contract business_key drives SCD2 write when metadata has no business_key.

        Pre-populates with two products, re-writes one with a name change, and
        verifies that SCD2 versioning used the contract-specified business key.
        """
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "product_id", "type": "string", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "price", "type": "decimal", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_scd2",
                    "table": self.TABLE_NAME,
                    "mode": "scd2",
                    "business_key": ["product_id"],
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)

        # Initial load: two products, no business_key in metadata
        ctx1 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "scd2"},  # no business_key — contract provides it
        )
        df1 = pl.DataFrame(
            {
                "product_id": ["P001", "P002"],
                "name": ["Widget", "Gadget"],
                "price": [10.0, 20.0],
            }
        )
        io_mgr.handle_output(ctx1, df1)

        assert self.verifier.count_current(self.fqn) == 2

        # Second load: P001 name changed, P002 unchanged
        ctx2 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "scd2"},
        )
        df2 = pl.DataFrame(
            {
                "product_id": ["P001", "P002"],
                "name": ["Widget Pro", "Gadget"],
                "price": [10.0, 20.0],
            }
        )
        io_mgr.handle_output(ctx2, df2)

        # P001 should have a new version; P002 unchanged
        assert self.verifier.count_total(self.fqn) == 3
        assert self.verifier.count_current(self.fqn) == 2
        assert self.verifier.count_expired(self.fqn) == 1
        current_p001 = self.verifier.get_current_row(self.fqn, "product_id", "P001")
        assert current_p001 is not None
        assert current_p001["name"] == "Widget Pro"

    # ------------------------------------------------------------------
    # 9. Contract business_key conflicts with metadata — error before write
    # ------------------------------------------------------------------

    def test_contract_business_key_conflict_raises_error(self) -> None:
        """Contract business_key=[product_id] vs metadata business_key=[name] -> error."""
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "product_id", "type": "string", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "price", "type": "decimal", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_scd2",
                    "table": self.TABLE_NAME,
                    "mode": "scd2",
                    "business_key": ["product_id"],
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "scd2", "business_key": ["name"]},  # conflict
        )
        df = pl.DataFrame(
            {
                "product_id": ["P001"],
                "name": ["Widget"],
                "price": [10.0],
            }
        )

        with pytest.raises(ContractViolationError, match="business_key conflict"):
            io_mgr.handle_output(ctx, df)

        assert self.verifier.count_total(self.fqn) == 0

    # ------------------------------------------------------------------
    # 10. Contract tracked_columns match in metadata — warning + scoped versioning
    # ------------------------------------------------------------------

    def test_contract_tracked_columns_match_limits_versioning(self) -> None:
        """Contract and metadata both declare tracked_columns=[name].

        - Warning is logged (redundant config).
        - Only name changes trigger new SCD2 versions; price changes do not.
        """
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "product_id", "type": "string", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "price", "type": "decimal", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_scd2",
                    "table": self.TABLE_NAME,
                    "mode": "scd2",
                    "business_key": ["product_id"],
                    "tracked_columns": ["name"],
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)

        # Initial load — both contract and metadata specify tracked_columns=[name]
        ctx1 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
                "tracked_columns": ["name"],  # matches contract -> warning
            },
        )
        df1 = pl.DataFrame(
            {
                "product_id": ["P001"],
                "name": ["Widget"],
                "price": [10.0],
            }
        )
        io_mgr.handle_output(ctx1, df1)
        assert self.verifier.count_total(self.fqn) == 1

        # Price change only — should NOT trigger a new version
        ctx2 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
                "tracked_columns": ["name"],
            },
        )
        df2 = pl.DataFrame(
            {
                "product_id": ["P001"],
                "name": ["Widget"],
                "price": [99.0],  # price changed, but price is not tracked
            }
        )
        io_mgr.handle_output(ctx2, df2)
        assert self.verifier.count_total(self.fqn) == 1  # no new version

        # Name change — SHOULD trigger a new version
        ctx3 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={
                "write_mode": "scd2",
                "business_key": ["product_id"],
                "tracked_columns": ["name"],
            },
        )
        df3 = pl.DataFrame(
            {
                "product_id": ["P001"],
                "name": ["Widget Pro"],  # name changed -> new version
                "price": [99.0],
            }
        )
        io_mgr.handle_output(ctx3, df3)
        assert self.verifier.count_total(self.fqn) == 2
        assert self.verifier.count_current(self.fqn) == 1
        assert self.verifier.count_expired(self.fqn) == 1

        # Verify warning was logged for the redundant tracked_columns config
        # (bare table name, not schema-qualified)
        ctx1.log.warning.assert_any_call(
            f"tracked_columns ['name'] is declared in both the asset "
            f"metadata and the contract's sink definition for "
            f"'{self.TABLE_NAME}'. "
            f"Remove it from one location to avoid ambiguity."
        )

    # ------------------------------------------------------------------
    # 11. Contract tracked_columns overrides default when not in metadata
    # ------------------------------------------------------------------

    def test_contract_tracked_columns_overrides_default(self) -> None:
        """Contract declares tracked_columns=[name] and metadata omits it -> silent override.

        The contract is authoritative; only changes to tracked columns (name)
        should trigger a new SCD2 version. Price changes are ignored.
        """
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "product_id", "type": "string", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "price", "type": "decimal", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_scd2",
                    "table": self.TABLE_NAME,
                    "mode": "scd2",
                    "business_key": ["product_id"],
                    "tracked_columns": ["name"],
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)

        # Initial load — contract is authoritative for all SCD2 config
        ctx1 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={},
        )
        df1 = pl.DataFrame(
            {
                "product_id": ["P001"],
                "name": ["Widget"],
                "price": [10.0],
            }
        )
        io_mgr.handle_output(ctx1, df1)
        assert self.verifier.count_total(self.fqn) == 1

        # Price change only — should NOT trigger a new version
        ctx2 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={},
        )
        df2 = pl.DataFrame(
            {
                "product_id": ["P001"],
                "name": ["Widget"],
                "price": [99.0],  # price changed, but price is not tracked
            }
        )
        io_mgr.handle_output(ctx2, df2)
        assert self.verifier.count_total(self.fqn) == 1  # no new version

        # Name change — SHOULD trigger a new version
        ctx3 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={},
        )
        df3 = pl.DataFrame(
            {
                "product_id": ["P001"],
                "name": ["Widget Pro"],  # name changed -> new version
                "price": [99.0],
            }
        )
        io_mgr.handle_output(ctx3, df3)
        assert self.verifier.count_total(self.fqn) == 2
        assert self.verifier.count_current(self.fqn) == 1
        assert self.verifier.count_expired(self.fqn) == 1

        # No SCD2 reconciliation warnings should have been logged (silent
        # override, not redundant).  PII metadata sync warnings and
        # tracked_columns coverage warnings are expected in this test.
        scd2_warnings = [
            c
            for c in ctx1.log.warning.call_args_list
            if "PII metadata" not in str(c) and "tracked_columns" not in str(c)
        ]
        assert scd2_warnings == []

    # ------------------------------------------------------------------
    # 12. Contract detect_deletes=True overrides default (False)
    # ------------------------------------------------------------------

    def test_contract_detect_deletes_overrides_default(self) -> None:
        """Contract detect_deletes=true overrides the default (False).

        When contract supplies detect_deletes=true and metadata doesn't set it,
        the contract value is used. Absent records get expired.
        """
        _write_contract(
            self.tmp_path,
            self.TABLE_NAME,
            columns=[
                {"name": "product_id", "type": "string", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "price", "type": "decimal", "nullable": True},
            ],
            sinks=[
                {
                    "type": "table",
                    "schema": "test_scd2",
                    "table": self.TABLE_NAME,
                    "mode": "scd2",
                    "business_key": ["product_id"],
                    "detect_deletes": True,
                },
            ],
        )

        io_mgr = self._make_io_manager(self.tmp_path)

        # Initial load: products A and B
        ctx1 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "scd2", "business_key": ["product_id"]},
        )
        df1 = pl.DataFrame(
            {
                "product_id": ["P001", "P002"],
                "name": ["Widget", "Gadget"],
                "price": [10.0, 20.0],
            }
        )
        io_mgr.handle_output(ctx1, df1)
        assert self.verifier.count_current(self.fqn) == 2

        # Second load: only P001 present (P002 absent)
        # With detect_deletes=True from contract, P002 should be expired
        ctx2 = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "scd2", "business_key": ["product_id"]},
            # no detect_deletes in metadata — contract provides it
        )
        df2 = pl.DataFrame(
            {
                "product_id": ["P001"],
                "name": ["Widget"],
                "price": [10.0],
            }
        )
        io_mgr.handle_output(ctx2, df2)

        assert self.verifier.count_current(self.fqn) == 1
        assert self.verifier.count_expired(self.fqn) == 1
        assert self.verifier.get_current_row(self.fqn, "product_id", "P001") is not None
        assert self.verifier.get_current_row(self.fqn, "product_id", "P002") is None
