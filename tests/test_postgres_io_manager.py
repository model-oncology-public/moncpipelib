"""Tests for PostgresIOManager.

Tests focus on IO Manager responsibilities after refactor:
- Target resolution (schema cascade, table prefix, suffix stripping)
- Metadata extraction and write config building
- Contract loading and search path wiring
- Delegation to PostgresResource._write_single / _write_batched
- Deprecation warnings on legacy constructor arguments
- PII drift detection on load_input
- for_testing() factory method
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from dagster import InputContext, OutputContext

from moncpipelib.contracts import ContractEnforcementMode, DataContract
from moncpipelib.io_managers.postgres import (
    PostgresIOManager,
    WriteMode,
)
from moncpipelib.resources.postgres import PostgresResource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_resource(**overrides) -> PostgresResource:
    """Create a PostgresResource for IO Manager tests.

    Returns a real PostgresResource instance (required by Dagster's config
    validation) with test-friendly defaults.
    """
    defaults: dict = {
        "host": "localhost",
        "port": 5432,
        "user": "user",
        "password": "pass",
        "database": "testdb",
    }
    defaults.update(overrides)
    return PostgresResource(**defaults)


# ---------------------------------------------------------------------------
# TestGetWriteConfig
# ---------------------------------------------------------------------------


class TestGetWriteConfig:
    """Tests for _get_write_config method."""

    @pytest.fixture
    def io_manager(self):
        """Create IO manager for testing."""
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
            write_mode=WriteMode.FULL_REFRESH,
            primary_key=["id"],
        )

    def test_get_write_config_defaults(self, io_manager):
        """Test default write config from IO Manager."""
        context = MagicMock()
        context.metadata = None
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["write_mode"] == WriteMode.FULL_REFRESH
        assert config["primary_key"] == ["id"]

    def test_get_write_config_override_from_metadata_string(self, io_manager):
        """Test write config override from metadata with string value."""
        context = MagicMock()
        context.metadata = {
            "write_mode": "upsert",
            "primary_key": ["order_id"],
        }
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["write_mode"] == WriteMode.UPSERT
        assert config["primary_key"] == ["order_id"]

    def test_get_write_config_override_from_metadata_enum(self, io_manager):
        """Test write config override from metadata with enum value."""
        context = MagicMock()
        context.metadata = {
            "write_mode": WriteMode.APPEND,
        }
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["write_mode"] == WriteMode.APPEND

    def test_get_write_config_primary_key_string(self, io_manager):
        """Test primary key conversion from string."""
        context = MagicMock()
        context.metadata = {
            "primary_key": "order_id",
        }
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["primary_key"] == ["order_id"]

    def test_get_write_config_update_columns(self, io_manager):
        """Test update columns from metadata."""
        context = MagicMock()
        context.metadata = {
            "update_columns": ["name", "value"],
        }
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["update_columns"] == ["name", "value"]

    def test_get_write_config_partition_column(self, io_manager):
        """Test partition column from metadata."""
        context = MagicMock()
        context.metadata = {
            "partition_column": "order_date",
        }
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["partition_column"] == "order_date"

    def test_get_write_config_analyze_after_write_defaults_to_none(self, io_manager):
        """Without metadata, analyze_after_write defers to the resource setting."""
        context = MagicMock()
        context.metadata = None
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["analyze_after_write"] is None

    def test_get_write_config_analyze_after_write_from_metadata(self, io_manager):
        """Per-asset analyze_after_write metadata flows into write config."""
        context = MagicMock()
        context.metadata = {
            "analyze_after_write": "never",
        }
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["analyze_after_write"] == "never"

    def test_get_write_config_rejects_unrecognized_metadata_key(self, io_manager):
        """Test that unrecognized metadata keys raise ValueError."""
        context = MagicMock()
        context.metadata = {"mode": "append"}  # typo: should be "write_mode"
        context.asset_key.to_user_string.return_value = "test_asset"
        context.has_partition_key = False

        with pytest.raises(ValueError, match="Unrecognized metadata key.*mode"):
            io_manager._get_write_config(context)

    def test_get_write_config_rejects_multiple_unrecognized_keys(self, io_manager):
        """Test that multiple unrecognized keys are all reported."""
        context = MagicMock()
        context.metadata = {"mode": "append", "pk": "id"}
        context.asset_key.to_user_string.return_value = "test_asset"
        context.has_partition_key = False

        with pytest.raises(ValueError, match="Unrecognized metadata key"):
            io_manager._get_write_config(context)

    def test_get_write_config_accepts_all_recognized_keys(self, io_manager):
        """Test that all recognized metadata keys are accepted."""
        context = MagicMock()
        context.metadata = {
            "write_mode": "append",
            "primary_key": ["id"],
            "update_columns": ["name"],
            "partition_column": "date",
            "source_file": "data.csv",
            "data_date": "2024-01-15",
        }
        context.has_partition_key = False

        config = io_manager._get_write_config(context)
        assert config["write_mode"] == WriteMode.APPEND


class TestGetWriteConfigSCD2:
    """Tests for _get_write_config with SCD2 metadata."""

    @pytest.fixture
    def io_manager(self):
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )

    def test_scd2_defaults(self, io_manager):
        """SCD2 default column names are applied."""
        context = MagicMock()
        context.metadata = {
            "write_mode": "scd2",
            "business_key": "product_id",
        }
        context.has_partition_key = False
        config = io_manager._get_write_config(context)
        assert config["write_mode"] == WriteMode.SCD2
        assert config["business_key"] == ["product_id"]
        assert config["tracked_columns"] is None
        assert config["effective_from_col"] == "effective_from"
        assert config["effective_to_col"] == "effective_to"
        assert config["is_current_col"] == "is_current"
        assert config["hash_col"] == "row_hash"

    def test_scd2_custom_col_names(self, io_manager):
        """Custom SCD2 column names from metadata."""
        context = MagicMock()
        context.metadata = {
            "write_mode": "scd2",
            "business_key": ["pk1", "pk2"],
            "tracked_columns": ["col_a", "col_b"],
            "effective_from_col": "valid_from",
            "effective_to_col": "valid_to",
            "is_current_col": "active",
            "hash_col": "checksum",
        }
        context.has_partition_key = False
        config = io_manager._get_write_config(context)
        assert config["business_key"] == ["pk1", "pk2"]
        assert config["tracked_columns"] == ["col_a", "col_b"]
        assert config["effective_from_col"] == "valid_from"
        assert config["effective_to_col"] == "valid_to"
        assert config["is_current_col"] == "active"
        assert config["hash_col"] == "checksum"

    def test_scd2_business_key_string_to_list(self, io_manager):
        """Single string business key is coerced to list."""
        context = MagicMock()
        context.metadata = {
            "write_mode": "scd2",
            "business_key": "product_id",
        }
        context.has_partition_key = False
        config = io_manager._get_write_config(context)
        assert config["business_key"] == ["product_id"]

    def test_scd2_tracked_columns_string_to_list(self, io_manager):
        """Single string tracked_columns is coerced to list."""
        context = MagicMock()
        context.metadata = {
            "write_mode": "scd2",
            "business_key": "pk",
            "tracked_columns": "col_a",
        }
        context.has_partition_key = False
        config = io_manager._get_write_config(context)
        assert config["tracked_columns"] == ["col_a"]

    def test_detect_deletes_defaults_to_false(self, io_manager):
        """detect_deletes defaults to False when not in metadata."""
        context = MagicMock()
        context.metadata = {
            "write_mode": "scd2",
            "business_key": "product_id",
        }
        context.has_partition_key = False
        config = io_manager._get_write_config(context)
        assert config["detect_deletes"] is False

    def test_detect_deletes_override_true(self, io_manager):
        """detect_deletes can be overridden to True via metadata."""
        context = MagicMock()
        context.metadata = {
            "write_mode": "scd2",
            "business_key": "product_id",
            "detect_deletes": True,
        }
        context.has_partition_key = False
        config = io_manager._get_write_config(context)
        assert config["detect_deletes"] is True


class TestGetWriteConfigPartitionColumnExplicit:
    """Tests for partition_column_explicit tracking in _get_write_config."""

    @pytest.fixture
    def io_manager(self):
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
        )

    def test_partition_column_explicit_false_by_default(self, io_manager):
        """IO manager default does not set partition_column_explicit."""
        context = MagicMock()
        context.metadata = {}
        context.asset_key.to_user_string.return_value = "test_asset"
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["partition_column_explicit"] is False

    def test_partition_column_explicit_true_when_set_in_metadata(self, io_manager):
        """Asset metadata partition_column sets partition_column_explicit=True."""
        context = MagicMock()
        context.metadata = {"partition_column": "report_date"}
        context.asset_key.to_user_string.return_value = "test_asset"
        context.has_partition_key = False

        config = io_manager._get_write_config(context)

        assert config["partition_column_explicit"] is True
        assert config["partition_column"] == "report_date"


# ---------------------------------------------------------------------------
# TestMetadataKeyValidation
# ---------------------------------------------------------------------------


class TestMetadataKeyValidation:
    """Tests for _validate_metadata_keys method."""

    @pytest.fixture
    def io_manager(self):
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
        )

    def test_no_metadata_passes(self, io_manager):
        """No metadata should not raise."""
        context = MagicMock()
        context.metadata = None
        io_manager._validate_metadata_keys(context)

    def test_empty_metadata_passes(self, io_manager):
        """Empty metadata dict should not raise."""
        context = MagicMock()
        context.metadata = {}
        io_manager._validate_metadata_keys(context)

    def test_unrecognized_key_raises(self, io_manager):
        """Unrecognized metadata key raises ValueError."""
        context = MagicMock()
        context.metadata = {"mode": "append"}
        context.asset_key.to_user_string.return_value = "test_asset"

        with pytest.raises(ValueError, match="Unrecognized metadata key"):
            io_manager._validate_metadata_keys(context)


# ---------------------------------------------------------------------------
# TestSchemaOverride
# ---------------------------------------------------------------------------


class TestSchemaOverride:
    """Tests for schema_override and table_prefix fields."""

    def test_schema_override_changes_schema(self):
        """Test schema_override replaces default_schema in table name."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            schema_override="integration_tests",
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.has_partition_key = False

        table_name = io_manager._get_table_name(context)
        assert table_name == "integration_tests.orders"

    def test_table_prefix_prepends_to_table_name(self):
        """Test table_prefix is prepended to table name."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            table_prefix="test_run_",
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.has_partition_key = False

        table_name = io_manager._get_table_name(context)
        assert table_name == "silver.test_run_orders"

    def test_schema_override_with_prefix(self):
        """Test both schema_override and table_prefix together."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            schema_override="integration_tests",
            table_prefix="johndoe_abc123_",
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.has_partition_key = False

        table_name = io_manager._get_table_name(context)
        assert table_name == "integration_tests.johndoe_abc123_orders"

    def test_schema_override_with_suffix_strip_and_prefix(self):
        """Test schema_override + table_prefix + suffix stripping."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            schema_override="integration_tests",
            table_prefix="test_",
            table_suffix_to_strip="_silver",
        )
        context = MagicMock()
        context.asset_key.path = ["orders_silver"]
        context.has_partition_key = False

        table_name = io_manager._get_table_name(context)
        assert table_name == "integration_tests.test_orders"

    def test_no_override_uses_default_schema(self):
        """Test default behavior without overrides."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.has_partition_key = False

        table_name = io_manager._get_table_name(context)
        assert table_name == "silver.orders"

    def test_none_prefix_no_effect(self):
        """Test that table_prefix=None has no effect."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            table_prefix=None,
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.has_partition_key = False

        table_name = io_manager._get_table_name(context)
        assert table_name == "silver.orders"

    def test_empty_string_prefix_no_effect(self):
        """Test that table_prefix='' has no effect."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            table_prefix="",
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.has_partition_key = False

        table_name = io_manager._get_table_name(context)
        assert table_name == "silver.orders"

    def test_default_schema_override_is_none(self):
        """Test that schema_override defaults to None."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        assert io_manager.schema_override is None

    def test_default_table_prefix_is_none(self):
        """Test that table_prefix defaults to None."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        assert io_manager.table_prefix is None


# ---------------------------------------------------------------------------
# TestCanonicalTableResolution
# ---------------------------------------------------------------------------


class TestCanonicalTableResolution:
    """Tests for canonical_table field on ResolvedTarget.

    When table_prefix is set for integration test isolation, contract sink
    matching must use the canonical (unprefixed) table name. The bare_table
    field includes the prefix and is used for SQL; canonical_table excludes
    it and is used for contract sink lookups.
    """

    def _make_io_manager(self, **kwargs):
        defaults = {
            "postgres_resource": _make_mock_resource(),
            "default_schema": "silver",
        }
        defaults.update(kwargs)
        return PostgresIOManager(**defaults)

    def _make_context(self, asset_name: str = "orders"):
        context = MagicMock()
        context.asset_key.path = [asset_name]
        context.asset_key.to_user_string.return_value = asset_name
        context.has_partition_key = False
        context.metadata = {}
        context.log = MagicMock()
        return context

    def test_canonical_equals_bare_when_no_prefix(self):
        """Without table_prefix, canonical_table equals bare_table."""
        io_mgr = self._make_io_manager()
        context = self._make_context("orders")
        assert io_mgr._resolve_canonical_table_name(context) == "orders"
        assert io_mgr._resolve_bare_table_name(context) == "orders"

    def test_canonical_excludes_prefix(self):
        """canonical_table omits table_prefix; bare_table includes it."""
        io_mgr = self._make_io_manager(table_prefix="test_")
        context = self._make_context("orders")
        assert io_mgr._resolve_canonical_table_name(context) == "orders"
        assert io_mgr._resolve_bare_table_name(context) == "test_orders"

    def test_canonical_with_suffix_strip(self):
        """Suffix stripping applies to canonical_table."""
        io_mgr = self._make_io_manager(table_suffix_to_strip="_silver")
        context = self._make_context("orders_silver")
        assert io_mgr._resolve_canonical_table_name(context) == "orders"
        assert io_mgr._resolve_bare_table_name(context) == "orders"

    def test_canonical_with_prefix_and_suffix_strip(self):
        """Suffix is stripped, then prefix is applied only to bare_table."""
        io_mgr = self._make_io_manager(table_prefix="test_", table_suffix_to_strip="_silver")
        context = self._make_context("orders_silver")
        assert io_mgr._resolve_canonical_table_name(context) == "orders"
        assert io_mgr._resolve_bare_table_name(context) == "test_orders"

    def test_resolve_target_populates_canonical_table(self):
        """_resolve_target sets canonical_table distinct from bare_table."""
        io_mgr = self._make_io_manager(
            table_prefix="integration_",
            schema_override="integration_tests",
        )
        context = self._make_context("orders")
        target = io_mgr._resolve_target(context)
        assert target.bare_table == "integration_orders"
        assert target.canonical_table == "orders"
        assert target.table_name == "integration_tests.integration_orders"

    def test_contract_sink_matches_with_table_prefix(self):
        """Contract sink schema is used when table_prefix is set but schema_override is not."""
        from moncpipelib.contracts import Schema

        io_mgr = self._make_io_manager(
            table_prefix="test_",
        )
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="orders",
            layer="silver",
            schema=Schema(columns=[]),
            sinks=[{"type": "table", "schema": "silver_prod", "table": "orders"}],
        )
        context = self._make_context("orders")

        # Without schema_override, the contract sink schema should be used
        schema = io_mgr._resolve_schema(context, contract=contract)
        assert schema == "silver_prod"

    def test_schema_override_beats_contract_sink_schema(self):
        """schema_override takes priority over contract sink schema for test isolation."""
        from moncpipelib.contracts import Schema

        io_mgr = self._make_io_manager(
            table_prefix="test_",
            schema_override="integration_tests",
        )
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="orders",
            layer="silver",
            schema=Schema(columns=[]),
            sinks=[{"type": "table", "schema": "silver_prod", "table": "orders"}],
        )
        context = self._make_context("orders")

        # schema_override must win to ensure test isolation
        schema = io_mgr._resolve_schema(context, contract=contract)
        assert schema == "integration_tests"


# ---------------------------------------------------------------------------
# TestResolveTarget
# ---------------------------------------------------------------------------


class TestResolveTarget:
    """Tests for _resolve_target and the schema/layer resolution cascade."""

    def test_default_schema_resolves_table_name(self):
        """default_schema is used when no per-asset override exists."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context)

        assert target.table_name == "silver.orders"
        assert target.schema == "silver"
        assert target.bare_table == "orders"
        assert target.layer == "silver"

    def test_target_schema_metadata_takes_priority_over_default(self):
        """Per-asset target_schema metadata overrides default_schema."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
        )
        context = MagicMock(spec=OutputContext)
        context.asset_key.path = ["orders"]
        context.metadata = {"target_schema": "silver"}
        context.has_partition_key = False

        target = io_manager._resolve_target(context)

        assert target.table_name == "silver.orders"
        assert target.schema == "silver"
        assert target.layer == "silver"

    def test_contract_sink_schema_used_when_metadata_absent(self):
        """Contract sink schema resolves the target when metadata has no target_schema."""
        from moncpipelib.contracts import DataContract, Schema

        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
        )
        context = MagicMock(spec=OutputContext)
        context.asset_key.path = ["orders"]
        context.metadata = {}
        context.has_partition_key = False

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="gold",
            schema=Schema(columns=[]),
            sinks=[{"type": "table", "schema": "gold", "table": "orders"}],
        )

        target = io_manager._resolve_target(context, contract=contract)

        assert target.table_name == "gold.orders"
        assert target.schema == "gold"
        assert target.layer == "gold"

    def test_metadata_schema_wins_over_mismatched_sink_schema(self):
        """A sink declaring a DIFFERENT schema than metadata target_schema is not applied (#405).

        Pre-#405 the sink schema silently won, so a write whose asset metadata
        named reference_gold could be redirected into the sink's schema when the
        wrong contract resolved. On mismatch the sink is now skipped with a
        warning and the metadata schema is used.
        """
        from moncpipelib.contracts import DataContract, Schema

        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
        )
        context = MagicMock(spec=OutputContext)
        context.asset_key.path = ["orders"]
        context.metadata = {"target_schema": "silver"}
        context.has_partition_key = False

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="gold",
            schema=Schema(columns=[]),
            sinks=[{"type": "table", "schema": "gold", "table": "orders"}],
        )

        target = io_manager._resolve_target(context, contract=contract)

        assert target.table_name == "silver.orders"
        assert target.schema == "silver"
        context.log.warning.assert_called()

    def test_metadata_schema_agreeing_with_sink_schema_unchanged(self):
        """When metadata and the contract sink agree, resolution is unchanged."""
        from moncpipelib.contracts import DataContract, Schema

        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
        )
        context = MagicMock(spec=OutputContext)
        context.asset_key.path = ["orders"]
        context.metadata = {"target_schema": "gold"}
        context.has_partition_key = False

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="gold",
            schema=Schema(columns=[]),
            sinks=[{"type": "table", "schema": "gold", "table": "orders"}],
        )

        target = io_manager._resolve_target(context, contract=contract)

        assert target.table_name == "gold.orders"
        assert target.schema == "gold"

    def test_db_schema_backwards_compat(self):
        """Legacy db_schema still resolves correctly during deprecation."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            db_schema="bronze",
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context)

        assert target.table_name == "bronze.orders"
        assert target.schema == "bronze"

    def test_schema_override_takes_priority_for_testing(self):
        """schema_override (integration testing) takes priority over default_schema."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            schema_override="integration_tests",
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context)

        assert target.table_name == "integration_tests.orders"
        assert target.schema == "integration_tests"

    def test_no_schema_anywhere_raises_value_error(self):
        """Raises ValueError when no schema source is configured."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            db_schema="",
            default_schema=None,
        )
        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.asset_key.to_user_string.return_value = "orders"
        context.metadata = {}
        context.has_partition_key = False

        with pytest.raises(ValueError, match="No target schema resolved"):
            io_manager._resolve_target(context)

    def test_input_context_uses_upstream_metadata(self):
        """InputContext reads target_schema from upstream_output.metadata."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
        )
        context = MagicMock(spec=InputContext)
        context.asset_key.path = ["orders"]
        context.has_partition_key = False
        # Simulate InputContext: upstream_output has metadata
        context.upstream_output = MagicMock()
        context.upstream_output.metadata = {"target_schema": "silver"}

        target = io_manager._resolve_target(context)

        assert target.table_name == "silver.orders"
        assert target.schema == "silver"


# ---------------------------------------------------------------------------
# TestLayerDerivation
# ---------------------------------------------------------------------------


class TestLayerDerivation:
    """Tests for _resolve_layer automatic layer derivation from schema."""

    def test_schema_silver_derives_layer_silver(self):
        """Schema 'silver' in VALID_LAYERS -> layer 'silver'."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        context = MagicMock()
        context.metadata = {}
        context.has_partition_key = False

        layer = io_manager._resolve_layer(context, "silver")
        assert layer == "silver"

    def test_schema_staging_derives_layer_none(self):
        """Schema 'staging' not in VALID_LAYERS -> layer None."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="staging",
        )
        context = MagicMock()
        context.metadata = {}
        context.has_partition_key = False

        layer = io_manager._resolve_layer(context, "staging")
        assert layer is None

    def test_layer_override_metadata_overrides_derivation(self):
        """layer_override metadata takes priority over auto-derivation."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        context = MagicMock(spec=OutputContext)
        context.metadata = {"layer_override": "custom_layer"}
        context.has_partition_key = False

        layer = io_manager._resolve_layer(context, "silver")
        assert layer == "custom_layer"

    def test_deprecated_self_layer_still_works(self):
        """Legacy self.layer is used when schema is not a valid layer."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            db_schema="staging",
            layer="silver",
        )
        context = MagicMock()
        context.metadata = {}
        context.has_partition_key = False

        layer = io_manager._resolve_layer(context, "staging")
        assert layer == "silver"

    def test_layer_override_works_on_bare_magicmock_context(self):
        """Regression: ``layer_override`` must work on a bare ``MagicMock``
        context (no ``spec=OutputContext``). The integration test
        harness uses such mocks; without the duck-type fallback in
        ``_get_context_metadata``, ``isinstance(context, OutputContext)``
        is False, the auto-attribute ``upstream_output`` is truthy, and
        the metadata is never read. This regressed migration 018's
        lineage integration tests post-merge of Phase 5."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="test_write",
        )
        # Bare MagicMock -- isinstance(context, OutputContext) is False.
        context = MagicMock()
        context.metadata = {"layer_override": "bronze"}
        context.has_partition_key = False

        layer = io_manager._resolve_layer(context, "test_write")
        assert layer == "bronze"

    def test_get_context_metadata_returns_real_mapping_only(self):
        """Pin the type-strict return: a ``MagicMock`` context without
        explicit ``.metadata`` setup must produce ``None`` (not a child
        mock) so downstream callers see a deterministic "no metadata"
        signal."""
        # No explicit .metadata setup -> MagicMock auto-attribute.
        context = MagicMock()
        # The auto-attribute is a MagicMock (not a Mapping), so the
        # helper must reject it.
        result = PostgresIOManager._get_context_metadata(context)
        assert result is None or isinstance(result, dict)
        if result is not None:
            # If anything is returned, it must be a real Mapping with
            # no spurious child-mock keys.
            assert "layer_override" not in result


# ---------------------------------------------------------------------------
# TestDeprecationWarnings
# ---------------------------------------------------------------------------


class TestDeprecationWarnings:
    """Tests for deprecation warnings on legacy constructor arguments."""

    def test_db_schema_emits_deprecation_warning(self):
        """Setting db_schema triggers a DeprecationWarning."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            db_schema="bronze",
        )
        with pytest.warns(DeprecationWarning, match="db_schema.*deprecated"):
            io_manager.setup_for_execution(MagicMock())

    def test_layer_emits_deprecation_warning(self):
        """Setting layer triggers a DeprecationWarning."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            db_schema="bronze",
            layer="bronze",
        )
        with pytest.warns(DeprecationWarning, match="layer.*deprecated"):
            io_manager.setup_for_execution(MagicMock())

    def test_table_suffix_to_strip_emits_deprecation_warning(self):
        """Setting table_suffix_to_strip triggers a DeprecationWarning."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            table_suffix_to_strip="_silver",
        )
        with pytest.warns(DeprecationWarning, match="table_suffix_to_strip.*deprecated"):
            io_manager.setup_for_execution(MagicMock())

    def test_default_schema_alone_no_warnings(self):
        """Using only default_schema produces no deprecation warnings."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            io_manager.setup_for_execution(MagicMock())

    def test_db_schema_copies_to_default_schema(self):
        """When db_schema is set, it's copied to default_schema if not already set."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            db_schema="bronze",
        )
        with pytest.warns(DeprecationWarning):
            io_manager.setup_for_execution(MagicMock())

        assert io_manager.default_schema == "bronze"

    def test_no_schema_at_init_allowed_for_contract_sink_pattern(self):
        """No db_schema or default_schema is allowed; schema validated per-asset at write time."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
        )
        # setup_for_execution should NOT raise -- contracts may provide schema via sinks
        io_manager.setup_for_execution(MagicMock())


# ---------------------------------------------------------------------------
# TestContractSearchPathAutoWire
# ---------------------------------------------------------------------------


class TestContractSearchPathAutoWire:
    """Tests for automatic contract_search_paths from make_contract_checks().

    The auto-wire sets the real Pydantic field (contract_search_paths) so
    the value survives Dagster resource serialization across K8s step pods.
    """

    def test_make_contract_checks_sets_contract_search_paths(self, tmp_path):
        """make_contract_checks() auto-wires contract_search_paths for write-time discovery."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        assert io_manager.contract_search_paths is None

        # Create a minimal contract file
        (tmp_path / "test.contract.yaml").write_text(
            'version: "1.0"\n'
            'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
            "asset: test_asset\nlayer: bronze\n"
            "schema:\n  columns:\n    - name: id\n      type: integer\n      nullable: false\n"
        )

        io_manager.make_contract_checks(tmp_path)

        assert io_manager.contract_search_paths == [str(tmp_path.resolve())]

    def test_auto_wired_path_survives_model_dump(self, tmp_path):
        """contract_search_paths set by make_contract_checks() is in model_dump().

        This ensures the value survives Dagster resource serialization across
        process boundaries (e.g. k8s_job_executor step pods).
        """
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )

        (tmp_path / "test.contract.yaml").write_text(
            'version: "1.0"\n'
            'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
            "asset: test_asset\nlayer: bronze\n"
            "schema:\n  columns:\n    - name: id\n      type: integer\n      nullable: false\n"
        )

        io_manager.make_contract_checks(tmp_path)

        dumped = io_manager.model_dump()
        assert dumped["contract_search_paths"] == [str(tmp_path.resolve())]

    def test_explicit_search_paths_not_overwritten(self, tmp_path):
        """Explicit contract_search_paths is not overwritten by make_contract_checks()."""
        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            contract_search_paths=[str(explicit_dir)],
        )

        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()
        (contracts_dir / "test.contract.yaml").write_text(
            'version: "1.0"\n'
            'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
            "asset: test_asset\nlayer: bronze\n"
            "schema:\n  columns:\n    - name: id\n      type: integer\n      nullable: false\n"
        )

        io_manager.make_contract_checks(contracts_dir)

        # Explicit value preserved, not overwritten
        assert io_manager.contract_search_paths == [str(explicit_dir)]

    def test_get_contract_search_paths_returns_auto_wired(self, tmp_path):
        """_get_contract_search_paths() returns auto-wired path from make_contract_checks()."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )
        assert io_manager._get_contract_search_paths() is None

        (tmp_path / "test.contract.yaml").write_text(
            'version: "1.0"\n'
            'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
            "asset: test_asset\nlayer: bronze\n"
            "schema:\n  columns:\n    - name: id\n      type: integer\n      nullable: false\n"
        )

        io_manager.make_contract_checks(tmp_path)

        paths = io_manager._get_contract_search_paths()
        assert paths is not None
        assert len(paths) == 1
        assert paths[0] == tmp_path.resolve()


# ---------------------------------------------------------------------------
# TestSinkTableOverride
# ---------------------------------------------------------------------------


class TestSinkTableOverride:
    """Tests for contract sink table field overriding canonical table in _resolve_target."""

    @pytest.fixture
    def io_manager(self):
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
        )

    def test_sink_table_overrides_canonical(self, io_manager):
        """Sink table field becomes the physical table name for SQL."""
        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="fda_ndc_directory_silver",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "reference_silver",
                    "table": "fda_ndc_directory",
                }
            ],
        )

        context = MagicMock()
        context.asset_key.path = ["fda_ndc_directory_silver"]
        context.asset_key.to_user_string.return_value = "fda_ndc_directory_silver"
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context, contract=contract)

        assert target.bare_table == "fda_ndc_directory"
        assert target.table_name == "reference_silver.fda_ndc_directory"
        assert target.schema == "reference_silver"
        # canonical_table still reflects the asset-derived value
        assert target.canonical_table == "fda_ndc_directory_silver"

    def test_no_contract_preserves_asset_derived_name(self, io_manager):
        """Without a contract, bare_table is derived from asset key."""
        context = MagicMock()
        context.asset_key.path = ["patient_claims"]
        context.asset_key.to_user_string.return_value = "patient_claims"
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context, contract=None)

        assert target.bare_table == "patient_claims"
        assert target.table_name == "silver.patient_claims"

    def test_canonical_table_unchanged_with_sink_override(self, io_manager):
        """canonical_table reflects asset key, not the sink override."""
        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="claims_silver",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[{"type": "table", "schema": "silver", "table": "claims"}],
        )

        context = MagicMock()
        context.asset_key.path = ["claims_silver"]
        context.asset_key.to_user_string.return_value = "claims_silver"
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context, contract=contract)

        assert target.canonical_table == "claims_silver"
        assert target.bare_table == "claims"

    def test_sink_table_with_prefix(self):
        """Sink table name gets table_prefix applied."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            table_prefix="ci_test_",
        )

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders_silver",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[{"type": "table", "schema": "silver", "table": "orders"}],
        )

        context = MagicMock()
        context.asset_key.path = ["orders_silver"]
        context.asset_key.to_user_string.return_value = "orders_silver"
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context, contract=contract)

        assert target.bare_table == "ci_test_orders"
        assert target.table_name == "silver.ci_test_orders"

    def test_no_sink_table_uses_canonical(self, io_manager):
        """Sink exists but has no table field; falls back to canonical table."""
        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[{"type": "table", "schema": "silver", "mode": "upsert"}],
        )

        context = MagicMock()
        context.asset_key.path = ["orders"]
        context.asset_key.to_user_string.return_value = "orders"
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context, contract=contract)

        assert target.bare_table == "orders"
        assert target.table_name == "silver.orders"

    def test_strict_match_asset_equals_sink_table(self, io_manager):
        """When asset name matches sink table, resolution is the same as before."""
        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="patient_claims",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "patient_claims",
                    "mode": "full_refresh",
                }
            ],
        )

        context = MagicMock()
        context.asset_key.path = ["patient_claims"]
        context.asset_key.to_user_string.return_value = "patient_claims"
        context.metadata = {}
        context.has_partition_key = False

        target = io_manager._resolve_target(context, contract=contract)

        assert target.bare_table == "patient_claims"
        assert target.canonical_table == "patient_claims"
        assert target.table_name == "silver.patient_claims"


# ---------------------------------------------------------------------------
# TestCheckPiiDrift
# ---------------------------------------------------------------------------


class TestCheckPiiDrift:
    """Tests for _check_pii_drift method."""

    @pytest.fixture
    def io_manager(self):
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
            enforce_contracts=ContractEnforcementMode.WARN,
        )

    def _make_contract(self, asset, columns):
        from moncpipelib.contracts import Column, ColumnType, DataContract, Schema

        cols = [
            Column(name=c["name"], type=ColumnType.STRING, nullable=True, pii=c.get("pii", True))
            for c in columns
        ]
        return DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset=asset,
            layer="bronze",
            schema=Schema(columns=cols),
        )

    def _make_input_context(
        self, upstream_asset="upstream_asset", downstream_asset="downstream_asset"
    ):
        context = MagicMock(spec=InputContext)
        context.upstream_output.asset_key.to_user_string.return_value = upstream_asset
        context.asset_key.to_user_string.return_value = downstream_asset
        context.log = MagicMock()
        return context

    def test_warns_when_downstream_has_no_contract(self, io_manager):
        """Warns when upstream has PII but downstream has no contract."""
        upstream_contract = self._make_contract("upstream", [{"name": "patient_id", "pii": True}])
        context = self._make_input_context()

        with patch("moncpipelib.io_managers.postgres.load_contract_for_asset") as mock_load:
            mock_load.side_effect = lambda name, **_kwargs: (
                upstream_contract if name == "upstream_asset" else None
            )
            io_manager._check_pii_drift(context)

        context.log.warning.assert_called_once()
        assert "no data contract" in context.log.warning.call_args.args[0]

    def test_warns_on_pii_drift(self, io_manager):
        """Warns when upstream PII column is not marked PII in downstream."""
        upstream_contract = self._make_contract(
            "upstream",
            [
                {"name": "patient_id", "pii": True},
                {"name": "claim_id", "pii": False},
            ],
        )
        downstream_contract = self._make_contract(
            "downstream",
            [
                {"name": "patient_id", "pii": False},  # drift!
                {"name": "claim_id", "pii": False},
            ],
        )
        context = self._make_input_context()

        with patch("moncpipelib.io_managers.postgres.load_contract_for_asset") as mock_load:
            mock_load.side_effect = lambda name, **_kwargs: (
                upstream_contract if name == "upstream_asset" else downstream_contract
            )
            io_manager._check_pii_drift(context)

        context.log.warning.assert_called_once()
        assert "patient_id" in context.log.warning.call_args.args[0]
        assert "PII drift" in context.log.warning.call_args.args[0]

    def test_no_warning_when_pii_aligned(self, io_manager):
        """No warning when both contracts agree on PII annotations."""
        upstream_contract = self._make_contract(
            "upstream",
            [
                {"name": "patient_id", "pii": True},
            ],
        )
        downstream_contract = self._make_contract(
            "downstream",
            [
                {"name": "patient_id", "pii": True},
            ],
        )
        context = self._make_input_context()

        with patch("moncpipelib.io_managers.postgres.load_contract_for_asset") as mock_load:
            mock_load.side_effect = lambda name, **_kwargs: (
                upstream_contract if name == "upstream_asset" else downstream_contract
            )
            io_manager._check_pii_drift(context)

        context.log.warning.assert_not_called()

    def test_skipped_when_silent(self):
        """Drift check is skipped when enforce_contracts is SILENT."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
            enforce_contracts=ContractEnforcementMode.SILENT,
        )
        context = self._make_input_context()

        with patch("moncpipelib.io_managers.postgres.load_contract_for_asset") as mock_load:
            io_manager._check_pii_drift(context)
            mock_load.assert_not_called()

    def test_skipped_when_no_upstream_output(self, io_manager):
        """Drift check is skipped when there is no upstream output."""
        context = MagicMock(spec=InputContext)
        context.upstream_output = None

        with patch("moncpipelib.io_managers.postgres.load_contract_for_asset") as mock_load:
            io_manager._check_pii_drift(context)
            mock_load.assert_not_called()

    def test_skipped_when_upstream_has_no_pii(self, io_manager):
        """Drift check is skipped when upstream has no PII columns."""
        upstream_contract = self._make_contract(
            "upstream",
            [
                {"name": "claim_id", "pii": False},
            ],
        )
        context = self._make_input_context()

        with patch("moncpipelib.io_managers.postgres.load_contract_for_asset") as mock_load:
            mock_load.side_effect = lambda name, **_kwargs: (
                upstream_contract if name == "upstream_asset" else None
            )
            io_manager._check_pii_drift(context)

        context.log.warning.assert_not_called()

    def test_passes_contract_search_paths(self):
        """Drift check forwards contract_search_paths to loader."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
            enforce_contracts=ContractEnforcementMode.WARN,
            contract_search_paths=["/some/path"],
        )
        upstream_contract = self._make_contract("upstream", [{"name": "patient_id", "pii": True}])
        context = self._make_input_context()

        with patch("moncpipelib.io_managers.postgres.load_contract_for_asset") as mock_load:
            mock_load.side_effect = lambda name, **_kwargs: (
                upstream_contract if name == "upstream_asset" else None
            )
            io_manager._check_pii_drift(context)

        # Verify search_paths was passed to both calls
        for call in mock_load.call_args_list:
            assert call.kwargs.get("search_paths") is not None
            assert len(call.kwargs["search_paths"]) == 1


# ---------------------------------------------------------------------------
# TestContractSinkModeEnforcement
# ---------------------------------------------------------------------------


class TestContractSinkModeEnforcement:
    """Tests for reconciliation of contract sink 'mode' vs IO Manager write mode.

    The contract sink 'mode' field is the authoritative spec for how data should
    be written. These tests verify that the IO Manager passes write config and
    contract data to the resource, which handles the actual reconciliation.
    """

    @pytest.fixture
    def io_manager(self):
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
            layer="bronze",
            enforce_contracts=ContractEnforcementMode.ERROR,
        )

    @pytest.fixture
    def sample_df(self):
        return pl.DataFrame({"id": [1], "name": ["a"]})

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.run_id = "test-run-123"
        context.asset_key.path = ["orders"]
        context.asset_key.to_user_string.return_value = "orders"
        context.metadata = {}
        context.log = MagicMock()
        context.add_output_metadata = MagicMock()
        context.has_partition_key = False
        return context

    # --- _get_write_config: write_mode_explicit flag ---

    def test_write_mode_explicit_false_for_io_manager_default(self, io_manager, mock_context):
        """IO manager class-level default does not set write_mode_explicit."""
        config = io_manager._get_write_config(mock_context)
        assert config["write_mode_explicit"] is False

    def test_write_mode_explicit_true_when_set_in_asset_metadata(self, io_manager, mock_context):
        """Asset metadata write_mode sets write_mode_explicit=True."""
        mock_context.metadata = {"write_mode": "append"}
        config = io_manager._get_write_config(mock_context)
        assert config["write_mode_explicit"] is True
        assert config["write_mode"] == WriteMode.APPEND

    def test_write_mode_explicit_true_for_enum_value_in_metadata(self, io_manager, mock_context):
        """WriteMode enum value in asset metadata also sets write_mode_explicit=True."""
        mock_context.metadata = {"write_mode": WriteMode.UPSERT}
        config = io_manager._get_write_config(mock_context)
        assert config["write_mode_explicit"] is True


# ---------------------------------------------------------------------------
# TestContractPrimaryKeyEnforcement
# ---------------------------------------------------------------------------


class TestContractPrimaryKeyEnforcement:
    """Tests for primary_key_explicit tracking in _get_write_config."""

    @pytest.fixture
    def io_manager(self):
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
            layer="bronze",
            enforce_contracts=ContractEnforcementMode.ERROR,
        )

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.run_id = "test-run-123"
        context.asset_key.path = ["orders"]
        context.asset_key.to_user_string.return_value = "orders"
        context.metadata = {}
        context.log = MagicMock()
        context.add_output_metadata = MagicMock()
        context.has_partition_key = False
        return context

    # --- _get_write_config: primary_key_explicit flag ---

    def test_pk_explicit_false_for_io_manager_default(self, io_manager, mock_context):
        """IO manager class-level default does not set primary_key_explicit."""
        config = io_manager._get_write_config(mock_context)
        assert config["primary_key_explicit"] is False

    def test_pk_explicit_true_when_set_in_asset_metadata(self, io_manager, mock_context):
        """Asset metadata primary_key sets primary_key_explicit=True."""
        mock_context.metadata = {"primary_key": ["id"]}
        config = io_manager._get_write_config(mock_context)
        assert config["primary_key_explicit"] is True
        assert config["primary_key"] == ["id"]

    def test_pk_explicit_true_for_string_value(self, io_manager, mock_context):
        """String primary_key in metadata is normalised to list and flagged explicit."""
        mock_context.metadata = {"primary_key": "id"}
        config = io_manager._get_write_config(mock_context)
        assert config["primary_key_explicit"] is True
        assert config["primary_key"] == ["id"]


# ---------------------------------------------------------------------------
# TestContractSCD2Enforcement
# ---------------------------------------------------------------------------


class TestContractSCD2Enforcement:
    """Tests for SCD2 explicit flag tracking in _get_write_config.

    Covers business_key, tracked_columns, and detect_deletes.
    """

    @pytest.fixture
    def io_manager(self):
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="test_scd2",
            layer="silver",
            enforce_contracts=ContractEnforcementMode.ERROR,
        )

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.run_id = "test-run-123"
        context.asset_key.path = ["dim_product"]
        context.asset_key.to_user_string.return_value = "dim_product"
        context.metadata = {}
        context.log = MagicMock()
        context.add_output_metadata = MagicMock()
        context.has_partition_key = False
        return context

    def _make_write_config(self, io_manager: PostgresIOManager, context: MagicMock) -> dict:
        return io_manager._get_write_config(context)

    # -----------------------------------------------------------------------
    # _get_write_config: explicit flag defaults and metadata-set behaviour
    # -----------------------------------------------------------------------

    def test_business_key_explicit_false_by_default(
        self, io_manager: PostgresIOManager, mock_context: MagicMock
    ) -> None:
        """business_key_explicit defaults to False when not in metadata."""
        cfg = self._make_write_config(io_manager, mock_context)
        assert cfg["business_key"] is None
        assert cfg["business_key_explicit"] is False

    def test_business_key_explicit_true_when_set_in_metadata(
        self, io_manager: PostgresIOManager, mock_context: MagicMock
    ) -> None:
        """business_key_explicit=True when key is present in asset metadata."""
        mock_context.metadata = {"business_key": ["product_id"]}
        cfg = self._make_write_config(io_manager, mock_context)
        assert cfg["business_key"] == ["product_id"]
        assert cfg["business_key_explicit"] is True

    def test_business_key_string_normalised_to_list(
        self, io_manager: PostgresIOManager, mock_context: MagicMock
    ) -> None:
        """A bare string in metadata is normalised to a single-element list."""
        mock_context.metadata = {"business_key": "product_id"}
        cfg = self._make_write_config(io_manager, mock_context)
        assert cfg["business_key"] == ["product_id"]
        assert cfg["business_key_explicit"] is True

    def test_tracked_columns_explicit_false_by_default(
        self, io_manager: PostgresIOManager, mock_context: MagicMock
    ) -> None:
        """tracked_columns_explicit defaults to False when not in metadata."""
        cfg = self._make_write_config(io_manager, mock_context)
        assert cfg["tracked_columns"] is None
        assert cfg["tracked_columns_explicit"] is False

    def test_tracked_columns_explicit_true_when_set_in_metadata(
        self, io_manager: PostgresIOManager, mock_context: MagicMock
    ) -> None:
        """tracked_columns_explicit=True when key is present in asset metadata."""
        mock_context.metadata = {"tracked_columns": ["name", "price"]}
        cfg = self._make_write_config(io_manager, mock_context)
        assert cfg["tracked_columns"] == ["name", "price"]
        assert cfg["tracked_columns_explicit"] is True

    def test_detect_deletes_explicit_false_by_default(
        self, io_manager: PostgresIOManager, mock_context: MagicMock
    ) -> None:
        """detect_deletes_explicit defaults to False; detect_deletes defaults to False."""
        cfg = self._make_write_config(io_manager, mock_context)
        assert cfg["detect_deletes"] is False
        assert cfg["detect_deletes_explicit"] is False

    def test_detect_deletes_explicit_true_when_set_in_metadata(
        self, io_manager: PostgresIOManager, mock_context: MagicMock
    ) -> None:
        """detect_deletes_explicit=True when key is present in asset metadata."""
        mock_context.metadata = {"detect_deletes": True}
        cfg = self._make_write_config(io_manager, mock_context)
        assert cfg["detect_deletes"] is True
        assert cfg["detect_deletes_explicit"] is True


# ---------------------------------------------------------------------------
# TestInputValidation (keep only dict type error test)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input type validation in handle_output."""

    @pytest.fixture
    def io_manager(self):
        """Create IO manager for testing."""
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="bronze",
            layer="bronze",
        )

    @pytest.fixture
    def mock_context(self):
        """Create a mock OutputContext."""
        context = MagicMock()
        context.asset_key.to_user_string.return_value = "test_asset"
        context.run_id = "test-run-123"
        context.metadata = {}
        context.log = MagicMock()
        context.has_partition_key = False
        return context

    def test_handle_output_with_dict_raises_type_error(self, io_manager, mock_context):
        """Ensure clear error when asset returns dict instead of DataFrame."""
        with pytest.raises(TypeError) as exc_info:
            io_manager.handle_output(mock_context, {"col1": [1, 2, 3]})

        assert "expected a Polars DataFrame" in str(exc_info.value)
        assert "received dict" in str(exc_info.value)
        assert "test_asset" in str(exc_info.value)  # Asset name should be included


# ---------------------------------------------------------------------------
# TestExtractPartitionValues
# ---------------------------------------------------------------------------


class TestExtractPartitionValues:
    """Tests for _extract_partition_values()."""

    def test_returns_none_for_non_partitioned_context(self):
        """Non-partitioned context returns None."""
        ctx = MagicMock(spec=OutputContext)
        ctx.has_partition_key = False
        assert PostgresIOManager._extract_partition_values(ctx) is None

    def test_returns_single_partition_key(self):
        """Single partition key returns list with one element."""
        ctx = MagicMock(spec=OutputContext)
        ctx.has_partition_key = True
        ctx.asset_partition_keys = ["2024-01-15"]
        result = PostgresIOManager._extract_partition_values(ctx)
        assert result == ["2024-01-15"]

    def test_returns_multiple_partition_keys(self):
        """Multiple partition keys (backfill) returns full list."""
        ctx = MagicMock(spec=OutputContext)
        ctx.has_partition_key = True
        ctx.asset_partition_keys = ["2024-01-15", "2024-01-16", "2024-01-17"]
        result = PostgresIOManager._extract_partition_values(ctx)
        assert result == ["2024-01-15", "2024-01-16", "2024-01-17"]


# ---------------------------------------------------------------------------
# TestReconcilePartitionColumn
# ---------------------------------------------------------------------------


class TestReconcilePartitionColumn:
    """Tests for partition_column reconciliation (four-way pattern).

    These test the ContractReconciler directly since reconciliation
    was moved from the IO Manager to shared utilities.
    """

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.log = MagicMock()
        context.has_partition_key = False
        return context

    def test_contract_only_overrides_default(self, mock_context):
        """Contract partition_column silently overrides IO manager default (None)."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "partition_column": "report_date",
                }
            ],
        )

        write_config = {
            "partition_column": None,
            "partition_column_explicit": False,
        }

        result = ContractReconciler.reconcile_partition_column(
            contract, "orders", write_config, mock_context
        )

        assert result == "report_date"

    def test_metadata_only_unchanged(self, mock_context):
        """Metadata partition_column used when contract has no sink match."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        result = ContractReconciler.reconcile_partition_column(
            None,
            "orders",
            {"partition_column": "report_date", "partition_column_explicit": True},
            mock_context,
        )

        assert result == "report_date"

    def test_both_same_value_logs_warning(self, mock_context):
        """Both set to same value -> warning logged, proceeds."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "partition_column": "report_date",
                }
            ],
        )

        write_config = {
            "partition_column": "report_date",
            "partition_column_explicit": True,
        }

        result = ContractReconciler.reconcile_partition_column(
            contract, "orders", write_config, mock_context
        )

        assert result == "report_date"
        mock_context.log.warning.assert_called_once()
        assert "declared in both" in mock_context.log.warning.call_args[0][0]

    def test_different_values_raises_error(self, mock_context):
        """Different values -> ContractViolationError."""
        from moncpipelib.contracts.exceptions import ContractViolationError
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "partition_column": "report_date",
                }
            ],
        )

        write_config = {
            "partition_column": "created_at",
            "partition_column_explicit": True,
        }

        with pytest.raises(ContractViolationError, match="partition_column conflict"):
            ContractReconciler.reconcile_partition_column(
                contract, "orders", write_config, mock_context
            )


# ---------------------------------------------------------------------------
# TestReconcileSequenceColumn
# ---------------------------------------------------------------------------


class TestReconcileSequenceColumn:
    """Tests for sequence_column reconciliation (four-way pattern)."""

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.log = MagicMock()
        return context

    def test_contract_only_overrides_default(self, mock_context):
        """Contract sequence_column silently overrides the SCD2_DEFAULTS value."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "sequence_column": "version_num",
                }
            ],
        )

        write_config = {
            "sequence_col": "seq_id",
            "sequence_col_explicit": False,
        }

        result = ContractReconciler.reconcile_sequence_column(
            contract, "orders", write_config, mock_context
        )

        assert result == "version_num"

    def test_contract_null_opts_out(self, mock_context):
        """Contract setting sequence_column: null explicitly opts out."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "sequence_column": None,
                }
            ],
        )

        write_config = {
            "sequence_col": "seq_id",
            "sequence_col_explicit": False,
        }

        result = ContractReconciler.reconcile_sequence_column(
            contract, "orders", write_config, mock_context
        )

        assert result is None

    def test_metadata_only_unchanged(self, mock_context):
        """Metadata sequence_col used when contract has no sink match."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        result = ContractReconciler.reconcile_sequence_column(
            None,
            "orders",
            {"sequence_col": "my_seq", "sequence_col_explicit": True},
            mock_context,
        )

        assert result == "my_seq"

    def test_both_same_value_logs_warning(self, mock_context):
        """Both set to same value -> warning logged, proceeds."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "sequence_column": "seq_id",
                }
            ],
        )

        write_config = {
            "sequence_col": "seq_id",
            "sequence_col_explicit": True,
        }

        result = ContractReconciler.reconcile_sequence_column(
            contract, "orders", write_config, mock_context
        )

        assert result == "seq_id"
        mock_context.log.warning.assert_called_once()
        assert "declared in both" in mock_context.log.warning.call_args[0][0]

    def test_different_values_raises_error(self, mock_context):
        """Different values -> ContractViolationError."""
        from moncpipelib.contracts.exceptions import ContractViolationError
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "sequence_column": "version_num",
                }
            ],
        )

        write_config = {
            "sequence_col": "seq_id",
            "sequence_col_explicit": True,
        }

        with pytest.raises(ContractViolationError, match="sequence_column conflict"):
            ContractReconciler.reconcile_sequence_column(
                contract, "orders", write_config, mock_context
            )

    def test_contract_no_sequence_column_key_preserves_default(self, mock_context):
        """Contract sink without sequence_column key preserves write_config default."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "mode": "scd2",
                }
            ],
        )

        write_config = {
            "sequence_col": "seq_id",
            "sequence_col_explicit": False,
        }

        result = ContractReconciler.reconcile_sequence_column(
            contract, "orders", write_config, mock_context
        )

        assert result == "seq_id"


# ---------------------------------------------------------------------------
# TestReconcileSkipUnchanged
# ---------------------------------------------------------------------------


class TestReconcileSkipUnchanged:
    """Tests for skip_unchanged reconciliation (four-way pattern)."""

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.log = MagicMock()
        return context

    @staticmethod
    def _contract(skip_unchanged: bool) -> DataContract:
        return DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "orders",
                    "mode": "upsert",
                    "primary_key": ["id"],
                    "skip_unchanged": skip_unchanged,
                }
            ],
        )

    def test_contract_only_overrides_default(self, mock_context):
        """Contract skip_unchanged silently overrides the default (False)."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        write_config = {
            "skip_unchanged": False,
            "skip_unchanged_explicit": False,
        }

        result = ContractReconciler.reconcile_skip_unchanged(
            self._contract(True), "orders", write_config, mock_context
        )

        assert result is True

    def test_metadata_only_unchanged(self, mock_context):
        """Caller-supplied skip_unchanged used when no contract sink matches."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        result = ContractReconciler.reconcile_skip_unchanged(
            None,
            "orders",
            {"skip_unchanged": True, "skip_unchanged_explicit": True},
            mock_context,
        )

        assert result is True

    def test_sink_without_field_leaves_caller_value(self, mock_context):
        """A matching sink that omits skip_unchanged does not touch the value."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(True)
        del contract.sinks[0]["skip_unchanged"]

        result = ContractReconciler.reconcile_skip_unchanged(
            contract,
            "orders",
            {"skip_unchanged": True, "skip_unchanged_explicit": True},
            mock_context,
        )

        assert result is True
        mock_context.log.warning.assert_not_called()

    def test_both_same_value_logs_warning(self, mock_context):
        """Both set to the same value -> warning logged, proceeds."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        result = ContractReconciler.reconcile_skip_unchanged(
            self._contract(True),
            "orders",
            {"skip_unchanged": True, "skip_unchanged_explicit": True},
            mock_context,
        )

        assert result is True
        mock_context.log.warning.assert_called_once()
        assert "declared in both" in mock_context.log.warning.call_args[0][0]

    def test_conflict_raises(self, mock_context):
        """Different values -> ContractViolationError."""
        from moncpipelib.contracts.exceptions import ContractViolationError
        from moncpipelib.contracts.reconciliation import ContractReconciler

        with pytest.raises(ContractViolationError, match="skip_unchanged conflict"):
            ContractReconciler.reconcile_skip_unchanged(
                self._contract(False),
                "orders",
                {"skip_unchanged": True, "skip_unchanged_explicit": True},
                mock_context,
            )

    def test_config_without_key_defaults_to_false(self, mock_context):
        """Pre-existing config dicts without the key reconcile as default-off."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        result = ContractReconciler.reconcile_skip_unchanged(None, "orders", {}, mock_context)

        assert result is False


# ---------------------------------------------------------------------------
# TestFindMatchingSinkBareTable
# ---------------------------------------------------------------------------


class TestFindMatchingSinkBareTable:
    """Tests for find_matching_sink matching by bare table name."""

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.log = MagicMock()
        context.has_partition_key = False
        return context

    def test_matches_by_bare_table_only(self, mock_context):
        """Sink matching uses bare table name, not schema-qualified."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[{"type": "table", "schema": "silver", "table": "orders", "mode": "upsert"}],
        )

        result = ContractReconciler.find_matching_sink(contract, "orders", mock_context)

        assert result is not None
        assert result["table"] == "orders"
        assert result["mode"] == "upsert"

    def test_single_sink_fallback_when_name_mismatch(self, mock_context):
        """Single table sink matches via lenient fallback when name differs."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="fda_ndc_directory_silver",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[{"type": "table", "schema": "reference_silver", "table": "fda_ndc_directory"}],
        )

        # bare_table doesn't match sink table, but single-sink fallback kicks in
        result = ContractReconciler.find_matching_sink(
            contract, "fda_ndc_directory_silver", mock_context
        )

        assert result is not None
        assert result["table"] == "fda_ndc_directory"
        assert result["schema"] == "reference_silver"

    def test_multi_sink_no_strict_match_returns_none(self, mock_context):
        """Multiple table sinks with no strict match returns None (no fallback)."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {"type": "table", "schema": "silver", "table": "products"},
                {"type": "table", "schema": "gold", "table": "items"},
            ],
        )

        result = ContractReconciler.find_matching_sink(contract, "orders", mock_context)

        assert result is None

    def test_single_sink_strict_match_preferred(self, mock_context):
        """When strict match works, it's used over the fallback path."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[{"type": "table", "schema": "silver", "table": "orders", "mode": "upsert"}],
        )

        result = ContractReconciler.find_matching_sink(contract, "orders", mock_context)

        assert result is not None
        assert result["table"] == "orders"
        assert result["mode"] == "upsert"

    def test_non_table_sinks_ignored_in_fallback(self, mock_context):
        """Non-table sinks are not considered for the single-sink fallback."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {"type": "s3", "bucket": "my-bucket"},
                {"type": "table", "schema": "silver", "table": "orders_raw"},
            ],
        )

        # Only one table-type sink -> fallback matches it
        result = ContractReconciler.find_matching_sink(contract, "orders", mock_context)

        assert result is not None
        assert result["table"] == "orders_raw"

    def test_multiple_sinks_matching_same_table_raises(self, mock_context):
        """Multiple sinks matching same bare table -> ContractViolationError."""
        from moncpipelib.contracts.exceptions import ContractViolationError
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=MagicMock(columns=[]),
            sinks=[
                {"type": "table", "schema": "silver", "table": "orders"},
                {"type": "table", "schema": "gold", "table": "orders"},
            ],
        )

        with pytest.raises(ContractViolationError, match="Multiple contract sinks"):
            ContractReconciler.find_matching_sink(contract, "orders", mock_context)

    def test_no_contract_returns_none(self, mock_context):
        """None contract returns None."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        result = ContractReconciler.find_matching_sink(None, "orders", mock_context)

        assert result is None


# ---------------------------------------------------------------------------
# TestFindMatchingSinkSchemaComparison (#405)
# ---------------------------------------------------------------------------


class TestFindMatchingSinkSchemaComparison:
    """Tests for schema-aware sink matching (#405).

    When the write target carries a schema, a sink declaring a different
    schema must never match -- applying synthetic_gold's sink configuration
    to a reference_gold write is the #405 failure mode.
    """

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.log = MagicMock()
        context.has_partition_key = False
        return context

    def _contract(self, sinks):
        return DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="dim_provider",
            layer="gold",
            schema=MagicMock(columns=[]),
            sinks=sinks,
        )

    def test_matching_schema_returns_sink(self, mock_context):
        """Sink schema equal to target_schema matches normally."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(
            [{"type": "table", "schema": "reference_gold", "table": "dim_provider"}]
        )

        result = ContractReconciler.find_matching_sink(
            contract, "dim_provider", mock_context, target_schema="reference_gold"
        )

        assert result is not None
        assert result["schema"] == "reference_gold"

    def test_mismatched_schema_rejected_with_warning(self, mock_context):
        """A name match in a DIFFERENT schema is rejected, never applied."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(
            [{"type": "table", "schema": "synthetic_gold", "table": "dim_provider"}]
        )

        result = ContractReconciler.find_matching_sink(
            contract, "dim_provider", mock_context, target_schema="reference_gold"
        )

        assert result is None
        mock_context.log.warning.assert_called_once()
        assert "NOT applied" in mock_context.log.warning.call_args[0][0]

    def test_sink_without_schema_never_excluded(self, mock_context):
        """Sinks that do not declare a schema are not filtered out."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract([{"type": "table", "table": "dim_provider"}])

        result = ContractReconciler.find_matching_sink(
            contract, "dim_provider", mock_context, target_schema="reference_gold"
        )

        assert result is not None
        assert result["table"] == "dim_provider"

    def test_no_target_schema_preserves_existing_behavior(self, mock_context):
        """target_schema=None disables the filter entirely."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(
            [{"type": "table", "schema": "synthetic_gold", "table": "dim_provider"}]
        )

        result = ContractReconciler.find_matching_sink(contract, "dim_provider", mock_context)

        assert result is not None

    def test_same_table_multi_schema_sinks_disambiguated(self, mock_context):
        """One contract with same-named sinks in two schemas resolves by schema."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(
            [
                {"type": "table", "schema": "synthetic_gold", "table": "dim_provider"},
                {"type": "table", "schema": "reference_gold", "table": "dim_provider"},
            ]
        )

        result = ContractReconciler.find_matching_sink(
            contract, "dim_provider", mock_context, target_schema="reference_gold"
        )

        assert result is not None
        assert result["schema"] == "reference_gold"

    def test_single_sink_fallback_requires_schema_compatibility(self, mock_context):
        """The lenient single-sink fallback never returns a schema-mismatched sink."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(
            [{"type": "table", "schema": "synthetic_gold", "table": "dim_provider"}]
        )

        # Name mismatch (e.g. table_prefix) AND schema mismatch -> no fallback
        result = ContractReconciler.find_matching_sink(
            contract, "itest_dim_provider", mock_context, target_schema="reference_gold"
        )

        assert result is None

    def test_prefixed_table_with_schema_filter_falls_back(self, mock_context):
        """Schema filter narrows same-named multi-schema sinks for prefixed tables."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(
            [
                {"type": "table", "schema": "synthetic_gold", "table": "dim_provider"},
                {"type": "table", "schema": "reference_gold", "table": "dim_provider"},
            ]
        )

        result = ContractReconciler.find_matching_sink(
            contract, "itest_dim_provider", mock_context, target_schema="reference_gold"
        )

        assert result is not None
        assert result["schema"] == "reference_gold"


# ---------------------------------------------------------------------------
# TestForTesting
# ---------------------------------------------------------------------------


class TestForTesting:
    """Tests for the for_testing() factory method."""

    @pytest.fixture
    def production_io_manager(self):
        """Create a production-like IO manager with all fields populated."""
        return PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            default_schema="silver",
            write_mode=WriteMode.UPSERT,
            primary_key=["claim_id"],
            update_columns=["status", "updated_at"],
            partition_column="process_date",
            enforce_contracts=ContractEnforcementMode.ERROR,
            contract_search_paths=["/app/assets/silver"],
        )

    def test_test_schema_sets_schema_override(self, production_io_manager):
        """test_schema parameter maps to schema_override."""
        test_io = production_io_manager.for_testing(test_schema="integration_tests")
        assert test_io.schema_override == "integration_tests"

    def test_preserves_postgres_resource(self, production_io_manager):
        """postgres_resource is preserved from original."""
        test_io = production_io_manager.for_testing(test_schema="integration_tests")
        assert test_io.postgres_resource is production_io_manager.postgres_resource

    def test_preserves_write_behavior(self, production_io_manager):
        """Write mode and related fields are preserved."""
        test_io = production_io_manager.for_testing(test_schema="integration_tests")
        assert test_io.write_mode == WriteMode.UPSERT
        assert test_io.primary_key == ["claim_id"]
        assert test_io.update_columns == ["status", "updated_at"]
        assert test_io.partition_column == "process_date"

    def test_preserves_enforce_contracts(self, production_io_manager):
        """enforce_contracts is NOT overridden -- inherited from original."""
        test_io = production_io_manager.for_testing(test_schema="integration_tests")
        assert test_io.enforce_contracts == ContractEnforcementMode.ERROR

    def test_table_prefix(self, production_io_manager):
        """table_prefix is applied to the test clone."""
        test_io = production_io_manager.for_testing(
            test_schema="integration_tests",
            table_prefix="ci_abc123_",
        )
        assert test_io.table_prefix == "ci_abc123_"

    def test_contract_search_paths_override(self, production_io_manager):
        """contract_search_paths can be overridden for test contracts."""
        test_io = production_io_manager.for_testing(
            test_schema="integration_tests",
            contract_search_paths=["/tests/contracts"],
        )
        assert test_io.contract_search_paths == ["/tests/contracts"]

    def test_contract_search_paths_preserved_when_not_overridden(self, production_io_manager):
        """contract_search_paths is preserved from original when not explicitly overridden."""
        test_io = production_io_manager.for_testing(test_schema="integration_tests")
        assert test_io.contract_search_paths == ["/app/assets/silver"]

    def test_kwargs_override(self, production_io_manager):
        """Additional overrides can be passed via **overrides."""
        test_io = production_io_manager.for_testing(
            test_schema="integration_tests",
            write_mode=WriteMode.FULL_REFRESH,
        )
        assert test_io.write_mode == WriteMode.FULL_REFRESH

    def test_original_not_mutated(self, production_io_manager):
        """for_testing() returns a new instance; original is unchanged."""
        test_io = production_io_manager.for_testing(
            test_schema="integration_tests",
            table_prefix="test_",
        )
        assert production_io_manager.schema_override is None
        assert production_io_manager.table_prefix is None
        assert test_io is not production_io_manager

    def test_preserves_deprecated_fields(self):
        """Deprecated fields (db_schema, layer, table_suffix_to_strip) are preserved."""
        io_manager = PostgresIOManager(
            postgres_resource=_make_mock_resource(),
            db_schema="bronze",
            layer="bronze",
            table_suffix_to_strip="_bronze",
        )
        test_io = io_manager.for_testing(test_schema="integration_tests")
        assert test_io.db_schema == "bronze"
        assert test_io.layer == "bronze"
        assert test_io.table_suffix_to_strip == "_bronze"


# ===========================================================================
# Resource delegation tests
# ===========================================================================


class TestResourceDelegation:
    """Tests for PostgresIOManager delegating writes to a PostgresResource."""

    @pytest.fixture
    def resource(self):
        """Create a real PostgresResource for delegation."""
        return _make_mock_resource(user="testuser", password="testpass")

    @pytest.fixture
    def io_manager_with_resource(self, resource):
        """Create a PostgresIOManager configured for delegation."""
        return PostgresIOManager(
            postgres_resource=resource,
            default_schema="silver",
            enforce_contracts=ContractEnforcementMode.SILENT,
        )

    @pytest.fixture
    def mock_context(self):
        """Create a mock OutputContext."""
        context = MagicMock(spec=OutputContext)
        context.asset_key.path = ["test_asset"]
        context.asset_key.to_user_string.return_value = "test_asset"
        context.run_id = "test-run-123"
        context.has_partition_key = False
        context.metadata = {}
        context.log = MagicMock()
        return context

    def test_handle_output_routes_to_resource_write_single(
        self, io_manager_with_resource, mock_context
    ):
        """When postgres_resource is set, handle_output delegates through _write_single."""
        from moncpipelib.resources.types import WriteResult

        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mock_result = WriteResult(
            table_name="silver.test_asset",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.FULL_REFRESH,
            stats={"rows_deleted": 0, "rows_inserted": 3},
            row_count=3,
        )

        with patch.object(
            PostgresResource, "_write_single", return_value=mock_result
        ) as mock_write:
            io_manager_with_resource.handle_output(mock_context, df)
            mock_write.assert_called_once()

        mock_context.add_output_metadata.assert_called_once()

    def test_handle_output_none_skips_without_delegation(self, io_manager_with_resource):
        """handle_output(None) should skip without delegating."""
        context = MagicMock(spec=OutputContext)
        context.log = MagicMock()

        io_manager_with_resource.handle_output(context, None)

        context.log.warning.assert_called_once_with("Received None, skipping write")

    def test_delegated_write_single_receives_correct_args(
        self, io_manager_with_resource, mock_context
    ):
        """Delegation should call resource._write_single with correct target and config."""
        from moncpipelib.resources.types import WriteResult

        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mock_result = WriteResult(
            table_name="silver.test_asset",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.FULL_REFRESH,
            stats={"rows_deleted": 0, "rows_inserted": 3},
            row_count=3,
            columns=["id", "name"],
        )

        with patch.object(
            PostgresResource, "_write_single", return_value=mock_result
        ) as mock_write:
            io_manager_with_resource.handle_output(mock_context, df)

            mock_write.assert_called_once()
            call_kwargs = mock_write.call_args.kwargs
            assert call_kwargs["table_name"] == "silver.test_asset"
            assert call_kwargs["schema"] == "silver"
            assert call_kwargs["layer"] == "silver"
            assert call_kwargs["wctx"].asset_name == "test_asset"
            assert call_kwargs["wctx"].run_id == "test-run-123"
            assert call_kwargs["write_config"]["write_mode"] == WriteMode.FULL_REFRESH

        # Verify Dagster metadata was set
        mock_context.add_output_metadata.assert_called_once()
        metadata = mock_context.add_output_metadata.call_args[0][0]
        assert "write_mode" in metadata
        assert "target_table" in metadata

    def test_delegated_uses_io_manager_target_resolution(self, resource, mock_context):
        """Delegation should use IO manager's schema cascade, not the resource's."""
        from moncpipelib.resources.types import WriteResult

        io_manager = PostgresIOManager(
            postgres_resource=resource,
            default_schema="gold",
            table_prefix="test_",
            enforce_contracts=ContractEnforcementMode.SILENT,
        )

        df = pl.DataFrame({"id": [1]})
        mock_result = WriteResult(
            table_name="gold.test_test_asset",
            schema="gold",
            layer="gold",
            write_mode=WriteMode.FULL_REFRESH,
            stats={"rows_deleted": 0, "rows_inserted": 1},
            row_count=1,
        )

        with patch.object(
            PostgresResource, "_write_single", return_value=mock_result
        ) as mock_write:
            io_manager.handle_output(mock_context, df)

            call_kwargs = mock_write.call_args.kwargs
            # Should use IO manager's schema cascade (gold) and table_prefix (test_)
            assert call_kwargs["table_name"] == "gold.test_test_asset"
            assert call_kwargs["schema"] == "gold"
            assert call_kwargs["layer"] == "gold"
            assert call_kwargs["bare_table"] == "test_test_asset"

    def test_delegated_passes_metadata_write_config(self, io_manager_with_resource, mock_context):
        """Delegation should pass write config built from asset metadata."""
        from moncpipelib.resources.types import WriteResult

        mock_context.metadata = {
            "write_mode": "upsert",
            "primary_key": ["id"],
            "source_file": "data.csv",
        }

        df = pl.DataFrame({"id": [1], "value": ["x"]})
        mock_result = WriteResult(
            table_name="silver.test_asset",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.UPSERT,
            stats={"rows_upserted": 1},
            row_count=1,
        )

        with patch.object(
            PostgresResource, "_write_single", return_value=mock_result
        ) as mock_write:
            io_manager_with_resource.handle_output(mock_context, df)

            call_kwargs = mock_write.call_args.kwargs
            assert call_kwargs["write_config"]["write_mode"] == WriteMode.UPSERT
            assert call_kwargs["write_config"]["primary_key"] == ["id"]
            assert call_kwargs["write_config"]["write_mode_explicit"] is True
            assert call_kwargs["source_file"] == "data.csv"

    def test_batched_delegation_routes_to_resource(self, io_manager_with_resource, mock_context):
        """BatchedDataFrame should route to resource._write_batched when resource is set."""
        from moncpipelib.resources.types import WriteResult
        from moncpipelib.streaming import BatchedDataFrame

        batched = BatchedDataFrame(batches=iter([]), total_rows_hint=0)
        mock_result = WriteResult(
            table_name="silver.test_asset",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.FULL_REFRESH,
            stats={},
            row_count=0,
            batch_count=0,
        )

        with patch.object(
            PostgresResource, "_write_batched", return_value=mock_result
        ) as mock_write:
            io_manager_with_resource.handle_output(mock_context, batched)
            mock_write.assert_called_once()

    def test_for_testing_preserves_postgres_resource(self, io_manager_with_resource, resource):
        """for_testing() should preserve the postgres_resource field."""
        test_io = io_manager_with_resource.for_testing(test_schema="integration_tests")
        assert test_io.postgres_resource is resource


# ---------------------------------------------------------------------------
# TestReconcilePrimaryKeySinkLevel (#401)
# ---------------------------------------------------------------------------


class TestReconcilePrimaryKeySinkLevel:
    """Sink-level primary_key participates in reconciliation (#401).

    The spec has documented sink `primary_key` as an alternative to
    schema-level `primary_key: true` columns since the sinks section landed,
    but reconcile_primary_key only ever read the schema flags -- a sink-level
    declaration was silently ignored.
    """

    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.log = MagicMock()
        return context

    def _contract(self, sink_pk=None, schema_pk_col=None) -> DataContract:
        from moncpipelib.contracts.models import Column, ColumnType, Schema

        sink: dict = {"type": "table", "schema": "silver", "table": "orders"}
        if sink_pk is not None:
            sink["primary_key"] = sink_pk

        columns = []
        if schema_pk_col is not None:
            columns.append(
                Column(
                    name=schema_pk_col,
                    type=ColumnType.STRING,
                    nullable=False,
                    primary_key=True,
                    pii=False,
                )
            )

        return DataContract(
            version="1.0",
            pipeline_id="test-pipeline",
            asset="orders",
            layer="silver",
            schema=Schema(columns=columns),
            sinks=[sink],
        )

    def test_sink_pk_used_when_metadata_unset(self, mock_context):
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(sink_pk=["order_id", "load_period"])
        result = ContractReconciler.reconcile_primary_key(
            contract,
            {"primary_key": None, "primary_key_explicit": False},
            mock_context,
            bare_table="orders",
        )
        assert result == ["order_id", "load_period"]

    def test_sink_pk_string_normalised(self, mock_context):
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(sink_pk="order_id")
        result = ContractReconciler.reconcile_primary_key(
            contract,
            {"primary_key": None, "primary_key_explicit": False},
            mock_context,
            bare_table="orders",
        )
        assert result == ["order_id"]

    def test_sink_pk_takes_precedence_over_schema_flags(self, mock_context):
        """Sink-level names the upsert conflict key; schema-level often marks
        a surrogate identifier (data-platform dim_hcpcs). Sink wins."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(sink_pk=["hcpcs_code"], schema_pk_col="hcpcs_key")
        result = ContractReconciler.reconcile_primary_key(
            contract,
            {"primary_key": None, "primary_key_explicit": False},
            mock_context,
            bare_table="orders",
        )
        assert result == ["hcpcs_code"]

    def test_schema_flags_remain_fallback_without_sink_pk(self, mock_context):
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(schema_pk_col="order_id")
        result = ContractReconciler.reconcile_primary_key(
            contract,
            {"primary_key": None, "primary_key_explicit": False},
            mock_context,
            bare_table="orders",
        )
        assert result == ["order_id"]

    def test_sink_pk_conflicting_metadata_raises(self, mock_context):
        from moncpipelib.contracts.exceptions import ContractViolationError
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(sink_pk=["order_id"])
        with pytest.raises(ContractViolationError, match="Primary key conflict"):
            ContractReconciler.reconcile_primary_key(
                contract,
                {"primary_key": ["other_id"], "primary_key_explicit": True},
                mock_context,
                bare_table="orders",
            )

    def test_sink_pk_matching_metadata_warns_redundant(self, mock_context):
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(sink_pk=["order_id"])
        result = ContractReconciler.reconcile_primary_key(
            contract,
            {"primary_key": ["order_id"], "primary_key_explicit": True},
            mock_context,
            bare_table="orders",
        )
        assert result == ["order_id"]
        mock_context.log.warning.assert_called_once()

    def test_without_bare_table_sink_pk_not_consulted(self, mock_context):
        """Backward compatibility: pre-#401 callers that omit bare_table get
        the schema-flags-only behaviour."""
        from moncpipelib.contracts.reconciliation import ContractReconciler

        contract = self._contract(sink_pk=["order_id"])
        result = ContractReconciler.reconcile_primary_key(
            contract,
            {"primary_key": None, "primary_key_explicit": False},
            mock_context,
        )
        assert result is None
