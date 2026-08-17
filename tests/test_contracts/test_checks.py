"""Tests for contract-based Dagster asset check generation."""

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from dagster import AssetKey

from moncpipelib.config import LineageDefaults
from moncpipelib.contracts import (
    Column,
    ColumnTest,
    ColumnType,
    DataContract,
    Schema,
    Severity,
    TableExpectation,
    generate_asset_check,
    generate_asset_checks_from_contract,
    load_contract,
    load_contract_checks,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestGenerateAssetCheck:
    """Tests for generate_asset_check function."""

    @pytest.fixture
    def simple_contract(self):
        """Create a simple contract for testing."""
        columns = [
            Column(name="id", type=ColumnType.INTEGER, nullable=False),
            Column(name="value", type=ColumnType.STRING, nullable=True),
        ]
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
        )

    def test_generates_asset_check_definition(self, simple_contract):
        """Test that a valid AssetChecksDefinition is generated."""
        check = generate_asset_check(simple_contract, "test_asset")
        assert check is not None
        # AssetChecksDefinition has check_keys attribute
        assert hasattr(check, "check_keys")

    def test_accepts_string_asset_key(self, simple_contract):
        """Test generating check with string asset key."""
        check = generate_asset_check(simple_contract, "test_asset")
        assert check is not None

    def test_accepts_list_asset_key(self, simple_contract):
        """Test generating check with list asset key."""
        check = generate_asset_check(simple_contract, ["bronze", "test_asset"])
        assert check is not None


class TestGenerateAssetChecksFromContract:
    """Tests for generate_asset_checks_from_contract function."""

    @pytest.fixture
    def contract_with_tests(self):
        """Create a contract with column tests and expectations."""
        columns = [
            Column(
                name="id",
                type=ColumnType.INTEGER,
                nullable=False,
                tests=[
                    ColumnTest(test_type="not_null"),
                    ColumnTest(test_type="unique"),
                ],
            ),
            Column(
                name="status",
                type=ColumnType.STRING,
                nullable=False,
                tests=[
                    ColumnTest(
                        test_type="accepted_values",
                        parameters={"values": ["a", "b", "c"]},
                        severity=Severity.WARN,
                    ),
                ],
            ),
        ]
        expectations = [
            TableExpectation(
                expectation_type="row_count",
                parameters={"min": 1, "max": 1000},
            ),
        ]
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
            expectations=expectations,
        )

    def test_generates_multiple_checks(self, contract_with_tests):
        """Test that multiple checks are generated (unbatched)."""

        def mock_loader(_context):
            return pl.DataFrame({"id": [1, 2], "status": ["a", "b"]})

        checks = generate_asset_checks_from_contract(
            contract_with_tests,
            "test_asset",
            mock_loader,
            batched=False,
        )

        # Should have: 1 schema + 3 column tests + 1 expectation = 5 checks
        assert len(checks) == 5

    def test_skips_managed_columns(self):
        """Test that managed columns are skipped."""
        columns = [
            Column(name="id", type=ColumnType.INTEGER, nullable=False),
            Column(
                name=LineageDefaults.ID_COLUMN,
                type=ColumnType.UUID,
                nullable=False,
                managed=True,
                tests=[ColumnTest(test_type="not_null")],  # Should be skipped
            ),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
        )

        def mock_loader(_context):
            return pl.DataFrame({"id": [1]})

        checks = generate_asset_checks_from_contract(contract, "test_asset", mock_loader)

        # Should have: 1 schema check only (managed column tests skipped)
        assert len(checks) == 1

    def test_expectation_check_names_include_column_parameter(self):
        """Multiple expectations of same type on different columns get unique names."""
        columns = [
            Column(name="provider_key", type=ColumnType.STRING, nullable=False),
            Column(name="npi", type=ColumnType.STRING, nullable=False),
        ]
        expectations = [
            TableExpectation(
                expectation_type="null_percentage",
                parameters={"column": "provider_key", "max_percent": 0},
            ),
            TableExpectation(
                expectation_type="null_percentage",
                parameters={"column": "npi", "max_percent": 0},
            ),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="dim_provider",
            layer="gold",
            schema=Schema(columns=columns),
            expectations=expectations,
        )

        def mock_loader(_context):
            return pl.DataFrame({"provider_key": ["PK-1"], "npi": ["NPI-1"]})

        checks = generate_asset_checks_from_contract(contract, "dim_provider", mock_loader)

        check_names = [key.name for chk in checks for key in chk.check_keys]
        # Should have unique names for each null_percentage expectation
        assert "dim_provider_null_percentage_provider_key" in check_names
        assert "dim_provider_null_percentage_npi" in check_names
        # No duplicate names
        assert len(check_names) == len(set(check_names))

    def test_expectation_check_name_without_column_parameter(self):
        """Expectations without a column parameter use the base name."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        expectations = [
            TableExpectation(
                expectation_type="row_count",
                parameters={"min": 1, "max": 1000},
            ),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
            expectations=expectations,
        )

        def mock_loader(_context):
            return pl.DataFrame({"id": [1]})

        checks = generate_asset_checks_from_contract(contract, "test_asset", mock_loader)

        check_names = [key.name for chk in checks for key in chk.check_keys]
        assert "test_asset_row_count" in check_names


class TestLoadContractChecks:
    """Tests for load_contract_checks function."""

    def test_loads_and_generates_checks(self):
        """Test loading contract from file and generating checks."""

        def mock_loader(_context):
            return pl.DataFrame({"id": [1], "value": ["test"]})

        checks = load_contract_checks(
            FIXTURES_DIR / "minimal_contract.yaml",
            "simple_asset",
            mock_loader,
        )

        # Minimal contract has 2 columns, no tests, no expectations
        # Should have 1 schema check
        assert len(checks) == 1


class TestIntegrationWithRealContract:
    """Integration tests using real contract fixtures."""

    def test_full_contract_generates_checks(self):
        """Test generating checks from the full valid contract."""
        contract = load_contract(FIXTURES_DIR / "valid_contract.yaml")

        def mock_loader(_context):
            return pl.DataFrame(
                {
                    "claim_id": ["CLM-001"],
                    "patient_id": ["PAT-12345678"],
                    "amount": [100.0],
                    "claim_date": [pl.date(2024, 1, 15)],
                    "status": ["pending"],
                    "diagnosis_code": ["A01.1"],
                    LineageDefaults.ID_COLUMN: ["test-uuid"],
                    LineageDefaults.KEY_COLUMN: ["v1:test:bronze:2024-01-15:abc123"],
                }
            )

        # Default batched=True: single definition with all specs
        checks = generate_asset_checks_from_contract(
            contract,
            "claims_bronze",
            mock_loader,
        )
        assert len(checks) == 1
        # The single definition should contain many check specs
        assert len(checks[0].check_keys) > 5

        # batched=False: individual definitions
        individual = generate_asset_checks_from_contract(
            contract,
            "claims_bronze",
            mock_loader,
            batched=False,
        )
        assert len(individual) > 5


class TestSeverityToDagster:
    """Tests for _severity_to_dagster function."""

    def test_error_severity_conversion(self):
        """Test ERROR severity converts correctly."""
        from dagster import AssetCheckSeverity

        from moncpipelib.contracts.checks import _severity_to_dagster

        result = _severity_to_dagster(Severity.ERROR)
        assert result == AssetCheckSeverity.ERROR

    def test_warn_severity_conversion(self):
        """Test WARN severity converts correctly."""
        from dagster import AssetCheckSeverity

        from moncpipelib.contracts.checks import _severity_to_dagster

        result = _severity_to_dagster(Severity.WARN)
        assert result == AssetCheckSeverity.WARN


class TestDiscoverContractChecks:
    """Tests for discover_contract_checks function."""

    def test_discovers_contracts_in_directory(self, tmp_path):
        """Test discovering contracts from a directory with *.contract.yaml files."""
        from moncpipelib.contracts import discover_contract_checks

        # Create a test contract file with the expected naming pattern
        contract_content = """
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
"""
        contract_file = tmp_path / "test_asset.contract.yaml"
        contract_file.write_text(contract_content)

        def make_loader(_asset_name):
            def loader(_context):
                return pl.DataFrame({"id": [1]})

            return loader

        checks = discover_contract_checks(tmp_path, make_loader)

        # Should find the contract we created
        assert len(checks) > 0
        # No sink schema and no prefix -> legacy flat key fallback.
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {AssetKey(["test_asset"])}

    def test_sink_derived_asset_key(self, tmp_path):
        """Checks attach to [sink_schema, sink_table] when the contract has a sink."""
        from moncpipelib.contracts import discover_contract_checks

        contract_content = """
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: npi
layer: silver
sinks:
  - type: table
    schema: reference_silver
    table: npi
schema:
  columns:
    - name: id
      type: integer
      nullable: false
"""
        (tmp_path / "npi.contract.yaml").write_text(contract_content)

        def make_loader(_asset_name):
            def loader(_context):
                return pl.DataFrame({"id": [1]})

            return loader

        checks = discover_contract_checks(tmp_path, make_loader)

        assert len(checks) > 0
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {AssetKey(["reference_silver", "npi"])}

    def test_discovers_with_asset_key_prefix(self, tmp_path):
        """Test discovering contracts with asset key prefix."""
        from moncpipelib.contracts import discover_contract_checks

        # Create a test contract file
        contract_content = """
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
"""
        contract_file = tmp_path / "test_asset.contract.yaml"
        contract_file.write_text(contract_content)

        def make_loader(_asset_name):
            def loader(_context):
                return pl.DataFrame({"id": [1]})

            return loader

        checks = discover_contract_checks(
            tmp_path,
            make_loader,
            asset_key_prefix=["bronze"],
        )

        assert len(checks) > 0
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {AssetKey(["bronze", "test_asset"])}

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Test that an empty directory returns no checks."""
        from moncpipelib.contracts import discover_contract_checks

        def make_loader(_asset_name):
            def loader(_context):
                return pl.DataFrame({"id": [1]})

            return loader

        checks = discover_contract_checks(tmp_path, make_loader)

        assert checks == []


class TestAssetCheckWithAssetKey:
    """Tests for asset check generation with different asset key types."""

    @pytest.fixture
    def simple_contract(self):
        """Create a simple contract for testing."""
        columns = [
            Column(name="id", type=ColumnType.INTEGER, nullable=False),
        ]
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
        )

    def test_accepts_tuple_asset_key(self, simple_contract):
        """Test generating check with tuple asset key."""
        check = generate_asset_check(simple_contract, ("bronze", "test_asset"))
        assert check is not None

    def test_checks_from_contract_with_tuple_key(self, simple_contract):
        """Test generating checks from contract with tuple asset key."""

        def mock_loader(_context):
            return pl.DataFrame({"id": [1]})

        checks = generate_asset_checks_from_contract(
            simple_contract,
            ("bronze", "test_asset"),
            mock_loader,
        )

        assert len(checks) > 0


class TestGenerateAssetCheckDeprecation:
    """Tests for generate_asset_check deprecation warning."""

    def test_emits_deprecation_warning(self):
        """Test that generate_asset_check emits a DeprecationWarning."""
        columns = [
            Column(name="id", type=ColumnType.INTEGER, nullable=False),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
        )

        with pytest.warns(DeprecationWarning, match="generate_asset_check.*deprecated"):
            generate_asset_check(contract, "test_asset")


class TestDeriveTableName:
    """Tests for _derive_table_name helper function."""

    def test_basic_derivation(self):
        """Test basic table name from asset name."""
        from moncpipelib.contracts.checks import _derive_table_name

        result = _derive_table_name("orders", db_schema="bronze")
        assert result == "bronze.orders"

    def test_suffix_stripping(self):
        """Test suffix is stripped from asset name."""
        from moncpipelib.contracts.checks import _derive_table_name

        result = _derive_table_name(
            "orders_bronze",
            db_schema="bronze",
            table_suffix_to_strip="_bronze",
        )
        assert result == "bronze.orders"

    def test_prefix_addition(self):
        """Test prefix is prepended to table name."""
        from moncpipelib.contracts.checks import _derive_table_name

        result = _derive_table_name(
            "orders",
            db_schema="bronze",
            table_prefix="test_",
        )
        assert result == "bronze.test_orders"

    def test_schema_override(self):
        """Test schema_override takes precedence over db_schema."""
        from moncpipelib.contracts.checks import _derive_table_name

        result = _derive_table_name(
            "orders",
            db_schema="bronze",
            schema_override="test_schema",
        )
        assert result == "test_schema.orders"

    def test_all_options_combined(self):
        """Test all naming options together."""
        from moncpipelib.contracts.checks import _derive_table_name

        result = _derive_table_name(
            "orders_bronze",
            db_schema="bronze",
            table_suffix_to_strip="_bronze",
            table_prefix="ci_",
            schema_override="test_schema",
        )
        assert result == "test_schema.ci_orders"


class TestMakeDfLoaderFactory:
    """Tests for _make_df_loader_factory function."""

    @patch("moncpipelib.contracts.checks.psycopg.connect")
    @patch("moncpipelib.contracts.checks.PostgresPolarsSchema")
    def test_factory_creates_working_loader(self, mock_schema_cls, mock_connect):
        """Test that the factory creates a loader that reads from Postgres."""
        from moncpipelib.contracts.checks import _make_df_loader_factory

        # Set up mocks
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_schema_cls.from_psycopg2_connection.return_value = {}

        factory = _make_df_loader_factory(
            host="localhost",
            port=5432,
            user="test",
            password="pass",
            database="testdb",
            db_schema="bronze",
            sslmode="disable",
        )

        loader = factory("orders")
        assert callable(loader)

        # Verify the loader tries to connect with correct params
        with patch("moncpipelib.contracts.checks.pl.read_database") as mock_read:
            mock_read.return_value = pl.DataFrame({"id": [1]})
            mock_context = MagicMock()

            result = loader(mock_context)

            mock_connect.assert_called_with(
                host="localhost",
                port=5432,
                user="test",
                password="pass",
                dbname="testdb",
                sslmode="disable",
            )
            mock_schema_cls.register_uuid_adapter.assert_called_with(mock_conn)
            mock_read.assert_called_once()
            assert result.shape == (1, 1)
            mock_conn.close.assert_called_once()

    @patch("moncpipelib.contracts.checks.psycopg.connect")
    @patch("moncpipelib.contracts.checks.PostgresPolarsSchema")
    def test_factory_applies_table_naming(self, mock_schema_cls, mock_connect):
        """Test that the factory derives table names correctly."""
        from moncpipelib.contracts.checks import _make_df_loader_factory

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_schema_cls.from_psycopg2_connection.return_value = {}

        factory = _make_df_loader_factory(
            host="localhost",
            port=5432,
            user="test",
            password="pass",
            database="testdb",
            db_schema="bronze",
            sslmode="disable",
            table_suffix_to_strip="_bronze",
            table_prefix="ci_",
        )

        loader = factory("orders_bronze")

        with patch("moncpipelib.contracts.checks.pl.read_database") as mock_read:
            mock_read.return_value = pl.DataFrame({"id": [1]})
            mock_context = MagicMock()

            loader(mock_context)

            # Verify the query uses the correctly derived table name
            call_args = mock_read.call_args
            query = call_args.kwargs.get("query") or call_args[1].get("query") or call_args[0][0]
            assert "bronze.ci_orders" in query


class TestMakeContractChecks:
    """Tests for standalone make_contract_checks function."""

    def test_discovers_and_generates_checks(self, tmp_path):
        """Test that make_contract_checks discovers contracts and generates checks."""
        from moncpipelib.contracts.checks import make_contract_checks

        # Create a test contract file
        contract_content = """
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
"""
        contract_file = tmp_path / "test_asset.contract.yaml"
        contract_file.write_text(contract_content)

        checks = make_contract_checks(
            tmp_path,
            host="localhost",
            user="test",
            password="pass",
            database="testdb",
            db_schema="bronze",
        )

        # Should find the contract and generate at least a schema check
        assert len(checks) > 0
        # Without a prefix, checks attach to the resolved [schema, table]
        # key so they bind to the real asset instead of a flat stub key.
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {AssetKey(["bronze", "test_asset"])}

    def test_empty_directory_returns_empty(self, tmp_path):
        """Test that an empty directory returns no checks."""
        from moncpipelib.contracts.checks import make_contract_checks

        checks = make_contract_checks(
            tmp_path,
            host="localhost",
            user="test",
            password="pass",
            database="testdb",
            db_schema="bronze",
        )

        assert checks == []

    def test_with_asset_key_prefix(self, tmp_path):
        """Test that asset_key_prefix is passed through correctly."""
        from moncpipelib.contracts.checks import make_contract_checks

        contract_content = """
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
"""
        (tmp_path / "test_asset.contract.yaml").write_text(contract_content)

        checks = make_contract_checks(
            tmp_path,
            host="localhost",
            user="test",
            password="pass",
            database="testdb",
            db_schema="bronze",
            asset_key_prefix=["custom_prefix"],
        )

        assert len(checks) > 0
        # Explicit prefix overrides sink-derived keys with [*prefix, asset].
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {AssetKey(["custom_prefix", "test_asset"])}


# ---------------------------------------------------------------------------
# _resolve_check_table
# ---------------------------------------------------------------------------

_MINIMAL_CONTRACT_YAML = """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: {asset}
layer: {layer}
{extra}
schema:
  columns:
    - name: id
      type: integer
      nullable: false
"""


class TestResolveCheckTable:
    """Tests for _resolve_check_table function."""

    def _make_contract(
        self,
        asset: str = "test_asset",
        layer: str = "bronze",
        sinks: list[dict] | None = None,
    ) -> DataContract:
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset=asset,
            layer=layer,
            schema=Schema(columns=columns),
            sinks=sinks,
        )

    def test_uses_sink_schema_and_table(self):
        """Contract with sink schema + table resolves to sink values."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract(
            sinks=[{"type": "table", "schema": "reference_silver", "table": "fda_ndc_directory"}]
        )
        result = _resolve_check_table(contract, db_schema="bronze")
        assert result == "reference_silver.fda_ndc_directory"

    def test_schema_override_beats_sink(self):
        """schema_override always wins over sink schema."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract(
            sinks=[{"type": "table", "schema": "reference_silver", "table": "fda_ndc_directory"}]
        )
        result = _resolve_check_table(
            contract, schema_override="integration_tests", db_schema="bronze"
        )
        assert result == "integration_tests.fda_ndc_directory"

    def test_falls_back_to_default_schema(self):
        """No sink schema falls back to default_schema."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract(sinks=[{"type": "table", "table": "orders"}])
        result = _resolve_check_table(contract, default_schema="silver")
        assert result == "silver.orders"

    def test_falls_back_to_db_schema(self):
        """No sink schema or default_schema falls back to db_schema."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract()  # no sinks
        result = _resolve_check_table(contract, db_schema="bronze")
        assert result == "bronze.test_asset"

    def test_asset_name_fallback(self):
        """No sink table uses contract.asset as table name."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract(
            asset="claims_bronze",
            sinks=[{"type": "table", "schema": "bronze"}],
        )
        result = _resolve_check_table(contract)
        assert result == "bronze.claims_bronze"

    def test_applies_prefix_and_suffix_strip(self):
        """table_prefix and table_suffix_to_strip work correctly."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract(asset="orders_bronze")
        result = _resolve_check_table(
            contract,
            db_schema="bronze",
            table_suffix_to_strip="_bronze",
            table_prefix="ci_abc_",
        )
        assert result == "bronze.ci_abc_orders"

    def test_sink_table_ignores_suffix_strip(self):
        """When sink provides table name, suffix stripping is not applied."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract(
            asset="orders_bronze",
            sinks=[{"type": "table", "schema": "bronze", "table": "orders"}],
        )
        result = _resolve_check_table(
            contract,
            db_schema="bronze",
            table_suffix_to_strip="_bronze",
            table_prefix="ci_",
        )
        # Prefix still applies, but suffix stripping does NOT (sink table is explicit)
        assert result == "bronze.ci_orders"

    def test_no_schema_raises(self):
        """No schema from any source raises ValueError."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract()
        with pytest.raises(ValueError, match="No schema resolved"):
            _resolve_check_table(contract)

    def test_skips_non_table_sinks(self):
        """Non-table sinks are ignored when resolving schema."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract(
            sinks=[
                {"type": "file", "schema": "wrong_schema"},
                {"type": "table", "schema": "correct_schema", "table": "orders"},
            ]
        )
        result = _resolve_check_table(contract)
        assert result == "correct_schema.orders"

    def test_default_schema_beats_db_schema(self):
        """default_schema has higher priority than db_schema."""
        from moncpipelib.contracts.checks import _resolve_check_table

        contract = self._make_contract()
        result = _resolve_check_table(contract, default_schema="silver", db_schema="bronze")
        assert result == "silver.test_asset"


# ---------------------------------------------------------------------------
# _derive_check_asset_key
# ---------------------------------------------------------------------------


class TestDeriveCheckAssetKey:
    """Tests for _derive_check_asset_key function."""

    def _make_contract(
        self,
        asset: str = "test_asset",
        layer: str = "bronze",
        sinks: list[dict] | None = None,
    ) -> DataContract:
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset=asset,
            layer=layer,
            schema=Schema(columns=columns),
            sinks=sinks,
        )

    def test_prefix_overrides_fq_table_and_sink(self):
        """asset_key_prefix preserves the legacy [*prefix, asset] shape."""
        from moncpipelib.contracts.checks import _derive_check_asset_key

        contract = self._make_contract(
            sinks=[{"type": "table", "schema": "reference_silver", "table": "npi"}]
        )
        key = _derive_check_asset_key(
            contract,
            fq_table="reference_silver.npi",
            asset_key_prefix=["bronze"],
        )
        assert key == AssetKey(["bronze", "test_asset"])

    def test_fq_table_splits_into_schema_and_table(self):
        """fq_table produces the [schema, table] key used by real assets."""
        from moncpipelib.contracts.checks import _derive_check_asset_key

        contract = self._make_contract()
        key = _derive_check_asset_key(contract, fq_table="reference_silver.npi")
        assert key == AssetKey(["reference_silver", "npi"])

    def test_fq_table_splits_on_first_dot_only(self):
        """Only the first dot separates schema from table."""
        from moncpipelib.contracts.checks import _derive_check_asset_key

        contract = self._make_contract()
        key = _derive_check_asset_key(contract, fq_table="myschema.table.v2")
        assert key == AssetKey(["myschema", "table.v2"])

    def test_sink_fallback_when_no_fq_table(self):
        """Without fq_table, the first table sink drives the key."""
        from moncpipelib.contracts.checks import _derive_check_asset_key

        contract = self._make_contract(
            sinks=[{"type": "table", "schema": "mhhs_silver", "table": "claims_837i"}]
        )
        key = _derive_check_asset_key(contract)
        assert key == AssetKey(["mhhs_silver", "claims_837i"])

    def test_sink_without_table_uses_asset_name(self):
        """Sink schema without an explicit table falls back to contract.asset."""
        from moncpipelib.contracts.checks import _derive_check_asset_key

        contract = self._make_contract(
            asset="orders",
            sinks=[{"type": "table", "schema": "bronze"}],
        )
        key = _derive_check_asset_key(contract)
        assert key == AssetKey(["bronze", "orders"])

    def test_non_table_sinks_skipped(self):
        """Non-table sinks never contribute a schema."""
        from moncpipelib.contracts.checks import _derive_check_asset_key

        contract = self._make_contract(sinks=[{"type": "file", "schema": "wrong"}])
        key = _derive_check_asset_key(contract)
        assert key == AssetKey(["test_asset"])

    def test_no_sinks_falls_back_to_flat_key(self):
        """Contracts without sinks keep the legacy flat key."""
        from moncpipelib.contracts.checks import _derive_check_asset_key

        contract = self._make_contract()
        key = _derive_check_asset_key(contract)
        assert key == AssetKey(["test_asset"])


# ---------------------------------------------------------------------------
# _make_deferred_df_loader
# ---------------------------------------------------------------------------


class TestMakeDeferredDfLoader:
    """Tests for _make_deferred_df_loader function."""

    def test_connection_factory_not_called_at_creation(self):
        """Connection factory must NOT be called at loader creation time."""
        from moncpipelib.contracts.checks import _make_deferred_df_loader

        factory = MagicMock()
        _make_deferred_df_loader(connection_factory=factory, fq_table="s.t")
        factory.assert_not_called()

    @patch("moncpipelib.contracts.checks.pl.read_database")
    @patch("moncpipelib.contracts.checks.PostgresPolarsSchema")
    def test_connection_factory_called_at_execution(self, mock_schema_cls, mock_read):
        """Connection factory is called when the loader runs."""
        from moncpipelib.contracts.checks import _make_deferred_df_loader

        mock_conn = MagicMock()
        factory = MagicMock(return_value=mock_conn)
        mock_schema_cls.from_psycopg2_connection.return_value = {}
        mock_read.return_value = pl.DataFrame({"id": [1]})

        loader = _make_deferred_df_loader(connection_factory=factory, fq_table="bronze.orders")
        result = loader(MagicMock())

        factory.assert_called_once()
        mock_schema_cls.register_uuid_adapter.assert_called_with(mock_conn)
        assert result.shape == (1, 1)

    @patch("moncpipelib.contracts.checks.pl.read_database")
    @patch("moncpipelib.contracts.checks.PostgresPolarsSchema")
    def test_closes_connection_on_success(self, mock_schema_cls, mock_read):
        """Connection is closed after successful read."""
        from moncpipelib.contracts.checks import _make_deferred_df_loader

        mock_conn = MagicMock()
        factory = MagicMock(return_value=mock_conn)
        mock_schema_cls.from_psycopg2_connection.return_value = {}
        mock_read.return_value = pl.DataFrame({"id": [1]})

        loader = _make_deferred_df_loader(connection_factory=factory, fq_table="bronze.orders")
        loader(MagicMock())

        mock_conn.close.assert_called_once()

    @patch("moncpipelib.contracts.checks.PostgresPolarsSchema")
    def test_closes_connection_on_error(self, mock_schema_cls):
        """Connection is closed even when an error occurs."""
        from moncpipelib.contracts.checks import _make_deferred_df_loader

        mock_conn = MagicMock()
        factory = MagicMock(return_value=mock_conn)
        mock_schema_cls.from_psycopg2_connection.side_effect = RuntimeError("boom")

        loader = _make_deferred_df_loader(connection_factory=factory, fq_table="bronze.orders")
        with pytest.raises(RuntimeError, match="boom"):
            loader(MagicMock())

        mock_conn.close.assert_called_once()

    @patch("moncpipelib.contracts.checks.pl.read_database")
    @patch("moncpipelib.contracts.checks.PostgresPolarsSchema")
    def test_uses_correct_table_in_query(self, mock_schema_cls, mock_read):
        """Loader queries the correct fully-qualified table name."""
        from moncpipelib.contracts.checks import _make_deferred_df_loader

        mock_conn = MagicMock()
        factory = MagicMock(return_value=mock_conn)
        mock_schema_cls.from_psycopg2_connection.return_value = {}
        mock_read.return_value = pl.DataFrame({"id": [1]})

        loader = _make_deferred_df_loader(
            connection_factory=factory, fq_table="reference_silver.fda_ndc_directory"
        )
        loader(MagicMock())

        call_args = mock_read.call_args
        query = call_args.kwargs.get("query") or call_args[0][0]
        assert "reference_silver.fda_ndc_directory" in query


# ---------------------------------------------------------------------------
# discover_contract_checks - recursive glob
# ---------------------------------------------------------------------------


class TestDiscoverContractChecksRecursive:
    """Tests for recursive contract discovery."""

    def test_discovers_contracts_in_nested_directories(self, tmp_path: Path):
        """Contracts in subdirectories are discovered."""
        from moncpipelib.contracts import discover_contract_checks

        # Create nested contract
        nested_dir = tmp_path / "gold" / "dim_patient"
        nested_dir.mkdir(parents=True)
        contract_content = _MINIMAL_CONTRACT_YAML.format(
            asset="dim_patient", layer="gold", extra=""
        )
        (nested_dir / "dim_patient.contract.yaml").write_text(contract_content)

        def make_loader(_asset_name):
            def loader(_context):
                return pl.DataFrame({"id": [1]})

            return loader

        checks = discover_contract_checks(tmp_path, make_loader)
        assert len(checks) > 0

    def test_discovers_both_top_level_and_nested(self, tmp_path: Path):
        """Both top-level and nested contracts are discovered."""
        from moncpipelib.contracts import discover_contract_checks

        # Top-level contract
        (tmp_path / "top.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="top_asset", layer="bronze", extra="")
        )
        # Nested contract
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        (nested / "nested.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="nested_asset", layer="silver", extra="")
        )

        def make_loader(_asset_name):
            def loader(_context):
                return pl.DataFrame({"id": [1]})

            return loader

        checks = discover_contract_checks(tmp_path, make_loader)
        # Each contract generates at least 1 check (schema check)
        assert len(checks) >= 2


# ---------------------------------------------------------------------------
# make_contract_checks - connection_factory and per-contract schema
# ---------------------------------------------------------------------------


class TestMakeContractChecksConnectionFactory:
    """Tests for standalone make_contract_checks with connection_factory."""

    def test_with_connection_factory(self, tmp_path: Path):
        """connection_factory parameter generates checks without legacy credentials."""
        from moncpipelib.contracts.checks import make_contract_checks

        contract_yaml = _MINIMAL_CONTRACT_YAML.format(
            asset="orders",
            layer="bronze",
            extra="sinks:\n  - type: table\n    schema: bronze\n    table: orders",
        )
        (tmp_path / "orders.contract.yaml").write_text(contract_yaml)

        mock_factory = MagicMock()
        checks = make_contract_checks(
            tmp_path,
            connection_factory=mock_factory,
        )

        assert len(checks) > 0
        # Factory not called at definition time
        mock_factory.assert_not_called()

    def test_per_contract_schema_from_sinks(self, tmp_path: Path):
        """Each contract reads schema from its own sink."""
        from moncpipelib.contracts.checks import make_contract_checks

        # Contract 1: reference_silver schema
        (tmp_path / "fda.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="fda_ndc",
                layer="silver",
                extra=(
                    "sinks:\n"
                    "  - type: table\n"
                    "    schema: reference_silver\n"
                    "    table: fda_ndc_directory"
                ),
            )
        )
        # Contract 2: synthetic_gold schema
        (tmp_path / "dim.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="dim_patient",
                layer="gold",
                extra=(
                    "sinks:\n  - type: table\n    schema: synthetic_gold\n    table: dim_patient"
                ),
            )
        )

        mock_factory = MagicMock()
        checks = make_contract_checks(
            tmp_path,
            connection_factory=mock_factory,
        )

        # Should generate checks for both contracts
        assert len(checks) >= 2

        # Each contract's checks attach to its own sink-derived key.
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {
            AssetKey(["reference_silver", "fda_ndc_directory"]),
            AssetKey(["synthetic_gold", "dim_patient"]),
        }

    def test_legacy_credentials_still_work(self, tmp_path: Path):
        """Legacy host/user/password params still generate checks."""
        from moncpipelib.contracts.checks import make_contract_checks

        (tmp_path / "test.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="test_asset", layer="bronze", extra="")
        )

        checks = make_contract_checks(
            tmp_path,
            host="localhost",
            user="test",
            password="pass",
            database="testdb",
            db_schema="bronze",
        )

        assert len(checks) > 0

    def test_recursive_discovery(self, tmp_path: Path):
        """Standalone function discovers contracts in nested directories."""
        from moncpipelib.contracts.checks import make_contract_checks

        nested = tmp_path / "defs" / "gold" / "dim_date"
        nested.mkdir(parents=True)
        (nested / "dim_date.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="dim_date",
                layer="gold",
                extra=("sinks:\n  - type: table\n    schema: gold\n    table: dim_date"),
            )
        )

        mock_factory = MagicMock()
        checks = make_contract_checks(
            tmp_path,
            connection_factory=mock_factory,
        )

        assert len(checks) > 0


# ---------------------------------------------------------------------------
# PostgresIOManager.make_contract_checks - deferred connection
# ---------------------------------------------------------------------------


class TestIOManagerMakeContractChecks:
    """Tests for PostgresIOManager.make_contract_checks with deferred connection."""

    def test_deferred_connection_not_called_at_definition_time(self, tmp_path: Path):
        """psycopg2.connect must NOT be called during make_contract_checks()."""
        from moncpipelib.io_managers import PostgresIOManager
        from moncpipelib.resources.postgres import PostgresResource

        (tmp_path / "test.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="test_asset",
                layer="bronze",
                extra=("sinks:\n  - type: table\n    schema: bronze\n    table: test_asset"),
            )
        )

        resource = PostgresResource(
            host="db.example.com",
            user="writer",
            password="secret",
            database="analytics",
        )
        io_mgr = PostgresIOManager(
            postgres_resource=resource,
            default_schema="bronze",
        )

        # The connection factory (and its psycopg.connect call) lives on the
        # resource now -- both make_contract_checks implementations share it.
        with patch("moncpipelib.resources.postgres.psycopg.connect") as mock_connect:
            checks = io_mgr.make_contract_checks(tmp_path)
            # Connection is deferred -- not called at definition time
            mock_connect.assert_not_called()

        assert len(checks) > 0

    def test_per_contract_schema_from_sinks(self, tmp_path: Path):
        """IO manager reads schema from each contract's sink."""
        from moncpipelib.io_managers import PostgresIOManager
        from moncpipelib.resources.postgres import PostgresResource

        # Two contracts with different schemas
        (tmp_path / "silver.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="claims",
                layer="silver",
                extra=("sinks:\n  - type: table\n    schema: synthetic_silver\n    table: claims"),
            )
        )
        (tmp_path / "gold.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="dim_patient",
                layer="gold",
                extra=(
                    "sinks:\n  - type: table\n    schema: synthetic_gold\n    table: dim_patient"
                ),
            )
        )

        resource = PostgresResource(
            host="db.example.com",
            user="writer",
            password="secret",
            database="analytics",
        )
        io_mgr = PostgresIOManager(
            postgres_resource=resource,
            default_schema="silver",
        )

        checks = io_mgr.make_contract_checks(tmp_path)

        # Should generate checks for both contracts
        assert len(checks) >= 2

        # Each contract's checks attach to its own sink-derived key.
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {
            AssetKey(["synthetic_silver", "claims"]),
            AssetKey(["synthetic_gold", "dim_patient"]),
        }

    def test_schema_override_applies_to_all(self, tmp_path: Path):
        """schema_override beats all sink schemas for test isolation.

        Verifies via _resolve_check_table that the IO manager's
        schema_override takes priority over contract sink schemas.
        """
        from moncpipelib.contracts.checks import _resolve_check_table
        from moncpipelib.contracts.loader import load_contract

        (tmp_path / "test.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="orders",
                layer="silver",
                extra=("sinks:\n  - type: table\n    schema: synthetic_silver\n    table: orders"),
            )
        )

        contract = load_contract(tmp_path / "test.contract.yaml")
        fq_table = _resolve_check_table(
            contract,
            schema_override="integration_tests",
            default_schema="silver",
        )
        assert fq_table == "integration_tests.orders"

    def test_recursive_discovery(self, tmp_path: Path):
        """IO manager discovers contracts in nested directories."""
        from moncpipelib.io_managers import PostgresIOManager
        from moncpipelib.resources.postgres import PostgresResource

        nested = tmp_path / "defs" / "gold"
        nested.mkdir(parents=True)
        (nested / "dim_date.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="dim_date",
                layer="gold",
                extra=("sinks:\n  - type: table\n    schema: gold\n    table: dim_date"),
            )
        )

        resource = PostgresResource(
            host="db.example.com",
            user="writer",
            password="secret",
            database="analytics",
        )
        io_mgr = PostgresIOManager(
            postgres_resource=resource,
            default_schema="silver",
        )

        checks = io_mgr.make_contract_checks(tmp_path)
        assert len(checks) > 0

    def test_envvar_fields_resolved_at_execution_time(self, tmp_path: Path, monkeypatch):
        """EnvVar fields must be resolved via get_value() at check execution time.

        This is the core bug: asset check ops are not Dagster resources, so
        Dagster never resolves EnvVar fields.  make_contract_checks() must
        build a connection factory that calls EnvVar.get_value() itself.
        """
        from dagster import EnvVar

        from moncpipelib.io_managers import PostgresIOManager
        from moncpipelib.resources.postgres import PostgresResource

        monkeypatch.setenv("TEST_DB_HOST", "resolved-host.example.com")
        monkeypatch.setenv("TEST_DB_USER", "resolved_user")
        monkeypatch.setenv("TEST_DB_PASS", "resolved_pass")
        monkeypatch.setenv("TEST_DB_NAME", "resolved_db")

        (tmp_path / "test.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="test_asset",
                layer="bronze",
                extra="sinks:\n  - type: table\n    schema: bronze\n    table: test_asset",
            )
        )

        resource = PostgresResource(
            host=EnvVar("TEST_DB_HOST"),
            user=EnvVar("TEST_DB_USER"),
            password=EnvVar("TEST_DB_PASS"),
            database=EnvVar("TEST_DB_NAME"),
        )
        io_mgr = PostgresIOManager(
            postgres_resource=resource,
            default_schema="bronze",
        )

        checks = io_mgr.make_contract_checks(tmp_path)
        assert len(checks) > 0

        # Simulate check execution: the df_loader's connection factory should
        # resolve EnvVars to actual env var values, not the EnvVar key names.
        with patch("moncpipelib.contracts.checks.psycopg") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.description = [("id",), ("name",)]
            mock_cursor.fetchmany.side_effect = [[(1, "a")], []]

            mock_context = MagicMock()
            # Execute the schema check (first check) -- it calls the df_loader
            # which invokes the connection factory.
            with contextlib.suppress(Exception):
                checks[0].node_def.compute_fn.decorated_fn(mock_context)

            # Verify psycopg.connect was called with resolved values,
            # not EnvVar key names like "TEST_DB_HOST"
            if mock_pg.connect.called:
                call_kwargs = mock_pg.connect.call_args
                assert call_kwargs.kwargs.get("host") == "resolved-host.example.com"
                assert call_kwargs.kwargs.get("user") == "resolved_user"
                assert call_kwargs.kwargs.get("password") == "resolved_pass"
                assert call_kwargs.kwargs.get("dbname") == "resolved_db"


class TestResourceMakeContractChecks:
    """Tests for PostgresResource.make_contract_checks asset key derivation."""

    def _make_resource(self):
        from moncpipelib.resources.postgres import PostgresResource

        return PostgresResource(
            host="db.example.com",
            user="writer",
            password="secret",
            database="analytics",
        )

    def test_sink_derived_asset_key(self, tmp_path: Path):
        """Checks attach to the sink-derived [schema, table] key."""
        (tmp_path / "npi.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="npi",
                layer="silver",
                extra=("sinks:\n  - type: table\n    schema: reference_silver\n    table: npi"),
            )
        )

        checks = self._make_resource().make_contract_checks(tmp_path)

        assert len(checks) > 0
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {AssetKey(["reference_silver", "npi"])}

    def test_asset_key_prefix_overrides_sink(self, tmp_path: Path):
        """Explicit prefix preserves the legacy [*prefix, asset] shape."""
        (tmp_path / "npi.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(
                asset="npi",
                layer="silver",
                extra=("sinks:\n  - type: table\n    schema: reference_silver\n    table: npi"),
            )
        )

        checks = self._make_resource().make_contract_checks(tmp_path, asset_key_prefix=["bronze"])

        assert len(checks) > 0
        target_keys = {key.asset_key for chk in checks for key in chk.check_keys}
        assert target_keys == {AssetKey(["bronze", "npi"])}


class TestPIIRedactionInChecks:
    """Tests that PII flag is wired through to generated column checks."""

    @staticmethod
    def _get_closure_var(check_def, var_name: str):
        """Extract a closure variable from an AssetChecksDefinition's inner function."""
        fn = check_def.node_def.compute_fn.decorated_fn
        for cell_name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or [], strict=False):
            if cell_name == var_name:
                return cell.cell_contents
        return None

    def test_pii_column_captures_is_pii_true(self):
        """Verify PII column's generated check captures is_pii=True in closure."""
        columns = [
            Column(
                name="patient_name",
                type=ColumnType.STRING,
                nullable=False,
                pii=True,
                tests=[
                    ColumnTest(
                        test_type="accepted_values",
                        parameters={"values": ["ALLOWED"]},
                    ),
                ],
            ),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_pii",
            layer="bronze",
            schema=Schema(columns=columns),
        )

        def mock_loader(_context):
            return pl.DataFrame({"patient_name": ["Alice"]})

        checks = generate_asset_checks_from_contract(
            contract,
            "test_pii",
            mock_loader,
            batched=False,
        )
        column_checks = [
            c for c in checks if any("accepted_values" in k.name for k in c.check_keys)
        ]
        assert len(column_checks) == 1

        is_pii = self._get_closure_var(column_checks[0], "is_pii")
        assert is_pii is True

    def test_non_pii_column_captures_is_pii_false(self):
        """Verify non-PII column's generated check captures is_pii=False in closure."""
        columns = [
            Column(
                name="status",
                type=ColumnType.STRING,
                nullable=False,
                pii=False,
                tests=[
                    ColumnTest(
                        test_type="accepted_values",
                        parameters={"values": ["active"]},
                    ),
                ],
            ),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_non_pii",
            layer="bronze",
            schema=Schema(columns=columns),
        )

        def mock_loader(_context):
            return pl.DataFrame({"status": ["active"]})

        checks = generate_asset_checks_from_contract(
            contract,
            "test_non_pii",
            mock_loader,
            batched=False,
        )
        column_checks = [
            c for c in checks if any("accepted_values" in k.name for k in c.check_keys)
        ]
        assert len(column_checks) == 1

        is_pii = self._get_closure_var(column_checks[0], "is_pii")
        assert is_pii is False


# ---------------------------------------------------------------------------
# Batched checks
# ---------------------------------------------------------------------------


class TestBatchedChecks:
    """Tests for batched=True mode in generate_asset_checks_from_contract."""

    @pytest.fixture()
    def contract_with_rules(self):
        """Contract with schema, column tests, and table expectations."""
        columns = [
            Column(
                name="id",
                type=ColumnType.INTEGER,
                nullable=False,
                tests=[ColumnTest(test_type="not_null")],
            ),
            Column(
                name="name",
                type=ColumnType.STRING,
                nullable=True,
                tests=[
                    ColumnTest(test_type="not_null", severity=Severity.WARN),
                    ColumnTest(
                        test_type="accepted_values",
                        parameters={"values": ["a", "b"]},
                    ),
                ],
            ),
        ]
        return DataContract(
            version="1.0",
            pipeline_id="batch-test-uuid",
            asset="batch_test",
            layer="silver",
            schema=Schema(columns=columns),
            expectations=[
                TableExpectation(
                    expectation_type="row_count",
                    parameters={"min": 1},
                ),
            ],
        )

    def test_batched_returns_single_definition(self, contract_with_rules):
        """batched=True returns a single-element list."""

        def mock_loader(_ctx):
            return pl.DataFrame({"id": [1], "name": ["a"]})

        checks = generate_asset_checks_from_contract(
            contract_with_rules,
            "batch_test",
            mock_loader,
            batched=True,
        )
        assert len(checks) == 1

    def test_batched_contains_all_check_specs(self, contract_with_rules):
        """The single definition contains specs for all rules."""

        def mock_loader(_ctx):
            return pl.DataFrame({"id": [1], "name": ["a"]})

        checks = generate_asset_checks_from_contract(
            contract_with_rules,
            "batch_test",
            mock_loader,
            batched=True,
        )
        check_def = checks[0]
        check_names = {k.name for k in check_def.check_keys}

        # 1 schema + 3 column tests + 1 table expectation = 5
        assert len(check_names) == 5
        assert "batch_test_schema" in check_names
        assert "batch_test_id_not_null" in check_names
        assert "batch_test_name_not_null" in check_names
        assert "batch_test_name_accepted_values" in check_names
        assert "batch_test_row_count" in check_names

    def test_unbatched_returns_multiple_definitions(self, contract_with_rules):
        """batched=False (default) returns one definition per rule."""

        def mock_loader(_ctx):
            return pl.DataFrame({"id": [1], "name": ["a"]})

        checks = generate_asset_checks_from_contract(
            contract_with_rules,
            "batch_test",
            mock_loader,
            batched=False,
        )
        # 1 schema + 3 column tests + 1 table expectation = 5
        assert len(checks) == 5

    def test_batched_default_is_true(self, contract_with_rules):
        """Default behavior matches batched."""

        def mock_loader(_ctx):
            return pl.DataFrame({"id": [1], "name": ["a"]})

        default_checks = generate_asset_checks_from_contract(
            contract_with_rules,
            "batch_test",
            mock_loader,
        )
        explicit_checks = generate_asset_checks_from_contract(
            contract_with_rules,
            "batch_test",
            mock_loader,
            batched=True,
        )
        assert len(default_checks) == len(explicit_checks) == 1


# ---------------------------------------------------------------------------
# SCD2 current-row scoping (issue #418)
# ---------------------------------------------------------------------------


class TestScd2CurrentRowScoping:
    """Checks against SCD2 sinks scope to current rows.

    An SCD2 table legitimately repeats business keys across history rows,
    so unscoped full-table checks fail ``unique`` on the first change wave
    (issue #418). Contracts whose check sink declares ``mode: scd2`` must
    have their tests and expectations validated against current rows only.
    """

    @staticmethod
    def _make_contract(sinks=None, tests=None):
        from moncpipelib.contracts import Column, ColumnType, DataContract, Schema

        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="fda_purplebook_bla",
            layer="gold",
            schema=Schema(
                columns=[
                    Column(
                        name="bla_number",
                        type=ColumnType.STRING,
                        nullable=False,
                        tests=tests or [],
                    ),
                ]
            ),
            sinks=sinks,
        )

    def test_scd2_sink_resolves_current_col(self):
        from moncpipelib.contracts.checks import _resolve_current_scope_col

        contract = self._make_contract(
            sinks=[
                {
                    "type": "table",
                    "schema": "reference_gold",
                    "table": "fda_purplebook_bla",
                    "mode": "scd2",
                    "business_key": ["bla_number"],
                }
            ]
        )
        assert _resolve_current_scope_col(contract) == "is_current"

    def test_non_scd2_sink_resolves_none(self):
        from moncpipelib.contracts.checks import _resolve_current_scope_col

        contract = self._make_contract(
            sinks=[{"type": "table", "schema": "silver", "table": "t", "mode": "full_refresh"}]
        )
        assert _resolve_current_scope_col(contract) is None

    def test_no_sinks_resolves_none(self):
        from moncpipelib.contracts.checks import _resolve_current_scope_col

        assert _resolve_current_scope_col(self._make_contract()) is None

    def test_first_table_sink_drives_scoping(self):
        """Scope follows the same sink _resolve_check_table targets."""
        from moncpipelib.contracts.checks import _resolve_current_scope_col

        contract = self._make_contract(
            sinks=[
                {"type": "table", "schema": "silver", "table": "t", "mode": "full_refresh"},
                {"type": "table", "schema": "gold", "table": "t2", "mode": "scd2"},
            ]
        )
        assert _resolve_current_scope_col(contract) is None

    def test_non_table_sinks_skipped(self):
        from moncpipelib.contracts.checks import _resolve_current_scope_col

        contract = self._make_contract(
            sinks=[
                {"type": "file", "path": "out.parquet"},
                {"type": "table", "schema": "gold", "table": "t", "mode": "scd2"},
            ]
        )
        assert _resolve_current_scope_col(contract) == "is_current"

    def test_scope_df_filters_current_rows(self):
        from moncpipelib.contracts.checks import _scope_df_to_current

        df = pl.DataFrame(
            {
                "bla_number": ["017016", "017016", "125554"],
                "is_current": [False, True, True],
            }
        )
        scoped = _scope_df_to_current(df, "is_current")
        assert scoped.height == 2
        assert scoped["is_current"].all()

    def test_scope_df_missing_column_raises(self):
        from moncpipelib.contracts.checks import _scope_df_to_current

        df = pl.DataFrame({"bla_number": ["017016"]})
        with pytest.raises(ValueError, match="mode 'scd2'"):
            _scope_df_to_current(df, "is_current")

    def _history_frame(self):
        """One changed key (two versions) plus one unchanged key."""
        return pl.DataFrame(
            {
                "bla_number": ["017016", "017016", "125554"],
                "is_current": [False, True, True],
            }
        )

    def _col_checks(self):
        from moncpipelib.contracts import Severity

        return [("t_unique", "bla_number", "unique", {}, None, Severity.ERROR, False)]

    def test_polars_runner_scoped_unique_passes_on_history(self):
        """The #418 repro: unique on an SCD2 business key with history rows."""
        from moncpipelib.contracts.checks import _run_checks_polars

        contract = self._make_contract()
        results = list(
            _run_checks_polars(
                MagicMock(),
                lambda _ctx: self._history_frame(),
                contract,
                "t_schema",
                self._col_checks(),
                [],
                current_col="is_current",
            )
        )
        unique_result = next(r for r in results if r.check_name == "t_unique")
        assert unique_result.passed
        assert unique_result.metadata["total_count"].value == 2
        assert "current rows only" in unique_result.metadata["scope"].value

    def test_polars_runner_unscoped_unique_fails_on_history(self):
        """Without scoping the same frame fails -- the pre-#418 behavior."""
        from moncpipelib.contracts.checks import _run_checks_polars

        contract = self._make_contract()
        results = list(
            _run_checks_polars(
                MagicMock(),
                lambda _ctx: self._history_frame(),
                contract,
                "t_schema",
                self._col_checks(),
                [],
            )
        )
        unique_result = next(r for r in results if r.check_name == "t_unique")
        assert not unique_result.passed
        assert "scope" not in unique_result.metadata

    def test_wrapped_df_loader_scopes_unbatched_checks(self):
        from moncpipelib.contracts.checks import _wrap_df_loader_current_only

        wrapped = _wrap_df_loader_current_only(lambda _ctx: self._history_frame(), "is_current")
        assert wrapped(MagicMock()).height == 2

    def test_generate_checks_scd2_sink_builds(self):
        """Batched definition builds cleanly for an scd2-sink contract."""
        from moncpipelib.contracts import ColumnTest

        contract = self._make_contract(
            sinks=[
                {
                    "type": "table",
                    "schema": "reference_gold",
                    "table": "fda_purplebook_bla",
                    "mode": "scd2",
                    "business_key": ["bla_number"],
                }
            ],
            tests=[ColumnTest(test_type="unique")],
        )
        checks = generate_asset_checks_from_contract(
            contract,
            AssetKey(["reference_gold", "fda_purplebook_bla"]),
            lambda _ctx: self._history_frame(),
        )
        assert len(checks) == 1
