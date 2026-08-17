"""Tests for contract models."""

from typing import Any

import pytest

from moncpipelib.config import LineageDefaults
from moncpipelib.contracts.models import (
    Column,
    ColumnTest,
    ColumnType,
    ContractEnforcementMode,
    DataContract,
    LineageConfig,
    Owner,
    Schema,
    Severity,
    TableExpectation,
    TableReference,
    TestingConfig,
    ValidationResult,
)


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self):
        """Test severity enum has correct values."""
        assert Severity.ERROR.value == "error"
        assert Severity.WARN.value == "warn"

    def test_severity_from_string(self):
        """Test creating severity from string."""
        assert Severity("error") == Severity.ERROR
        assert Severity("warn") == Severity.WARN


class TestColumnType:
    """Tests for ColumnType enum."""

    def test_column_type_values(self):
        """Test all column types are defined."""
        types = [
            ("string", ColumnType.STRING),
            ("integer", ColumnType.INTEGER),
            ("decimal", ColumnType.DECIMAL),
            ("boolean", ColumnType.BOOLEAN),
            ("date", ColumnType.DATE),
            ("datetime", ColumnType.DATETIME),
            ("uuid", ColumnType.UUID),
        ]
        for value, expected in types:
            assert ColumnType(value) == expected


class TestContractEnforcementMode:
    """Tests for ContractEnforcementMode enum."""

    def test_enforcement_modes(self):
        """Test enforcement mode values."""
        assert ContractEnforcementMode.ERROR.value == "error"
        assert ContractEnforcementMode.WARN.value == "warn"
        assert ContractEnforcementMode.SILENT.value == "silent"


class TestColumnTest:
    """Tests for ColumnTest dataclass."""

    def test_simple_test(self):
        """Test creating a simple test without parameters."""
        test = ColumnTest(test_type="not_null")
        assert test.test_type == "not_null"
        assert test.parameters == {}
        assert test.severity == Severity.ERROR
        assert test.when is None

    def test_test_with_parameters(self):
        """Test creating a test with parameters."""
        test = ColumnTest(
            test_type="accepted_values",
            parameters={"values": ["a", "b", "c"]},
            severity=Severity.WARN,
        )
        assert test.test_type == "accepted_values"
        assert test.parameters == {"values": ["a", "b", "c"]}
        assert test.severity == Severity.WARN

    def test_test_with_when_condition(self):
        """Test creating a test with when condition."""
        test = ColumnTest(test_type="pattern", parameters={"value": "^foo$"}, when="not_null")
        assert test.when == "not_null"


class TestColumn:
    """Tests for Column dataclass."""

    def test_required_fields(self):
        """Test column with required fields only."""
        col = Column(name="id", type=ColumnType.INTEGER, nullable=False)
        assert col.name == "id"
        assert col.type == ColumnType.INTEGER
        assert col.nullable is False
        assert col.description is None
        assert col.primary_key is False
        assert col.managed is False
        assert col.pii is True  # Safe default
        assert col.tests == []

    def test_all_fields(self):
        """Test column with all fields."""
        tests = [ColumnTest(test_type="not_null")]
        col = Column(
            name="claim_id",
            type=ColumnType.STRING,
            nullable=False,
            description="Unique identifier",
            primary_key=True,
            managed=False,
            pii=False,
            tests=tests,
        )
        assert col.description == "Unique identifier"
        assert col.primary_key is True
        assert col.pii is False
        assert col.tests == tests

    def test_managed_column(self):
        """Test managed column flag."""
        col = Column(
            name=LineageDefaults.ID_COLUMN, type=ColumnType.UUID, nullable=False, managed=True
        )
        assert col.managed is True

    def test_pii_default_true(self):
        """Test column defaults to PII (safe by default)."""
        col = Column(name="patient_name", type=ColumnType.STRING, nullable=False)
        assert col.pii is True

    def test_pii_explicit_false(self):
        """Test column can be explicitly opted out of PII."""
        col = Column(name="event_type", type=ColumnType.STRING, nullable=False, pii=False)
        assert col.pii is False

    def test_phi_defaults_to_pii_true(self):
        """Test phi mirrors an unannotated (default-true) pii flag."""
        col = Column(name="patient_name", type=ColumnType.STRING, nullable=False)
        assert col.phi is True

    def test_phi_defaults_to_pii_false(self):
        """Test phi mirrors an explicit pii: false."""
        col = Column(name="event_type", type=ColumnType.STRING, nullable=False, pii=False)
        assert col.phi is False

    def test_phi_explicit_overrides_pii(self):
        """Test explicit phi diverges from pii in both directions."""
        provider = Column(
            name="provider_npi", type=ColumnType.STRING, nullable=False, pii=True, phi=False
        )
        assert provider.pii is True
        assert provider.phi is False

        clinical = Column(
            name="lab_value", type=ColumnType.STRING, nullable=False, pii=False, phi=True
        )
        assert clinical.pii is False
        assert clinical.phi is True


class TestTableExpectation:
    """Tests for TableExpectation dataclass."""

    def test_row_count_expectation(self):
        """Test row count expectation."""
        exp = TableExpectation(
            expectation_type="row_count",
            parameters={"min": 1, "max": 1000000},
            severity=Severity.ERROR,
        )
        assert exp.expectation_type == "row_count"
        assert exp.parameters["min"] == 1
        assert exp.parameters["max"] == 1000000

    def test_freshness_expectation(self):
        """Test freshness expectation."""
        exp = TableExpectation(
            expectation_type="freshness",
            parameters={"column": "claim_date", "max_age_hours": 48},
            severity=Severity.WARN,
        )
        assert exp.expectation_type == "freshness"
        assert exp.severity == Severity.WARN


class TestOwner:
    """Tests for Owner dataclass."""

    def test_owner_with_team_only(self):
        """Test owner with only required team field."""
        owner = Owner(team="data-engineering")
        assert owner.team == "data-engineering"
        assert owner.contact is None
        assert owner.slack_channel is None

    def test_owner_with_all_fields(self):
        """Test owner with all fields."""
        owner = Owner(
            team="data-engineering",
            contact="team@example.com",
            slack_channel="#data-alerts",
        )
        assert owner.contact == "team@example.com"
        assert owner.slack_channel == "#data-alerts"


class TestSchema:
    """Tests for Schema dataclass."""

    def test_schema_defaults(self):
        """Test schema default values."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        schema = Schema(columns=columns)
        assert schema.strict is True
        assert len(schema.columns) == 1

    def test_relaxed_schema(self):
        """Test relaxed (non-strict) schema mode."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        schema = Schema(columns=columns, strict=False)
        assert schema.strict is False


class TestDataContract:
    """Tests for DataContract dataclass."""

    @pytest.fixture
    def sample_contract(self):
        """Create a sample contract for testing."""
        columns = [
            Column(
                name="claim_id",
                type=ColumnType.STRING,
                nullable=False,
                primary_key=True,
            ),
            Column(
                name="patient_id",
                type=ColumnType.STRING,
                nullable=False,
                primary_key=True,
            ),
            Column(
                name="amount",
                type=ColumnType.DECIMAL,
                nullable=True,
            ),
            Column(
                name=LineageDefaults.ID_COLUMN,
                type=ColumnType.UUID,
                nullable=False,
                managed=True,
            ),
        ]
        schema = Schema(columns=columns, strict=True)
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="claims_bronze",
            layer="bronze",
            schema=schema,
            description="Test contract",
            owner=Owner(team="test-team"),
        )

    def test_get_primary_key_columns(self, sample_contract):
        """Test getting primary key column names."""
        pk_cols = sample_contract.get_primary_key_columns()
        assert pk_cols == ["claim_id", "patient_id"]

    def test_get_column(self, sample_contract):
        """Test getting a column by name."""
        col = sample_contract.get_column("amount")
        assert col is not None
        assert col.name == "amount"
        assert col.type == ColumnType.DECIMAL

    def test_get_column_not_found(self, sample_contract):
        """Test getting a non-existent column."""
        col = sample_contract.get_column("nonexistent")
        assert col is None

    def test_get_non_managed_columns(self, sample_contract):
        """Test getting non-managed columns."""
        non_managed = sample_contract.get_non_managed_columns()
        assert len(non_managed) == 3
        assert all(not c.managed for c in non_managed)
        assert LineageDefaults.ID_COLUMN not in [c.name for c in non_managed]

    def test_get_pii_columns(self):
        """Test getting PII columns (default + explicit)."""
        columns = [
            Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
            Column(name="event_type", type=ColumnType.STRING, nullable=False, pii=False),
            Column(name="ssn", type=ColumnType.STRING, nullable=True),  # default pii=True
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns),
        )
        pii_cols = contract.get_pii_columns()
        assert len(pii_cols) == 2
        assert pii_cols[0].name == "patient_id"
        assert pii_cols[1].name == "ssn"

    def test_get_pii_column_names(self):
        """Test getting PII column names."""
        columns = [
            Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
            Column(name="event_type", type=ColumnType.STRING, nullable=False, pii=False),
            Column(name="amount", type=ColumnType.DECIMAL, nullable=True),  # default pii=True
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns),
        )
        assert contract.get_pii_column_names() == ["patient_id", "amount"]

    def test_get_non_pii_column_names(self):
        """Test getting non-PII column names."""
        columns = [
            Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
            Column(name="event_type", type=ColumnType.STRING, nullable=False, pii=False),
            Column(name="status", type=ColumnType.STRING, nullable=False, pii=False),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns),
        )
        assert contract.get_non_pii_column_names() == ["event_type", "status"]

    def test_all_columns_pii_by_default(self):
        """Test that unannotated columns are all treated as PII."""
        columns = [
            Column(name="col_a", type=ColumnType.STRING, nullable=False),
            Column(name="col_b", type=ColumnType.INTEGER, nullable=True),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns),
        )
        assert contract.get_pii_column_names() == ["col_a", "col_b"]
        assert contract.get_non_pii_column_names() == []

    def test_get_phi_column_names_mirrors_pii_when_unset(self):
        """Test phi helpers follow pii when phi is never annotated."""
        columns = [
            Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
            Column(name="event_type", type=ColumnType.STRING, nullable=False, pii=False),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns),
        )
        assert contract.get_phi_column_names() == ["patient_id"]
        assert contract.get_non_phi_column_names() == ["event_type"]

    def test_get_phi_column_names_with_explicit_phi(self):
        """Test explicit phi annotations override the pii mirror."""
        columns = [
            Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
            Column(
                name="provider_npi", type=ColumnType.STRING, nullable=False, pii=True, phi=False
            ),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns),
        )
        assert contract.get_pii_column_names() == ["patient_id", "provider_npi"]
        assert contract.get_phi_column_names() == ["patient_id"]
        assert contract.get_non_phi_column_names() == ["provider_npi"]
        phi_cols = contract.get_phi_columns()
        assert len(phi_cols) == 1
        assert phi_cols[0].name == "patient_id"


class TestGetParameter:
    """Tests for DataContract.get_parameter()."""

    def _make_contract(self, parameters: dict[str, Any] | None = None) -> DataContract:
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
            parameters=parameters or {},
        )

    def test_get_existing_parameter(self):
        contract = self._make_contract({"days_of_tolerance": 30})
        assert contract.get_parameter("days_of_tolerance") == 30

    def test_get_parameter_preserves_types(self):
        params = {
            "threshold": 0.95,
            "enabled": False,
            "tags": ["a", "b"],
            "nested": {"key": "value"},
        }
        contract = self._make_contract(params)
        assert contract.get_parameter("threshold") == 0.95
        assert contract.get_parameter("enabled") is False
        assert contract.get_parameter("tags") == ["a", "b"]
        assert contract.get_parameter("nested") == {"key": "value"}

    def test_missing_key_no_default_with_parameters_raises(self):
        contract = self._make_contract({"days_of_tolerance": 30, "max_retries": 3})
        with pytest.raises(
            KeyError, match="not found.*Available parameters.*days_of_tolerance.*max_retries.*typo"
        ):
            contract.get_parameter("days_of_tolerence")  # typo

    def test_missing_key_no_default_no_parameters_raises(self):
        contract = self._make_contract()
        with pytest.raises(
            KeyError, match="no parameters are defined.*Add a 'parameters:' section"
        ):
            contract.get_parameter("days_of_tolerance")

    def test_missing_key_with_default_returns_default(self):
        contract = self._make_contract({"other": 5})
        result = contract.get_parameter("days_of_tolerance", 30)
        assert result == 30

    def test_missing_key_with_default_warns_when_parameters_exist(self, caplog):
        contract = self._make_contract({"other": 5})
        with caplog.at_level("WARNING"):
            result = contract.get_parameter("days_of_tolerance", 30)
        assert result == 30
        assert "not found" in caplog.text
        assert "typo" in caplog.text.lower()
        assert "30" in caplog.text

    def test_missing_key_with_default_warns_when_no_parameters(self, caplog):
        contract = self._make_contract()
        with caplog.at_level("WARNING"):
            result = contract.get_parameter("days_of_tolerance", 30)
        assert result == 30
        assert "no parameters are defined" in caplog.text
        assert "30" in caplog.text

    def test_default_none_is_valid(self):
        contract = self._make_contract()
        result = contract.get_parameter("optional_key", None)
        assert result is None

    def test_existing_key_does_not_warn(self, caplog):
        contract = self._make_contract({"days_of_tolerance": 30})
        with caplog.at_level("WARNING"):
            contract.get_parameter("days_of_tolerance")
        assert caplog.text == ""


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_passed_result(self):
        """Test a passed validation result."""
        result = ValidationResult(
            passed=True,
            message="All values are not null",
            failed_count=0,
            total_count=100,
        )
        assert result.passed is True
        assert result.failed_count == 0

    def test_failed_result_with_samples(self):
        """Test a failed validation result with sample failures."""
        result = ValidationResult(
            passed=False,
            message="Found 5 null values",
            failed_count=5,
            total_count=100,
            sample_failures=[{"id": 1, "value": None}, {"id": 2, "value": None}],
        )
        assert result.passed is False
        assert result.failed_count == 5
        assert len(result.sample_failures) == 2


class TestTableReference:
    """Tests for TableReference dataclass."""

    def test_fully_qualified_name(self):
        """Test fully_qualified_name property."""
        ref = TableReference(database="analytics", schema="bronze", table="orders")
        assert ref.fully_qualified_name == "analytics.bronze.orders"

    def test_schema_qualified_name(self):
        """Test schema_qualified_name property."""
        ref = TableReference(database="analytics", schema="bronze", table="orders")
        assert ref.schema_qualified_name == "bronze.orders"

    def test_field_values(self):
        """Test individual field access."""
        ref = TableReference(database="mydb", schema="raw", table="events")
        assert ref.database == "mydb"
        assert ref.schema == "raw"
        assert ref.table == "events"


class TestTestingConfig:
    """Tests for TestingConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = TestingConfig()
        assert config.enabled is True
        assert config.source_row_limit == 1000
        assert config.source_where_clause is None
        assert config.expected_min_rows is None
        assert config.expected_max_rows is None
        assert config.timeout_seconds == 300

    def test_custom_values(self):
        """Test custom values."""
        config = TestingConfig(
            enabled=False,
            source_row_limit=500,
            source_where_clause="date > '2024-01-01'",
            expected_min_rows=1,
            expected_max_rows=1000,
            timeout_seconds=600,
        )
        assert config.enabled is False
        assert config.source_row_limit == 500
        assert config.source_where_clause == "date > '2024-01-01'"
        assert config.expected_min_rows == 1
        assert config.expected_max_rows == 1000
        assert config.timeout_seconds == 600


class TestLineageConfig:
    """Tests for LineageConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = LineageConfig()
        assert config.enabled is True
        assert config.source_system is None
        assert config.transformation_type is None

    def test_custom_values(self):
        """Test custom values."""
        config = LineageConfig(
            enabled=False,
            source_system="openfda",
            transformation_type="ingest",
        )
        assert config.enabled is False
        assert config.source_system == "openfda"
        assert config.transformation_type == "ingest"

    def test_disabled_only(self):
        """Test setting only enabled=False leaves other fields as None."""
        config = LineageConfig(enabled=False)
        assert config.enabled is False
        assert config.source_system is None
        assert config.transformation_type is None


class TestDataContractSources:
    """Tests for DataContract source/sink/testing extensions."""

    @pytest.fixture
    def contract_with_sources(self):
        """Contract with sources, sinks, and testing config."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[
                {"type": "table", "schema": "bronze", "table": "orders"},
                {"type": "table", "database": "mydb", "schema": "raw", "table": "events"},
                {"type": "external", "system": "sftp"},  # Not a table
            ],
            sinks=[
                {"type": "table", "schema": "silver", "table": "orders"},
            ],
        )

    def test_get_source_tables(self, contract_with_sources):
        """Test extracting source table references."""
        refs = contract_with_sources.get_source_tables()
        assert len(refs) == 2
        assert refs[0].database == "analytics"  # Default
        assert refs[0].schema == "bronze"
        assert refs[0].table == "orders"
        assert refs[1].database == "mydb"
        assert refs[1].schema == "raw"

    def test_get_sink_tables(self, contract_with_sources):
        """Test extracting sink table references."""
        refs = contract_with_sources.get_sink_tables()
        assert len(refs) == 1
        assert refs[0].schema == "silver"
        assert refs[0].table == "orders"

    def test_get_source_tables_empty(self):
        """Test with no sources."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
        )
        assert contract.get_source_tables() == []

    def test_get_sink_tables_empty(self):
        """Test with no sinks."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
        )
        assert contract.get_sink_tables() == []

    def test_non_table_sources_excluded(self, contract_with_sources):
        """Test that non-table sources are excluded."""
        refs = contract_with_sources.get_source_tables()
        # The "external" source should not be included
        assert all(isinstance(ref, TableReference) for ref in refs)
        assert len(refs) == 2  # Only the 2 table sources

    def test_default_sources_and_sinks(self):
        """Test that sources and sinks default to empty lists."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
        )
        assert contract.sources == []
        assert contract.sinks == []
        assert contract.testing is None


class TestDataContractValidateForIntegrationTesting:
    """Tests for validate_for_integration_testing method."""

    def test_valid_contract(self):
        """Test a valid contract passes validation."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[{"type": "table", "schema": "bronze", "table": "raw_data"}],
            sinks=[{"type": "table", "schema": "silver", "table": "clean_data"}],
        )
        errors = contract.validate_for_integration_testing()
        assert errors == []

    def test_no_sources_error(self):
        """Test error when no table sources."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sinks=[{"type": "table", "schema": "silver", "table": "x"}],
        )
        errors = contract.validate_for_integration_testing()
        assert any("source" in e.lower() for e in errors)

    def test_no_sinks_error(self):
        """Test error when no table sinks."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[{"type": "table", "schema": "bronze", "table": "x"}],
        )
        errors = contract.validate_for_integration_testing()
        assert any("sink" in e.lower() for e in errors)

    def test_source_missing_schema(self):
        """Test error when source missing schema field."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[{"type": "table", "table": "x"}],
            sinks=[{"type": "table", "schema": "silver", "table": "y"}],
        )
        errors = contract.validate_for_integration_testing()
        assert any("schema" in e.lower() for e in errors)

    def test_source_missing_table(self):
        """Test error when source missing table field."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[{"type": "table", "schema": "bronze"}],
            sinks=[{"type": "table", "schema": "silver", "table": "y"}],
        )
        errors = contract.validate_for_integration_testing()
        assert any("table" in e.lower() for e in errors)

    def test_invalid_testing_config_min_exceeds_max(self):
        """Test error when expected_min_rows > expected_max_rows."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[{"type": "table", "schema": "bronze", "table": "x"}],
            sinks=[{"type": "table", "schema": "silver", "table": "y"}],
            testing=TestingConfig(
                expected_min_rows=1000,
                expected_max_rows=10,
            ),
        )
        errors = contract.validate_for_integration_testing()
        assert any("expected_min_rows" in e for e in errors)

    def test_invalid_testing_config_negative_row_limit(self):
        """Test error when source_row_limit is not positive."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[{"type": "table", "schema": "bronze", "table": "x"}],
            sinks=[{"type": "table", "schema": "silver", "table": "y"}],
            testing=TestingConfig(source_row_limit=0),
        )
        errors = contract.validate_for_integration_testing()
        assert any("source_row_limit" in e for e in errors)

    def test_invalid_testing_config_negative_timeout(self):
        """Test error when timeout_seconds is not positive."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[{"type": "table", "schema": "bronze", "table": "x"}],
            sinks=[{"type": "table", "schema": "silver", "table": "y"}],
            testing=TestingConfig(timeout_seconds=-1),
        )
        errors = contract.validate_for_integration_testing()
        assert any("timeout_seconds" in e for e in errors)

    def test_valid_testing_config(self):
        """Test valid testing config passes."""
        columns = [Column(name="id", type=ColumnType.INTEGER, nullable=False)]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="silver",
            schema=Schema(columns=columns),
            sources=[{"type": "table", "schema": "bronze", "table": "x"}],
            sinks=[{"type": "table", "schema": "silver", "table": "y"}],
            testing=TestingConfig(
                source_row_limit=500,
                expected_min_rows=1,
                expected_max_rows=1000,
            ),
        )
        errors = contract.validate_for_integration_testing()
        assert errors == []
