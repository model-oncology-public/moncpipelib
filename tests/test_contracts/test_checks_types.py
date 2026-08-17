"""Tests for typed check classes."""

import polars as pl
import pytest

from moncpipelib.contracts import (
    # Column tests
    AcceptedValues,
    Between,
    Column,
    ColumnType,
    DataContract,
    # Table expectations
    Freshness,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    MaxLength,
    MinLength,
    NotIn,
    NotInFuture,
    NotNull,
    NullPercentage,
    Pattern,
    RowCount,
    Schema,
    Severity,
    Unique,
    UniqueCombination,
    WithinDays,
    # Validators
    run_column_test,
    run_table_expectation,
)


class TestColumnTestTypes:
    """Tests for typed column test classes."""

    def test_not_null_type_and_parameters(self):
        """Test NotNull has correct type and empty parameters."""
        test = NotNull()
        assert test.test_type == "not_null"
        assert test.parameters == {}
        assert test.severity == Severity.ERROR

    def test_not_null_with_severity(self):
        """Test NotNull with custom severity."""
        test = NotNull(severity=Severity.WARN)
        assert test.severity == Severity.WARN

    def test_not_null_with_when(self):
        """Test NotNull with conditional 'when' clause."""
        test = NotNull(when="not_null")
        assert test.when == "not_null"

    def test_unique_type_and_parameters(self):
        """Test Unique has correct type and empty parameters."""
        test = Unique()
        assert test.test_type == "unique"
        assert test.parameters == {}

    def test_accepted_values_parameters(self):
        """Test AcceptedValues has correct parameters."""
        test = AcceptedValues(values=["a", "b", "c"])
        assert test.test_type == "accepted_values"
        assert test.parameters == {"values": ["a", "b", "c"]}

    def test_accepted_values_with_numbers(self):
        """Test AcceptedValues with numeric values."""
        test = AcceptedValues(values=[1, 2, 3])
        assert test.parameters == {"values": [1, 2, 3]}

    def test_not_in_parameters(self):
        """Test NotIn has correct parameters."""
        test = NotIn(values=[-1, -999])
        assert test.test_type == "not_in"
        assert test.parameters == {"values": [-1, -999]}

    def test_not_in_with_strings(self):
        """Test NotIn with string values."""
        test = NotIn(values=["UNKNOWN", "N/A"])
        assert test.parameters == {"values": ["UNKNOWN", "N/A"]}

    def test_pattern_parameters(self):
        """Test Pattern has correct parameters."""
        test = Pattern(regex=r"^[A-Z]{2}-\d+$")
        assert test.test_type == "pattern"
        assert test.parameters == {"regex": r"^[A-Z]{2}-\d+$"}

    def test_greater_than_parameters(self):
        """Test GreaterThan has correct parameters."""
        test = GreaterThan(threshold=0)
        assert test.test_type == "greater_than"
        assert test.parameters == {"threshold": 0}

    def test_greater_than_or_equal_parameters(self):
        """Test GreaterThanOrEqual has correct parameters."""
        test = GreaterThanOrEqual(threshold=10)
        assert test.test_type == "greater_than_or_equal"
        assert test.parameters == {"threshold": 10}

    def test_less_than_parameters(self):
        """Test LessThan has correct parameters."""
        test = LessThan(threshold=100)
        assert test.test_type == "less_than"
        assert test.parameters == {"threshold": 100}

    def test_less_than_or_equal_parameters(self):
        """Test LessThanOrEqual has correct parameters."""
        test = LessThanOrEqual(threshold=5)
        assert test.test_type == "less_than_or_equal"
        assert test.parameters == {"threshold": 5}

    def test_between_parameters(self):
        """Test Between has correct parameters."""
        test = Between(min=0, max=100)
        assert test.test_type == "between"
        assert test.parameters == {"min": 0, "max": 100}

    def test_min_length_parameters(self):
        """Test MinLength has correct parameters."""
        test = MinLength(length=3)
        assert test.test_type == "min_length"
        assert test.parameters == {"length": 3}

    def test_max_length_parameters(self):
        """Test MaxLength has correct parameters."""
        test = MaxLength(length=50)
        assert test.test_type == "max_length"
        assert test.parameters == {"length": 50}

    def test_not_in_future_parameters(self):
        """Test NotInFuture has correct type and empty parameters."""
        test = NotInFuture()
        assert test.test_type == "not_in_future"
        assert test.parameters == {}

    def test_within_days_parameters(self):
        """Test WithinDays has correct parameters."""
        test = WithinDays(days=30)
        assert test.test_type == "within_days"
        assert test.parameters == {"days": 30}


class TestTableExpectationTypes:
    """Tests for typed table expectation classes."""

    def test_row_count_parameters(self):
        """Test RowCount has correct parameters."""
        test = RowCount(min=1, max=1000)
        assert test.expectation_type == "row_count"
        assert test.parameters == {"min": 1, "max": 1000}

    def test_row_count_min_only(self):
        """Test RowCount with only min."""
        test = RowCount(min=1)
        assert test.parameters == {"min": 1}

    def test_row_count_max_only(self):
        """Test RowCount with only max."""
        test = RowCount(max=1000)
        assert test.parameters == {"max": 1000}

    def test_freshness_parameters(self):
        """Test Freshness has correct parameters."""
        test = Freshness(column="updated_at", max_age_hours=24)
        assert test.expectation_type == "freshness"
        assert test.parameters == {"column": "updated_at", "max_age_hours": 24}

    def test_null_percentage_parameters(self):
        """Test NullPercentage has correct parameters."""
        test = NullPercentage(column="email", max_percent=5.0)
        assert test.expectation_type == "null_percentage"
        assert test.parameters == {"column": "email", "max_percent": 5.0}

    def test_unique_combination_parameters(self):
        """Test UniqueCombination has correct parameters."""
        test = UniqueCombination(columns=["order_id", "line_item"])
        assert test.expectation_type == "unique_combination"
        assert test.parameters == {"columns": ["order_id", "line_item"]}


class TestTypedChecksWithValidators:
    """Tests that typed checks work with the validator functions."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "status": ["active", "inactive", "active"],
                "amount": [100.0, 50.0, 75.0],
            }
        )

    def test_not_null_works_with_validator(self, sample_df):
        """Test NotNull works with run_column_test."""
        test = NotNull()
        result = run_column_test(sample_df, "id", test.test_type, test.parameters)
        assert result.passed

    def test_unique_works_with_validator(self, sample_df):
        """Test Unique works with run_column_test."""
        test = Unique()
        result = run_column_test(sample_df, "id", test.test_type, test.parameters)
        assert result.passed

    def test_accepted_values_works_with_validator(self, sample_df):
        """Test AcceptedValues works with run_column_test."""
        test = AcceptedValues(values=["active", "inactive", "pending"])
        result = run_column_test(sample_df, "status", test.test_type, test.parameters)
        assert result.passed

    def test_greater_than_works_with_validator(self, sample_df):
        """Test GreaterThan works with run_column_test."""
        test = GreaterThan(threshold=0)
        result = run_column_test(sample_df, "amount", test.test_type, test.parameters)
        assert result.passed

    def test_row_count_works_with_validator(self, sample_df):
        """Test RowCount works with run_table_expectation."""
        test = RowCount(min=1, max=100)
        result = run_table_expectation(sample_df, test.expectation_type, test.parameters)
        assert result.passed

    def test_unique_combination_works_with_validator(self, sample_df):
        """Test UniqueCombination works with run_table_expectation."""
        test = UniqueCombination(columns=["id", "name"])
        result = run_table_expectation(sample_df, test.expectation_type, test.parameters)
        assert result.passed


class TestTypedChecksInContract:
    """Tests that typed checks work in DataContract definitions."""

    def test_contract_with_typed_column_tests(self):
        """Test creating a DataContract with typed column tests."""
        columns = [
            Column(
                name="id",
                type=ColumnType.INTEGER,
                nullable=False,
                tests=[NotNull(), Unique()],
            ),
            Column(
                name="status",
                type=ColumnType.STRING,
                nullable=False,
                tests=[AcceptedValues(values=["active", "inactive"])],
            ),
            Column(
                name="amount",
                type=ColumnType.DECIMAL,
                nullable=False,
                tests=[
                    GreaterThan(threshold=0),
                    LessThan(threshold=10000),
                ],
            ),
        ]

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
        )

        # Verify tests are properly attached
        id_col = contract.get_column("id")
        assert id_col is not None
        assert len(id_col.tests) == 2
        assert id_col.tests[0].test_type == "not_null"
        assert id_col.tests[1].test_type == "unique"

        status_col = contract.get_column("status")
        assert status_col is not None
        assert len(status_col.tests) == 1
        assert status_col.tests[0].test_type == "accepted_values"
        assert status_col.tests[0].parameters == {"values": ["active", "inactive"]}

        amount_col = contract.get_column("amount")
        assert amount_col is not None
        assert len(amount_col.tests) == 2

    def test_contract_with_typed_expectations(self):
        """Test creating a DataContract with typed table expectations."""
        columns = [
            Column(name="id", type=ColumnType.INTEGER, nullable=False),
        ]

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns),
            expectations=[
                RowCount(min=1, max=1000000),
                UniqueCombination(columns=["id"]),
            ],
        )

        assert len(contract.expectations) == 2
        assert contract.expectations[0].expectation_type == "row_count"
        assert contract.expectations[0].parameters == {"min": 1, "max": 1000000}
        assert contract.expectations[1].expectation_type == "unique_combination"
        assert contract.expectations[1].parameters == {"columns": ["id"]}

    def test_full_contract_example(self):
        """Test a complete contract with all typed check features."""
        columns = [
            Column(
                name="order_id",
                type=ColumnType.STRING,
                nullable=False,
                description="Unique order identifier",
                primary_key=True,
                tests=[
                    NotNull(),
                    Unique(),
                    Pattern(regex=r"^ORD-\d{6}$"),
                ],
            ),
            Column(
                name="customer_id",
                type=ColumnType.STRING,
                nullable=False,
                tests=[NotNull(), MinLength(length=5), MaxLength(length=20)],
            ),
            Column(
                name="amount",
                type=ColumnType.DECIMAL,
                nullable=False,
                tests=[
                    NotNull(),
                    Between(min=0.01, max=999999.99),
                ],
            ),
            Column(
                name="status",
                type=ColumnType.STRING,
                nullable=False,
                tests=[
                    AcceptedValues(
                        values=["pending", "confirmed", "shipped", "delivered", "cancelled"],
                        severity=Severity.ERROR,
                    ),
                ],
            ),
            Column(
                name="priority",
                type=ColumnType.INTEGER,
                nullable=True,
                tests=[
                    GreaterThanOrEqual(threshold=1),
                    LessThanOrEqual(threshold=5),
                ],
            ),
        ]

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="orders_bronze",
            layer="bronze",
            description="Raw orders data",
            schema=Schema(columns=columns),
            expectations=[
                RowCount(min=1),
                UniqueCombination(columns=["order_id"]),
                NullPercentage(column="priority", max_percent=50.0),
            ],
        )

        # Verify structure
        assert contract.asset == "orders_bronze"
        assert len(contract.schema.columns) == 5
        assert len(contract.expectations) == 3
        assert contract.get_primary_key_columns() == ["order_id"]

        # Verify test types are correct Literal types
        order_id_col = contract.get_column("order_id")
        assert order_id_col is not None
        assert order_id_col.tests[2].test_type == "pattern"
        assert order_id_col.tests[2].parameters["regex"] == r"^ORD-\d{6}$"
