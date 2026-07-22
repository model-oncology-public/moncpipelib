"""Tests for contract validators."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import polars as pl
import pytest

from moncpipelib.config import LineageDefaults
from moncpipelib.contracts import (
    Column,
    ColumnType,
    DataContract,
    Schema,
    run_column_test,
    run_post_write_expectations,
    run_table_expectation,
    validate_accepted_values,
    validate_between,
    validate_freshness,
    validate_greater_than,
    validate_greater_than_or_equal,
    validate_less_than,
    validate_less_than_or_equal,
    validate_max_length,
    validate_min_length,
    validate_not_in,
    validate_not_in_future,
    validate_not_null,
    validate_null_percentage,
    validate_pattern,
    validate_row_count,
    validate_row_count_raw,
    validate_schema,
    validate_unique,
    validate_unique_combination,
    validate_within_days,
)
from moncpipelib.contracts.models import TableExpectation


class TestValidateSchema:
    """Tests for schema validation."""

    @pytest.fixture
    def sample_contract(self):
        """Create a sample contract for testing."""
        columns = [
            Column(name="id", type=ColumnType.INTEGER, nullable=False),
            Column(name="name", type=ColumnType.STRING, nullable=False),
            Column(name="amount", type=ColumnType.DECIMAL, nullable=True),
            Column(
                name=LineageDefaults.ID_COLUMN, type=ColumnType.UUID, nullable=False, managed=True
            ),
        ]
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=columns, strict=False),
        )

    def test_valid_schema(self, sample_contract):
        """Test validation passes for matching schema."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
                "amount": [1.5, 2.5, None],
            }
        )
        result = validate_schema(df, sample_contract)
        assert result.passed is True

    def test_missing_column(self, sample_contract):
        """Test validation fails for missing column."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                # "name" is missing
                "amount": [1.5, 2.5, 3.5],
            }
        )
        result = validate_schema(df, sample_contract)
        assert result.passed is False
        assert "name" in result.message

    def test_type_mismatch(self, sample_contract):
        """Test validation fails for type mismatch."""
        df = pl.DataFrame(
            {
                "id": ["a", "b", "c"],  # Should be integer
                "name": ["x", "y", "z"],
                "amount": [1.5, 2.5, 3.5],
            }
        )
        result = validate_schema(df, sample_contract)
        assert result.passed is False
        assert "id" in result.message

    def test_strict_mode_extra_columns(self):
        """Test strict mode fails on unexpected columns."""
        columns = [
            Column(name="id", type=ColumnType.INTEGER, nullable=False),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns, strict=True),
        )
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "extra_col": ["a", "b", "c"],  # Unexpected
            }
        )
        result = validate_schema(df, contract)
        assert result.passed is False
        assert "extra_col" in result.message

    def test_non_strict_allows_extra_columns(self, sample_contract):
        """Test non-strict mode allows extra columns."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
                "amount": [1.5, 2.5, 3.5],
                "extra_col": ["x", "y", "z"],
            }
        )
        result = validate_schema(df, sample_contract)
        assert result.passed is True

    @pytest.mark.parametrize("col_type", [ColumnType.JSON, ColumnType.JSONB])
    def test_json_types_accept_string_column(self, col_type: ColumnType) -> None:
        """JSON and JSONB contract types accept String DataFrame columns."""
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=[Column(name="payload", type=col_type, nullable=True)]),
        )
        df = pl.DataFrame({"payload": ['{"key": "value"}', None, "[]"]})
        result = validate_schema(df, contract)
        assert result.passed is True

    def test_jsonb_type_rejects_non_string(self) -> None:
        """JSONB contract type rejects non-String DataFrame columns."""
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=[Column(name="payload", type=ColumnType.JSONB, nullable=False)]),
        )
        df = pl.DataFrame({"payload": [1, 2, 3]})
        result = validate_schema(df, contract)
        assert result.passed is False
        assert "payload" in result.message


class TestValidateNotNull:
    """Tests for not_null validation."""

    def test_no_nulls(self):
        """Test passes when no nulls present."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = validate_not_null(df, "col")
        assert result.passed is True
        assert result.failed_count == 0

    def test_has_nulls(self):
        """Test fails when nulls present."""
        df = pl.DataFrame({"col": [1, None, 3, None]})
        result = validate_not_null(df, "col")
        assert result.passed is False
        assert result.failed_count == 2

    def test_column_not_found(self):
        """Test fails for non-existent column."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = validate_not_null(df, "missing")
        assert result.passed is False
        assert "not found" in result.message


class TestValidateUnique:
    """Tests for unique validation."""

    def test_all_unique(self):
        """Test passes when all values are unique."""
        df = pl.DataFrame({"col": [1, 2, 3, 4, 5]})
        result = validate_unique(df, "col")
        assert result.passed is True

    def test_has_duplicates(self):
        """Test fails when duplicates present."""
        df = pl.DataFrame({"col": [1, 2, 2, 3, 3, 3]})
        result = validate_unique(df, "col")
        assert result.passed is False
        assert result.failed_count == 3  # 3 duplicate rows

    def test_column_not_found(self):
        """Test fails for non-existent column."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = validate_unique(df, "missing")
        assert result.passed is False


class TestValidateAcceptedValues:
    """Tests for accepted_values validation."""

    def test_all_accepted(self):
        """Test passes when all values are accepted."""
        df = pl.DataFrame({"status": ["pending", "approved", "denied"]})
        result = validate_accepted_values(df, "status", ["pending", "approved", "denied"])
        assert result.passed is True

    def test_invalid_values(self):
        """Test fails when invalid values present."""
        df = pl.DataFrame({"status": ["pending", "invalid", "unknown"]})
        result = validate_accepted_values(df, "status", ["pending", "approved", "denied"])
        assert result.passed is False
        assert result.failed_count == 2

    def test_null_values_ignored(self):
        """Test null values are ignored."""
        df = pl.DataFrame({"status": ["pending", None, "approved"]})
        result = validate_accepted_values(df, "status", ["pending", "approved"])
        assert result.passed is True


class TestValidateNotIn:
    """Tests for not_in (rejected values) validation."""

    def test_no_rejected_values(self):
        """Test passes when no rejected values present."""
        df = pl.DataFrame({"score": [1, 2, 3, 4, 5]})
        result = validate_not_in(df, "score", [-1, -999])
        assert result.passed is True

    def test_has_rejected_values(self):
        """Test fails when rejected values present."""
        df = pl.DataFrame({"score": [1, -1, 3, -999, 5]})
        result = validate_not_in(df, "score", [-1, -999])
        assert result.passed is False
        assert result.failed_count == 2
        assert "rejected" in result.message

    def test_nulls_ignored(self):
        """Test null values are not treated as matches."""
        df = pl.DataFrame({"score": [1, None, 3]})
        result = validate_not_in(df, "score", [-1])
        assert result.passed is True

    def test_empty_rejected_list(self):
        """Test passes when rejected list is empty."""
        df = pl.DataFrame({"score": [1, 2, 3]})
        result = validate_not_in(df, "score", [])
        assert result.passed is True

    def test_column_not_found(self):
        """Test fails gracefully for missing column."""
        df = pl.DataFrame({"other": [1, 2]})
        result = validate_not_in(df, "score", [-1])
        assert result.passed is False
        assert "not found" in result.message

    def test_string_values(self):
        """Test works with string rejected values."""
        df = pl.DataFrame({"status": ["active", "UNKNOWN", "pending", "N/A"]})
        result = validate_not_in(df, "status", ["UNKNOWN", "N/A"])
        assert result.passed is False
        assert result.failed_count == 2

    def test_dispatcher_integration(self):
        """Test not_in works through run_column_test dispatcher."""
        df = pl.DataFrame({"score": [1, -1, 3]})
        result = run_column_test(df, "score", "not_in", {"values": [-1]})
        assert result.passed is False
        assert result.failed_count == 1


class TestValidatePattern:
    """Tests for pattern validation."""

    def test_all_match(self):
        """Test passes when all values match pattern."""
        df = pl.DataFrame({"code": ["PAT-12345678", "PAT-87654321"]})
        result = validate_pattern(df, "code", "^PAT-[0-9]{8}$")
        assert result.passed is True

    def test_some_dont_match(self):
        """Test fails when some values don't match."""
        df = pl.DataFrame({"code": ["PAT-12345678", "INVALID", "PAT-123"]})
        result = validate_pattern(df, "code", "^PAT-[0-9]{8}$")
        assert result.passed is False
        assert result.failed_count == 2


class TestValidateGreaterThan:
    """Tests for greater_than validation."""

    def test_all_greater(self):
        """Test passes when all values are greater."""
        df = pl.DataFrame({"amount": [1, 2, 3, 4, 5]})
        result = validate_greater_than(df, "amount", 0)
        assert result.passed is True

    def test_some_not_greater(self):
        """Test fails when some values are not greater."""
        df = pl.DataFrame({"amount": [-1, 0, 1, 2]})
        result = validate_greater_than(df, "amount", 0)
        assert result.passed is False
        assert result.failed_count == 2  # -1 and 0

    def test_null_values_ignored(self):
        """Test null values are ignored."""
        df = pl.DataFrame({"amount": [1, None, 2]})
        result = validate_greater_than(df, "amount", 0)
        assert result.passed is True


class TestValidateGreaterThanOrEqual:
    """Tests for greater_than_or_equal validation."""

    def test_all_greater_or_equal(self):
        """Test passes when all values are >= threshold."""
        df = pl.DataFrame({"amount": [0, 1, 2, 3]})
        result = validate_greater_than_or_equal(df, "amount", 0)
        assert result.passed is True

    def test_some_less(self):
        """Test fails when some values are less than threshold."""
        df = pl.DataFrame({"amount": [-1, 0, 1]})
        result = validate_greater_than_or_equal(df, "amount", 0)
        assert result.passed is False
        assert result.failed_count == 1


class TestValidateLessThan:
    """Tests for less_than validation."""

    def test_all_less(self):
        """Test passes when all values are less."""
        df = pl.DataFrame({"amount": [1, 2, 3]})
        result = validate_less_than(df, "amount", 10)
        assert result.passed is True

    def test_some_not_less(self):
        """Test fails when some values are not less."""
        df = pl.DataFrame({"amount": [5, 10, 15]})
        result = validate_less_than(df, "amount", 10)
        assert result.passed is False
        assert result.failed_count == 2  # 10 and 15


class TestValidateLessThanOrEqual:
    """Tests for less_than_or_equal validation."""

    def test_all_less_or_equal(self):
        """Test passes when all values are <= threshold."""
        df = pl.DataFrame({"amount": [5, 10]})
        result = validate_less_than_or_equal(df, "amount", 10)
        assert result.passed is True

    def test_some_greater(self):
        """Test fails when some values are greater."""
        df = pl.DataFrame({"amount": [5, 10, 15]})
        result = validate_less_than_or_equal(df, "amount", 10)
        assert result.passed is False
        assert result.failed_count == 1


class TestValidateBetween:
    """Tests for between validation."""

    def test_all_in_range(self):
        """Test passes when all values in range."""
        df = pl.DataFrame({"amount": [1, 5, 10]})
        result = validate_between(df, "amount", 1, 10)
        assert result.passed is True

    def test_some_outside_range(self):
        """Test fails when some values outside range."""
        df = pl.DataFrame({"amount": [0, 5, 15]})
        result = validate_between(df, "amount", 1, 10)
        assert result.passed is False
        assert result.failed_count == 2


class TestValidateNotInFuture:
    """Tests for not_in_future validation."""

    def test_all_past_dates(self):
        """Test passes when all dates are in the past."""
        yesterday = date.today() - timedelta(days=1)
        df = pl.DataFrame({"dt": [yesterday, yesterday]})
        result = validate_not_in_future(df, "dt")
        assert result.passed is True

    def test_has_future_dates(self):
        """Test fails when future dates present."""
        tomorrow = date.today() + timedelta(days=1)
        yesterday = date.today() - timedelta(days=1)
        df = pl.DataFrame({"dt": [yesterday, tomorrow]})
        result = validate_not_in_future(df, "dt")
        assert result.passed is False
        assert result.failed_count == 1

    def test_datetime_column(self):
        """Test works with datetime column."""
        past = datetime.now() - timedelta(hours=1)
        df = pl.DataFrame({"dt": [past]})
        result = validate_not_in_future(df, "dt")
        assert result.passed is True


class TestValidateWithinDays:
    """Tests for within_days validation."""

    def test_all_within_range(self):
        """Test passes when all dates within range."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        df = pl.DataFrame({"dt": [today, yesterday]})
        result = validate_within_days(df, "dt", 7)
        assert result.passed is True

    def test_some_outside_range(self):
        """Test fails when some dates outside range."""
        today = date.today()
        old = today - timedelta(days=100)
        df = pl.DataFrame({"dt": [today, old]})
        result = validate_within_days(df, "dt", 7)
        assert result.passed is False
        assert result.failed_count == 1


class TestValidateMinLength:
    """Tests for min_length validation."""

    def test_all_meet_length(self):
        """Test passes when all values meet minimum length."""
        df = pl.DataFrame({"code": ["abc", "abcd", "abcde"]})
        result = validate_min_length(df, "code", 3)
        assert result.passed is True

    def test_some_too_short(self):
        """Test fails when some values are too short."""
        df = pl.DataFrame({"code": ["a", "ab", "abc"]})
        result = validate_min_length(df, "code", 3)
        assert result.passed is False
        assert result.failed_count == 2


class TestValidateMaxLength:
    """Tests for max_length validation."""

    def test_all_within_length(self):
        """Test passes when all values within max length."""
        df = pl.DataFrame({"code": ["a", "ab", "abc"]})
        result = validate_max_length(df, "code", 3)
        assert result.passed is True

    def test_some_too_long(self):
        """Test fails when some values are too long."""
        df = pl.DataFrame({"code": ["abc", "abcdefg"]})
        result = validate_max_length(df, "code", 5)
        assert result.passed is False
        assert result.failed_count == 1


class TestValidateRowCount:
    """Tests for row_count validation."""

    def test_within_bounds(self):
        """Test passes when row count within bounds."""
        df = pl.DataFrame({"id": [1, 2, 3]})
        result = validate_row_count(df, min_count=1, max_count=10)
        assert result.passed is True

    def test_below_minimum(self):
        """Test fails when below minimum."""
        df = pl.DataFrame({"id": [1]})
        result = validate_row_count(df, min_count=5)
        assert result.passed is False
        assert "below minimum" in result.message

    def test_above_maximum(self):
        """Test fails when above maximum."""
        df = pl.DataFrame({"id": [1, 2, 3, 4, 5]})
        result = validate_row_count(df, max_count=3)
        assert result.passed is False
        assert "exceeds maximum" in result.message

    def test_no_bounds(self):
        """Test passes when no bounds specified."""
        df = pl.DataFrame({"id": [1, 2, 3]})
        result = validate_row_count(df)
        assert result.passed is True


class TestValidateFreshness:
    """Tests for freshness validation."""

    def test_fresh_data(self):
        """Test passes when data is fresh."""
        now = datetime.now()
        df = pl.DataFrame({"dt": [now - timedelta(hours=1)]})
        result = validate_freshness(df, "dt", max_age_hours=24)
        assert result.passed is True

    def test_stale_data(self):
        """Test fails when data is stale."""
        old = datetime.now() - timedelta(days=5)
        df = pl.DataFrame({"dt": [old]})
        result = validate_freshness(df, "dt", max_age_hours=24)
        assert result.passed is False
        assert "hours old" in result.message

    def test_empty_dataframe(self):
        """Test fails for empty DataFrame."""
        df = pl.DataFrame({"dt": []}).cast({"dt": pl.Datetime})
        result = validate_freshness(df, "dt", max_age_hours=24)
        assert result.passed is False


class TestValidateNullPercentage:
    """Tests for null_percentage validation."""

    def test_within_threshold(self):
        """Test passes when null percentage within threshold."""
        df = pl.DataFrame({"col": [1, 2, 3, None]})  # 25% null
        result = validate_null_percentage(df, "col", max_percent=50)
        assert result.passed is True

    def test_exceeds_threshold(self):
        """Test fails when null percentage exceeds threshold."""
        df = pl.DataFrame({"col": [None, None, None, 1]})  # 75% null
        result = validate_null_percentage(df, "col", max_percent=50)
        assert result.passed is False


class TestValidateUniqueCombination:
    """Tests for unique_combination validation."""

    def test_all_unique_combinations(self):
        """Test passes when all combinations are unique."""
        df = pl.DataFrame(
            {
                "col1": [1, 1, 2],
                "col2": ["a", "b", "a"],
            }
        )
        result = validate_unique_combination(df, ["col1", "col2"])
        assert result.passed is True

    def test_duplicate_combinations(self):
        """Test fails when duplicate combinations exist."""
        df = pl.DataFrame(
            {
                "col1": [1, 1, 1],
                "col2": ["a", "a", "b"],
            }
        )
        result = validate_unique_combination(df, ["col1", "col2"])
        assert result.passed is False
        assert result.failed_count == 1  # One duplicate


class TestRunColumnTest:
    """Tests for the run_column_test dispatcher."""

    def test_not_null_test(self):
        """Test dispatches to not_null validator."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = run_column_test(df, "col", "not_null", {})
        assert result.passed is True

    def test_pattern_test(self):
        """Test dispatches to pattern validator."""
        df = pl.DataFrame({"code": ["ABC123", "XYZ789"]})
        result = run_column_test(df, "code", "pattern", {"value": "^[A-Z]{3}[0-9]{3}$"})
        assert result.passed is True

    def test_accepted_values_test(self):
        """Test dispatches to accepted_values validator."""
        df = pl.DataFrame({"status": ["a", "b"]})
        result = run_column_test(df, "status", "accepted_values", {"values": ["a", "b", "c"]})
        assert result.passed is True

    def test_accepted_values_from_list_form(self):
        """Test accepted_values works with 'value' key (list-form YAML parsing).

        When YAML uses the natural list form:
            accepted_values:
              - January
              - February
        the parser stores under 'value' (singular), not 'values'.
        """
        df = pl.DataFrame({"month": ["January", "February"]})
        result = run_column_test(
            df, "month", "accepted_values", {"value": ["January", "February", "March"]}
        )
        assert result.passed is True

        # Verify failure case too
        result_fail = run_column_test(df, "month", "accepted_values", {"value": ["January"]})
        assert result_fail.passed is False

    def test_with_when_condition(self):
        """Test applies when condition before validation."""
        # Without when condition, this would fail because of null
        df = pl.DataFrame({"code": ["ABC", None, "XYZ"]})
        result = run_column_test(df, "code", "pattern", {"value": "^[A-Z]{3}$"}, when="not_null")
        assert result.passed is True

    def test_unknown_test_type(self):
        """Test returns error for unknown test type."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = run_column_test(df, "col", "unknown_test", {})
        assert result.passed is False
        assert "Unknown test type" in result.message

    def test_pii_redacts_sample_values(self):
        """Test that pii=True redacts sample values from failure messages."""
        df = pl.DataFrame({"ssn": ["123-45-6789", "987-65-4321"]})
        result = run_column_test(df, "ssn", "pattern", {"value": r"^XXX-XX-XXXX$"}, pii=True)
        assert result.passed is False
        assert "REDACTED" in result.message
        assert "column marked as PII" in result.message
        assert "123-45-6789" not in result.message
        assert "987-65-4321" not in result.message
        assert result.sample_failures is None

    def test_pii_false_shows_samples(self):
        """Test that pii=False (default) shows actual sample values."""
        df = pl.DataFrame({"name": ["Alice", "Bob"]})
        result = run_column_test(df, "name", "accepted_values", {"value": ["Charlie"]}, pii=False)
        assert result.passed is False
        assert "REDACTED" not in result.message
        assert result.sample_failures is not None

    def test_pii_passing_test_unchanged(self):
        """Test that pii=True does not alter passing test messages."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = run_column_test(df, "col", "not_null", {}, pii=True)
        assert result.passed is True
        assert "REDACTED" not in result.message


class TestRunTableExpectation:
    """Tests for the run_table_expectation dispatcher."""

    def test_row_count_expectation(self):
        """Test dispatches to row_count validator."""
        df = pl.DataFrame({"id": [1, 2, 3]})
        result = run_table_expectation(df, "row_count", {"min": 1, "max": 10})
        assert result.passed is True

    def test_null_percentage_expectation(self):
        """Test dispatches to null_percentage validator."""
        df = pl.DataFrame({"col": [1, 2, None]})
        result = run_table_expectation(df, "null_percentage", {"column": "col", "max_percent": 50})
        assert result.passed is True

    def test_unique_combination_expectation(self):
        """Test dispatches to unique_combination validator."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = run_table_expectation(df, "unique_combination", {"columns": ["a", "b"]})
        assert result.passed is True

    def test_unknown_expectation_type(self):
        """Test returns error for unknown expectation type."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = run_table_expectation(df, "unknown_expectation", {})
        assert result.passed is False
        assert "Unknown expectation type" in result.message


class TestValidateRowCountRaw:
    """Tests for validate_row_count_raw (integer-based)."""

    def test_within_bounds(self):
        result = validate_row_count_raw(500, min_count=100, max_count=1000)
        assert result.passed is True

    def test_below_min(self):
        result = validate_row_count_raw(50, min_count=100)
        assert result.passed is False
        assert "below minimum" in result.message

    def test_above_max(self):
        result = validate_row_count_raw(1500, max_count=1000)
        assert result.passed is False
        assert "exceeds maximum" in result.message

    def test_none_bounds(self):
        result = validate_row_count_raw(0)
        assert result.passed is True


def _mock_cursor(fetchone_value=None, fetchall_value=None, description=None):
    """Create a mock database cursor for post-write validation tests."""
    cursor = MagicMock()
    cursor.fetchone.return_value = fetchone_value
    cursor.fetchall.return_value = fetchall_value or []
    cursor.description = description
    return cursor


class TestPostWriteRowCount:
    """Tests for post-write row_count via SQL and SCD2 counter."""

    def test_lineage_scoped(self):
        """row_count queries with lineage ID filter."""
        cursor = _mock_cursor(fetchone_value=(1500,))
        exp = TableExpectation(
            expectation_type="row_count",
            parameters={"min": 1000, "max": 2000},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
        )
        assert len(results) == 1
        _, result = results[0]
        assert result.passed is True

        # Verify SQL used lineage scoping
        executed_sql = cursor.execute.call_args[0][0]
        assert "_lineage_id" in executed_sql
        assert cursor.execute.call_args[0][1] == ["abc-123"]

    def test_no_lineage_fallback(self):
        """row_count queries without lineage scoping when lineage_id is None."""
        cursor = _mock_cursor(fetchone_value=(500,))
        exp = TableExpectation(
            expectation_type="row_count",
            parameters={"min": 100},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id=None,
        )
        _, result = results[0]
        assert result.passed is True

        # Verify SQL has no WHERE clause
        executed_sql = cursor.execute.call_args[0][0]
        assert "_lineage_id" not in executed_sql

    def test_scd2_uses_total_rows(self):
        """SCD2 row_count uses accumulated counter, not SQL."""
        cursor = _mock_cursor()
        exp = TableExpectation(
            expectation_type="row_count",
            parameters={"min": 1000},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
            total_rows=1500,
            is_scd2=True,
        )
        _, result = results[0]
        assert result.passed is True
        # cursor.execute should NOT have been called for row_count
        cursor.execute.assert_not_called()

    def test_scd2_fails_below_min(self):
        """SCD2 row_count fails when total_rows is below min."""
        cursor = _mock_cursor()
        exp = TableExpectation(
            expectation_type="row_count",
            parameters={"min": 1000},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            total_rows=300,
            is_scd2=True,
        )
        _, result = results[0]
        assert result.passed is False
        assert "below minimum" in result.message

    def test_fails_below_min_via_sql(self):
        """Non-SCD2 row_count fails when SQL count is below min."""
        cursor = _mock_cursor(fetchone_value=(50,))
        exp = TableExpectation(
            expectation_type="row_count",
            parameters={"min": 100},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
        )
        _, result = results[0]
        assert result.passed is False
        assert "below minimum" in result.message


class TestPostWriteNullPercentage:
    """Tests for post-write null_percentage via SQL."""

    def test_within_threshold(self):
        cursor = _mock_cursor(fetchone_value=(5, 100))
        exp = TableExpectation(
            expectation_type="null_percentage",
            parameters={"column": "email", "max_percent": 10},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
        )
        _, result = results[0]
        assert result.passed is True

    def test_exceeds_threshold(self):
        cursor = _mock_cursor(fetchone_value=(50, 100))
        exp = TableExpectation(
            expectation_type="null_percentage",
            parameters={"column": "email", "max_percent": 10},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
        )
        _, result = results[0]
        assert result.passed is False
        assert "50.0% nulls" in result.message

    def test_missing_column_param(self):
        cursor = _mock_cursor()
        exp = TableExpectation(
            expectation_type="null_percentage",
            parameters={"max_percent": 10},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
        )
        _, result = results[0]
        assert result.passed is False
        assert "requires" in result.message


class TestPostWriteUniqueCombination:
    """Tests for post-write unique_combination via SQL."""

    def test_unique(self):
        cursor = _mock_cursor(fetchone_value=(100,), fetchall_value=[])
        # Need two calls: first for COUNT(*), second for GROUP BY
        cursor.fetchone.side_effect = [(100,)]
        cursor.fetchall.return_value = []
        exp = TableExpectation(
            expectation_type="unique_combination",
            parameters={"columns": ["first_name", "last_name"]},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
        )
        _, result = results[0]
        assert result.passed is True

    def test_duplicates_found(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (100,)
        cursor.fetchall.return_value = [("John", "Doe", 3)]
        cursor.description = [("first_name",), ("last_name",), ("cnt",)]
        exp = TableExpectation(
            expectation_type="unique_combination",
            parameters={"columns": ["first_name", "last_name"]},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
        )
        _, result = results[0]
        assert result.passed is False
        assert "duplicate" in result.message


class TestPostWriteFreshness:
    """Tests for post-write freshness via SQL."""

    def test_fresh_data(self):
        recent = datetime.now() - timedelta(hours=1)
        cursor = _mock_cursor(fetchone_value=(recent,))
        exp = TableExpectation(
            expectation_type="freshness",
            parameters={"column": "updated_at", "max_age_hours": 24},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
        )
        _, result = results[0]
        assert result.passed is True

    def test_stale_data(self):
        old = datetime.now() - timedelta(days=5)
        cursor = _mock_cursor(fetchone_value=(old,))
        exp = TableExpectation(
            expectation_type="freshness",
            parameters={"column": "updated_at", "max_age_hours": 24},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
            lineage_id="abc-123",
        )
        _, result = results[0]
        assert result.passed is False
        assert "hours old" in result.message

    def test_no_data(self):
        cursor = _mock_cursor(fetchone_value=(None,))
        exp = TableExpectation(
            expectation_type="freshness",
            parameters={"column": "updated_at", "max_age_hours": 24},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
        )
        _, result = results[0]
        assert result.passed is False
        assert "no non-null" in result.message


class TestPostWriteUnknownExpectation:
    """Test unknown expectation type in post-write."""

    def test_unknown_type(self):
        cursor = _mock_cursor()
        exp = TableExpectation(
            expectation_type="some_future_type",
            parameters={},
        )
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=[exp],
        )
        _, result = results[0]
        assert result.passed is False
        assert "Unknown expectation type" in result.message


class TestPostWriteMultipleExpectations:
    """Test running multiple expectations in one call."""

    def test_multiple_expectations(self):
        cursor = MagicMock()
        # row_count returns 500, null_percentage returns (2, 500)
        cursor.fetchone.side_effect = [(500,), (2, 500)]
        expectations = [
            TableExpectation(
                expectation_type="row_count",
                parameters={"min": 100},
            ),
            TableExpectation(
                expectation_type="null_percentage",
                parameters={"column": "email", "max_percent": 5},
            ),
        ]
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."customers"',
            expectations=expectations,
            lineage_id="abc-123",
        )
        assert len(results) == 2
        assert results[0][1].passed is True
        assert results[1][1].passed is True


class TestPostWriteHistoryCompleteness:
    """Tests for history_completeness post-write expectation."""

    @staticmethod
    def _make_periods():
        from moncpipelib.contracts.models import Period

        return [
            Period(source="a.csv", effective_from=date(2025, 1, 1), effective_to=date(2025, 7, 1)),
            Period(source="b.csv", effective_from=date(2025, 7, 1), effective_to=date(2026, 1, 1)),
            Period(source="c.csv", effective_from=date(2026, 1, 1)),
        ]

    def test_all_present(self):
        """All periods have rows -> PASS."""
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            (date(2025, 1, 1),),
            (date(2025, 7, 1),),
            (date(2026, 1, 1),),
        ]
        exp = TableExpectation(expectation_type="history_completeness")
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."products"',
            expectations=[exp],
            periods=self._make_periods(),
            effective_date=date(2026, 1, 1),
        )
        _, result = results[0]
        assert result.passed is True
        assert "complete" in result.message

    def test_missing_period_final_load(self):
        """Final period loaded but earlier one missing -> FAIL."""
        cursor = MagicMock()
        # Only 2 of 3 periods present (missing 2025-07-01)
        cursor.fetchall.return_value = [
            (date(2025, 1, 1),),
            (date(2026, 1, 1),),
        ]
        exp = TableExpectation(expectation_type="history_completeness")
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."products"',
            expectations=[exp],
            periods=self._make_periods(),
            effective_date=date(2026, 1, 1),  # final period
        )
        _, result = results[0]
        assert result.passed is False
        assert "2025-07-01" in result.message
        assert "incomplete" in result.message.lower()

    def test_missing_period_mid_load(self):
        """Non-final period loaded, others missing -> PASS (progress info)."""
        cursor = MagicMock()
        # Only first period present
        cursor.fetchall.return_value = [(date(2025, 1, 1),)]
        exp = TableExpectation(expectation_type="history_completeness")
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."products"',
            expectations=[exp],
            periods=self._make_periods(),
            effective_date=date(2025, 1, 1),  # first period, not final
        )
        _, result = results[0]
        assert result.passed is True
        assert "in progress" in result.message.lower()
        assert "1/3" in result.message

    def test_no_periods(self):
        """No periods in contract -> PASS (skipped)."""
        cursor = MagicMock()
        exp = TableExpectation(expectation_type="history_completeness")
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."products"',
            expectations=[exp],
            periods=None,
        )
        _, result = results[0]
        assert result.passed is True
        assert "skipped" in result.message

    def test_no_effective_date_checks_all(self):
        """effective_date=None (regular write) -> checks all, fails if missing."""
        cursor = MagicMock()
        cursor.fetchall.return_value = [(date(2025, 1, 1),)]  # only 1 of 3
        exp = TableExpectation(expectation_type="history_completeness")
        results = run_post_write_expectations(
            cursor=cursor,
            table_name='"silver"."products"',
            expectations=[exp],
            periods=self._make_periods(),
            effective_date=None,  # not a historical load
        )
        _, result = results[0]
        assert result.passed is False
        assert "incomplete" in result.message.lower()
