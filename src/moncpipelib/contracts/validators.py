"""Runtime validation functions for data contracts.

This module provides validation functions that check DataFrames
against contract expectations.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl

from moncpipelib.config import LineageDefaults
from moncpipelib.contracts.models import (
    ColumnType,
    DataContract,
    Period,
    ValidationResult,
)

if TYPE_CHECKING:
    pass

_MAX_SAMPLE_VALUES = 5
"""Maximum number of sample values to include in validation failure messages."""

# Mapping from contract types to Polars types
POLARS_TYPE_MAP: dict[ColumnType, set[type[pl.DataType]]] = {
    ColumnType.STRING: {pl.Utf8, pl.String},
    ColumnType.INTEGER: {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    },
    ColumnType.DECIMAL: {pl.Float32, pl.Float64, pl.Decimal},
    ColumnType.BOOLEAN: {pl.Boolean},
    ColumnType.DATE: {pl.Date},
    ColumnType.DATETIME: {pl.Datetime},
    ColumnType.UUID: {pl.Utf8, pl.String},  # UUIDs are stored as strings
    ColumnType.JSON: {pl.Utf8, pl.String},  # JSON stored as text in Polars
    ColumnType.JSONB: {pl.Utf8, pl.String},  # JSONB stored as text in Polars
}


def validate_schema(
    df: pl.DataFrame,
    contract: DataContract,
) -> ValidationResult:
    """Validate DataFrame schema against contract.

    Checks:
    - All non-managed columns in contract exist in DataFrame
    - Column types match (with type coercion tolerance)
    - No unexpected columns (if strict mode)

    Args:
        df: DataFrame to validate
        contract: Contract defining expected schema

    Returns:
        ValidationResult with pass/fail and details
    """
    errors: list[str] = []
    df_columns = set(df.columns)

    # Get expected columns (non-managed only)
    expected_columns = contract.get_non_managed_columns()
    expected_names = {c.name for c in expected_columns}

    # Check for missing columns
    missing = expected_names - df_columns
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    # Check column types for columns that exist
    for col in expected_columns:
        if col.name in df_columns:
            df_dtype = df.schema[col.name]
            expected_types = POLARS_TYPE_MAP.get(col.type, set())

            # Check if the actual type matches any of the expected types
            type_matches = any(isinstance(df_dtype, t) for t in expected_types)
            if not type_matches:
                errors.append(
                    f"Column '{col.name}' has type {df_dtype}, expected one of {col.type.value}"
                )

    # Check for unexpected columns in strict mode
    if contract.schema.strict:
        # In strict mode, also include managed columns as valid
        all_valid = expected_names | {c.name for c in contract.schema.columns if c.managed}
        unexpected = df_columns - all_valid
        if unexpected:
            errors.append(f"Unexpected columns (strict mode): {sorted(unexpected)}")

    if errors:
        return ValidationResult(
            passed=False,
            message="; ".join(errors),
            failed_count=len(errors),
            total_count=len(expected_columns),
        )

    return ValidationResult(
        passed=True,
        message="Schema validation passed",
        failed_count=0,
        total_count=len(expected_columns),
    )


def validate_not_null(
    df: pl.DataFrame,
    column: str,
) -> ValidationResult:
    """Check column contains no null values.

    Args:
        df: DataFrame to validate
        column: Column name to check

    Returns:
        ValidationResult with pass/fail and null count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    null_count = df.filter(pl.col(column).is_null()).height
    total_count = df.height

    if null_count > 0:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {null_count} null values out of {total_count} rows",
            failed_count=null_count,
            total_count=total_count,
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' has no null values",
        failed_count=0,
        total_count=total_count,
    )


def validate_unique(
    df: pl.DataFrame,
    column: str,
) -> ValidationResult:
    """Check column contains only unique values.

    Args:
        df: DataFrame to validate
        column: Column name to check

    Returns:
        ValidationResult with pass/fail and duplicate count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    total_count = df.height
    unique_count = df.select(pl.col(column)).n_unique()
    duplicate_count = total_count - unique_count

    if duplicate_count > 0:
        # Get sample of duplicate values
        duplicates = (
            df.group_by(column)
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") > 1)
            .head(_MAX_SAMPLE_VALUES)
        )
        sample_values = duplicates[column].to_list()

        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {duplicate_count} duplicate values. Sample duplicates: {sample_values}",
            failed_count=duplicate_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' has all unique values",
        failed_count=0,
        total_count=total_count,
    )


def validate_accepted_values(
    df: pl.DataFrame,
    column: str,
    values: list[Any],
) -> ValidationResult:
    """Check column values are in accepted list.

    Args:
        df: DataFrame to validate
        column: Column name to check
        values: List of accepted values

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    # Filter to non-null values not in the accepted list
    invalid_df = df.filter(pl.col(column).is_not_null() & ~pl.col(column).is_in(values))
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = (
            invalid_df.select(column).unique().head(_MAX_SAMPLE_VALUES)[column].to_list()
        )
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values not in {values}. Sample invalid: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' has all values in accepted list",
        failed_count=0,
        total_count=total_count,
    )


def validate_not_in(
    df: pl.DataFrame,
    column: str,
    values: list[Any],
) -> ValidationResult:
    """Check column does not contain any rejected values.

    Null values are ignored (not considered a match against the rejected list).

    Args:
        df: DataFrame to validate.
        column: Column name to check.
        values: List of rejected values that should not appear.

    Returns:
        ValidationResult with pass/fail and count of rejected value occurrences.
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    rejected_df = df.filter(pl.col(column).is_not_null() & pl.col(column).is_in(values))
    rejected_count = rejected_df.height
    total_count = df.height

    if rejected_count > 0:
        sample_values = (
            rejected_df.select(column).unique().head(_MAX_SAMPLE_VALUES)[column].to_list()
        )
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {rejected_count} rows with rejected values "
                f"{values}. Sample matches: {sample_values}"
            ),
            failed_count=rejected_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' has no rejected values from {values}",
        failed_count=0,
        total_count=total_count,
    )


def validate_pattern(
    df: pl.DataFrame,
    column: str,
    pattern: str,
) -> ValidationResult:
    """Check column values match regex pattern.

    Args:
        df: DataFrame to validate
        column: Column name to check
        pattern: Regex pattern to match

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    # Filter to non-null values that don't match the pattern
    invalid_df = df.filter(pl.col(column).is_not_null() & ~pl.col(column).str.contains(pattern))
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = (
            invalid_df.select(column).unique().head(_MAX_SAMPLE_VALUES)[column].to_list()
        )
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values not matching pattern '{pattern}'. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values match pattern '{pattern}'",
        failed_count=0,
        total_count=total_count,
    )


def validate_greater_than(
    df: pl.DataFrame,
    column: str,
    threshold: float,
) -> ValidationResult:
    """Check column values are greater than threshold.

    Args:
        df: DataFrame to validate
        column: Column name to check
        threshold: Minimum value (exclusive)

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    invalid_df = df.filter(pl.col(column).is_not_null() & (pl.col(column) <= threshold))
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values <= {threshold}. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all > {threshold}",
        failed_count=0,
        total_count=total_count,
    )


def validate_greater_than_or_equal(
    df: pl.DataFrame,
    column: str,
    threshold: float,
) -> ValidationResult:
    """Check column values are greater than or equal to threshold.

    Args:
        df: DataFrame to validate
        column: Column name to check
        threshold: Minimum value (inclusive)

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    invalid_df = df.filter(pl.col(column).is_not_null() & (pl.col(column) < threshold))
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values < {threshold}. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all >= {threshold}",
        failed_count=0,
        total_count=total_count,
    )


def validate_less_than(
    df: pl.DataFrame,
    column: str,
    threshold: float,
) -> ValidationResult:
    """Check column values are less than threshold.

    Args:
        df: DataFrame to validate
        column: Column name to check
        threshold: Maximum value (exclusive)

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    invalid_df = df.filter(pl.col(column).is_not_null() & (pl.col(column) >= threshold))
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values >= {threshold}. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all < {threshold}",
        failed_count=0,
        total_count=total_count,
    )


def validate_less_than_or_equal(
    df: pl.DataFrame,
    column: str,
    threshold: float,
) -> ValidationResult:
    """Check column values are less than or equal to threshold.

    Args:
        df: DataFrame to validate
        column: Column name to check
        threshold: Maximum value (inclusive)

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    invalid_df = df.filter(pl.col(column).is_not_null() & (pl.col(column) > threshold))
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values > {threshold}. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all <= {threshold}",
        failed_count=0,
        total_count=total_count,
    )


def validate_between(
    df: pl.DataFrame,
    column: str,
    min_value: float,
    max_value: float,
) -> ValidationResult:
    """Check column values are between min and max (inclusive).

    Args:
        df: DataFrame to validate
        column: Column name to check
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    invalid_df = df.filter(
        pl.col(column).is_not_null() & ((pl.col(column) < min_value) | (pl.col(column) > max_value))
    )
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values outside range [{min_value}, {max_value}]. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all in range [{min_value}, {max_value}]",
        failed_count=0,
        total_count=total_count,
    )


def validate_not_in_future(
    df: pl.DataFrame,
    column: str,
) -> ValidationResult:
    """Check date/datetime column has no future values.

    Args:
        df: DataFrame to validate
        column: Column name to check (must be Date or Datetime)

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    today = date.today()
    col_dtype = df.schema[column]

    if isinstance(col_dtype, pl.Datetime):
        # Compare datetime to current datetime
        now = datetime.now()
        invalid_df = df.filter(pl.col(column).is_not_null() & (pl.col(column) > now))
    elif isinstance(col_dtype, pl.Date):
        # Compare date to today
        invalid_df = df.filter(pl.col(column).is_not_null() & (pl.col(column) > today))
    else:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' is not a Date or Datetime type",
            failed_count=1,
            total_count=1,
        )

    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} future values. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: str(v)} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' has no future values",
        failed_count=0,
        total_count=total_count,
    )


def validate_within_days(
    df: pl.DataFrame,
    column: str,
    days: int,
) -> ValidationResult:
    """Check date column values are within N days of today.

    Args:
        df: DataFrame to validate
        column: Column name to check (must be Date or Datetime)
        days: Maximum number of days from today

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    today = date.today()
    min_date = today - timedelta(days=days)
    col_dtype = df.schema[column]

    if isinstance(col_dtype, pl.Datetime):
        # Convert date boundaries to datetime for comparison
        min_dt = datetime.combine(min_date, datetime.min.time())
        max_dt = datetime.now()
        invalid_df = df.filter(
            pl.col(column).is_not_null() & ((pl.col(column) < min_dt) | (pl.col(column) > max_dt))
        )
    elif isinstance(col_dtype, pl.Date):
        invalid_df = df.filter(
            pl.col(column).is_not_null() & ((pl.col(column) < min_date) | (pl.col(column) > today))
        )
    else:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' is not a Date or Datetime type",
            failed_count=1,
            total_count=1,
        )

    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values outside {days} day window. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: str(v)} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all within {days} days",
        failed_count=0,
        total_count=total_count,
    )


def validate_min_length(
    df: pl.DataFrame,
    column: str,
    length: int,
) -> ValidationResult:
    """Check string column values have minimum length.

    Args:
        df: DataFrame to validate
        column: Column name to check
        length: Minimum string length

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    invalid_df = df.filter(pl.col(column).is_not_null() & (pl.col(column).str.len_chars() < length))
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values shorter than {length} chars. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all >= {length} chars",
        failed_count=0,
        total_count=total_count,
    )


def validate_max_length(
    df: pl.DataFrame,
    column: str,
    length: int,
) -> ValidationResult:
    """Check string column values have maximum length.

    Args:
        df: DataFrame to validate
        column: Column name to check
        length: Maximum string length

    Returns:
        ValidationResult with pass/fail and invalid count
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    invalid_df = df.filter(pl.col(column).is_not_null() & (pl.col(column).str.len_chars() > length))
    invalid_count = invalid_df.height
    total_count = df.height

    if invalid_count > 0:
        sample_values = invalid_df.select(column).head(_MAX_SAMPLE_VALUES)[column].to_list()
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {invalid_count} values longer than {length} chars. Sample: {sample_values}",
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all <= {length} chars",
        failed_count=0,
        total_count=total_count,
    )


def validate_row_count(
    df: pl.DataFrame,
    min_count: int | None = None,
    max_count: int | None = None,
) -> ValidationResult:
    """Check table row count is within bounds.

    Args:
        df: DataFrame to validate
        min_count: Minimum row count (inclusive)
        max_count: Maximum row count (inclusive)

    Returns:
        ValidationResult with pass/fail
    """
    row_count = df.height
    errors: list[str] = []

    if min_count is not None and row_count < min_count:
        errors.append(f"Row count {row_count} is below minimum {min_count}")

    if max_count is not None and row_count > max_count:
        errors.append(f"Row count {row_count} exceeds maximum {max_count}")

    if errors:
        return ValidationResult(
            passed=False,
            message="; ".join(errors),
            failed_count=1,
            total_count=1,
        )

    return ValidationResult(
        passed=True,
        message=f"Row count {row_count} is within bounds",
        failed_count=0,
        total_count=1,
    )


def validate_freshness(
    df: pl.DataFrame,
    column: str,
    max_age_hours: int,
) -> ValidationResult:
    """Check most recent value in column is within max age.

    Args:
        df: DataFrame to validate
        column: Date/datetime column to check
        max_age_hours: Maximum age in hours

    Returns:
        ValidationResult with pass/fail
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    if df.height == 0:
        return ValidationResult(
            passed=False,
            message="DataFrame is empty, cannot check freshness",
            failed_count=1,
            total_count=1,
        )

    # Get the maximum (most recent) value
    max_value = df.select(pl.col(column).max()).item()

    if max_value is None:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has no non-null values",
            failed_count=1,
            total_count=1,
        )

    now = datetime.now()
    col_dtype = df.schema[column]

    if isinstance(col_dtype, pl.Date):
        # Convert date to datetime for comparison
        max_datetime = datetime.combine(max_value, datetime.min.time())
    elif isinstance(col_dtype, pl.Datetime):
        max_datetime = max_value
    else:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' is not a Date or Datetime type",
            failed_count=1,
            total_count=1,
        )

    age = now - max_datetime
    age_hours = age.total_seconds() / 3600

    if age_hours > max_age_hours:
        return ValidationResult(
            passed=False,
            message=f"Data is {age_hours:.1f} hours old, exceeds maximum {max_age_hours} hours. Most recent: {max_value}",
            failed_count=1,
            total_count=1,
        )

    return ValidationResult(
        passed=True,
        message=f"Data is {age_hours:.1f} hours old, within {max_age_hours} hour limit",
        failed_count=0,
        total_count=1,
    )


def validate_null_percentage(
    df: pl.DataFrame,
    column: str,
    max_percent: float,
) -> ValidationResult:
    """Check null percentage in column is below threshold.

    Args:
        df: DataFrame to validate
        column: Column name to check
        max_percent: Maximum percentage of nulls allowed (0-100)

    Returns:
        ValidationResult with pass/fail
    """
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found in DataFrame",
            failed_count=1,
            total_count=1,
        )

    total_count = df.height
    if total_count == 0:
        return ValidationResult(
            passed=True,
            message="DataFrame is empty, no nulls to check",
            failed_count=0,
            total_count=0,
        )

    null_count = df.filter(pl.col(column).is_null()).height
    null_percent = (null_count / total_count) * 100

    if null_percent > max_percent:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has {null_percent:.1f}% nulls, exceeds maximum {max_percent}%",
            failed_count=null_count,
            total_count=total_count,
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' has {null_percent:.1f}% nulls, within {max_percent}% limit",
        failed_count=null_count,
        total_count=total_count,
    )


def validate_unique_combination(
    df: pl.DataFrame,
    columns: list[str],
) -> ValidationResult:
    """Check combination of columns is unique.

    Args:
        df: DataFrame to validate
        columns: List of column names that should be unique together

    Returns:
        ValidationResult with pass/fail
    """
    # Check all columns exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return ValidationResult(
            passed=False,
            message=f"Columns not found: {missing}",
            failed_count=1,
            total_count=1,
        )

    total_count = df.height
    unique_count = df.select(columns).unique().height
    duplicate_count = total_count - unique_count

    if duplicate_count > 0:
        # Find duplicate combinations
        duplicates = (
            df.group_by(columns)
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") > 1)
            .head(_MAX_SAMPLE_VALUES)
        )
        sample_failures = duplicates.select(columns).to_dicts()

        return ValidationResult(
            passed=False,
            message=f"Columns {columns} have {duplicate_count} duplicate combinations",
            failed_count=duplicate_count,
            total_count=total_count,
            sample_failures=sample_failures,
        )

    return ValidationResult(
        passed=True,
        message=f"Columns {columns} are unique together",
        failed_count=0,
        total_count=total_count,
    )


def validate_row_count_raw(
    row_count: int,
    min_count: int | None = None,
    max_count: int | None = None,
) -> ValidationResult:
    """Check a raw row count is within bounds.

    Unlike ``validate_row_count`` which takes a DataFrame, this accepts a
    pre-computed integer count. Used for post-write validation where the
    count comes from SQL or an accumulated counter (e.g., SCD2 staging totals).

    Args:
        row_count: The row count to validate.
        min_count: Minimum row count (inclusive).
        max_count: Maximum row count (inclusive).

    Returns:
        ValidationResult with pass/fail.
    """
    errors: list[str] = []

    if min_count is not None and row_count < min_count:
        errors.append(f"Row count {row_count} is below minimum {min_count}")

    if max_count is not None and row_count > max_count:
        errors.append(f"Row count {row_count} exceeds maximum {max_count}")

    if errors:
        return ValidationResult(
            passed=False,
            message="; ".join(errors),
            failed_count=1,
            total_count=1,
        )

    return ValidationResult(
        passed=True,
        message=f"Row count {row_count} is within bounds",
        failed_count=0,
        total_count=1,
    )


def _redact_pii_samples(result: ValidationResult) -> ValidationResult:
    """Replace sample values in a validation result with a PII redaction notice.

    Used to prevent PII column values from leaking into Dagster metadata,
    logs, or the Dagster UI when a validation check fails.
    """
    redacted_msg = re.sub(
        r"(Sample[^:]*:) \[.*\]",
        r"\1 [REDACTED - column marked as PII]",
        result.message,
    )
    return ValidationResult(
        passed=result.passed,
        message=redacted_msg,
        failed_count=result.failed_count,
        total_count=result.total_count,
        sample_failures=None,
    )


def run_column_test(
    df: pl.DataFrame,
    column: str,
    test_type: str,
    parameters: dict[str, Any],
    when: str | None = None,
    pii: bool = False,
) -> ValidationResult:
    """Run a single column test.

    This is a dispatcher function that calls the appropriate validator
    based on the test type.

    Args:
        df: DataFrame to validate
        column: Column name to check
        test_type: Type of test (e.g., "not_null", "unique", "pattern")
        parameters: Test-specific parameters
        when: Optional condition (e.g., "not_null")
        pii: If True, redact sample values from failure messages to prevent
            PII leakage into logs and Dagster metadata.

    Returns:
        ValidationResult from the appropriate validator
    """
    # If there's a "when" condition, filter the dataframe first
    test_df = df
    if when == "not_null":
        test_df = df.filter(pl.col(column).is_not_null())

    # Dispatch to appropriate validator
    validators: dict[str, Callable[[], ValidationResult]] = {
        "not_null": lambda: validate_not_null(test_df, column),
        "unique": lambda: validate_unique(test_df, column),
        "accepted_values": lambda: validate_accepted_values(
            test_df, column, parameters.get("values", parameters.get("value", []))
        ),
        "not_in": lambda: validate_not_in(
            test_df, column, parameters.get("values", parameters.get("value", []))
        ),
        "pattern": lambda: validate_pattern(test_df, column, parameters.get("value", "")),
        "greater_than": lambda: validate_greater_than(test_df, column, parameters.get("value", 0)),
        "greater_than_or_equal": lambda: validate_greater_than_or_equal(
            test_df, column, parameters.get("value", 0)
        ),
        "less_than": lambda: validate_less_than(test_df, column, parameters.get("value", 0)),
        "less_than_or_equal": lambda: validate_less_than_or_equal(
            test_df, column, parameters.get("value", 0)
        ),
        "between": lambda: validate_between(
            test_df, column, parameters.get("min", 0), parameters.get("max", 0)
        ),
        "not_in_future": lambda: validate_not_in_future(test_df, column),
        "within_days": lambda: validate_within_days(
            test_df, column, parameters.get("value", parameters.get("days", 0))
        ),
        "min_length": lambda: validate_min_length(
            test_df, column, parameters.get("value", parameters.get("length", 0))
        ),
        "max_length": lambda: validate_max_length(
            test_df, column, parameters.get("value", parameters.get("length", 0))
        ),
    }

    validator = validators.get(test_type)
    if validator is None:
        return ValidationResult(
            passed=False,
            message=f"Unknown test type: {test_type}",
            failed_count=1,
            total_count=1,
        )

    result = validator()

    if pii and not result.passed:
        result = _redact_pii_samples(result)

    return result


def run_table_expectation(
    df: pl.DataFrame,
    expectation_type: str,
    parameters: dict[str, Any],
) -> ValidationResult:
    """Run a single table expectation.

    This is a dispatcher function that calls the appropriate validator
    based on the expectation type.

    Args:
        df: DataFrame to validate
        expectation_type: Type of expectation (e.g., "row_count", "freshness")
        parameters: Expectation-specific parameters

    Returns:
        ValidationResult from the appropriate validator
    """
    expectations: dict[str, Callable[[], ValidationResult]] = {
        "row_count": lambda: validate_row_count(df, parameters.get("min"), parameters.get("max")),
        "freshness": lambda: validate_freshness(
            df, parameters.get("column", ""), parameters.get("max_age_hours", 0)
        ),
        "null_percentage": lambda: validate_null_percentage(
            df, parameters.get("column", ""), parameters.get("max_percent", 0)
        ),
        "unique_combination": lambda: validate_unique_combination(
            df, parameters.get("columns", [])
        ),
        "history_completeness": lambda: ValidationResult(
            passed=True,
            message="history_completeness deferred to post-write SQL validation",
            failed_count=0,
            total_count=0,
        ),
    }

    validator = expectations.get(expectation_type)
    if validator is None:
        return ValidationResult(
            passed=False,
            message=f"Unknown expectation type: {expectation_type}",
            failed_count=1,
            total_count=1,
        )

    return validator()


def _build_lineage_clause(lineage_id: str | None) -> tuple[str, list[Any]]:
    """Build a WHERE clause scoped by lineage ID.

    Returns:
        Tuple of (SQL clause fragment, parameter list). The clause includes
        a leading ``WHERE`` when lineage_id is provided, or is empty otherwise.
    """
    if lineage_id is not None:
        return f' WHERE "{LineageDefaults.ID_COLUMN}" = %s', [lineage_id]
    return "", []


def _post_write_row_count(
    cursor: Any,
    table_name: str,
    parameters: dict[str, Any],
    lineage_id: str | None,
    total_rows: int | None,
    is_scd2: bool,
) -> ValidationResult:
    """Validate row count via SQL or accumulated counter.

    For SCD2 writes, uses the in-memory ``total_rows`` counter (rows staged
    across all batches) because the staging table is dropped after merge and
    the merged table includes historical records.

    For non-SCD2 writes, queries the table scoped by lineage ID to count
    the rows written in this run.
    """
    min_count = parameters.get("min")
    max_count = parameters.get("max")

    if is_scd2:
        count = total_rows if total_rows is not None else 0
        return validate_row_count_raw(count, min_count, max_count)

    where, params = _build_lineage_clause(lineage_id)
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}{where}", params)  # noqa: S608
    count = cursor.fetchone()[0]
    return validate_row_count_raw(count, min_count, max_count)


def _post_write_freshness(
    cursor: Any,
    table_name: str,
    parameters: dict[str, Any],
    lineage_id: str | None,
) -> ValidationResult:
    """Validate data freshness via SQL.

    Queries the maximum value of the specified column and checks it is
    within ``max_age_hours`` of the current time.
    """
    column = parameters.get("column", "")
    max_age_hours = parameters.get("max_age_hours", 0)

    if not column:
        return ValidationResult(
            passed=False,
            message="freshness expectation requires a 'column' parameter",
            failed_count=1,
            total_count=1,
        )

    where, params = _build_lineage_clause(lineage_id)
    cursor.execute(
        f'SELECT MAX("{column}") FROM {table_name}{where}',  # noqa: S608
        params,
    )
    row = cursor.fetchone()
    max_value = row[0] if row else None

    if max_value is None:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has no non-null values",
            failed_count=1,
            total_count=1,
        )

    now = datetime.now()
    if isinstance(max_value, date) and not isinstance(max_value, datetime):
        max_value = datetime.combine(max_value, datetime.min.time())

    age = now - max_value
    age_hours = age.total_seconds() / 3600

    if age_hours > max_age_hours:
        return ValidationResult(
            passed=False,
            message=(
                f"Data is {age_hours:.1f} hours old, exceeds maximum "
                f"{max_age_hours} hours. Most recent: {max_value}"
            ),
            failed_count=1,
            total_count=1,
        )

    return ValidationResult(
        passed=True,
        message=f"Data is {age_hours:.1f} hours old, within {max_age_hours} hour limit",
        failed_count=0,
        total_count=1,
    )


def _post_write_null_percentage(
    cursor: Any,
    table_name: str,
    parameters: dict[str, Any],
    lineage_id: str | None,
) -> ValidationResult:
    """Validate null percentage for a column via SQL."""
    column = parameters.get("column", "")
    max_percent = parameters.get("max_percent", 0)

    if not column:
        return ValidationResult(
            passed=False,
            message="null_percentage expectation requires a 'column' parameter",
            failed_count=1,
            total_count=1,
        )

    where, params = _build_lineage_clause(lineage_id)
    cursor.execute(
        f'SELECT COUNT(*) FILTER (WHERE "{column}" IS NULL), COUNT(*) '  # noqa: S608
        f"FROM {table_name}{where}",
        params,
    )
    null_count, total_count = cursor.fetchone()

    if total_count == 0:
        return ValidationResult(
            passed=True,
            message="No rows to check for nulls",
            failed_count=0,
            total_count=0,
        )

    null_percent = (null_count / total_count) * 100

    if null_percent > max_percent:
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {null_percent:.1f}% nulls, exceeds maximum {max_percent}%"
            ),
            failed_count=null_count,
            total_count=total_count,
        )

    return ValidationResult(
        passed=True,
        message=(f"Column '{column}' has {null_percent:.1f}% nulls, within {max_percent}% limit"),
        failed_count=null_count,
        total_count=total_count,
    )


def _post_write_unique_combination(
    cursor: Any,
    table_name: str,
    parameters: dict[str, Any],
    lineage_id: str | None,
) -> ValidationResult:
    """Validate column combination uniqueness via SQL."""
    columns: list[str] = parameters.get("columns", [])

    if not columns:
        return ValidationResult(
            passed=False,
            message="unique_combination expectation requires a 'columns' parameter",
            failed_count=1,
            total_count=1,
        )

    quoted_cols = ", ".join(f'"{c}"' for c in columns)
    where, params = _build_lineage_clause(lineage_id)

    # Count total rows
    cursor.execute(
        f"SELECT COUNT(*) FROM {table_name}{where}",  # noqa: S608
        params,
    )
    total_count = cursor.fetchone()[0]

    # Find duplicates
    cursor.execute(
        f"SELECT {quoted_cols}, COUNT(*) AS cnt "  # noqa: S608
        f"FROM {table_name}{where} "
        f"GROUP BY {quoted_cols} HAVING COUNT(*) > 1 "
        f"LIMIT {_MAX_SAMPLE_VALUES}",
        params,
    )
    duplicates = cursor.fetchall()

    if duplicates:
        # Get column names from cursor description for sample output
        col_names = [desc[0] for desc in cursor.description[:-1]]  # exclude cnt
        sample_failures = [dict(zip(col_names, row[:-1], strict=True)) for row in duplicates]
        duplicate_count = sum(row[-1] - 1 for row in duplicates)

        return ValidationResult(
            passed=False,
            message=f"Columns {columns} have {duplicate_count} duplicate combinations",
            failed_count=duplicate_count,
            total_count=total_count,
            sample_failures=sample_failures,
        )

    return ValidationResult(
        passed=True,
        message=f"Columns {columns} are unique together",
        failed_count=0,
        total_count=total_count,
    )


def _post_write_history_completeness(
    cursor: Any,
    table_name: str,
    periods: list[Period],
    effective_date: date | None,
    effective_from_col: str,
) -> ValidationResult:
    """Validate that the SCD2 table contains rows for all declared periods.

    Queries distinct ``effective_from`` values from the table and compares
    against the contract's period manifest. Smart about partial loading:
    passes during mid-sequence loads, fails only on the final period if
    earlier periods are missing.

    Args:
        cursor: Database cursor within the open transaction.
        table_name: Fully-qualified table name.
        periods: List of Period definitions from the contract.
        effective_date: The effective_date used for the current write, or
            None for a regular (non-historical) write.
        effective_from_col: Name of the effective_from column in the table.
    """
    cursor.execute(
        f'SELECT DISTINCT "{effective_from_col}" FROM {table_name}',  # noqa: S608
    )
    present_dates: set[date] = {row[0] for row in cursor.fetchall() if row[0] is not None}

    expected_dates = [p.effective_from for p in periods]
    loaded = [d for d in expected_dates if d in present_dates]
    missing = [d for d in expected_dates if d not in present_dates]

    total_periods = len(expected_dates)
    loaded_count = len(loaded)

    if not missing:
        return ValidationResult(
            passed=True,
            message=(
                f"History complete: all {total_periods} periods present "
                f"({', '.join(d.isoformat() for d in expected_dates)})"
            ),
            failed_count=0,
            total_count=total_periods,
        )

    # Determine if this is the final period load
    is_final = effective_date is not None and effective_date == expected_dates[-1]
    # Also treat non-historical writes (effective_date=None) as final checks
    is_final = is_final or effective_date is None

    missing_strs = [d.isoformat() for d in missing]

    if not is_final:
        # Mid-sequence: pass with progress info
        return ValidationResult(
            passed=True,
            message=(
                f"History in progress: {loaded_count}/{total_periods} periods loaded. "
                f"Missing: {missing_strs}"
            ),
            failed_count=len(missing),
            total_count=total_periods,
        )

    # Final period or regular write: fail if incomplete
    return ValidationResult(
        passed=False,
        message=(
            f"History incomplete: {loaded_count}/{total_periods} periods present. "
            f"Missing periods: {missing_strs}"
        ),
        failed_count=len(missing),
        total_count=total_periods,
    )


def run_post_write_expectations(
    cursor: Any,
    table_name: str,
    expectations: Sequence[Any],
    lineage_id: str | None = None,
    total_rows: int | None = None,
    is_scd2: bool = False,
    periods: list[Period] | None = None,
    effective_date: date | None = None,
    effective_from_col: str = "effective_from",
) -> list[tuple[Any, ValidationResult]]:
    """Run table expectations against the database after all batches are written.

    For batched writes, table expectations (row_count, freshness, null_percentage,
    unique_combination) cannot be validated on a single batch. This function runs
    them via SQL against the actual written data within the open transaction.

    Queries are scoped by ``_lineage_id`` when available, validating only the rows
    written in this run. When lineage is unavailable, queries are unscoped.

    For SCD2 writes, ``row_count`` uses the accumulated ``total_rows`` counter
    (rows staged across all batches) rather than querying the merged table,
    since the staging table is dropped after merge and the merged table includes
    historical records.

    Args:
        cursor: Database cursor within the open transaction.
        table_name: Fully-qualified table name (e.g., ``"silver"."customers"``).
        expectations: List of ``TableExpectation`` objects from the contract.
        lineage_id: Lineage UUID for scoping queries. ``None`` falls back to
            unscoped queries.
        total_rows: Accumulated row count from batch iteration. Used for SCD2
            row_count validation.
        is_scd2: Whether this is an SCD2 write mode.
        periods: Contract period definitions for history_completeness validation.
        effective_date: The effective_date used for the current write, for
            determining whether this is the final period in a historical load.
        effective_from_col: Name of the effective_from column in the table.

    Returns:
        List of (TableExpectation, ValidationResult) tuples.
    """
    results: list[tuple[Any, ValidationResult]] = []

    for exp in expectations:
        exp_type = exp.expectation_type
        params = exp.parameters

        if exp_type == "row_count":
            result = _post_write_row_count(
                cursor, table_name, params, lineage_id, total_rows, is_scd2
            )
        elif exp_type == "freshness":
            result = _post_write_freshness(cursor, table_name, params, lineage_id)
        elif exp_type == "null_percentage":
            result = _post_write_null_percentage(cursor, table_name, params, lineage_id)
        elif exp_type == "unique_combination":
            result = _post_write_unique_combination(cursor, table_name, params, lineage_id)
        elif exp_type == "history_completeness":
            if periods:
                result = _post_write_history_completeness(
                    cursor, table_name, periods, effective_date, effective_from_col
                )
            else:
                result = ValidationResult(
                    passed=True,
                    message="history_completeness skipped: no periods defined in contract",
                    failed_count=0,
                    total_count=0,
                )
        else:
            result = ValidationResult(
                passed=False,
                message=f"Unknown expectation type: {exp_type}",
                failed_count=1,
                total_count=1,
            )

        results.append((exp, result))

    return results
