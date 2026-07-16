"""Typed column test and table expectation classes.

This module provides strongly-typed classes for defining column tests
and table expectations, with explicit parameter requirements and
IDE autocomplete support.

Example:
    ```python
    from moncpipelib.contracts.checks_types import (
        NotNull,
        Unique,
        AcceptedValues,
        Pattern,
        GreaterThan,
        RowCount,
        UniqueCombination,
    )
    from moncpipelib.contracts import Column, ColumnType, DataContract, Schema

    columns = [
        Column(
            name="order_id",
            type=ColumnType.STRING,
            nullable=False,
            tests=[NotNull(), Unique()],
        ),
        Column(
            name="status",
            type=ColumnType.STRING,
            nullable=False,
            tests=[AcceptedValues(values=["pending", "completed", "cancelled"])],
        ),
        Column(
            name="amount",
            type=ColumnType.DECIMAL,
            nullable=False,
            tests=[GreaterThan(threshold=0)],
        ),
    ]

    contract = DataContract(
        version="1.0",
        asset="orders",
        layer="bronze",
        schema=Schema(columns=columns),
        expectations=[
            RowCount(min=1, max=1000000),
            UniqueCombination(columns=["order_id", "order_date"]),
        ],
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from moncpipelib.contracts.models import Severity

# =============================================================================
# Base Protocol for type checking
# =============================================================================

# These classes act as ColumnTest and TableExpectation compatible objects
# They have the same attributes but don't inherit to avoid dataclass issues


# =============================================================================
# Column Test Types
# =============================================================================


@dataclass
class NotNull:
    """Validates that column values are not null.

    Example:
        Column(name="id", type=ColumnType.STRING, nullable=False, tests=[NotNull()])
    """

    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["not_null"] = field(default="not_null", init=False)
    parameters: dict[str, Any] = field(default_factory=dict, init=False)


@dataclass
class Unique:
    """Validates that column values are unique (no duplicates).

    Example:
        Column(name="id", type=ColumnType.STRING, nullable=False, tests=[Unique()])
    """

    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["unique"] = field(default="unique", init=False)
    parameters: dict[str, Any] = field(default_factory=dict, init=False)


@dataclass
class AcceptedValues:
    """Validates that column values are in an allowed set.

    Args:
        values: List of allowed values

    Example:
        Column(
            name="status",
            type=ColumnType.STRING,
            nullable=False,
            tests=[AcceptedValues(values=["pending", "completed", "cancelled"])],
        )
    """

    values: list[str | int | float | bool]
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["accepted_values"] = field(default="accepted_values", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"values": self.values}


@dataclass
class NotIn:
    """Validates that column values are not in a rejected set.

    Args:
        values: List of rejected values that should not appear.

    Example:
        Column(
            name="match_score",
            type=ColumnType.INTEGER,
            nullable=False,
            tests=[NotIn(values=[-1, -999])],
        )
    """

    values: list[str | int | float | bool]
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["not_in"] = field(default="not_in", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"values": self.values}


@dataclass
class Pattern:
    """Validates that string column values match a regex pattern.

    Args:
        regex: Regular expression pattern to match

    Example:
        Column(
            name="email",
            type=ColumnType.STRING,
            nullable=True,
            tests=[Pattern(regex=r"^[^@]+@[^@]+\\.[^@]+$")],
        )
    """

    regex: str
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["pattern"] = field(default="pattern", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"regex": self.regex}


@dataclass
class GreaterThan:
    """Validates that numeric column values are greater than a threshold.

    Args:
        threshold: Minimum value (exclusive)

    Example:
        Column(
            name="amount",
            type=ColumnType.DECIMAL,
            nullable=False,
            tests=[GreaterThan(threshold=0)],
        )
    """

    threshold: int | float
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["greater_than"] = field(default="greater_than", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"threshold": self.threshold}


@dataclass
class GreaterThanOrEqual:
    """Validates that numeric column values are greater than or equal to a threshold.

    Args:
        threshold: Minimum value (inclusive)

    Example:
        Column(
            name="quantity",
            type=ColumnType.INTEGER,
            nullable=False,
            tests=[GreaterThanOrEqual(threshold=0)],
        )
    """

    threshold: int | float
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["greater_than_or_equal"] = field(default="greater_than_or_equal", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"threshold": self.threshold}


@dataclass
class LessThan:
    """Validates that numeric column values are less than a threshold.

    Args:
        threshold: Maximum value (exclusive)

    Example:
        Column(
            name="discount_percent",
            type=ColumnType.DECIMAL,
            nullable=False,
            tests=[LessThan(threshold=100)],
        )
    """

    threshold: int | float
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["less_than"] = field(default="less_than", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"threshold": self.threshold}


@dataclass
class LessThanOrEqual:
    """Validates that numeric column values are less than or equal to a threshold.

    Args:
        threshold: Maximum value (inclusive)

    Example:
        Column(
            name="rating",
            type=ColumnType.INTEGER,
            nullable=False,
            tests=[LessThanOrEqual(threshold=5)],
        )
    """

    threshold: int | float
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["less_than_or_equal"] = field(default="less_than_or_equal", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"threshold": self.threshold}


@dataclass
class Between:
    """Validates that numeric column values are within a range (inclusive).

    Args:
        min: Minimum value (inclusive)
        max: Maximum value (inclusive)

    Example:
        Column(
            name="percentage",
            type=ColumnType.DECIMAL,
            nullable=False,
            tests=[Between(min=0, max=100)],
        )
    """

    min: int | float
    max: int | float
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["between"] = field(default="between", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"min": self.min, "max": self.max}


@dataclass
class MinLength:
    """Validates that string column values have at least a minimum length.

    Args:
        length: Minimum string length (inclusive)

    Example:
        Column(
            name="code",
            type=ColumnType.STRING,
            nullable=False,
            tests=[MinLength(length=3)],
        )
    """

    length: int
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["min_length"] = field(default="min_length", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"length": self.length}


@dataclass
class MaxLength:
    """Validates that string column values don't exceed a maximum length.

    Args:
        length: Maximum string length (inclusive)

    Example:
        Column(
            name="abbreviation",
            type=ColumnType.STRING,
            nullable=False,
            tests=[MaxLength(length=10)],
        )
    """

    length: int
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["max_length"] = field(default="max_length", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"length": self.length}


@dataclass
class NotInFuture:
    """Validates that date/datetime column values are not in the future.

    Example:
        Column(
            name="birth_date",
            type=ColumnType.DATE,
            nullable=True,
            tests=[NotInFuture()],
        )
    """

    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["not_in_future"] = field(default="not_in_future", init=False)
    parameters: dict[str, Any] = field(default_factory=dict, init=False)


@dataclass
class WithinDays:
    """Validates that date column values are within N days of today.

    Args:
        days: Maximum number of days from today

    Example:
        Column(
            name="last_activity",
            type=ColumnType.DATE,
            nullable=True,
            tests=[WithinDays(days=30)],
        )
    """

    days: int
    severity: Severity = Severity.ERROR
    when: str | None = None
    test_type: Literal["within_days"] = field(default="within_days", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"days": self.days}


# =============================================================================
# Table Expectation Types
# =============================================================================


@dataclass
class RowCount:
    """Validates that the table row count is within a specified range.

    Args:
        min: Minimum row count (optional)
        max: Maximum row count (optional)

    Example:
        contract = DataContract(
            ...,
            expectations=[RowCount(min=1, max=1000000)],
        )
    """

    min: int | None = None
    max: int | None = None
    severity: Severity = Severity.ERROR
    expectation_type: Literal["row_count"] = field(default="row_count", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.min is not None:
            params["min"] = self.min
        if self.max is not None:
            params["max"] = self.max
        return params


@dataclass
class Freshness:
    """Validates that the most recent value in a column is within a time threshold.

    Args:
        column: Column name to check for freshness
        max_age_hours: Maximum age in hours

    Example:
        contract = DataContract(
            ...,
            expectations=[Freshness(column="updated_at", max_age_hours=24)],
        )
    """

    column: str
    max_age_hours: int
    severity: Severity = Severity.ERROR
    expectation_type: Literal["freshness"] = field(default="freshness", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"column": self.column, "max_age_hours": self.max_age_hours}


@dataclass
class NullPercentage:
    """Validates that the null percentage for a column is below a threshold.

    Args:
        column: Column name to check
        max_percent: Maximum null percentage (0-100)

    Example:
        contract = DataContract(
            ...,
            expectations=[NullPercentage(column="email", max_percent=5.0)],
        )
    """

    column: str
    max_percent: float
    severity: Severity = Severity.ERROR
    expectation_type: Literal["null_percentage"] = field(default="null_percentage", init=False)

    @property
    def parameters(self) -> dict[str, Any]:
        return {"column": self.column, "max_percent": self.max_percent}


@dataclass
class UniqueCombination:
    """Validates that a combination of columns is unique across all rows.

    Args:
        columns: List of column names that together must be unique

    Example:
        contract = DataContract(
            ...,
            expectations=[UniqueCombination(columns=["order_id", "line_item"])],
        )
    """

    columns: list[str]
    severity: Severity = Severity.ERROR
    expectation_type: Literal["unique_combination"] = field(
        default="unique_combination", init=False
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"columns": self.columns}


# =============================================================================
# Type aliases for convenience
# =============================================================================

# All column test types
ColumnTestType = (
    NotNull
    | Unique
    | AcceptedValues
    | Pattern
    | GreaterThan
    | GreaterThanOrEqual
    | LessThan
    | LessThanOrEqual
    | Between
    | MinLength
    | MaxLength
    | NotInFuture
    | WithinDays
)

# All table expectation types
TableExpectationType = RowCount | Freshness | NullPercentage | UniqueCombination
