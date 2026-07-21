"""SQL-pushdown contract validation.

Executes contract checks as SQL queries directly in PostgreSQL instead of
loading the full table into Python. This eliminates OOM issues for large
tables -- checks run as COUNT/aggregate queries that return scalar results.

Each function mirrors its Polars counterpart in ``validators.py`` and
returns the same ``ValidationResult`` dataclass with identical message
formatting.

Security: Table and column identifiers are validated against a strict
allowlist pattern. All parameter values use ``%s`` parameterization
(psycopg handles escaping). No string interpolation of user-provided
values into SQL.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from moncpipelib.contracts.models import ValidationResult

if TYPE_CHECKING:
    import psycopg

_MAX_SAMPLE_VALUES = 5
"""Maximum number of sample failure values to include in messages."""

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")
"""Regex for valid SQL identifiers (tables, columns, schemas)."""


def _validate_identifier(name: str, kind: str = "identifier") -> None:
    """Validate a SQL identifier against injection.

    Raises:
        ValueError: If the identifier contains unsafe characters.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL {kind}: {name!r}")


def _quote_column(column: str) -> str:
    """Quote a column name for safe SQL usage."""
    _validate_identifier(column, "column name")
    return f'"{column}"'


def _where_not_null(column: str) -> str:
    """Build a WHERE clause fragment for the 'when: not_null' condition."""
    return f"{_quote_column(column)} IS NOT NULL"


def _scope_predicate(current_col: str | None) -> str:
    """Build the SCD2 current-row scope predicate ('' when unscoped).

    ``current_col`` is validated as a SQL identifier before interpolation;
    the resulting predicate restricts every check query to rows where the
    boolean current-flag column is TRUE. SCD2 tables legitimately repeat
    business keys across history rows (issue #418), so unscoped full-table
    checks would fail ``unique`` on the first change wave and degrade every
    other test type as history accrues.
    """
    if current_col is None:
        return ""
    return f"{_quote_column(current_col)} = TRUE"


def _compose_where(*predicates: str) -> str:
    """Join non-empty predicates into a WHERE clause ('' when none)."""
    parts = [p for p in predicates if p]
    return f"WHERE {' AND '.join(parts)}" if parts else ""


def _and_scope(current_col: str | None) -> str:
    """Render the scope predicate as an ``AND ...`` suffix ('' when unscoped).

    For queries whose WHERE clause always starts with a column filter
    (``WHERE col IS NOT NULL AND ...``); bare-table queries use
    ``_compose_where`` instead.
    """
    scope = _scope_predicate(current_col)
    return f" AND {scope}" if scope else ""


def _redact_pii_message(message: str) -> str:
    """Replace sample values in a message with a PII redaction notice."""
    return re.sub(
        r"(Sample[^:]*:) \[.*\]",
        r"\1 [REDACTED - column marked as PII]",
        message,
    )


# ---------------------------------------------------------------------------
# Column test validators (SQL pushdown)
# ---------------------------------------------------------------------------


def sql_not_null(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column contains no null values via SQL COUNT."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    if when == "not_null":
        # when=not_null on not_null check is a no-op (filter to non-null, then check null)
        return ValidationResult(passed=True, message=f"Column '{column}' has no null values")

    cursor.execute(
        f"SELECT COUNT(*) FILTER (WHERE {col} IS NULL) AS null_count, "  # noqa: S608
        f"COUNT(*) AS total_count FROM {table} {_compose_where(_scope_predicate(current_col))}"
    )
    row = cursor.fetchone()
    assert row is not None
    null_count, total_count = row

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


def sql_unique(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column contains only unique values via SQL COUNT DISTINCT."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    where = _compose_where(
        _where_not_null(column) if when == "not_null" else "",
        _scope_predicate(current_col),
    )
    cursor.execute(
        f"SELECT COUNT(DISTINCT {col}) AS unique_count, "  # noqa: S608
        f"COUNT(*) AS total_count FROM {table} {where}"
    )
    row = cursor.fetchone()
    assert row is not None
    unique_count, total_count = row
    duplicate_count = total_count - unique_count

    if duplicate_count > 0:
        cursor.execute(
            f"SELECT {col}, COUNT(*) AS cnt FROM {table} {where} "  # noqa: S608
            f"GROUP BY {col} HAVING COUNT(*) > 1 LIMIT {_MAX_SAMPLE_VALUES}"
        )
        sample_values = [r[0] for r in cursor.fetchall()]
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {duplicate_count} duplicate values. "
                f"Sample duplicates: {sample_values}"
            ),
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


def sql_accepted_values(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    values: list[Any],
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column values are in accepted list via SQL NOT IN."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    placeholders = ", ".join(["%s"] * len(values))
    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND {col} NOT IN ({placeholders}) {when_clause}{scope_clause}",
        values,
    )
    invalid_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if invalid_count > 0:
        cursor.execute(
            f"SELECT DISTINCT {col} FROM {table} "  # noqa: S608
            f"WHERE {col} IS NOT NULL AND {col} NOT IN ({placeholders}) "
            f"{when_clause}{scope_clause} LIMIT {_MAX_SAMPLE_VALUES}",
            values,
        )
        sample_values = [r[0] for r in cursor.fetchall()]
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {invalid_count} values not in accepted list "
                f"{values}. Sample invalid values: {sample_values}"
            ),
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all in accepted list",
        failed_count=0,
        total_count=total_count,
    )


def sql_not_in(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    values: list[Any],
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column does not contain any rejected values via SQL IN."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    placeholders = ", ".join(["%s"] * len(values))
    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND {col} IN ({placeholders}) {when_clause}{scope_clause}",
        values,
    )
    rejected_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if rejected_count > 0:
        cursor.execute(
            f"SELECT DISTINCT {col} FROM {table} "  # noqa: S608
            f"WHERE {col} IS NOT NULL AND {col} IN ({placeholders}) "
            f"{when_clause}{scope_clause} LIMIT {_MAX_SAMPLE_VALUES}",
            values,
        )
        sample_values = [r[0] for r in cursor.fetchall()]
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


def sql_pattern(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    pattern: str,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column values match a regex pattern via PostgreSQL ~ operator."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND {col}::text !~ %s {when_clause}{scope_clause}",
        (pattern,),
    )
    invalid_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if invalid_count > 0:
        cursor.execute(
            f"SELECT DISTINCT {col} FROM {table} "  # noqa: S608
            f"WHERE {col} IS NOT NULL AND {col}::text !~ %s "
            f"{when_clause}{scope_clause} LIMIT {_MAX_SAMPLE_VALUES}",
            (pattern,),
        )
        sample_values = [r[0] for r in cursor.fetchall()]
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {invalid_count} values not matching pattern "
                f"'{pattern}'. Sample invalid values: {sample_values}"
            ),
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values all match pattern '{pattern}'",
        failed_count=0,
        total_count=total_count,
    )


def _sql_comparison(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    inverse_op: str,
    value: Any,
    op_name: str,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Generic comparison check (greater_than, less_than, etc.)."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND {col} {inverse_op} %s {when_clause}{scope_clause}",
        (value,),
    )
    invalid_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if invalid_count > 0:
        cursor.execute(
            f"SELECT {col} FROM {table} "  # noqa: S608
            f"WHERE {col} IS NOT NULL AND {col} {inverse_op} %s "
            f"{when_clause}{scope_clause} LIMIT {_MAX_SAMPLE_VALUES}",
            (value,),
        )
        sample_values = [r[0] for r in cursor.fetchall()]
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {invalid_count} values not {op_name} {value}. "
                f"Sample invalid values: {sample_values}"
            ),
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all {op_name} {value}",
        failed_count=0,
        total_count=total_count,
    )


def sql_greater_than(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    value: Any,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column values are greater than threshold."""
    return _sql_comparison(
        cursor, table, column, "<=", value, "greater than", when=when, current_col=current_col
    )


def sql_greater_than_or_equal(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    value: Any,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column values are >= threshold."""
    return _sql_comparison(
        cursor,
        table,
        column,
        "<",
        value,
        "greater than or equal to",
        when=when,
        current_col=current_col,
    )


def sql_less_than(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    value: Any,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column values are less than threshold."""
    return _sql_comparison(
        cursor, table, column, ">=", value, "less than", when=when, current_col=current_col
    )


def sql_less_than_or_equal(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    value: Any,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column values are <= threshold."""
    return _sql_comparison(
        cursor,
        table,
        column,
        ">",
        value,
        "less than or equal to",
        when=when,
        current_col=current_col,
    )


def sql_between(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    min_value: Any,
    max_value: Any,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column values are between min and max (inclusive)."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND ({col} < %s OR {col} > %s) {when_clause}{scope_clause}",
        (min_value, max_value),
    )
    invalid_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if invalid_count > 0:
        cursor.execute(
            f"SELECT {col} FROM {table} "  # noqa: S608
            f"WHERE {col} IS NOT NULL AND ({col} < %s OR {col} > %s) "
            f"{when_clause}{scope_clause} LIMIT {_MAX_SAMPLE_VALUES}",
            (min_value, max_value),
        )
        sample_values = [r[0] for r in cursor.fetchall()]
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {invalid_count} values outside range "
                f"[{min_value}, {max_value}]. Sample invalid values: {sample_values}"
            ),
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all between {min_value} and {max_value}",
        failed_count=0,
        total_count=total_count,
    )


def sql_not_in_future(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column values are not in the future via CURRENT_TIMESTAMP."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND {col} > CURRENT_TIMESTAMP {when_clause}{scope_clause}"
    )
    invalid_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if invalid_count > 0:
        cursor.execute(
            f"SELECT {col} FROM {table} "  # noqa: S608
            f"WHERE {col} IS NOT NULL AND {col} > CURRENT_TIMESTAMP "
            f"{when_clause}{scope_clause} LIMIT {_MAX_SAMPLE_VALUES}"
        )
        sample_values = [str(r[0]) for r in cursor.fetchall()]
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {invalid_count} future values. "
                f"Sample invalid values: {sample_values}"
            ),
            failed_count=invalid_count,
            total_count=total_count,
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' has no future values",
        failed_count=0,
        total_count=total_count,
    )


def sql_within_days(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    days: int,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column date values are within N days of today."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND "
        f"({col}::date < CURRENT_DATE - %s::int OR {col}::date > CURRENT_DATE) "
        f"{when_clause}{scope_clause}",
        (days,),
    )
    invalid_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if invalid_count > 0:
        return ValidationResult(
            passed=False,
            message=(f"Column '{column}' has {invalid_count} values outside the last {days} days"),
            failed_count=invalid_count,
            total_count=total_count,
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values are all within the last {days} days",
        failed_count=0,
        total_count=total_count,
    )


def sql_min_length(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    length: int,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column string values have minimum length."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND LENGTH({col}::text) < %s {when_clause}{scope_clause}",
        (length,),
    )
    invalid_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if invalid_count > 0:
        cursor.execute(
            f"SELECT {col} FROM {table} "  # noqa: S608
            f"WHERE {col} IS NOT NULL AND LENGTH({col}::text) < %s "
            f"{when_clause}{scope_clause} LIMIT {_MAX_SAMPLE_VALUES}",
            (length,),
        )
        sample_values = [r[0] for r in cursor.fetchall()]
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {invalid_count} values shorter than "
                f"{length} characters. Sample invalid values: {sample_values}"
            ),
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values all have at least {length} characters",
        failed_count=0,
        total_count=total_count,
    )


def sql_max_length(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    length: int,
    *,
    when: str | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check column string values have maximum length."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    when_clause = f"AND {_where_not_null(column)}" if when == "not_null" else ""
    scope_clause = _and_scope(current_col)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {col} IS NOT NULL AND LENGTH({col}::text) > %s {when_clause}{scope_clause}",
        (length,),
    )
    invalid_count = cursor.fetchone()[0]  # type: ignore[index]

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    total_count = cursor.fetchone()[0]  # type: ignore[index]

    if invalid_count > 0:
        cursor.execute(
            f"SELECT {col} FROM {table} "  # noqa: S608
            f"WHERE {col} IS NOT NULL AND LENGTH({col}::text) > %s "
            f"{when_clause}{scope_clause} LIMIT {_MAX_SAMPLE_VALUES}",
            (length,),
        )
        sample_values = [r[0] for r in cursor.fetchall()]
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {invalid_count} values longer than "
                f"{length} characters. Sample invalid values: {sample_values}"
            ),
            failed_count=invalid_count,
            total_count=total_count,
            sample_failures=[{column: v} for v in sample_values],
        )

    return ValidationResult(
        passed=True,
        message=f"Column '{column}' values all have at most {length} characters",
        failed_count=0,
        total_count=total_count,
    )


# ---------------------------------------------------------------------------
# Table expectation validators (SQL pushdown)
# ---------------------------------------------------------------------------


def sql_row_count(
    cursor: psycopg.Cursor,
    table: str,
    *,
    min_count: int | None = None,
    max_count: int | None = None,
    current_col: str | None = None,
) -> ValidationResult:
    """Check table row count is within bounds."""
    _validate_identifier(table, "table name")

    cursor.execute(
        f"SELECT COUNT(*) FROM {table} {_compose_where(_scope_predicate(current_col))}"  # noqa: S608
    )
    row_count = cursor.fetchone()[0]  # type: ignore[index]

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


def sql_freshness(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    max_age_hours: float,
    *,
    current_col: str | None = None,
) -> ValidationResult:
    """Check that the most recent value in a timestamp column is within max_age_hours."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    cursor.execute(
        f"SELECT MAX({col}), "  # noqa: S608
        f"EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX({col}))) / 3600.0 "
        f"FROM {table} WHERE {col} IS NOT NULL{_and_scope(current_col)}"
    )
    row = cursor.fetchone()
    assert row is not None
    most_recent, age_hours = row

    if most_recent is None:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' has no non-null values for freshness check",
            failed_count=1,
            total_count=1,
        )

    if age_hours > max_age_hours:
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' most recent value is {age_hours:.1f} hours old "
                f"(max allowed: {max_age_hours}h). Most recent: {most_recent}"
            ),
            failed_count=1,
            total_count=1,
        )

    return ValidationResult(
        passed=True,
        message=(
            f"Column '{column}' is fresh ({age_hours:.1f}h old, max allowed: {max_age_hours}h)"
        ),
        failed_count=0,
        total_count=1,
    )


def sql_null_percentage(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    max_percent: float,
    *,
    current_col: str | None = None,
) -> ValidationResult:
    """Check that null percentage for a column is within threshold."""
    _validate_identifier(table, "table name")
    col = _quote_column(column)

    cursor.execute(
        f"SELECT "  # noqa: S608
        f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count, "
        f"COUNT(*) AS total_count "
        f"FROM {table} {_compose_where(_scope_predicate(current_col))}"
    )
    row = cursor.fetchone()
    assert row is not None
    null_count, total_count = row

    if total_count == 0:
        return ValidationResult(
            passed=True,
            message=f"Column '{column}' has 0 rows (no nulls possible)",
            failed_count=0,
            total_count=0,
        )

    null_percent = (null_count / total_count) * 100

    if null_percent > max_percent:
        return ValidationResult(
            passed=False,
            message=(
                f"Column '{column}' has {null_percent:.1f}% null values "
                f"({null_count}/{total_count}), exceeds maximum {max_percent}%"
            ),
            failed_count=null_count,
            total_count=total_count,
        )

    return ValidationResult(
        passed=True,
        message=(
            f"Column '{column}' null percentage {null_percent:.1f}% "
            f"is within threshold ({max_percent}%)"
        ),
        failed_count=0,
        total_count=total_count,
    )


def sql_unique_combination(
    cursor: psycopg.Cursor,
    table: str,
    columns: list[str],
    *,
    current_col: str | None = None,
) -> ValidationResult:
    """Check that the combination of columns is unique."""
    _validate_identifier(table, "table name")
    col_list = ", ".join(_quote_column(c) for c in columns)
    col_names = ", ".join(columns)

    where = _compose_where(_scope_predicate(current_col))
    cursor.execute(
        f"SELECT COUNT(*) AS total, "  # noqa: S608
        f"COUNT(DISTINCT ({col_list})) AS unique_count "
        f"FROM {table} {where}"
    )
    row = cursor.fetchone()
    assert row is not None
    total_count, unique_count = row
    duplicate_count = total_count - unique_count

    if duplicate_count > 0:
        cursor.execute(
            f"SELECT {col_list}, COUNT(*) AS cnt FROM {table} {where} "  # noqa: S608
            f"GROUP BY {col_list} HAVING COUNT(*) > 1 "
            f"LIMIT {_MAX_SAMPLE_VALUES}"
        )
        sample_rows = cursor.fetchall()
        sample_values = [dict(zip(columns, row[:-1], strict=False)) for row in sample_rows]
        return ValidationResult(
            passed=False,
            message=(
                f"Columns ({col_names}) have {duplicate_count} duplicate combinations. "
                f"Sample duplicates: {sample_values}"
            ),
            failed_count=duplicate_count,
            total_count=total_count,
            sample_failures=sample_values,
        )

    return ValidationResult(
        passed=True,
        message=f"Columns ({col_names}) are all unique combinations",
        failed_count=0,
        total_count=total_count,
    )


# ---------------------------------------------------------------------------
# Dispatchers (mirror run_column_test / run_table_expectation from validators.py)
# ---------------------------------------------------------------------------


def run_column_test_sql(
    cursor: psycopg.Cursor,
    table: str,
    column: str,
    test_type: str,
    parameters: dict[str, Any],
    *,
    when: str | None = None,
    pii: bool = False,
    current_col: str | None = None,
) -> ValidationResult:
    """Run a single column test as a SQL query.

    Mirrors ``validators.run_column_test`` but executes in PostgreSQL.

    ``current_col`` scopes every query to rows where that boolean column
    is TRUE. Set for SCD2 sinks so tests validate the current snapshot
    rather than sweeping expired history rows (issue #418).
    """
    validators: dict[str, Callable[[], ValidationResult]] = {
        "not_null": lambda: sql_not_null(cursor, table, column, when=when, current_col=current_col),
        "unique": lambda: sql_unique(cursor, table, column, when=when, current_col=current_col),
        "accepted_values": lambda: sql_accepted_values(
            cursor,
            table,
            column,
            parameters.get("values", parameters.get("value", [])),
            when=when,
            current_col=current_col,
        ),
        "not_in": lambda: sql_not_in(
            cursor,
            table,
            column,
            parameters.get("values", parameters.get("value", [])),
            when=when,
            current_col=current_col,
        ),
        "pattern": lambda: sql_pattern(
            cursor,
            table,
            column,
            parameters.get("value", ""),
            when=when,
            current_col=current_col,
        ),
        "greater_than": lambda: sql_greater_than(
            cursor,
            table,
            column,
            parameters.get("value"),
            when=when,
            current_col=current_col,
        ),
        "greater_than_or_equal": lambda: sql_greater_than_or_equal(
            cursor,
            table,
            column,
            parameters.get("value"),
            when=when,
            current_col=current_col,
        ),
        "less_than": lambda: sql_less_than(
            cursor,
            table,
            column,
            parameters.get("value"),
            when=when,
            current_col=current_col,
        ),
        "less_than_or_equal": lambda: sql_less_than_or_equal(
            cursor,
            table,
            column,
            parameters.get("value"),
            when=when,
            current_col=current_col,
        ),
        "between": lambda: sql_between(
            cursor,
            table,
            column,
            parameters.get("min"),
            parameters.get("max"),
            when=when,
            current_col=current_col,
        ),
        "not_in_future": lambda: sql_not_in_future(
            cursor, table, column, when=when, current_col=current_col
        ),
        "within_days": lambda: sql_within_days(
            cursor,
            table,
            column,
            parameters.get("days", 0),
            when=when,
            current_col=current_col,
        ),
        "min_length": lambda: sql_min_length(
            cursor,
            table,
            column,
            parameters.get("value", 0),
            when=when,
            current_col=current_col,
        ),
        "max_length": lambda: sql_max_length(
            cursor,
            table,
            column,
            parameters.get("value", 0),
            when=when,
            current_col=current_col,
        ),
    }

    validator = validators.get(test_type)
    if validator is None:
        return ValidationResult(
            passed=False,
            message=f"Unknown SQL test type: {test_type}",
            failed_count=1,
            total_count=1,
        )

    result = validator()

    if pii and not result.passed:
        result = ValidationResult(
            passed=result.passed,
            message=_redact_pii_message(result.message),
            failed_count=result.failed_count,
            total_count=result.total_count,
        )

    return result


def run_table_expectation_sql(
    cursor: psycopg.Cursor,
    table: str,
    expectation_type: str,
    parameters: dict[str, Any],
    *,
    current_col: str | None = None,
) -> ValidationResult:
    """Run a single table expectation as a SQL query.

    Mirrors ``validators.run_table_expectation`` but executes in PostgreSQL.

    ``current_col`` scopes every query to rows where that boolean column
    is TRUE. Set for SCD2 sinks so expectations validate the current
    snapshot rather than sweeping expired history rows (issue #418).
    """
    expectations: dict[str, Callable[[], ValidationResult]] = {
        "row_count": lambda: sql_row_count(
            cursor,
            table,
            min_count=parameters.get("min"),
            max_count=parameters.get("max"),
            current_col=current_col,
        ),
        "freshness": lambda: sql_freshness(
            cursor,
            table,
            parameters["column"],
            parameters.get("max_age_hours", 24),
            current_col=current_col,
        ),
        "null_percentage": lambda: sql_null_percentage(
            cursor,
            table,
            parameters["column"],
            parameters.get("max_percent", 100),
            current_col=current_col,
        ),
        "unique_combination": lambda: sql_unique_combination(
            cursor,
            table,
            parameters["columns"],
            current_col=current_col,
        ),
    }

    validator = expectations.get(expectation_type)
    if validator is None:
        return ValidationResult(
            passed=False,
            message=f"Unknown SQL expectation type: {expectation_type}",
            failed_count=1,
            total_count=1,
        )

    return validator()
