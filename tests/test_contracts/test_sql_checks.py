"""Tests for SQL-pushdown contract validation (sql_checks.py).

Focuses on SCD2 current-row scoping (issue #418): when ``current_col``
is set, every check query must restrict to rows where that boolean
column is TRUE so history rows don't fail snapshot-semantics checks.
"""

from datetime import datetime

import pytest

from moncpipelib.contracts.sql_checks import (
    _and_scope,
    _compose_where,
    _scope_predicate,
    run_column_test_sql,
    run_table_expectation_sql,
    sql_accepted_values,
    sql_freshness,
    sql_greater_than,
    sql_not_null,
    sql_null_percentage,
    sql_row_count,
    sql_unique,
    sql_unique_combination,
)

SCOPE = '"is_current" = TRUE'


class _CaptureCursor:
    """Fake cursor that records normalized SQL and feeds queued results."""

    def __init__(self, fetchone_values=None, fetchall_value=None):
        self.queries: list[str] = []
        self.params: list = []
        self._fetchone_values = list(fetchone_values or [])
        self._fetchall_value = fetchall_value or []

    def execute(self, sql, params=None):
        self.queries.append(" ".join(sql.split()))
        self.params.append(params)

    def fetchone(self):
        return self._fetchone_values.pop(0)

    def fetchall(self):
        return self._fetchall_value


class TestScopeHelpers:
    """Tests for the scope predicate builders."""

    def test_scope_predicate_none_is_empty(self):
        assert _scope_predicate(None) == ""

    def test_scope_predicate_quotes_column(self):
        assert _scope_predicate("is_current") == SCOPE

    def test_scope_predicate_rejects_unsafe_identifier(self):
        with pytest.raises(ValueError, match="Invalid SQL"):
            _scope_predicate('is_current"; DROP TABLE x;--')

    def test_compose_where_empty(self):
        assert _compose_where("", "") == ""

    def test_compose_where_single(self):
        assert _compose_where("", SCOPE) == f"WHERE {SCOPE}"

    def test_compose_where_joins_with_and(self):
        assert _compose_where('"c" IS NOT NULL', SCOPE) == f'WHERE "c" IS NOT NULL AND {SCOPE}'

    def test_and_scope(self):
        assert _and_scope(None) == ""
        assert _and_scope("is_current") == f" AND {SCOPE}"


class TestUniqueScoping:
    """sql_unique with and without current-row scoping."""

    def test_unscoped_has_no_where(self):
        cursor = _CaptureCursor(fetchone_values=[(3, 3)])
        result = sql_unique(cursor, "ref.t", "bla_number")
        assert result.passed
        assert "WHERE" not in cursor.queries[0]

    def test_scoped_filters_to_current(self):
        cursor = _CaptureCursor(fetchone_values=[(3, 3)])
        result = sql_unique(cursor, "ref.t", "bla_number", current_col="is_current")
        assert result.passed
        assert f"WHERE {SCOPE}" in cursor.queries[0]

    def test_scoped_combines_with_when_not_null(self):
        cursor = _CaptureCursor(fetchone_values=[(3, 3)])
        sql_unique(cursor, "ref.t", "bla_number", when="not_null", current_col="is_current")
        assert f'WHERE "bla_number" IS NOT NULL AND {SCOPE}' in cursor.queries[0]

    def test_scoped_failure_sample_query_is_scoped(self):
        cursor = _CaptureCursor(fetchone_values=[(2, 3)], fetchall_value=[("017016",)])
        result = sql_unique(cursor, "ref.t", "bla_number", current_col="is_current")
        assert not result.passed
        assert result.failed_count == 1
        assert all(f"WHERE {SCOPE}" in q for q in cursor.queries)


class TestColumnTestScoping:
    """Scoping applied across representative column test validators."""

    def test_not_null_scoped(self):
        cursor = _CaptureCursor(fetchone_values=[(0, 10)])
        result = sql_not_null(cursor, "ref.t", "c", current_col="is_current")
        assert result.passed
        assert f"WHERE {SCOPE}" in cursor.queries[0]

    def test_not_null_unscoped(self):
        cursor = _CaptureCursor(fetchone_values=[(0, 10)])
        sql_not_null(cursor, "ref.t", "c")
        assert "is_current" not in cursor.queries[0]

    def test_accepted_values_scopes_all_queries(self):
        cursor = _CaptureCursor(fetchone_values=[(1,), (10,)], fetchall_value=[("z",)])
        result = sql_accepted_values(cursor, "ref.t", "c", ["a", "b"], current_col="is_current")
        assert not result.passed
        # Main count and sample carry AND-scope; total count carries WHERE-scope.
        assert f"AND {SCOPE}" in cursor.queries[0]
        assert f"WHERE {SCOPE}" in cursor.queries[1]
        assert f"AND {SCOPE}" in cursor.queries[2]

    def test_comparison_scopes_all_queries(self):
        cursor = _CaptureCursor(fetchone_values=[(1,), (10,)], fetchall_value=[(0,)])
        result = sql_greater_than(cursor, "ref.t", "c", 0, current_col="is_current")
        assert not result.passed
        assert f"AND {SCOPE}" in cursor.queries[0]
        assert f"WHERE {SCOPE}" in cursor.queries[1]
        assert f"AND {SCOPE}" in cursor.queries[2]

    def test_dispatcher_forwards_current_col(self):
        cursor = _CaptureCursor(fetchone_values=[(3, 3)])
        run_column_test_sql(cursor, "ref.t", "bla_number", "unique", {}, current_col="is_current")
        assert f"WHERE {SCOPE}" in cursor.queries[0]

    def test_dispatcher_unscoped_by_default(self):
        cursor = _CaptureCursor(fetchone_values=[(3, 3)])
        run_column_test_sql(cursor, "ref.t", "bla_number", "unique", {})
        assert "is_current" not in cursor.queries[0]


class TestTableExpectationScoping:
    """Scoping applied to table expectation validators."""

    def test_row_count_scoped(self):
        cursor = _CaptureCursor(fetchone_values=[(5,)])
        result = sql_row_count(cursor, "ref.t", min_count=1, current_col="is_current")
        assert result.passed
        assert f"WHERE {SCOPE}" in cursor.queries[0]

    def test_unique_combination_scoped(self):
        cursor = _CaptureCursor(fetchone_values=[(10, 10)])
        result = sql_unique_combination(cursor, "ref.t", ["a", "b"], current_col="is_current")
        assert result.passed
        assert f"WHERE {SCOPE}" in cursor.queries[0]

    def test_freshness_scoped(self):
        cursor = _CaptureCursor(fetchone_values=[(datetime(2026, 7, 8), 0.5)])
        result = sql_freshness(cursor, "ref.t", "updated_at", 24, current_col="is_current")
        assert result.passed
        assert f"AND {SCOPE}" in cursor.queries[0]

    def test_null_percentage_scoped(self):
        cursor = _CaptureCursor(fetchone_values=[(0, 10)])
        result = sql_null_percentage(cursor, "ref.t", "c", 5.0, current_col="is_current")
        assert result.passed
        assert f"WHERE {SCOPE}" in cursor.queries[0]

    def test_dispatcher_forwards_current_col(self):
        cursor = _CaptureCursor(fetchone_values=[(5,)])
        run_table_expectation_sql(
            cursor, "ref.t", "row_count", {"min": 1}, current_col="is_current"
        )
        assert f"WHERE {SCOPE}" in cursor.queries[0]
