"""Tests for SQL sanitizer module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from moncpipelib.testing.sql_sanitizer import (
    SafeWhereClauseBuilder,
    SQLSafetyError,
    build_safe_table_copy,
    safe_copy_table,
)


class TestSQLSafetyError:
    """Tests for SQLSafetyError exception."""

    def test_inherits_from_exception(self):
        assert issubclass(SQLSafetyError, Exception)

    def test_message(self):
        err = SQLSafetyError("test message")
        assert str(err) == "test message"


class TestSafeWhereClauseBuilderBasic:
    """Tests for basic SafeWhereClauseBuilder behavior."""

    def test_empty_where_clause(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize("")
        assert clause == ""
        assert params == {}

    def test_whitespace_only_where_clause(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize("   ")
        assert clause == ""
        assert params == {}

    def test_simple_string_equality(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize("status = 'active'")
        assert "param_0" in params
        assert params["param_0"] == "active"
        # Clause should use psycopg2-style parameters
        assert "%(param_0)s" in clause

    def test_date_comparison(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize("date >= '2024-01-01'")
        assert params["param_0"] == "2024-01-01"

    def test_multiple_conditions(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize(
            "date >= '2024-01-01' AND status = 'active'"
        )
        assert len(params) == 2
        # Both values should be captured
        values = set(params.values())
        assert "2024-01-01" in values
        assert "active" in values

    def test_integer_literal(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize("count > 100")
        assert any(v == 100 for v in params.values())

    def test_or_condition(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize(
            "status = 'active' OR status = 'pending'"
        )
        assert len(params) == 2

    def test_in_clause(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize("status IN ('active', 'pending')")
        values = set(params.values())
        assert "active" in values
        assert "pending" in values

    def test_between_clause(self):
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize("amount BETWEEN 100 AND 500")
        values = set(params.values())
        assert 100 in values
        assert 500 in values


class TestSafeWhereClauseBuilderDangerousKeywords:
    """Tests for dangerous keyword detection."""

    def test_blocks_drop(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="DROP"):
            builder.validate_and_parameterize("1=1; DROP TABLE users")

    def test_blocks_delete(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="DELETE"):
            builder.validate_and_parameterize("1=1; DELETE FROM users")

    def test_blocks_union(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="UNION"):
            builder.validate_and_parameterize("1=1 UNION SELECT * FROM passwords")

    def test_blocks_insert(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="INSERT|INTO"):
            builder.validate_and_parameterize("1=1; INSERT INTO x VALUES(1)")

    def test_blocks_truncate(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="TRUNCATE"):
            builder.validate_and_parameterize("1=1; TRUNCATE users")

    def test_blocks_update(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="UPDATE"):
            builder.validate_and_parameterize("1=1; UPDATE users SET admin = true")

    def test_blocks_alter(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="ALTER"):
            builder.validate_and_parameterize("1=1; ALTER TABLE users ADD col text")

    def test_blocks_create(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="CREATE"):
            builder.validate_and_parameterize("1=1; CREATE TABLE evil (id int)")

    def test_blocks_grant(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="GRANT"):
            builder.validate_and_parameterize("1=1; GRANT ALL ON users TO public")

    def test_blocks_execute(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="EXECUTE"):
            builder.validate_and_parameterize("1=1; EXECUTE dangerous_func()")

    def test_blocks_into(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="INTO"):
            builder.validate_and_parameterize("1=1 INTO OUTFILE '/etc/passwd'")

    def test_keyword_case_insensitive(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="DROP"):
            builder.validate_and_parameterize("1=1; drop table users")

    def test_keyword_mixed_case(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="DROP"):
            builder.validate_and_parameterize("1=1; DrOp table users")


class TestSafeWhereClauseBuilderColumnValidation:
    """Tests for column allowlist validation."""

    def test_allowed_columns_pass(self):
        builder = SafeWhereClauseBuilder(allowed_columns=["date", "status"])
        clause, params = builder.validate_and_parameterize(
            "date >= '2024-01-01' AND status = 'active'"
        )
        assert len(params) == 2

    def test_disallowed_column_rejected(self):
        builder = SafeWhereClauseBuilder(allowed_columns=["date"])
        with pytest.raises(SQLSafetyError, match="not in allowed columns"):
            builder.validate_and_parameterize("secret_column = 'value'")

    def test_no_column_restriction_when_none(self):
        builder = SafeWhereClauseBuilder(allowed_columns=None)
        clause, params = builder.validate_and_parameterize("any_column = 'value'")
        assert len(params) == 1

    def test_column_case_insensitive(self):
        builder = SafeWhereClauseBuilder(allowed_columns=["Status"])
        clause, params = builder.validate_and_parameterize("status = 'active'")
        assert len(params) == 1

    def test_multiple_disallowed_columns(self):
        builder = SafeWhereClauseBuilder(allowed_columns=["date"])
        with pytest.raises(SQLSafetyError, match="not in allowed columns"):
            builder.validate_and_parameterize("status = 'active' AND secret = 'x'")


class TestSafeWhereClauseBuilderInvalidSQL:
    """Tests for invalid SQL detection."""

    def test_invalid_sql_syntax(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError, match="Invalid SQL syntax"):
            builder.validate_and_parameterize("((((unclosed")

    def test_completely_nonsensical_input(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError):
            builder.validate_and_parameterize("!!!###$$$")


class TestSafeWhereClauseBuilderInjection:
    """Tests for SQL injection attempt detection."""

    def test_injection_semicolon_attack(self):
        builder = SafeWhereClauseBuilder()
        with pytest.raises(SQLSafetyError):
            builder.validate_and_parameterize("id = 1; DROP TABLE users")

    def test_injection_tautology(self):
        """Tautology is safe (no dangerous keywords), just parameterize it."""
        builder = SafeWhereClauseBuilder()
        clause, params = builder.validate_and_parameterize("1 = 1")
        # This is technically "safe" (no dangerous keywords), just a tautology
        assert params == {
            "param_0": 1,
            "param_1": 1,
        }


class TestBuildSafeTableCopy:
    """Tests for build_safe_table_copy function."""

    def test_basic_copy(self):
        from psycopg import sql as psycopg_sql

        query, params = build_safe_table_copy(
            source_schema="bronze",
            source_table="orders",
            target_schema="test",
            target_table="test_orders",
        )
        assert isinstance(query, psycopg_sql.Composed)
        assert params == {}

    def test_copy_with_limit(self):
        query, params = build_safe_table_copy(
            source_schema="bronze",
            source_table="orders",
            target_schema="test",
            target_table="test_orders",
            limit=1000,
        )
        assert query is not None
        assert params == {}

    def test_copy_with_where(self):
        query, params = build_safe_table_copy(
            source_schema="bronze",
            source_table="orders",
            target_schema="test",
            target_table="test_orders",
            where_clause="status = %(param_0)s",
            where_params={"param_0": "active"},
        )
        assert query is not None
        assert params == {"param_0": "active"}

    def test_copy_with_limit_and_where(self):
        query, params = build_safe_table_copy(
            source_schema="bronze",
            source_table="orders",
            target_schema="test",
            target_table="test_orders",
            limit=100,
            where_clause="status = %(param_0)s",
            where_params={"param_0": "active"},
        )
        assert query is not None
        assert params == {"param_0": "active"}


class TestSafeCopyTable:
    """Tests for safe_copy_table convenience function."""

    def test_basic_copy(self):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 100

        rows = safe_copy_table(
            cursor=mock_cursor,
            source_schema="bronze",
            source_table="orders",
            target_schema="test",
            target_table="test_orders",
        )

        mock_cursor.execute.assert_called_once()
        assert rows == 100

    def test_copy_with_limit(self):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 50

        rows = safe_copy_table(
            cursor=mock_cursor,
            source_schema="bronze",
            source_table="orders",
            target_schema="test",
            target_table="test_orders",
            limit=1000,
        )

        mock_cursor.execute.assert_called_once()
        assert rows == 50

    def test_copy_with_where_clause(self):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 50

        rows = safe_copy_table(
            cursor=mock_cursor,
            source_schema="bronze",
            source_table="orders",
            target_schema="test",
            target_table="test_orders",
            where_clause="status = 'active'",
            allowed_columns=["status"],
        )

        mock_cursor.execute.assert_called_once()
        assert rows == 50

    def test_unsafe_where_raises_without_executing(self):
        mock_cursor = MagicMock()

        with pytest.raises(SQLSafetyError):
            safe_copy_table(
                cursor=mock_cursor,
                source_schema="bronze",
                source_table="orders",
                target_schema="test",
                target_table="test_orders",
                where_clause="1=1; DROP TABLE orders",
            )

        mock_cursor.execute.assert_not_called()

    def test_disallowed_column_raises_without_executing(self):
        mock_cursor = MagicMock()

        with pytest.raises(SQLSafetyError, match="not in allowed columns"):
            safe_copy_table(
                cursor=mock_cursor,
                source_schema="bronze",
                source_table="orders",
                target_schema="test",
                target_table="test_orders",
                where_clause="secret = 'value'",
                allowed_columns=["status", "date"],
            )

        mock_cursor.execute.assert_not_called()

    def test_no_where_clause_no_validation(self):
        """Without a WHERE clause, no validation is needed."""
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 200

        rows = safe_copy_table(
            cursor=mock_cursor,
            source_schema="bronze",
            source_table="orders",
            target_schema="test",
            target_table="test_orders",
        )

        mock_cursor.execute.assert_called_once()
        assert rows == 200
