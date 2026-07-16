"""SQL safety utilities for integration test data copying.

Provides battle-tested SQL safety using:
- ``psycopg.sql``: safe SQL identifier quoting
- sqlglot: SQL parsing and validation

Prevents SQL injection while allowing flexible WHERE clauses
in test configurations.
"""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from psycopg import sql
from sqlglot import exp


class SQLSafetyError(Exception):
    """Raised when potentially unsafe SQL is detected."""


class SafeWhereClauseBuilder:
    """Validates and parameterizes WHERE clauses using sqlglot parsing.

    Example:
        builder = SafeWhereClauseBuilder(allowed_columns=['date', 'status'])
        clause, params = builder.validate_and_parameterize(
            "date >= '2024-01-01' AND status = 'active'"
        )
        # Returns:
        # clause = "date >= %(param_0)s AND status = %(param_1)s"
        # params = {'param_0': '2024-01-01', 'param_1': 'active'}
    """

    DANGEROUS_KEYWORDS: frozenset[str] = frozenset(
        {
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "TRUNCATE",
            "ALTER",
            "CREATE",
            "GRANT",
            "REVOKE",
            "EXEC",
            "EXECUTE",
            "UNION",
            "INTO",
        }
    )

    SAFE_FUNCTIONS: frozenset[str] = frozenset(
        {
            "CURRENT_DATE",
            "CURRENT_TIMESTAMP",
            "NOW",
            "DATE_TRUNC",
            "COALESCE",
            "NULLIF",
            "UPPER",
            "LOWER",
            "TRIM",
            "LENGTH",
            "CAST",
        }
    )

    def __init__(self, allowed_columns: list[str] | None = None) -> None:
        """Initialize the builder.

        Args:
            allowed_columns: Whitelist of column names. If None, all valid
                identifiers are allowed.
        """
        self._allowed_columns: frozenset[str] | None = (
            frozenset(c.lower() for c in allowed_columns) if allowed_columns else None
        )

    def validate_and_parameterize(
        self,
        where_clause: str,
    ) -> tuple[str, dict[str, Any]]:
        """Validate WHERE clause and convert literals to parameters.

        Process:
        1. Check for dangerous keywords (DROP, DELETE, etc.)
        2. Parse SQL using sqlglot to validate structure
        3. Validate column names against allowlist
        4. Extract literal values and replace with parameters
        5. Return parameterized clause + parameter dict

        Args:
            where_clause: WHERE clause without "WHERE" keyword

        Returns:
            (parameterized_clause, parameters_dict)

        Raises:
            SQLSafetyError: If clause is unsafe or invalid
        """
        if not where_clause or not where_clause.strip():
            return "", {}

        # Step 1: Check for dangerous keywords
        self._check_dangerous_keywords(where_clause)

        # Step 2: Parse SQL using sqlglot
        try:
            parsed = sqlglot.parse_one(
                f"SELECT * FROM t WHERE {where_clause}",
                read="postgres",
            )
        except sqlglot.errors.SqlglotError as e:
            raise SQLSafetyError(f"Invalid SQL syntax: {e}") from e

        # Step 3: Extract WHERE node
        where_node = parsed.find(exp.Where)
        if not where_node:
            raise SQLSafetyError("Failed to parse WHERE clause")

        # Step 4: Validate columns
        self._validate_columns(where_node)

        # Step 5: Extract and parameterize literals
        parameters: dict[str, Any] = {}
        param_counter = 0

        def replace_literal(node: exp.Expression) -> exp.Expression:
            nonlocal param_counter
            if isinstance(node, exp.Literal):
                param_name = f"param_{param_counter}"
                param_counter += 1
                if node.is_string:
                    parameters[param_name] = node.this
                elif node.is_int:
                    parameters[param_name] = int(node.this)
                else:
                    parameters[param_name] = node.this
                return exp.Placeholder(this=param_name)
            return node

        parameterized_where = where_node.this.transform(replace_literal)
        parameterized_clause = parameterized_where.sql(dialect="postgres")

        # Convert sqlglot :param style to psycopg %(param)s style
        for param_name in parameters:
            parameterized_clause = parameterized_clause.replace(
                f":{param_name}", f"%({param_name})s"
            )

        return parameterized_clause, parameters

    def _check_dangerous_keywords(self, where_clause: str) -> None:
        """Check for dangerous SQL keywords using word boundary matching."""
        where_upper = where_clause.upper()
        for keyword in self.DANGEROUS_KEYWORDS:
            if re.search(rf"\b{keyword}\b", where_upper):
                raise SQLSafetyError(f"Dangerous keyword '{keyword}' detected in WHERE clause")

    def _validate_columns(self, where_node: exp.Where) -> None:
        """Validate column names against the allowlist."""
        if self._allowed_columns is None:
            return

        for column in where_node.find_all(exp.Column):
            col_name = column.name.lower()
            if col_name not in self._allowed_columns:
                raise SQLSafetyError(
                    f"Column '{col_name}' not in allowed columns: {sorted(self._allowed_columns)}"
                )


def build_safe_table_copy(
    source_schema: str,
    source_table: str,
    target_schema: str,
    target_table: str,
    limit: int | None = None,
    where_clause: str | None = None,
    where_params: dict[str, Any] | None = None,
) -> tuple[sql.Composed, dict[str, Any]]:
    """Build safe CREATE TABLE AS SELECT query.

    Uses psycopg.sql.Identifier for safe schema/table name quoting.

    Args:
        source_schema: Source schema name
        source_table: Source table name
        target_schema: Target schema name
        target_table: Target table name
        limit: Optional row limit
        where_clause: Optional parameterized WHERE clause (without WHERE keyword)
        where_params: Parameters for the WHERE clause

    Returns:
        (query, parameters) where query is psycopg.sql.Composed object
    """
    params: dict[str, Any] = dict(where_params) if where_params else {}

    # Build base query: CREATE TABLE target AS SELECT * FROM source
    query_parts: list[sql.Composable] = [
        sql.SQL("CREATE TABLE {}.{} AS SELECT * FROM {}.{}").format(
            sql.Identifier(target_schema),
            sql.Identifier(target_table),
            sql.Identifier(source_schema),
            sql.Identifier(source_table),
        )
    ]

    # Add WHERE clause if provided
    if where_clause:
        query_parts.append(sql.SQL(" WHERE "))
        query_parts.append(sql.SQL(where_clause))

    # Add LIMIT if provided
    if limit is not None:
        query_parts.append(sql.SQL(" LIMIT {}").format(sql.Literal(limit)))

    return sql.Composed(query_parts), params


def safe_copy_table(
    cursor: Any,
    source_schema: str,
    source_table: str,
    target_schema: str,
    target_table: str,
    limit: int | None = None,
    where_clause: str | None = None,
    allowed_columns: list[str] | None = None,
) -> int:
    """High-level function: validate, build, and execute table copy.

    Validates the WHERE clause (if provided), builds a safe CREATE TABLE AS
    SELECT query, and executes it.

    Args:
        cursor: psycopg cursor
        source_schema: Source schema name
        source_table: Source table name
        target_schema: Target schema name
        target_table: Target table name
        limit: Optional row limit
        where_clause: Optional WHERE clause (will be validated and parameterized)
        allowed_columns: Optional column whitelist for WHERE clause validation

    Returns:
        Number of rows copied

    Raises:
        SQLSafetyError: If WHERE clause is unsafe

    Example:
        >>> import psycopg
        >>> conn = psycopg.connect(...)
        >>> cursor = conn.cursor()
        >>> rows = safe_copy_table(
        ...     cursor,
        ...     'synthetic_bronze', 'fda_ndc_package_raw',
        ...     'integration_tests', 'test_source',
        ...     limit=1000,
        ...     where_clause="created_date >= CURRENT_DATE - INTERVAL '7 days'",
        ...     allowed_columns=['created_date', 'status']
        ... )
    """
    parameterized_clause: str | None = None
    params: dict[str, Any] = {}

    if where_clause:
        builder = SafeWhereClauseBuilder(allowed_columns=allowed_columns)
        parameterized_clause, params = builder.validate_and_parameterize(where_clause)

    query, query_params = build_safe_table_copy(
        source_schema=source_schema,
        source_table=source_table,
        target_schema=target_schema,
        target_table=target_table,
        limit=limit,
        where_clause=parameterized_clause,
        where_params=params,
    )

    all_params = {**params, **query_params}
    cursor.execute(query, all_params if all_params else None)
    row_count: int = cursor.rowcount
    return row_count
