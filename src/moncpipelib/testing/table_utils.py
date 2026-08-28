"""Table manipulation utilities for integration test setup.

Provides utilities for creating test tables from SQLAlchemy model
definitions, enabling schema-driven test table creation without
parsing database migrations.
"""

from __future__ import annotations

from typing import Any

import psycopg
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable


def create_test_table_from_model(
    conn: psycopg.Connection,
    model: type[Any],
    target_schema: str,
    target_table: str,
) -> None:
    """Create a test table from a SQLAlchemy model definition.

    Uses SQLAlchemy DDL generation to create the table with the correct schema,
    applying the branch's schema changes without parsing migrations.

    Args:
        conn: psycopg database connection
        model: SQLAlchemy model class (must have ``__table__`` attribute)
        target_schema: Schema for the test table
        target_table: Name for the test table

    Raises:
        AttributeError: If model doesn't have a ``__table__`` attribute.

    Example:
        >>> import psycopg
        >>> from myapp.models import FdaNdcPackage
        >>> conn = psycopg.connect(...)
        >>> create_test_table_from_model(
        ...     conn=conn,
        ...     model=FdaNdcPackage,
        ...     target_schema="integration_tests",
        ...     target_table="test_fda_ndc_package",
        ... )
    """
    test_metadata = sa.MetaData()

    original_table: sa.Table = model.__table__

    def _copy_col_without_indexes(col: sa.Column) -> sa.Column:
        # col.copy() carries over the index=True flag. When placed in a test
        # table whose name includes a user/run prefix, SQLAlchemy generates
        # schema-qualified index names that can exceed PostgreSQL's 63-character
        # identifier limit. Test tables don't need indexes, so strip the index
        # flag from every column copy.
        new_col = col.copy()
        new_col.index = None
        return new_col

    test_table_obj = sa.Table(
        target_table,
        test_metadata,
        *(_copy_col_without_indexes(col) for col in original_table.columns),
        schema=target_schema,
    )

    create_ddl = CreateTable(test_table_obj).compile(
        dialect=postgresql.dialect(),  # type: ignore[no-untyped-call]
        compile_kwargs={"literal_binds": True},
    )

    with conn.cursor() as cur:
        cur.execute(str(create_ddl))
