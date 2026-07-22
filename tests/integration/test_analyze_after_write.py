"""Integration tests for the post-write ANALYZE step.

Tracks public mirror issue model-oncology-public/moncpipelib#1: autovacuum
never autoanalyzes a partitioned parent, so the write path refreshes its
aggregate statistics post-commit. These tests run against a real PostgreSQL
16 testcontainer, so they exercise the pre-PG18 fallback branch (plain
recursive ``ANALYZE``; the ``ANALYZE ONLY`` branch is unit-tested).

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Generator
from typing import Any

import polars as pl
import psycopg
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import make_mock_output_context

pytestmark = pytest.mark.integration

SCHEMA = "test_write"


def _reltuples(conn: psycopg.Connection, schema: str, table: str) -> float:
    """Committed ``pg_class.reltuples`` for a relation (-1 = never analyzed)."""
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.reltuples FROM pg_class c"
            " JOIN pg_namespace n ON n.oid = c.relnamespace"
            " WHERE n.nspname = %s AND c.relname = %s",
            (schema, table),
        )
        row = cur.fetchone()
        assert row is not None
        return float(row[0])


@pytest.mark.integration
class TestAnalyzeAfterWritePartitionedParent:
    """Partitioned parents get their aggregate stats refreshed post-write."""

    TABLE = f"analyze_part_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        pg_connection: psycopg.Connection,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Generator[None, None, None]:
        self.conn = pg_connection
        self.fqn = f"{SCHEMA}.{self.TABLE}"
        with pg_connection.cursor() as cur:
            cur.execute(
                f"CREATE TABLE {self.fqn} (id integer NOT NULL, region text NOT NULL)"
                f" PARTITION BY LIST (region)"
            )
            cur.execute(
                f"CREATE TABLE {self.fqn}_east PARTITION OF {self.fqn} FOR VALUES IN ('east')"
            )
            cur.execute(
                f"CREATE TABLE {self.fqn}_west PARTITION OF {self.fqn} FOR VALUES IN ('west')"
            )
        pg_connection.commit()
        self.io_mgr = io_manager_factory(db_schema=SCHEMA)
        yield
        pg_connection.rollback()
        with pg_connection.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.fqn} CASCADE")
        pg_connection.commit()

    def _df(self) -> pl.DataFrame:
        return pl.DataFrame({"id": [1, 2, 3, 4], "region": ["east", "east", "west", "west"]})

    def _write(self, metadata: dict[str, Any] | None = None) -> None:
        ctx = make_mock_output_context(
            asset_name=self.TABLE,
            metadata={"write_mode": "append", **(metadata or {})},
        )
        self.io_mgr.handle_output(ctx, self._df())

    def test_default_refreshes_parent_stats(self) -> None:
        """Under the 'partitioned' default, the parent goes from never-analyzed
        (reltuples = -1) to real estimates after an append."""
        assert _reltuples(self.conn, SCHEMA, self.TABLE) == -1.0

        self._write()

        assert _reltuples(self.conn, SCHEMA, self.TABLE) == 4.0

    def test_never_override_skips_parent(self) -> None:
        """Per-asset analyze_after_write='never' leaves the parent unanalyzed."""
        self._write(metadata={"analyze_after_write": "never"})

        assert _reltuples(self.conn, SCHEMA, self.TABLE) == -1.0


@pytest.mark.integration
class TestAnalyzeAfterWriteOrdinaryTable:
    """Ordinary tables stay autovacuum-owned unless 'always' is requested."""

    TABLE = f"analyze_plain_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        pg_connection: psycopg.Connection,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Generator[None, None, None]:
        self.conn = pg_connection
        self.fqn = f"{SCHEMA}.{self.TABLE}"
        with pg_connection.cursor() as cur:
            cur.execute(f"CREATE TABLE {self.fqn} (id integer NOT NULL, name text)")
        pg_connection.commit()
        self.io_mgr = io_manager_factory(db_schema=SCHEMA)
        yield
        pg_connection.rollback()
        with pg_connection.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.fqn}")
        pg_connection.commit()

    def _write(self, metadata: dict[str, Any] | None = None) -> None:
        ctx = make_mock_output_context(
            asset_name=self.TABLE,
            metadata={"write_mode": "append", **(metadata or {})},
        )
        self.io_mgr.handle_output(ctx, pl.DataFrame({"id": [1, 2], "name": ["a", "b"]}))

    def test_default_skips_ordinary_table(self) -> None:
        """The 'partitioned' default leaves ordinary tables to autovacuum."""
        self._write()

        assert _reltuples(self.conn, SCHEMA, self.TABLE) == -1.0

    def test_always_override_analyzes_ordinary_table(self) -> None:
        """analyze_after_write='always' refreshes stats on an ordinary table."""
        self._write(metadata={"analyze_after_write": "always"})

        assert _reltuples(self.conn, SCHEMA, self.TABLE) == 2.0
