"""Integration tests for the upsert skip_unchanged change-guard.

Opt-in behavior requested by a downstream consumer (mirror issue
model-oncology-public/moncpipelib#3): with ``skip_unchanged=True`` the merge
guards ``DO UPDATE`` with ``WHERE <target>.col IS DISTINCT FROM EXCLUDED.col
OR ...``, so a conflicting row whose update columns are all unchanged is not
rewritten -- no dead tuple, no index churn, no WAL for no-op updates.

Row rewrites are observed through ``xmin``: PostgreSQL stamps a new ``xmin``
on every new row version, so an elided update leaves ``xmin`` untouched while
a rewritten row (even with identical values) gets a fresh one. This is the
same observable the WAL/bloat claim rests on: no new row version means no
heap write and no full-page WAL for that row.

These tests exercise the resource-first path (``database.write(...)``).

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import psycopg
import pytest

from moncpipelib.resources.postgres import PostgresResource
from moncpipelib.resources.types import WriteContext

from .conftest import TableBuilder

pytestmark = pytest.mark.integration


@pytest.mark.integration
class TestUpsertSkipUnchanged:
    """Behavior of ``write(..., write_mode="upsert", skip_unchanged=True)``."""

    TABLE_NAME: str = f"ups_skip_unchanged_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        postgres_resource: PostgresResource,
        pg_connection: psycopg.Connection,
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT", "value": "NUMERIC"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.resource = postgres_resource
        self.conn = pg_connection
        yield
        self.builder.drop(self.fqn)

    # -- helpers ---------------------------------------------------------

    def _write(self, df: pl.DataFrame, **kwargs: Any) -> None:
        wctx = WriteContext(
            asset_name=self.TABLE_NAME,
            run_id=f"skip-unchanged-{uuid.uuid4().hex[:8]}",
            log=MagicMock(),
        )
        self.resource.write(
            df,
            target=self.fqn,
            context=wctx,
            write_mode="upsert",
            primary_key=["id"],
            contract=None,
            **kwargs,
        )

    def _xmins(self) -> dict[int, str]:
        """Map id -> xmin. A row rewrite (even to identical values) changes xmin."""
        self.conn.rollback()
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, xmin::text FROM {self.fqn}")  # noqa: S608
            return {row[0]: row[1] for row in cur.fetchall()}

    # -- tests -----------------------------------------------------------

    def test_noop_reupsert_skips_all_rows(self) -> None:
        """Re-upserting identical data with the guard leaves every xmin intact."""
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        self._write(df)
        before = self._xmins()

        self._write(df, skip_unchanged=True)

        assert self._xmins() == before
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert [(r["id"], r["name"], float(r["value"])) for r in rows] == [
            (1, "a", 1.0),
            (2, "b", 2.0),
            (3, "c", 3.0),
        ]

    def test_default_reupsert_rewrites_rows(self) -> None:
        """Pin today's default: without the guard, identical rows are rewritten."""
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "value": [1.0, 2.0]})
        self._write(df)
        before = self._xmins()

        self._write(df)

        after = self._xmins()
        assert all(after[i] != before[i] for i in before), (
            f"expected every row rewritten on default no-op re-upsert, "
            f"before={before} after={after}"
        )

    def test_changed_rows_written_unchanged_rows_skipped(self) -> None:
        """Only the row whose update columns actually changed gets a new version."""
        self._write(pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "value": [1.0, 2.0]}))
        before = self._xmins()

        self._write(
            pl.DataFrame({"id": [1, 2], "name": ["a", "CHANGED"], "value": [1.0, 2.0]}),
            skip_unchanged=True,
        )

        after = self._xmins()
        assert after[1] == before[1]
        assert after[2] != before[2]
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert rows[1]["name"] == "CHANGED"

    def test_null_comparisons_are_null_safe(self) -> None:
        """IS DISTINCT FROM: NULL->NULL skips; NULL->value and value->NULL write."""
        self._write(pl.DataFrame({"id": [1, 2, 3], "name": [None, None, "c"], "value": [1.0] * 3}))
        before = self._xmins()

        self._write(
            pl.DataFrame({"id": [1, 2, 3], "name": [None, "filled", None], "value": [1.0] * 3}),
            skip_unchanged=True,
        )

        after = self._xmins()
        assert after[1] == before[1]  # NULL -> NULL: not distinct, skipped
        assert after[2] != before[2]  # NULL -> value: distinct, written
        assert after[3] != before[3]  # value -> NULL: distinct, written
        got = {r["id"]: r["name"] for r in self.builder.read_all(self.fqn)}
        assert got == {1: None, 2: "filled", 3: None}

    def test_new_keys_insert_while_unchanged_skip(self) -> None:
        """The guard only affects the conflict path; new keys insert normally."""
        self._write(pl.DataFrame({"id": [1], "name": ["a"], "value": [1.0]}))
        before = self._xmins()

        self._write(
            pl.DataFrame({"id": [1, 2], "name": ["a", "new"], "value": [1.0, 2.0]}),
            skip_unchanged=True,
        )

        after = self._xmins()
        assert after[1] == before[1]
        assert self.builder.count(self.fqn) == 2
        got = {r["id"]: r["name"] for r in self.builder.read_all(self.fqn)}
        assert got[2] == "new"

    def test_guard_scoped_to_update_columns(self) -> None:
        """A change outside update_columns neither updates nor rewrites the row."""
        self._write(pl.DataFrame({"id": [1], "name": ["a"], "value": [1.0]}))
        before = self._xmins()

        # value changes in the input, but only name is an update column: the
        # row must be skipped entirely (no rewrite, value stays 1.0).
        self._write(
            pl.DataFrame({"id": [1], "name": ["a"], "value": [99.0]}),
            skip_unchanged=True,
            update_columns=["name"],
        )

        assert self._xmins() == before
        rows = self.builder.read_all(self.fqn)
        assert float(rows[0]["value"]) == 1.0
