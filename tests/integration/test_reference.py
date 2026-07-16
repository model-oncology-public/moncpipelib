"""Integration tests for :func:`moncpipelib.reference.read_latest_partition`.

Exercises the end-to-end streaming behaviour against a real PostgreSQL
container:

- Empty bronze raises :class:`EmptyPartitionedTableError`.
- Single-partition bronze streams all rows.
- Multi-partition bronze streams ONLY the latest partition.
- A row landing mid-stream (between the pre-check and the iterator's
  first ``next()``) is included -- the WHERE clause's MAX subquery is
  re-evaluated at main-SELECT execution time.
- Small ``batch_size`` yields multiple frames whose concatenation
  matches the latest-partition row set.
- ``columns=`` projection emits exactly those columns in order.

Requires Docker. Run with: ``uv run pytest -m integration -v``.
"""

from __future__ import annotations

import uuid

import polars as pl
import psycopg
import pytest

from moncpipelib.reference import EmptyPartitionedTableError, read_latest_partition
from moncpipelib.resources.postgres import PostgresResource

pytestmark = pytest.mark.integration


@pytest.fixture
def ref_table(
    pg_connection: psycopg.Connection,
) -> str:
    """Create a fresh ``test_write.ref_<suffix>`` table with the shape:

    - ``code`` TEXT
    - ``label`` TEXT
    - ``load_period`` DATE

    Index on ``load_period`` so the precheck ``MAX(...)`` is cheap.
    """
    suffix = uuid.uuid4().hex[:8]
    fqn = f"test_write.ref_{suffix}"
    with pg_connection.cursor() as cur:
        cur.execute(f'CREATE TABLE {fqn} (  "code" TEXT,  "label" TEXT,  "load_period" DATE)')
        cur.execute(f'CREATE INDEX ON {fqn} ("load_period")')
    pg_connection.commit()
    yield fqn
    pg_connection.rollback()
    with pg_connection.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {fqn} CASCADE")
    pg_connection.commit()


def _insert_rows(
    conn: psycopg.Connection,
    fqn: str,
    rows: list[tuple[str, str, str]],
) -> None:
    """Insert ``(code, label, load_period_iso)`` tuples."""
    with conn.cursor() as cur:
        cur.executemany(
            f'INSERT INTO {fqn} ("code", "label", "load_period") VALUES (%s, %s, %s)',
            rows,
        )
    conn.commit()


class TestReadLatestPartitionIntegration:
    """End-to-end behavioural tests against a real PostgreSQL."""

    def test_empty_bronze_raises(
        self,
        postgres_resource: PostgresResource,
        ref_table: str,
    ) -> None:
        """No rows -> ``EmptyPartitionedTableError`` (a ``LookupError``)."""
        with pytest.raises(EmptyPartitionedTableError) as excinfo:
            list(
                read_latest_partition(
                    postgres_resource,
                    source_table=ref_table,
                )
            )
        assert ref_table in str(excinfo.value)
        assert "load_period" in str(excinfo.value)

    def test_single_partition_streams_all_rows(
        self,
        postgres_resource: PostgresResource,
        ref_table: str,
        pg_connection: psycopg.Connection,
    ) -> None:
        _insert_rows(
            pg_connection,
            ref_table,
            [
                ("A", "alpha", "2026-05-01"),
                ("B", "beta", "2026-05-01"),
                ("C", "gamma", "2026-05-01"),
            ],
        )

        batches = list(
            read_latest_partition(
                postgres_resource,
                source_table=ref_table,
            )
        )
        assert len(batches) >= 1
        df = pl.concat(batches)
        assert df.shape == (3, 3)
        assert set(df["code"].to_list()) == {"A", "B", "C"}

    def test_multi_partition_streams_only_latest(
        self,
        postgres_resource: PostgresResource,
        ref_table: str,
        pg_connection: psycopg.Connection,
    ) -> None:
        _insert_rows(
            pg_connection,
            ref_table,
            [
                ("old1", "stale", "2026-03-01"),
                ("old2", "stale", "2026-03-01"),
                ("old3", "stale", "2026-04-01"),
                ("new1", "fresh", "2026-05-01"),
                ("new2", "fresh", "2026-05-01"),
            ],
        )

        df = pl.concat(
            list(
                read_latest_partition(
                    postgres_resource,
                    source_table=ref_table,
                )
            )
        )
        assert df.shape == (2, 3)
        assert set(df["code"].to_list()) == {"new1", "new2"}
        # No row from any older partition leaks through.
        assert "old1" not in df["code"].to_list()
        assert "old3" not in df["code"].to_list()

    def test_columns_projection_emits_specified_columns_in_order(
        self,
        postgres_resource: PostgresResource,
        ref_table: str,
        pg_connection: psycopg.Connection,
    ) -> None:
        _insert_rows(
            pg_connection,
            ref_table,
            [("A", "alpha", "2026-05-01"), ("B", "beta", "2026-05-01")],
        )

        df = pl.concat(
            list(
                read_latest_partition(
                    postgres_resource,
                    source_table=ref_table,
                    columns=("label", "code"),
                )
            )
        )
        # Order matches the ``columns=`` argument, not the table order.
        assert df.columns == ["label", "code"]
        assert df.shape == (2, 2)

    def test_subquery_picks_up_partition_landing_mid_stream(
        self,
        postgres_resource: PostgresResource,
        ref_table: str,
        pg_connection: psycopg.Connection,
    ) -> None:
        """The main SELECT's WHERE clause uses ``= (SELECT MAX(...))``,
        so a partition that lands between the precheck and the iterator
        is included.

        The pre-check connection is opened+closed inside
        :func:`read_latest_partition` before it returns the iterator,
        so by the time we ``next()`` the iterator we can insert a
        newer partition without contending with the pre-check
        transaction.  The streaming SELECT then re-evaluates ``MAX(...)``
        and yields rows from the new partition.
        """
        _insert_rows(
            pg_connection,
            ref_table,
            [("old", "stale", "2026-04-01")],
        )

        # Build the iterator (runs the precheck).
        iterator = read_latest_partition(
            postgres_resource,
            source_table=ref_table,
        )

        # Insert a newer partition before consuming.  The streaming
        # SELECT in ``read_batched`` runs at first ``next(...)`` so its
        # MAX subquery sees this row.
        _insert_rows(
            pg_connection,
            ref_table,
            [("new", "fresh", "2026-05-15")],
        )

        df = pl.concat(list(iterator))
        # Only the new partition should be yielded -- not the old one,
        # not both.
        assert df.shape == (1, 3)
        assert df["code"].to_list() == ["new"]

    def test_multi_batch_streaming_with_small_batch_size(
        self,
        postgres_resource: PostgresResource,
        ref_table: str,
        pg_connection: psycopg.Connection,
    ) -> None:
        """A small ``batch_size`` yields multiple frames whose
        concatenation reconstitutes the latest-partition row set."""
        rows = [(f"c{i:03}", f"l{i:03}", "2026-05-01") for i in range(15)]
        _insert_rows(pg_connection, ref_table, rows)

        batches = list(
            read_latest_partition(
                postgres_resource,
                source_table=ref_table,
                batch_size=4,
            )
        )
        assert len(batches) >= 2  # 15 rows / batch_size 4 -> at least 4 batches

        df = pl.concat(batches)
        assert df.shape == (15, 3)
        assert set(df["code"].to_list()) == {f"c{i:03}" for i in range(15)}

    def test_reserved_word_partition_column(
        self,
        postgres_resource: PostgresResource,
        pg_connection: psycopg.Connection,
    ) -> None:
        """Identifier-quoting acceptance test: a partition column named
        ``order`` (a reserved word) must round-trip without a SQL error."""
        suffix = uuid.uuid4().hex[:8]
        fqn = f"test_write.ref_order_{suffix}"
        with pg_connection.cursor() as cur:
            cur.execute(f'CREATE TABLE {fqn} ("code" TEXT, "order" INTEGER)')
            cur.execute(
                f'INSERT INTO {fqn} ("code", "order") VALUES '
                "('a', 1), ('b', 1), ('c', 2), ('d', 2)"
            )
        pg_connection.commit()

        try:
            df = pl.concat(
                list(
                    read_latest_partition(
                        postgres_resource,
                        source_table=fqn,
                        partition_column="order",
                    )
                )
            )
            assert df.shape == (2, 2)
            assert set(df["code"].to_list()) == {"c", "d"}
        finally:
            with pg_connection.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {fqn} CASCADE")
            pg_connection.commit()

    def test_context_log_propagation(
        self,
        postgres_resource: PostgresResource,
        ref_table: str,
        pg_connection: psycopg.Connection,
    ) -> None:
        """``context=`` propagates: helper-side "Reading latest
        partition" line + per-batch progress logs from ``read_batched``."""
        from unittest.mock import MagicMock

        _insert_rows(
            pg_connection,
            ref_table,
            [("A", "alpha", "2026-05-01")],
        )

        context = MagicMock()
        context.log = MagicMock()

        list(
            read_latest_partition(
                postgres_resource,
                source_table=ref_table,
                context=context,
            )
        )

        info_strs = [str(c) for c in context.log.info.call_args_list]
        assert any("Reading latest partition" in s for s in info_strs)
        # ``read_batched`` streaming path emits "Starting streaming read"
        # + per-batch logs when ``context`` is passed.
        assert any("streaming read" in s.lower() for s in info_strs)
