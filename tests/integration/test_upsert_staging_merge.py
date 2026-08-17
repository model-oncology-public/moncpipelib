"""Integration tests for the upsert staging-COPY + merge path (#375 lever 2).

Grows phase by phase alongside docs/migrations/20260626_375-upsert-staging-merge.md.

Phase 1: the CSV serialization primitive (``serialize_for_staging_copy`` +
``COPY_STAGING_OPTIONS``) round-trips SQL NULL, empty string, and the literal
text ``\\N`` distinctly through a real COPY -- the fidelity contract #377 pins
that the param-bound path satisfies and a naive ``NULL '\\N'`` COPY would break.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import polars as pl
import psycopg
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager
from moncpipelib.io_managers.writers import (
    COPY_STAGING_OPTIONS,
    serialize_for_staging_copy,
)
from moncpipelib.streaming import BatchedDataFrame

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


@pytest.mark.integration
class TestStagingCopySerialization:
    """``serialize_for_staging_copy`` round-trips the NULL / '' / literal-\\N trilemma."""

    def test_trilemma_and_hostile_chars_roundtrip(
        self, pg_connection_params: dict[str, Any]
    ) -> None:
        table = f"test_write.stage_csv_{uuid.uuid4().hex[:8]}"
        # A string column carrying every hard value, plus a numeric column to
        # prove numeric NULLs survive the same encoding (the "always"-quote
        # alternative breaks integer NULL).
        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8],
                "s": ["plain", None, "", r"\N", "a,b", 'q"q', "line\nbreak", "  pad  "],
                "n": [10, None, 30, 40, 50, 60, 70, 80],
            }
        )
        expected_s = {
            1: "plain",
            2: None,
            3: "",
            4: r"\N",
            5: "a,b",
            6: 'q"q',
            7: "line\nbreak",
            8: "  pad  ",
        }
        expected_n = {1: 10, 2: None, 3: 30, 4: 40, 5: 50, 6: 60, 7: 70, 8: 80}

        conn = psycopg.connect(**pg_connection_params)
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
                cur.execute(f"CREATE TABLE {table} (id int, s text, n int)")
            conn.commit()

            payload = serialize_for_staging_copy(df)
            with (
                conn.cursor() as cur,
                cur.copy(f"COPY {table} (id, s, n) FROM STDIN WITH ({COPY_STAGING_OPTIONS})") as cp,
            ):
                cp.write(payload)
            conn.commit()

            with conn.cursor() as cur:
                cur.execute(f"SELECT id, s, n FROM {table} ORDER BY id")
                rows = cur.fetchall()
            got_s = {r[0]: r[1] for r in rows}
            got_n = {r[0]: r[2] for r in rows}

            assert got_s == expected_s
            assert got_n == expected_n
            # The three values that motivate the encoding must be DISTINCT.
            assert got_s[2] is None  # SQL NULL
            assert got_s[3] == ""  # empty string, not NULL
            assert got_s[4] == r"\N"  # literal text, not NULL

            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
        finally:
            conn.close()


@pytest.mark.integration
class TestBatchedUpsertCrossBatch:
    """Phase 3 / D2: cross-batch dedup under the per-batch staging-merge path.

    Each batch runs its own COPY-into-staging + merge inside one transaction.
    A key present in an early batch and again in a later batch must resolve to
    the later batch's value (last-write-wins across batches), and the merge must
    not raise on keys repeated across batches.
    """

    TABLE_NAME: str = f"ups_xbatch_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT", "value": "NUMERIC"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def test_same_key_across_batches_last_batch_wins(self) -> None:
        # id=1 appears in batch 1 and again in batch 3; id=2 only in batch 1.
        df1 = pl.DataFrame({"id": [1, 2], "name": ["b1_one", "b1_two"], "value": [1.0, 2.0]})
        df2 = pl.DataFrame({"id": [3], "name": ["b2_three"], "value": [3.0]})
        df3 = pl.DataFrame({"id": [1], "name": ["b3_one_WINS"], "value": [99.0]})
        batched = BatchedDataFrame(batches=iter([df1, df2, df3]), total_rows_hint=4)

        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        self.io_mgr.handle_output(ctx, batched)

        rows = self.builder.read_all(self.fqn, order_by="id")
        assert len(rows) == 3
        by_id = {r["id"]: r["name"] for r in rows}
        assert by_id[1] == "b3_one_WINS"  # later batch overwrote the earlier
        assert by_id[2] == "b1_two"
        assert by_id[3] == "b2_three"
