"""Characterization tests for the upsert write path (#375 lever 2 prep).

These tests freeze the *current* observable behavior of upsert-mode writes so
that the planned staging-COPY + single-merge rewrite (#375) can be developed
against a known-good baseline. They deliberately target the two behaviors that
differ between the current per-row ``executemany`` path and a single
``INSERT ... SELECT ... ON CONFLICT`` over a CSV-COPY'd staging table:

1. **In-batch duplicate conflict keys.** The current path applies rows
   sequentially, so duplicate keys within one write resolve last-write-wins
   with no error. A single merge statement over a staging table raises
   ``ON CONFLICT DO UPDATE command cannot affect row a second time`` unless
   staging is deduped on the conflict key first. These tests pin the
   last-write-wins contract the rewrite must preserve (via ``DISTINCT ON``).

2. **CSV-hostile value fidelity.** The current path binds parameters, so
   commas, quotes, newlines, backslashes, the literal NULL sentinel ``\\N``,
   and the empty-string-vs-NULL distinction round-trip exactly. The rewrite
   routes the same values through CSV COPY for the first time; these tests
   pin the exact values that must survive that encoding.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# TestUpsertInBatchDuplicates
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUpsertInBatchDuplicates:
    """Duplicate conflict keys within a single upsert write.

    Current contract: last occurrence in input order wins, no error. The
    staging-merge rewrite must reproduce this with a deterministic
    ``DISTINCT ON (pk) ... ORDER BY <input-order>`` dedup.
    """

    SINGLE_PK_TABLE: str = f"ups_char_dup_single_{uuid.uuid4().hex[:8]}"
    COMPOSITE_PK_TABLE: str = f"ups_char_dup_comp_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.single_fqn = table_builder.create_table(
            self.SINGLE_PK_TABLE,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT", "value": "NUMERIC"},
            primary_key=["id"],
        )
        self.composite_fqn = table_builder.create_table(
            self.COMPOSITE_PK_TABLE,
            columns={
                "tenant": "TEXT NOT NULL",
                "id": "INTEGER NOT NULL",
                "name": "TEXT",
            },
            primary_key=["tenant", "id"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.single_fqn)
        self.builder.drop(self.composite_fqn)

    def test_in_batch_duplicate_single_key_last_wins(self) -> None:
        """Two rows with the same PK in one write collapse to the last one."""
        ctx = make_mock_output_context(
            asset_name=self.SINGLE_PK_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        # id=1 appears three times; id=2 once. Input order is significant:
        # the LAST occurrence of id=1 ("third") must win.
        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 1],
                "name": ["first", "second", "solo", "third"],
                "value": [1.0, 2.0, 9.0, 3.0],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.single_fqn, order_by="id")
        assert len(rows) == 2
        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "third"
        assert float(rows[0]["value"]) == 3.0
        assert rows[1]["id"] == 2
        assert rows[1]["name"] == "solo"

    def test_in_batch_duplicate_against_existing_row(self) -> None:
        """Dupes in input still resolve last-wins on top of a pre-existing row."""
        self.builder.insert_rows(
            self.single_fqn,
            columns=["id", "name", "value"],
            rows=[(1, "preexisting", 100.0)],
        )
        ctx = make_mock_output_context(
            asset_name=self.SINGLE_PK_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame(
            {"id": [1, 1], "name": ["update_one", "update_two"], "value": [10.0, 20.0]}
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.single_fqn, order_by="id")
        assert len(rows) == 1
        assert rows[0]["name"] == "update_two"
        assert float(rows[0]["value"]) == 20.0

    def test_in_batch_duplicate_composite_key_last_wins(self) -> None:
        """Last-wins holds for composite conflict keys."""
        ctx = make_mock_output_context(
            asset_name=self.COMPOSITE_PK_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["tenant", "id"]},
        )
        df = pl.DataFrame(
            {
                "tenant": ["acme", "acme", "beta", "acme"],
                "id": [1, 1, 1, 1],
                "name": ["a1", "a2", "b1", "a3"],
            }
        )
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.composite_fqn, order_by="tenant")
        # (acme, 1) collapses to last ("a3"); (beta, 1) is distinct.
        assert len(rows) == 2
        by_key = {(r["tenant"], r["id"]): r["name"] for r in rows}
        assert by_key[("acme", 1)] == "a3"
        assert by_key[("beta", 1)] == "b1"

    def test_in_batch_duplicate_with_chunk_size_set_last_wins(self) -> None:
        """Last-wins holds when ``insert_chunk_size`` is set and dupes span the input.

        Today ``insert_chunk_size`` is inert on the upsert path -- ``_write_single``
        passes the whole DataFrame to ``execute_upsert`` in a single
        ``executemany`` (no slicing). This test sets it anyway and places the
        duplicate keys far apart in input order, so it guards the invariant the
        staging-merge rewrite must preserve once it *does* chunk the CSV COPY
        into staging: dedup the conflict key across the entire staging table
        (last input position wins), never per-chunk.
        """
        ctx = make_mock_output_context(
            asset_name=self.SINGLE_PK_TABLE,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        io_mgr = self.io_mgr.postgres_resource
        chunked_mgr = PostgresIOManager(
            postgres_resource=io_mgr,
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            insert_chunk_size=2,
        )
        # id=1 appears at input positions 0 and 4 (a chunk apart at size 2);
        # the later occurrence ("c3_wins") must win.
        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 1],
                "name": ["c1a", "c1b", "c2a", "c2b", "c3_wins"],
                "value": [1.0, 2.0, 3.0, 4.0, 99.0],
            }
        )
        chunked_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.single_fqn, order_by="id")
        assert len(rows) == 4
        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "c3_wins"
        assert float(rows[0]["value"]) == 99.0


# ---------------------------------------------------------------------------
# TestUpsertCsvHostileValues
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUpsertCsvHostileValues:
    """Text/NULL values that a CSV-COPY staging path could mangle.

    Pinned via the current (param-binding) path so the staging-COPY rewrite
    must round-trip them identically.
    """

    TABLE_NAME: str = f"ups_char_csv_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={"id": "INTEGER NOT NULL", "payload": "TEXT"},
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

    def test_csv_hostile_text_roundtrips_exactly(self) -> None:
        """Delimiters, quotes, newlines, backslashes, and the NULL sentinel survive."""
        payloads = {
            1: "has,comma",
            2: 'has"double"quote',
            3: "has\nnewline",
            4: "has\ttab",
            5: "has\\backslash",
            6: r"\N",  # literal NULL sentinel as a real string, must NOT become NULL
            7: "",  # empty string, must stay '' and not collapse to NULL
            8: "  leading and trailing  ",
            9: "unicode_é中文",
        }
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": list(payloads.keys()), "payload": list(payloads.values())})
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        got = {r["id"]: r["payload"] for r in rows}
        assert got == payloads

    def test_empty_string_vs_null_distinction(self) -> None:
        """Empty string and NULL are stored as distinct values, not coalesced."""
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": [1, 2], "payload": ["", None]})
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(self.fqn, order_by="id")
        got = {r["id"]: r["payload"] for r in rows}
        assert got[1] == ""
        assert got[2] is None
