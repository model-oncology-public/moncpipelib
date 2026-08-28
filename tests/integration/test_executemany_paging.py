"""Integration test: parametrized multi-row paging on the executemany insert path.

``_executemany_bulk`` pages multiple rows into each ``executemany`` /
``execute`` statement instead of one execution per row (#456). This test
exercises the real path end-to-end against PostgreSQL: append mode,
``bulk_insert_method="execute_values"``, and a small ``executemany_page_rows``
override so an 11-row insert spans 3 full pages plus a 2-row tail. Content
includes NULLs, unicode, and quote-containing strings to confirm the
parametrized-page rewrite round-trips exactly like the original per-row path.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import moncpipelib.io_managers.writers as writers_mod
from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


class TestExecutemanyPagingFidelity:
    """Small ``executemany_page_rows`` override, exact content round-trip."""

    TABLE_NAME_PREFIX = "executemany_paging"
    PAGE_ROWS = 3

    # 11 rows / page_rows=3 -> 3 full pages (9 rows) + a 2-row tail.
    NOTES: list[str | None] = [
        "alpha",
        None,
        "héllo wörld 世界 🎉",
        'O\'Brien said "hi" -- quotes',
        "",
        None,
        "tab\tand\nnewline",
        "beta",
        "quote's inside 'single' and \"double\"",
        None,
        "omega -- em dash, café",
    ]

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.table_name = f"{self.TABLE_NAME_PREFIX}_{uuid.uuid4().hex[:8]}"
        self.fqn = table_builder.create_table(
            self.table_name,
            columns={"id": "INTEGER NOT NULL", "note": "TEXT"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            bulk_insert_method="execute_values",
            executemany_page_rows=self.PAGE_ROWS,
        )
        yield
        self.builder.drop(self.fqn)

    def test_paged_append_round_trips_exactly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ids = list(range(1, len(self.NOTES) + 1))
        assert len(ids) == 11  # sanity: 3 full pages of 3 + a 2-row tail

        # Spy on the real paging entry point (still delegating to it) so the
        # test fails loudly if the resource's ``executemany_page_rows``
        # override is not actually threaded down to ``_executemany_bulk`` --
        # content round-tripping alone can't distinguish "paged correctly"
        # from "silently ignored the override and fell back to one row per
        # execution", since both produce identical rows in the table.
        page_rows_seen: list[int] = []
        original = writers_mod._executemany_bulk

        def _spy(cursor: Any, sql: str, rows: Any, *, page_rows: int = 1) -> None:
            page_rows_seen.append(page_rows)
            return original(cursor, sql, rows, page_rows=page_rows)

        monkeypatch.setattr(writers_mod, "_executemany_bulk", _spy)

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "append"},
        )
        df = pl.DataFrame({"id": ids, "note": self.NOTES})
        self.io_mgr.handle_output(ctx, df)

        assert page_rows_seen == [self.PAGE_ROWS]

        assert self.builder.count(self.fqn) == len(ids)
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert [r["id"] for r in rows] == ids
        assert [r["note"] for r in rows] == self.NOTES

        meta = ctx.add_output_metadata.call_args[0][0]
        assert meta["insert_method"].value == "execute_values"
