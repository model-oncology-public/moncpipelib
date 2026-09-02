"""Unit tests for parametrized multi-row paging on the executemany insert path.

``_executemany_bulk`` (writers.py) used to issue one ``INSERT`` execution per
row via ``cursor.executemany``. Each execution is one pgaudit session-audit
line, so a 392k-row streamed load produced roughly 731k pgaudit lines -- a
volume that itself became an operational and storage problem, independent of
the load's own cost (issue #456). Paging multiple rows into each execution
divides the audited-statement count by the page size.

These are statement-shape tests against a fake cursor -- no DB -- matching
``tests/test_writers_clear_table.py`` / ``tests/test_writers_upsert_guards.py``.
Behavioral / round-trip coverage against a real PostgreSQL lives in
``tests/integration/test_executemany_paging.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from moncpipelib.io_managers.writers import PG_MAX_BIND_PARAMS, _executemany_bulk

_BASE_SQL = "INSERT INTO s.t (a, b, c) VALUES %s"


class _FakeCursor:
    """Records ``execute`` / ``executemany`` calls; never touches a DB."""

    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Sequence[Any] | None]] = []
        self.executemany_calls: list[tuple[str, list[Sequence[Any]]]] = []

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        self.execute_calls.append((sql, params))

    def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self.executemany_calls.append((sql, list(params_seq)))


def _row_placeholder(n_cols: int) -> str:
    return "(" + ", ".join(["%s"] * n_cols) + ")"


def _rows(n_rows: int, n_cols: int) -> list[tuple[Any, ...]]:
    """Distinctive, order-traceable row values: row i, col j -> f"r{i}c{j}"."""
    return [tuple(f"r{i}c{j}" for j in range(n_cols)) for i in range(n_rows)]


def _flatten(rows: Sequence[tuple[Any, ...]]) -> list[Any]:
    return [v for row in rows for v in row]


class TestExecutemanyBulkPaging:
    """10 rows / 3 cols / page_rows=4 -> 2 full pages + a 2-row tail."""

    def test_full_pages_via_single_executemany_call(self) -> None:
        cursor = _FakeCursor()
        rows = _rows(10, 3)

        _executemany_bulk(cursor, _BASE_SQL, rows, page_rows=4)

        assert len(cursor.executemany_calls) == 1
        page_sql, param_lists = cursor.executemany_calls[0]
        assert page_sql.count(_row_placeholder(3)) == 4
        assert len(param_lists) == 2
        assert all(len(params) == 12 for params in param_lists)
        # Values arrive in row order across both pages.
        assert param_lists[0] == _flatten(rows[0:4])
        assert param_lists[1] == _flatten(rows[4:8])

    def test_remainder_via_single_execute_call(self) -> None:
        cursor = _FakeCursor()
        rows = _rows(10, 3)

        _executemany_bulk(cursor, _BASE_SQL, rows, page_rows=4)

        assert len(cursor.execute_calls) == 1
        tail_sql, tail_params = cursor.execute_calls[0]
        assert tail_sql.count(_row_placeholder(3)) == 2
        assert tail_params == _flatten(rows[8:10])


class TestExecutemanyBulkNoRemainder:
    def test_evenly_divisible_rows_have_no_tail_execute(self) -> None:
        """8 rows / page_rows=4 -> exactly 2 pages, no execute() call at all."""
        cursor = _FakeCursor()
        rows = _rows(8, 3)

        _executemany_bulk(cursor, _BASE_SQL, rows, page_rows=4)

        assert len(cursor.executemany_calls) == 1
        _, param_lists = cursor.executemany_calls[0]
        assert len(param_lists) == 2
        assert cursor.execute_calls == []


class TestExecutemanyBulkFewerRowsThanPage:
    def test_no_full_page_just_a_tail_execute(self) -> None:
        """3 rows / page_rows=10 -> zero full pages, one 3-row-arity execute."""
        cursor = _FakeCursor()
        rows = _rows(3, 3)

        _executemany_bulk(cursor, _BASE_SQL, rows, page_rows=10)

        assert cursor.executemany_calls == []
        assert len(cursor.execute_calls) == 1
        tail_sql, tail_params = cursor.execute_calls[0]
        assert tail_sql.count(_row_placeholder(3)) == 3
        assert tail_params == _flatten(rows)


class TestExecutemanyBulkParamCeiling:
    def test_wide_table_caps_effective_page_at_bind_param_ceiling(self) -> None:
        """200 cols / page_rows=500 -> capped to floor(65535 / 200) = 327.

        Each page statement then binds 327 * 200 = 65,400 parameters, under
        the PostgreSQL extended-query protocol's 65,535 (Int16) ceiling.
        """
        n_cols = 200
        expected_effective = PG_MAX_BIND_PARAMS // n_cols
        assert expected_effective == 327  # sanity-pin the worked example

        cursor = _FakeCursor()
        rows = _rows(400, n_cols)

        _executemany_bulk(cursor, _BASE_SQL, rows, page_rows=500)

        assert len(cursor.executemany_calls) == 1
        page_sql, param_lists = cursor.executemany_calls[0]
        page_tuple_count = page_sql.count(_row_placeholder(n_cols))
        assert page_tuple_count == expected_effective
        assert page_tuple_count * n_cols <= PG_MAX_BIND_PARAMS
        assert all(len(params) == expected_effective * n_cols for params in param_lists)

        # Remainder (400 - 327 = 73 rows) is a tail execute, also within
        # the ceiling.
        assert len(cursor.execute_calls) == 1
        tail_sql, tail_params = cursor.execute_calls[0]
        tail_tuple_count = tail_sql.count(_row_placeholder(n_cols))
        assert tail_tuple_count == 400 - expected_effective
        assert tail_tuple_count * n_cols <= PG_MAX_BIND_PARAMS
        assert len(tail_params) == tail_tuple_count * n_cols


class TestExecutemanyBulkLegacyBehavior:
    def test_page_rows_one_matches_pre_456_shape(self) -> None:
        """page_rows=1 (the old hard-coded shape) is byte-identical to before.

        Single-row placeholder rewrite, ``executemany`` called once with the
        row tuples passed through unchanged (not flattened), and no
        ``execute`` call.
        """
        cursor = _FakeCursor()
        rows = _rows(5, 3)

        _executemany_bulk(cursor, _BASE_SQL, rows, page_rows=1)

        assert cursor.execute_calls == []
        assert len(cursor.executemany_calls) == 1
        sql1, param_seq = cursor.executemany_calls[0]
        assert sql1.count(_row_placeholder(3)) == 1
        assert param_seq == list(rows)


class TestExecutemanyBulkEmptyRows:
    def test_empty_rows_issues_no_cursor_calls(self) -> None:
        cursor = _FakeCursor()

        _executemany_bulk(cursor, _BASE_SQL, [], page_rows=4)

        assert cursor.execute_calls == []
        assert cursor.executemany_calls == []


class TestExecutemanyBulkZeroWidthRows:
    """#456 review Fix 9: a zero-column row shape has no VALUES placeholder
    to build, so it must raise rather than silently emitting malformed SQL."""

    def test_zero_width_rows_raises(self) -> None:
        cursor = _FakeCursor()

        with pytest.raises(ValueError, match="_executemany_bulk"):
            _executemany_bulk(cursor, _BASE_SQL, [()], page_rows=4)


class TestExecutemanyBulkParametrizationGuarantee:
    """Row values must never be inlined into the audited statement text."""

    def test_sentinel_values_never_appear_in_sql_text(self) -> None:
        sentinel = "SENTINEL_PHI_9f3c"
        rows: list[tuple[Any, ...]] = [
            ("a", "b", "c"),
            (sentinel, "d", "e"),
            ("f", "g", "h"),
            ("i", "j", sentinel),
            ("k", "l", "m"),
        ]
        cursor = _FakeCursor()

        _executemany_bulk(cursor, _BASE_SQL, rows, page_rows=2)

        all_sql = [sql for sql, _ in cursor.execute_calls]
        all_sql += [sql for sql, _ in cursor.executemany_calls]
        assert all_sql, "expected at least one recorded statement"
        for sql_text in all_sql:
            assert sentinel not in sql_text
