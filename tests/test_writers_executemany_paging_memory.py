"""Memory-profile test for ``_executemany_bulk``'s paged branch (#456 review Fix 3/4).

Pre-fix, the paged branch built ``page_params`` as an eager list-of-lists
comprehension: every full page's flattened parameter list was materialized
into one Python ``list`` *before* ``cursor.executemany`` was even called.
psycopg3's real ``executemany`` consumes ``params_seq`` lazily, pipelining
one page per iteration -- so that eager list was a second full copy of the
paged row data held in memory for no benefit, and its size scaled with the
total row count.

Post-fix, ``_page_params()`` is a generator that yields one flattened page
at a time, computed via direct index arithmetic (no intermediate per-page
slice list, no upfront list-of-pages). Peak heap during the call should
therefore be a small, roughly constant multiple of one page's size,
regardless of how many rows are paged -- matching the boundary-streaming
acceptance pattern in ``tests/test_ingest_streaming_memory.py``.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterable, Sequence
from typing import Any

from moncpipelib.io_managers.writers import _executemany_bulk

_N_COLS = 5
_PAGE_ROWS = 100
# Distinctive fixed-width string values so per-row/per-page byte size is
# predictable when reasoning about the "clearly regressed" ceiling below.
_VALUE_WIDTH = 32


class _LazyCursor:
    """Drains ``params_seq`` one page at a time.

    Mirrors psycopg3's real ``executemany``, which pipelines pages rather
    than materializing the whole sequence up front. Deliberately does NOT
    do ``list(params_seq)`` (unlike the unit-test ``_FakeCursor`` in
    ``tests/test_writers_executemany_paging.py``) -- that would absorb an
    upstream eager-list regression into this cursor's own materialization
    and hide exactly the allocation pattern this test pins against.
    """

    def __init__(self) -> None:
        self.n_pages_drained = 0

    def executemany(self, sql: str, params_seq: Iterable[list[Any]]) -> None:
        del sql
        for _page in params_seq:
            self.n_pages_drained += 1

    def execute(self, sql: str, params: Sequence[Any]) -> None:
        del sql, params


def _rows(n_rows: int, n_cols: int) -> list[tuple[Any, ...]]:
    return [
        tuple(f"r{i}c{j}".ljust(_VALUE_WIDTH, "x") for j in range(n_cols)) for i in range(n_rows)
    ]


def _peak_heap_for_paging(n_rows: int) -> int:
    """Peak additional Python heap (bytes) allocated during one paged call.

    ``rows`` is built and ``tracemalloc.reset_peak()`` is called *before*
    entering the timed section, so constructing the input itself is not
    counted -- only what ``_executemany_bulk`` allocates while paging it.
    ``page_rows=100`` divides both row counts evenly, so every row is paged
    (no tail-execute contribution to compare across sizes).
    """
    rows = _rows(n_rows, _N_COLS)
    cursor = _LazyCursor()
    sql = "INSERT INTO s.t (" + ", ".join(f"c{j}" for j in range(_N_COLS)) + ") VALUES %s"

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        _executemany_bulk(cursor, sql, rows, page_rows=_PAGE_ROWS)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert cursor.n_pages_drained == n_rows // _PAGE_ROWS
    return peak


def test_paging_peak_heap_is_constant_not_proportional_to_row_count() -> None:
    """200k rows must not peak materially higher than 50k rows.

    Pre-fix, the eager ``page_params`` list comprehension flattened every
    full page up front: one Python list whose total size scales with the
    row count being paged. Post-fix, ``_page_params()`` yields one
    flattened page at a time, so peak heap should track a small, roughly
    constant number of page-sizes -- not the input size.

    If a future change reintroduces eager materialization of every page
    (e.g. ``list(_page_params())`` before ``executemany``), peak at 200k
    rows will jump to roughly 4x the 50k-row peak and this test will fail.
    """
    small_peak = _peak_heap_for_paging(50_000)
    large_peak = _peak_heap_for_paging(200_000)

    # A full eager flatten of 200k rows x 5 cols of ~32-byte string values
    # is on the order of several MB (200_000 * 5 * ~90 bytes/str object
    # overhead-inclusive, plus the wrapping list-of-lists). Use a fraction
    # of that as the "clearly regressed" ceiling -- generous enough to
    # absorb tracemalloc/interpreter overhead, but well below what a full
    # eager flatten of the paged data would allocate.
    clearly_regressed_ceiling_bytes = 200_000 * _N_COLS * 90 // 4
    assert large_peak < clearly_regressed_ceiling_bytes, (
        f"peak heap for 200k rows was {large_peak:,} bytes -- looks like a "
        f"full eager flatten of the paged data, not page-at-a-time streaming "
        f"(ceiling: {clearly_regressed_ceiling_bytes:,} bytes)"
    )

    # The core assertion: quadrupling the row count should not materially
    # change peak heap. A generous additive + multiplicative margin absorbs
    # tracemalloc/GC/allocator noise while still catching an O(n) regression
    # (which would show up as roughly a 4x jump, not noise).
    assert large_peak <= small_peak * 2 + 200_000, (
        f"peak heap scaled with row count: {small_peak:,} bytes at 50k rows "
        f"vs {large_peak:,} bytes at 200k rows (4x the rows) -- expected a "
        f"roughly constant peak, not one proportional to input size"
    )
