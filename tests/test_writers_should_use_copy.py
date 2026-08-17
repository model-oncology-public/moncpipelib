"""Unit tests for bulk-insert COPY-vs-execute_values selection.

``should_use_copy``'s AUTO branch previously sized the decision on the
per-call ``row_count`` alone. Streamed batched writes call ``insert_rows``
once per batch (typically ~1,000-row parts), so a large streamed load never
reached the AUTO threshold even when its total was hundreds of thousands of
rows -- the same shape of bug as the full-refresh clear-method sizing fixed by
model-oncology-public/moncpipelib#4, and the root cause of the pgaudit-volume
incident in #456 (a 392k-row streamed load produced ~731k pgaudit lines, one
per per-row INSERT execution). ``total_rows_hint`` lets the caller pass the
estimated stream total so AUTO sizes on ``max(row_count, total_rows_hint)``
instead of the individual batch alone.

There was previously no test coverage of the AUTO branch at all; this file
closes that gap alongside the new ``total_rows_hint`` parameter.

Statement-shape / pure-predicate tests only -- no cursor, no DB. Matches
``tests/test_writers_clear_table.py``. Behavioral coverage against a real
PostgreSQL lives in ``tests/integration/test_stream_total_copy_sizing.py``.
"""

from __future__ import annotations

import polars as pl
import pytest

from moncpipelib.io_managers.enums import BulkInsertMethod, FullRefreshMethod, WriteMode
from moncpipelib.io_managers.writers import WriterConfig, _frame_supports_copy, should_use_copy


def _config(
    bulk_insert_method: BulkInsertMethod = BulkInsertMethod.AUTO,
    bulk_insert_threshold: int = 10_000,
) -> WriterConfig:
    return WriterConfig(
        bulk_insert_method=bulk_insert_method,
        bulk_insert_threshold=bulk_insert_threshold,
        full_refresh_method=FullRefreshMethod.AUTO,
        full_refresh_threshold=10_000,
        insert_chunk_size=None,
    )


class TestShouldUseCopyStreamedHintReachesThreshold:
    """The #456 regression surface: sub-threshold batches, large stream total."""

    @pytest.mark.parametrize("write_mode", [WriteMode.FULL_REFRESH, WriteMode.APPEND])
    def test_small_batch_with_large_hint_uses_copy(self, write_mode: WriteMode) -> None:
        assert should_use_copy(_config(), 1_300, write_mode, total_rows_hint=392_000) is True


class TestShouldUseCopyAutoSizing:
    """``max(row_count, total_rows_hint or 0) >= threshold`` -- hint never downgrades."""

    def test_small_batch_no_hint_stays_execute_values(self) -> None:
        assert should_use_copy(_config(), 1_300, WriteMode.APPEND, total_rows_hint=None) is False

    def test_small_batch_small_hint_stays_execute_values(self) -> None:
        assert should_use_copy(_config(), 1_300, WriteMode.APPEND, total_rows_hint=5_000) is False

    def test_large_batch_no_hint_uses_copy(self) -> None:
        assert should_use_copy(_config(), 12_000, WriteMode.APPEND, total_rows_hint=None) is True

    def test_large_batch_small_hint_still_uses_copy(self) -> None:
        """A batch already at threshold is never downgraded by a small hint."""
        assert should_use_copy(_config(), 12_000, WriteMode.APPEND, total_rows_hint=100) is True


class TestShouldUseCopyHintZeroMatchesNone:
    """A hint of ``0`` contributes nothing to the max, same as no hint at all.

    Contrast with ``clear_table``'s ``row_count_hint``, where an explicit
    ``0`` IS taken at face value as a real measured count. There is no
    equivalent "real zero" case for this predicate: a hint of 0 and an
    absent hint both leave sizing entirely to ``row_count``.
    """

    @pytest.mark.parametrize(("row_count", "expected"), [(1_300, False), (12_000, True)])
    def test_hint_zero_behaves_like_hint_none(self, row_count: int, expected: bool) -> None:
        with_zero = should_use_copy(_config(), row_count, WriteMode.APPEND, total_rows_hint=0)
        with_none = should_use_copy(_config(), row_count, WriteMode.APPEND, total_rows_hint=None)
        assert with_zero is expected
        assert with_none is expected


class TestShouldUseCopyModeGate:
    """UPSERT / SCD2 never use COPY, regardless of hint size."""

    @pytest.mark.parametrize("write_mode", [WriteMode.UPSERT, WriteMode.SCD2])
    def test_incompatible_mode_wins_over_huge_hint(self, write_mode: WriteMode) -> None:
        assert should_use_copy(_config(), 100, write_mode, total_rows_hint=10_000_000) is False


class TestShouldUseCopyExplicitMethod:
    """An explicit method short-circuits AUTO sizing entirely, hint included."""

    def test_explicit_execute_values_ignores_huge_hint(self) -> None:
        config = _config(BulkInsertMethod.EXECUTE_VALUES)
        assert should_use_copy(config, 100, WriteMode.APPEND, total_rows_hint=10_000_000) is False

    @pytest.mark.parametrize("write_mode", [WriteMode.APPEND, WriteMode.FULL_REFRESH])
    def test_explicit_copy_ignores_small_batch(self, write_mode: WriteMode) -> None:
        config = _config(BulkInsertMethod.COPY)
        assert should_use_copy(config, 5, write_mode) is True


class TestFrameSupportsCopy:
    """#456 review Fix 2: ``df.write_csv`` raises ``ComputeError`` for any
    nested-dtype column (List/Array/Struct), so a schema carrying one must
    never be routed to the COPY path regardless of what ``should_use_copy``
    alone decides.
    """

    def test_plain_scalar_columns_support_copy(self) -> None:
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert _frame_supports_copy(df) is True

    def test_list_column_does_not_support_copy(self) -> None:
        df = pl.DataFrame({"a": [1, 2], "tags": [["x"], ["y", "z"]]})
        assert _frame_supports_copy(df) is False

    def test_struct_column_does_not_support_copy(self) -> None:
        df = pl.DataFrame({"a": [1], "meta": [{"k": "v"}]})
        assert _frame_supports_copy(df) is False

    def test_array_column_does_not_support_copy(self) -> None:
        df = pl.DataFrame(
            {"a": [1, 2], "fixed": [[1, 2], [3, 4]]},
            schema={"a": pl.Int64, "fixed": pl.Array(pl.Int64, 2)},
        )
        assert _frame_supports_copy(df) is False
