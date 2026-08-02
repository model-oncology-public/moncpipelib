"""Tests for the first-class parquet reader (#439).

Real parquet fixtures written with pyarrow; the reader is exercised for
single- and multi-file scans, the bronze all-text cast, batch sizing,
column subsetting, and the BatchedDataFrame wrapper.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from moncpipelib.streaming import (
    BatchedDataFrame,
    read_parquet_batched,
    stream_parquet_batches,
)


def _write_parquet(path: Path, table: pa.Table) -> Path:
    pq.write_table(table, path)
    return path


def _typed_table(ids: list[int], names: list[str], amts: list[float]) -> pa.Table:
    return pa.table({"id": ids, "name": names, "amt": amts})


def test_single_file_all_text(tmp_path: Path) -> None:
    p = _write_parquet(tmp_path / "a.parquet", _typed_table([1, 2], ["x", "y"], [1.5, 2.5]))
    batches = list(stream_parquet_batches([p]))
    assert len(batches) == 1
    df = batches[0]
    # bronze-verbatim: every column cast to String
    assert df.dtypes == [pl.String, pl.String, pl.String]
    assert df.to_dicts() == [
        {"id": "1", "name": "x", "amt": "1.5"},
        {"id": "2", "name": "y", "amt": "2.5"},
    ]


def test_all_text_false_preserves_dtypes(tmp_path: Path) -> None:
    p = _write_parquet(tmp_path / "a.parquet", _typed_table([1], ["x"], [1.5]))
    [df] = list(stream_parquet_batches([p], all_text=False))
    assert df.dtypes == [pl.Int64, pl.String, pl.Float64]


def test_batch_size_splits_rows(tmp_path: Path) -> None:
    p = _write_parquet(
        tmp_path / "a.parquet",
        _typed_table([1, 2, 3, 4, 5], ["a", "b", "c", "d", "e"], [1.0] * 5),
    )
    batches = list(stream_parquet_batches([p], batch_size=2))
    assert [b.height for b in batches] == [2, 2, 1]
    combined = pl.concat(batches)
    assert combined.height == 5


def test_multi_file_scan_is_one_logical_stream(tmp_path: Path) -> None:
    p1 = _write_parquet(tmp_path / "part-00001.parquet", _typed_table([1], ["a"], [1.0]))
    p2 = _write_parquet(tmp_path / "part-00002.parquet", _typed_table([2], ["b"], [2.0]))
    batches = list(stream_parquet_batches([p1, p2]))
    combined = pl.concat(batches)
    assert combined["id"].to_list() == ["1", "2"]  # order preserved across parts


def test_column_subset(tmp_path: Path) -> None:
    p = _write_parquet(tmp_path / "a.parquet", _typed_table([1], ["x"], [1.5]))
    [df] = list(stream_parquet_batches([p], columns=["id", "amt"]))
    assert df.columns == ["id", "amt"]


def test_empty_file_yields_no_batches(tmp_path: Path) -> None:
    empty = pa.table({"id": pa.array([], type=pa.int64()), "name": pa.array([], type=pa.string())})
    p = _write_parquet(tmp_path / "empty.parquet", empty)
    assert list(stream_parquet_batches([p])) == []


def test_read_parquet_batched_wraps_batched_dataframe(tmp_path: Path) -> None:
    p = _write_parquet(tmp_path / "a.parquet", _typed_table([1, 2, 3], ["a", "b", "c"], [1.0] * 3))
    result = read_parquet_batched([p], batch_size=2, total_rows_hint=3)
    assert isinstance(result, BatchedDataFrame)
    assert result.total_rows_hint == 3
    total = sum(b.height for b in result.batches)
    assert total == 3
