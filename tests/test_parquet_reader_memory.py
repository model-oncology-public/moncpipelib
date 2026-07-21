"""Streaming-memory acceptance test for the parquet reader (#439).

Reads a multi-row-group parquet whose fully-materialized all-text frame
would be tens of MiB, and asserts the batched reader's peak heap tracks
one batch, not the whole file.  This is the bound that lets the same
reader serve Trilliant's 100M-13B-row assets (this repo's use is a
sampled subset, but the reader must not assume small).

If the reader materialized the whole file (e.g.
``scan_parquet(...).collect()``), peak would scale with the file and
blow the threshold.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from moncpipelib.streaming import stream_parquet_batches

_ROWS = 1_000_000
_ROW_GROUP = 50_000
_BATCH = 50_000
# One all-text batch is ~1.5-3 MiB; the full 1M-row all-text frame is
# tens of MiB.  A 16 MiB ceiling is comfortably above one batch yet well
# below a full materialization.
_PEAK_THRESHOLD_BYTES = 16 * 1024 * 1024


def _write_large_parquet(path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array(range(_ROWS), type=pa.int64()),
            "name": pa.array([f"row-{i:09d}" for i in range(_ROWS)], type=pa.string()),
            "amt": pa.array([float(i) for i in range(_ROWS)], type=pa.float64()),
        }
    )
    pq.write_table(table, path, row_group_size=_ROW_GROUP)


def test_parquet_reader_peak_memory_is_bounded(tmp_path: Path) -> None:
    p = tmp_path / "big.parquet"
    _write_large_parquet(p)

    total_rows = 0
    tracemalloc.start()
    tracemalloc.reset_peak()
    for batch in stream_parquet_batches([p], batch_size=_BATCH):
        total_rows += batch.height
        # Do NOT retain the batch -- a consumer streams it onward (COPY to
        # Postgres) and lets it fall out of scope.
        del batch
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert total_rows == _ROWS
    assert peak < _PEAK_THRESHOLD_BYTES, (
        f"peak {peak / 1024 / 1024:.1f} MiB exceeded "
        f"{_PEAK_THRESHOLD_BYTES / 1024 / 1024:.0f} MiB for a {_ROWS:,}-row file "
        f"-- parquet reader is materializing more than one batch"
    )
