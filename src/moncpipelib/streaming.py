"""Streaming utilities for batched data processing in Dagster pipelines.

Provides types and helpers for end-to-end streaming pipelines that process
large datasets in batches without materializing the full dataset in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import polars as pl
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class BatchedDataFrame:
    """An iterator of Polars DataFrames representing a streaming result set.

    This type is recognized by PostgresIOManager.handle_output() and triggers
    batch-by-batch writing instead of single-DataFrame writing, enabling
    pipelines to process datasets of arbitrary size within fixed memory limits.

    The iterator is consumed exactly once during handle_output(). After
    consumption, summary statistics are available via the instance attributes.

    Attributes:
        batches: Iterator yielding pl.DataFrame objects. Each batch should
            have identical schemas (same columns, types).
        total_rows_hint: Optional estimated total rows (for progress logging).
            If provided, log messages include percentage progress. Does not
            need to be exact.
        rows_written: Total rows written. Populated after iteration by the
            IO manager (initially 0).
        batches_written: Total batches written. Populated after iteration by
            the IO manager (initially 0).

    Example:
        ```python
        from moncpipelib import BatchedDataFrame, transform_batched

        @op(
            required_resource_keys={"database"},
            out=Out(io_manager_key="silver_io_manager"),
        )
        def extract_and_transform(context: OpExecutionContext) -> BatchedDataFrame:
            database = context.resources.database
            batches = database.read_batched(
                "SELECT * FROM bronze.large_table",
                batch_size=50_000,
                context=context,
            )
            return transform_batched(batches, transform_fn=clean_batch)

        def clean_batch(df: pl.DataFrame) -> pl.DataFrame:
            return df.lazy().select([
                safe_int("id"),
                clean_text("name"),
            ]).collect(engine="streaming")
        ```
    """

    batches: Iterator[pl.DataFrame]
    total_rows_hint: int | None = None

    # Populated after iteration (by IO manager)
    rows_written: int = field(default=0, init=False)
    batches_written: int = field(default=0, init=False)


def transform_batched(
    batches: Iterator[pl.DataFrame],
    transform_fn: Callable[[pl.DataFrame], pl.DataFrame],
    *,
    total_rows_hint: int | None = None,
) -> BatchedDataFrame:
    """Apply a transform function to each batch in a streaming iterator.

    This connects read_batched() to the IO manager without materializing
    the full dataset. Each batch is transformed independently as it passes
    through the pipeline.

    Args:
        batches: Iterator of DataFrames (from database.read_batched()).
        transform_fn: Function that transforms a single batch DataFrame.
            Must accept and return a pl.DataFrame. Applied to each batch
            independently. The function should be pure (no side effects)
            and should not depend on state from other batches.
        total_rows_hint: Optional estimated total rows for progress logging.
            Passed through to the BatchedDataFrame for IO manager progress
            tracking.

    Returns:
        BatchedDataFrame suitable for returning from an op/asset. The IO
        manager will consume the iterator, writing each transformed batch.

    Warning:
        The transform_fn must work on individual batches independently.
        Cross-batch operations (global deduplication, percentile
        calculations, running aggregates) are not supported -- and they
        fail *silently*, not loudly. A ``df.unique(subset=business_key)``
        inside transform_fn dedupes within each batch only; duplicate keys
        spanning batch boundaries pass straight through (observed in
        production as data-platform's umls_vocabularies incident, #401
        item 5). The output is plausible enough that nothing downstream
        flags it.

        For operations that need the whole dataset, either:

        - push the operation into the extraction SQL (``SELECT DISTINCT
          ON (key) ... ORDER BY key, ...`` dedupes before batching),
        - write with ``mode="upsert"`` and the dedup key as
          ``primary_key`` (the staging merge collapses cross-batch
          duplicates last-write-wins), or
        - materialize via ``read_batched_to_dataframe()`` and accept the
          memory cost.

    Example:
        ```python
        from moncpipelib import transform_batched, safe_int, clean_text

        @op(out=Out(io_manager_key="silver_io_manager"),
            required_resource_keys={"database"})
        def extract_and_transform(context: OpExecutionContext) -> BatchedDataFrame:
            database = context.resources.database
            batches = database.read_batched(
                "SELECT * FROM bronze.large_table",
                batch_size=50_000,
                context=context,
            )
            return transform_batched(batches, transform_fn=clean_batch)

        def clean_batch(df: pl.DataFrame) -> pl.DataFrame:
            '''Transform a single batch of data.'''
            return (
                df.lazy()
                .select([
                    safe_int("base_account_no"),
                    clean_text("name"),
                    safe_date("created_at"),
                ])
                .collect(engine="streaming")
            )
        ```
    """

    def _transformed_iter() -> Iterator[pl.DataFrame]:
        for batch in batches:
            yield transform_fn(batch)

    return BatchedDataFrame(
        batches=_transformed_iter(),
        total_rows_hint=total_rows_hint,
    )


_DEFAULT_PARQUET_BATCH_ROWS: int = 50_000
"""Default parquet read batch size (rows).

Matches the DB ``read_batched`` default; a middle ground between per-batch
overhead and peak memory.  A parquet row group is read at most one batch
at a time, so peak heap tracks the batch, not the file (#439)."""


def stream_parquet_batches(
    paths: Sequence[str | Path],
    *,
    batch_size: int = _DEFAULT_PARQUET_BATCH_ROWS,
    columns: list[str] | None = None,
    all_text: bool = True,
) -> Iterator[pl.DataFrame]:
    """Stream one or more parquet files as row-bounded ``pl.DataFrame`` batches.

    First-class parquet read support for bronze consumers (#439): the
    Trilliant ``visits_oncology`` feed lands as N snappy-parquet parts per
    partition (see the multi-file resolver), and this reader scans the
    part list as one logical stream into ``BatchedDataFrame`` for
    ``PostgresResource.write``.

    Memory is bounded by ``batch_size`` rows regardless of file size: each
    file is read via ``pyarrow.parquet.ParquetFile.iter_batches`` (a
    row-group-bounded iterator), so the whole asset -- their assets run to
    100M-13B rows -- is never materialized.  ``pl.scan_parquet(...).collect()``
    is deliberately NOT used: ``collect`` returns a single materialized
    frame and would defeat the bound (design decision D4).

    Args:
        paths: Ordered parquet file paths.  Read in the given order so a
            multi-part asset streams deterministically; the resolver
            returns parts sorted (``part-00001`` ...).  Paths must be
            seekable local files -- parquet's footer is at end-of-file, so
            a forward-only blob stream cannot be scanned; download parts to
            disk first.
        batch_size: Rows per yielded batch.
        columns: Optional column subset to read; ``None`` reads all
            columns (the bronze default -- the expected-columns guard runs
            downstream).
        all_text: When ``True`` (default), every column is cast to
            ``pl.String`` -- the bronze-verbatim contract (bronze is
            all-text; typing belongs in silver).  Set ``False`` to
            preserve the parquet's native dtypes.

    Yields:
        ``pl.DataFrame`` batches.  An empty / zero-row file yields nothing.
    """
    for path in paths:
        parquet_file = pq.ParquetFile(str(path))
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            frame = cast("pl.DataFrame", pl.from_arrow(record_batch))
            if all_text:
                frame = frame.select(pl.all().cast(pl.String))
            yield frame


def read_parquet_batched(
    paths: Sequence[str | Path],
    *,
    batch_size: int = _DEFAULT_PARQUET_BATCH_ROWS,
    columns: list[str] | None = None,
    all_text: bool = True,
    total_rows_hint: int | None = None,
) -> BatchedDataFrame:
    """Wrap :func:`stream_parquet_batches` as a :class:`BatchedDataFrame`.

    Convenience for the common asset shape::

        @asset(...)
        def bronze_visits_oncology(context, database) -> WriteResult:
            batched = read_parquet_batched(part_paths)
            return database.write(batched, target="bronze.visits_oncology", context=context)

    Peak memory is bounded by ``batch_size`` rows (see
    :func:`stream_parquet_batches`).
    """
    return BatchedDataFrame(
        batches=stream_parquet_batches(
            paths, batch_size=batch_size, columns=columns, all_text=all_text
        ),
        total_rows_hint=total_rows_hint,
    )
