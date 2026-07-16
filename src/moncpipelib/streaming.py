"""Streaming utilities for batched data processing in Dagster pipelines.

Provides types and helpers for end-to-end streaming pipelines that process
large datasets in batches without materializing the full dataset in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import polars as pl


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
