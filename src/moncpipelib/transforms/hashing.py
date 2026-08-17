"""Deterministic hashing utilities for Polars DataFrames."""

from __future__ import annotations

import hashlib

import polars as pl


def compute_row_hash(
    columns: list[str],
    *,
    alias: str = "row_hash",
    separator: str = "|",
    null_sentinel: str = "\x00",
) -> pl.Expr:
    """Compute a deterministic SHA-256 hash over the specified columns.

    Creates a canonical string representation by casting all columns to String,
    replacing nulls with a sentinel, concatenating with a separator, then
    hashing with SHA-256. The result is a 64-character hex string.

    This hash is deterministic across sessions and platforms, making it
    suitable for change detection in SCD2 pipelines.

    Args:
        columns: Column names to include in the hash. Order matters -- the
            same columns in a different order will produce different hashes.
        alias: Output column name. Defaults to ``"row_hash"``.
        separator: Delimiter between column values. Defaults to ``"|"``.
        null_sentinel: Replacement for null values. Defaults to ``"\\x00"``.

    Returns:
        Polars expression producing a String column of 64-character SHA-256
        hex digests.

    Example:
        .. code-block:: python

            df = df.with_columns(
                compute_row_hash(["col_a", "col_b", "col_c"]).alias("row_hash")
            )
    """
    if not columns:
        msg = "columns must be a non-empty list"
        raise ValueError(msg)

    # Build expressions: cast each column to String, replace nulls with sentinel
    col_exprs = [pl.col(c).cast(pl.String).fill_null(null_sentinel) for c in columns]

    # Concatenate all columns with separator, then hash via map_batches
    concatenated = pl.concat_str(col_exprs, separator=separator)

    return concatenated.map_batches(lambda s: _sha256_series(s), return_dtype=pl.String).alias(
        alias
    )


def _sha256_series(series: pl.Series) -> pl.Series:
    """Apply SHA-256 to each value in a Polars String Series.

    Args:
        series: A String Series where each element is a concatenated
            row representation.

    Returns:
        A String Series of 64-character hex digests.
    """
    return pl.Series(
        [
            hashlib.sha256(val.encode("utf-8")).hexdigest() if val is not None else None
            for val in series.to_list()
        ],
        dtype=pl.String,
    )
