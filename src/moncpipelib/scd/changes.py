"""Standalone SCD Type 2 change detection for Polars DataFrames.

This module provides Python-side change detection for pipelines that need
pre-write visibility into what changed. The actual atomic write is handled
by the IO manager's Postgres-side CTE -- this utility is informational.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True)
class SCD2ChangeResult:
    """Result of SCD2 change detection between incoming and current data.

    Attributes:
        new_records: Rows with business keys not present in current target.
        changed_records: Rows with business keys present but hash differs.
        unchanged_records: Rows with business keys present and hash matches.
        deleted_keys: Business keys in current but not in incoming. Only
            populated when ``detect_deletes=True``; otherwise an empty DataFrame.
        summary: Counts for logging: ``new``, ``changed``, ``unchanged``,
            ``deleted``, ``total_incoming``.
    """

    new_records: pl.DataFrame
    changed_records: pl.DataFrame
    unchanged_records: pl.DataFrame
    deleted_keys: pl.DataFrame
    summary: dict[str, int] = field(default_factory=dict)


def detect_changes(
    incoming: pl.DataFrame,
    current: pl.DataFrame,
    business_key: str | list[str],
    hash_col: str = "row_hash",
    *,
    detect_deletes: bool = False,
) -> SCD2ChangeResult:
    """Detect SCD2 changes between incoming data and current dimension state.

    Performs Python-side (Polars) change detection for informational or
    logging purposes. The actual atomic write should be handled by the
    IO manager's Postgres-side CTE.

    Use this when you need to:

    - Log change summaries before writing
    - Apply custom filtering to changed/new rows
    - Feed change statistics to external monitoring

    Both DataFrames must contain the ``hash_col``. Use
    :func:`~moncpipelib.transforms.compute_row_hash` to add it.

    Args:
        incoming: New data with row hash computed.
        current: Current dimension state (``is_current=True`` rows) with
            row hash.
        business_key: Column(s) that uniquely identify a business entity.
            Accepts a single string or a list of strings for composite keys.
        hash_col: Name of the hash column. Defaults to ``"row_hash"``.
        detect_deletes: If True, also identify keys present in current
            but missing from incoming. Defaults to False.

    Returns:
        :class:`SCD2ChangeResult` with categorized rows and summary counts.

    Raises:
        ValueError: If ``hash_col`` is missing from either DataFrame, or
            if ``business_key`` columns are missing.
    """
    if isinstance(business_key, str):
        business_key = [business_key]

    # Validate required columns
    for label, df in [("incoming", incoming), ("current", current)]:
        missing_bk = [k for k in business_key if k not in df.columns]
        if missing_bk:
            msg = (
                f"business_key column(s) {missing_bk} not found in {label} DataFrame. "
                f"Available columns: {sorted(df.columns)}"
            )
            raise ValueError(msg)
        if hash_col not in df.columns:
            msg = (
                f"hash_col '{hash_col}' not found in {label} DataFrame. "
                f"Available columns: {sorted(df.columns)}. "
                f"Use compute_row_hash() to add it."
            )
            raise ValueError(msg)

    # Use lazy evaluation with streaming engine for reduced memory usage.
    # collect_all() enables Common Subplan Elimination so the shared join
    # is computed once, and streaming caps intermediate memory via morsels.
    from moncpipelib.config import POLARS_ENGINE

    # New records: business keys in incoming but not in current
    new_records = (
        incoming.lazy()
        .join(current.lazy(), on=business_key, how="anti")
        .collect(engine=POLARS_ENGINE)
    )

    # Matched records: inner join on business key (shared lazy plan)
    current_subset = current.select([*business_key, hash_col])
    matched_lf = incoming.lazy().join(
        current_subset.lazy(),
        on=business_key,
        how="inner",
        suffix="_current",
    )

    current_hash_col = f"{hash_col}_current"

    # Changed: matched rows where hash differs
    q_changed = matched_lf.filter(pl.col(hash_col) != pl.col(current_hash_col)).drop(
        current_hash_col
    )

    # Unchanged: matched rows where hash is the same
    q_unchanged = matched_lf.filter(pl.col(hash_col) == pl.col(current_hash_col)).drop(
        current_hash_col
    )

    # Collect both from the shared join plan (CSPE deduplicates the join)
    changed_records, unchanged_records = pl.collect_all(
        [q_changed, q_unchanged], engine=POLARS_ENGINE
    )

    # Deleted keys: in current but not in incoming
    if detect_deletes:
        deleted_keys = (
            current.lazy()
            .select(business_key)
            .join(incoming.lazy().select(business_key), on=business_key, how="anti")
            .collect(engine=POLARS_ENGINE)
        )
    else:
        deleted_keys = pl.DataFrame(
            {k: pl.Series([], dtype=current[k].dtype) for k in business_key}
        )

    summary = {
        "new": len(new_records),
        "changed": len(changed_records),
        "unchanged": len(unchanged_records),
        "deleted": len(deleted_keys),
        "total_incoming": len(incoming),
    }

    return SCD2ChangeResult(
        new_records=new_records,
        changed_records=changed_records,
        unchanged_records=unchanged_records,
        deleted_keys=deleted_keys,
        summary=summary,
    )
