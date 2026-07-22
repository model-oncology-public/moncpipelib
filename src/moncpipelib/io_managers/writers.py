"""Write execution functions for PostgresIOManager.

Contains the actual SQL execution logic for each write mode:
full_refresh, upsert, append, and SCD2. Also provides partition-scoped
write support for partition-aware Dagster assets. All functions are
stateless -- configuration is passed via WriterConfig rather than
accessing IO manager instance attributes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Any

import polars as pl
import psycopg
from psycopg import sql

from moncpipelib.config import (
    AUTO_BATCH_THRESHOLD,
    SCD2Config,
    parse_schema_table,
)
from moncpipelib.io_managers.enums import (
    BulkInsertMethod,
    FullRefreshMethod,
    WriteMode,
)
from moncpipelib.resources.types import LoggingContext


def _executemany_bulk(
    cursor: psycopg.Cursor,
    sql: str,
    rows: Sequence[Sequence[Any]],
) -> None:
    """Execute a bulk INSERT via psycopg3's pipelined ``executemany``.

    The SQL templates this module emits use the legacy ``VALUES %s``
    placeholder shape -- the multi-row form psycopg2's ``execute_values``
    expanded.  psycopg3 has no equivalent expansion; instead its rewritten
    ``executemany`` pipelines per-row INSERT statements over the
    extended-query protocol, which requires per-column placeholders.
    This helper rewrites ``VALUES %s`` to ``VALUES (%s, %s, ..., %s)``
    (one placeholder per column, inferred from the first row) before
    calling ``cursor.executemany``.
    """
    if not rows:
        return
    n_cols = len(rows[0])
    placeholder = "(" + ", ".join(["%s"] * n_cols) + ")"
    rewritten_sql = sql.replace("VALUES %s", f"VALUES {placeholder}", 1)
    cursor.executemany(rewritten_sql, rows)


@dataclass(frozen=True, slots=True)
class WriterConfig:
    """Bundled configuration for write execution methods.

    Constructed by PostgresIOManager from its constructor attributes
    and passed to writer functions to avoid coupling writers to
    the IO manager instance.
    """

    bulk_insert_method: BulkInsertMethod
    bulk_insert_threshold: int
    full_refresh_method: FullRefreshMethod
    full_refresh_threshold: int
    insert_chunk_size: int | None


def should_use_truncate(config: WriterConfig, row_count: int | None) -> bool:
    """Determine whether to use TRUNCATE or DELETE for full refresh.

    Args:
        config: Writer configuration.
        row_count: Estimated number of rows the clear will have to remove, or
            ``None`` when no estimate is available. ``0`` is a real count and
            is treated as such; only ``None`` means "unknown".

    Returns:
        True if TRUNCATE should be used, False for DELETE. Under AUTO an
        unknown row count resolves to DELETE, the lower-lock option.
    """
    if config.full_refresh_method == FullRefreshMethod.TRUNCATE:
        return True
    elif config.full_refresh_method == FullRefreshMethod.DELETE:
        return False
    else:  # AUTO - decide based on how much data the clear has to remove
        if row_count is None:
            return False
        return row_count >= config.full_refresh_threshold


def should_use_copy(config: WriterConfig, row_count: int, write_mode: WriteMode) -> bool:
    """Determine whether to use COPY protocol for bulk insert.

    COPY is only compatible with append and full_refresh modes since it
    doesn't support ON CONFLICT clauses.

    Args:
        config: Writer configuration.
        row_count: Number of rows in the incoming DataFrame.
        write_mode: The write mode being used.

    Returns:
        True if COPY protocol should be used, False for execute_values.
    """
    # COPY doesn't support ON CONFLICT, so exclude upsert and scd2
    if write_mode not in (WriteMode.APPEND, WriteMode.FULL_REFRESH):
        return False

    if config.bulk_insert_method == BulkInsertMethod.COPY:
        return True
    elif config.bulk_insert_method == BulkInsertMethod.EXECUTE_VALUES:
        return False
    else:  # AUTO - decide based on DataFrame size
        return row_count >= config.bulk_insert_threshold


def get_effective_chunk_size(config: WriterConfig, row_count: int) -> int | None:
    """Determine chunk size for insert operation.

    Args:
        config: Writer configuration.
        row_count: Number of rows in the DataFrame.

    Returns:
        Chunk size to use, or None to process all rows at once.
    """
    # Default threshold and chunk size for auto mode
    auto_threshold = AUTO_BATCH_THRESHOLD
    auto_chunk_size = AUTO_BATCH_THRESHOLD

    if config.insert_chunk_size == 0:
        # Explicitly disabled
        return None
    elif config.insert_chunk_size is not None:
        # Explicit chunk size set
        return config.insert_chunk_size
    else:
        # Auto mode: chunk large DataFrames
        if row_count >= auto_threshold:
            return auto_chunk_size
        return None


def insert_with_copy(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    df: pl.DataFrame,
    context: LoggingContext,
) -> int:
    """Insert DataFrame using PostgreSQL COPY protocol.

    COPY is significantly faster than the per-row INSERT path for large
    datasets (4-5x improvement), but doesn't support ON CONFLICT
    clauses.

    A single ``cursor.copy(sql)`` context streams every CSV-encoded
    chunk into one COPY invocation; for DataFrames above the
    auto-batch threshold (or whenever ``config.insert_chunk_size`` is
    set explicitly), the serialization is sliced into chunks so peak
    Python heap during the COPY tracks the chunk size rather than the
    full DataFrame's serialized size (#245).

    Args:
        config: Writer configuration.  ``insert_chunk_size`` controls
            chunked behavior; see :func:`get_effective_chunk_size`.
        cursor: Database cursor.
        table_name: Target table name.
        df: DataFrame to insert.
        context: Dagster context for logging.

    Returns:
        Number of rows inserted.
    """
    row_count = len(df)
    if row_count == 0:
        return 0

    columns_str = ", ".join(f'"{col}"' for col in df.columns)
    copy_sql = f"COPY {table_name} ({columns_str}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"  # noqa: S608

    chunk_size = get_effective_chunk_size(config, row_count)

    if chunk_size is None:
        # Single chunk -- DataFrame is small enough that the BytesIO
        # buffer doubling is acceptable, OR the operator explicitly
        # disabled chunking.
        buffer = BytesIO()
        df.write_csv(buffer, include_header=False, null_value="\\N")
        with cursor.copy(copy_sql) as copy:
            copy.write(buffer.getvalue())
        context.log.info(f"COPY inserted {row_count} rows into {table_name}")
        return row_count

    # Chunked: serialize each slice on demand and stream into the COPY.
    # ``total_inserted`` accumulates as the generator drains; the COPY
    # consumes one chunk at a time so peak heap tracks ``chunk_size``
    # rows' worth of CSV regardless of total DataFrame size.
    total_inserted = 0
    num_chunks = (row_count + chunk_size - 1) // chunk_size

    def _csv_chunks() -> Iterator[bytes]:
        nonlocal total_inserted
        for i, chunk_df in enumerate(df.iter_slices(n_rows=chunk_size)):
            buffer = BytesIO()
            chunk_df.write_csv(buffer, include_header=False, null_value="\\N")
            total_inserted += len(chunk_df)
            context.log.debug(
                f"COPY chunk {i + 1}/{num_chunks}: {len(chunk_df):,} rows "
                f"(total: {total_inserted:,}/{row_count:,})"
            )
            yield buffer.getvalue()

    with cursor.copy(copy_sql) as copy:
        for chunk in _csv_chunks():
            copy.write(chunk)

    context.log.info(
        f"COPY inserted {total_inserted} rows into {table_name} in "
        f"{num_chunks} chunks (chunk_size={chunk_size})"
    )
    return total_inserted


# CSV encoding for COPY into an upsert staging table. Round-trips the three
# values the param-bound path preserves and #377 pins: SQL NULL, empty string,
# and the literal text "\N". quote_style="non_numeric" quotes every string
# field, so an unquoted empty field is unambiguously NULL (paired with
# COPY ... NULL '') while a quoted "" is the empty string and "\N" is literal
# text; numeric NULLs stay unquoted-empty => NULL. Verified empirically in
# docs/migrations/20260626_375-upsert-staging-merge.md (D1). Deliberately
# distinct from insert_with_copy's NULL '\N' encoding, which collapses a
# literal "\N" to NULL (latent quirk on the append path; tracked as a
# follow-up, out of scope for #375).
COPY_STAGING_OPTIONS = "FORMAT CSV, NULL ''"


def serialize_for_staging_copy(df: pl.DataFrame) -> bytes:
    r"""Serialize a DataFrame to CSV bytes for COPY into an upsert staging table.

    Uses the fidelity-preserving encoding from migration 375 (D1): SQL NULL,
    empty string, and the literal text ``\N`` all round-trip distinctly. Pair
    with ``COPY ... WITH (<COPY_STAGING_OPTIONS>)``.
    """
    buffer = BytesIO()
    df.write_csv(buffer, include_header=False, null_value="", quote_style="non_numeric")
    return buffer.getvalue()


def insert_with_execute_values(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    columns: list[str],
    df: pl.DataFrame,
    context: LoggingContext,
) -> int:
    """Insert DataFrame via psycopg3's pipelined ``executemany`` with optional chunking.

    For large DataFrames, processes in chunks to limit memory usage.
    The function name is preserved for back-compat -- it names the
    intent ("bulk INSERT via the executemany family"), not the
    underlying driver function.

    Args:
        config: Writer configuration.
        cursor: Database cursor.
        table_name: Target table name.
        columns: Column names for INSERT.
        df: DataFrame to insert.
        context: Dagster context for logging.

    Returns:
        Number of rows inserted.
    """
    row_count = len(df)
    if row_count == 0:
        return 0

    insert_sql = f"""
        INSERT INTO {table_name} ({", ".join(columns)})
        VALUES %s
    """  # noqa: S608

    chunk_size = get_effective_chunk_size(config, row_count)

    if chunk_size is None:
        # Process all at once
        rows = df.rows()
        _executemany_bulk(cursor, insert_sql, rows)
        return row_count
    else:
        # Process in chunks to limit memory usage
        total_inserted = 0
        num_chunks = (row_count + chunk_size - 1) // chunk_size

        for i, chunk_df in enumerate(df.iter_slices(n_rows=chunk_size)):
            rows = chunk_df.rows()
            _executemany_bulk(cursor, insert_sql, rows)
            total_inserted += len(rows)

            context.log.debug(
                f"Inserted chunk {i + 1}/{num_chunks}: {len(rows):,} rows "
                f"(total: {total_inserted:,}/{row_count:,})"
            )

        return total_inserted


def _estimate_existing_rows(cursor: psycopg.Cursor, table_name: str) -> int | None:
    """Estimate the target's current row count from ``pg_class.reltuples``.

    Used to size the AUTO clear-method decision when the caller has no count
    to offer (the batched path never does -- see
    model-oncology-public/moncpipelib#4). Mirrors the catalog lookup in
    ``resources/_analyze_helpers.py``, including its treatment of
    ``reltuples = -1`` as "never analyzed" rather than "empty": every
    partitioned parent reports -1 until an explicit ANALYZE, and reading that
    as zero would bias toward the more disruptive lock on exactly the tables
    where it hurts most.

    Returns ``None`` when no estimate is available (relation absent, or never
    analyzed), which callers must treat as unknown rather than zero.

    Unlike ``_analyze_helpers``, this runs *inside* the write transaction, so
    it deliberately does not swallow errors: a failed statement has already
    aborted the transaction, and suppressing that would only defer the failure
    to the clear itself. The probe is a plain catalog read, and ``to_regclass``
    yields NULL (not an error) for a name that does not resolve.
    """
    cursor.execute("SELECT reltuples FROM pg_class WHERE oid = to_regclass(%s)", (table_name,))
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return None
    estimate = float(row[0])
    if estimate < 0:
        return None
    return int(estimate)


def clear_table(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    row_count_hint: int | None,
    context: LoggingContext,
) -> tuple[int, str]:
    """Clear a table using TRUNCATE or DELETE based on config.

    This is the "clear" half of a full refresh, separated for use
    by the batched handler which clears once then inserts per-batch.

    Args:
        config: Writer configuration.
        cursor: Database cursor.
        table_name: Target table name.
        row_count_hint: Estimated row count for auto-selection, or ``None`` if
            the caller has none. Under AUTO an absent hint falls back to the
            target's ``pg_class.reltuples``; an explicit ``0`` is honoured as a
            real count and does not trigger that fallback.
        context: Dagster context for logging.

    Returns:
        Tuple of (deleted_count, clear_method). deleted_count is 0 for TRUNCATE.
    """
    effective_row_count = row_count_hint
    if effective_row_count is None and config.full_refresh_method == FullRefreshMethod.AUTO:
        effective_row_count = _estimate_existing_rows(cursor, table_name)
    use_truncate = should_use_truncate(config, effective_row_count)
    if use_truncate:
        cursor.execute(f"TRUNCATE {table_name}")  # noqa: S608
        context.log.info(f"Truncated {table_name}")
        return 0, "truncate"
    else:
        cursor.execute(f"DELETE FROM {table_name}")  # noqa: S608
        deleted_count = cursor.rowcount
        context.log.info(f"Deleted {deleted_count} existing rows from {table_name}")
        return deleted_count, "delete"


def insert_rows(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    df: pl.DataFrame,
    write_mode: WriteMode,
    context: LoggingContext,
) -> int:
    """Insert DataFrame rows using COPY or execute_values based on config.

    This is the "insert" step separated for use by the batched handler
    which inserts each batch independently.

    Args:
        config: Writer configuration.
        cursor: Database cursor.
        table_name: Target table name.
        df: DataFrame to insert.
        write_mode: The write mode (affects COPY eligibility).
        context: Dagster context for logging.

    Returns:
        Number of rows inserted.
    """
    row_count = len(df)
    if row_count == 0:
        return 0

    use_copy = should_use_copy(config, row_count, write_mode)
    if use_copy:
        return insert_with_copy(config, cursor, table_name, df, context)
    else:
        return insert_with_execute_values(config, cursor, table_name, df.columns, df, context)


def execute_full_refresh(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    df: pl.DataFrame,
    context: LoggingContext,
) -> dict[str, int | str]:
    """Execute full refresh: DELETE or TRUNCATE (based on config) + INSERT.

    Returns:
        Dict with rows_deleted (0 for TRUNCATE), rows_inserted, clear_method,
        and insert_method.
    """
    row_count = len(df)
    use_truncate = should_use_truncate(config, row_count)

    if use_truncate:
        cursor.execute(f"TRUNCATE {table_name}")  # noqa: S608
        context.log.info(f"Truncated {table_name} (inserting {row_count} rows)")
        deleted_count = 0  # TRUNCATE doesn't report row count
        clear_method = "truncate"
    else:
        cursor.execute(f"DELETE FROM {table_name}")  # noqa: S608
        deleted_count = cursor.rowcount
        context.log.info(f"Deleted {deleted_count} existing rows from {table_name}")
        clear_method = "delete"

    # Insert new data using appropriate method
    inserted_count = 0
    insert_method = "none"
    if row_count > 0:
        use_copy = should_use_copy(config, row_count, WriteMode.FULL_REFRESH)
        if use_copy:
            inserted_count = insert_with_copy(config, cursor, table_name, df, context)
            insert_method = "copy"
        else:
            inserted_count = insert_with_execute_values(
                config, cursor, table_name, df.columns, df, context
            )
            insert_method = "execute_values"

    return {
        "rows_deleted": deleted_count,
        "rows_inserted": inserted_count,
        "clear_method": clear_method,
        "insert_method": insert_method,
    }


def execute_upsert(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    df: pl.DataFrame,
    primary_key: list[str],
    update_columns: list[str] | None,
    context: LoggingContext,
    *,
    skip_unchanged: bool = False,
) -> dict[str, int | str]:
    r"""Execute upsert via staging-COPY + a single DISTINCT ON merge (#375).

    COPYs the batch into a fresh TEMP staging table carrying an input-order
    ordinal (one audited ``WRITE,COPY``), then runs one
    ``INSERT ... SELECT DISTINCT ON (pk) ... ON CONFLICT`` merge (one audited
    ``WRITE,INSERT``) -- two audited statements regardless of row count, where
    the former per-row ``executemany`` path emitted one per row.

    Duplicate conflict keys within the batch resolve last-input-wins via
    ``ORDER BY pk, _ord DESC`` (the contract pinned by the #377 characterization
    tests), so the single merge never raises "ON CONFLICT DO UPDATE command
    cannot affect row a second time". The staging COPY streams in chunks
    (``config.insert_chunk_size``) so peak heap stays bounded (#239); CSV
    encoding round-trips NULL / '' / literal ``\N`` distinctly (see
    :func:`serialize_for_staging_copy`).

    Target columns the DataFrame does not supply (GENERATED ... STORED,
    identity, server defaults) are dropped from the staging table after the
    ``LIKE`` clone -- ``LIKE`` keeps their NOT NULL constraint but not the
    expression that would satisfy it, so staging them breaks the COPY while
    the final merge would have handled them fine (#400).

    Rows carrying NULL in any ``primary_key`` column are rejected with
    ``ValueError`` before staging: SQL NULLs never match ``ON CONFLICT``,
    so such rows would silently duplicate on every re-run (#401).

    Args:
        config: Writer configuration (``insert_chunk_size`` controls COPY chunking).
        cursor: Database cursor.
        table_name: Target table (schema-qualified).
        df: DataFrame to upsert. Column order defines the INSERT column list.
        primary_key: Columns for conflict detection.
        update_columns: Columns to update on conflict. None = all non-key
            columns; empty list = ON CONFLICT DO NOTHING.
        context: Dagster context for logging.
        skip_unchanged: When True, guard the ``DO UPDATE`` with
            ``WHERE <target>.col IS DISTINCT FROM EXCLUDED.col OR ...`` over
            the update columns, so a conflicting row whose update columns all
            already match is not rewritten -- no dead tuple, no index churn,
            no WAL (mirror issue model-oncology-public/moncpipelib#3).
            ``IS DISTINCT FROM`` is NULL-safe. Behavioral caveats: row-level
            ``ON UPDATE`` triggers (e.g. ``updated_at`` touch triggers) no
            longer fire for unchanged rows, which is why this is opt-in;
            conflicting rows are still locked to evaluate the guard, so
            concurrency semantics are unchanged. ``rows_upserted`` still
            reports the incoming row count. No effect when ``update_columns``
            is empty (``DO NOTHING`` already writes nothing on conflict).
            Every update column's type must have an equality operator:
            ``json`` (unlike ``jsonb``), ``xml``, and some geometric types
            do not, and the merge then fails with "could not identify an
            equality operator" -- on such tables scope ``update_columns``
            to exclude those columns or leave the guard off.

    Returns:
        Dict with rows_upserted count (incoming row count, pre-dedup).
    """
    row_count = len(df)
    if row_count == 0:
        return {"rows_upserted": 0}

    # NULL never matches ON CONFLICT (SQL NULL <> NULL), so a NULL-keyed row
    # bypasses conflict detection and inserts a fresh duplicate on every
    # re-materialization -- silent data corruption rather than an error.
    # Fail fast instead (#401 item 3, data-platform dim_hcpcs).
    null_keyed = {c: n for c in primary_key if (n := df[c].null_count()) > 0}
    if null_keyed:
        raise ValueError(
            f"Upsert into {table_name} rejected: primary_key column(s) contain "
            f"NULLs ({', '.join(f'{c}: {n} row(s)' for c, n in null_keyed.items())}). "
            f"NULLs never match ON CONFLICT, so NULL-keyed rows would silently "
            f"duplicate on every run. Make the key column(s) non-null or choose "
            f"a different primary_key."
        )

    columns = list(df.columns)
    if update_columns is None:
        # Update all columns except primary key
        update_columns = [c for c in columns if c not in primary_key]

    col_list = ", ".join(f'"{c}"' for c in columns)
    pk_list = ", ".join(f'"{c}"' for c in primary_key)
    stage = "_ups_stage"

    # 1) COPY the batch into a fresh TEMP staging table carrying an input-order
    #    ordinal. LIKE (without INCLUDING CONSTRAINTS) gives column types but no
    #    PK, so duplicate keys are allowed until the merge dedupes them.
    cursor.execute(f"DROP TABLE IF EXISTS {stage}")
    cursor.execute(f"CREATE TEMP TABLE {stage} (LIKE {table_name})")  # noqa: S608
    # LIKE copies NOT NULL constraints but not GENERATED expressions, identity,
    # or (without INCLUDING DEFAULTS) default expressions. Any target column the
    # DataFrame doesn't supply would therefore stage as a plain NOT NULL column
    # with no way to receive a value -> NotNullViolation at COPY time, even
    # though the final INSERT computes/fills it fine (#400). Such columns are
    # never read by the merge (its column list is the DataFrame's), so drop
    # them from staging entirely.
    cursor.execute(f"SELECT * FROM {stage} WHERE false")  # noqa: S608
    df_columns = set(columns)
    stage_cols = [d.name for d in cursor.description or []]
    for extra_col in (c for c in stage_cols if c not in df_columns):
        cursor.execute(f'ALTER TABLE {stage} DROP COLUMN "{extra_col}"')  # noqa: S608
    cursor.execute(f"ALTER TABLE {stage} ADD COLUMN _ord bigint")

    df_ord = df.with_row_index("_ord").select([*columns, "_ord"])
    copy_sql = f'COPY {stage} ({col_list}, "_ord") FROM STDIN WITH ({COPY_STAGING_OPTIONS})'  # noqa: S608
    chunk_size = get_effective_chunk_size(config, row_count)
    with cursor.copy(copy_sql) as copy:
        if chunk_size is None:
            copy.write(serialize_for_staging_copy(df_ord))
        else:
            for chunk_df in df_ord.iter_slices(n_rows=chunk_size):
                copy.write(serialize_for_staging_copy(chunk_df))

    # 2) One merge: dedupe staging last-input-wins, then upsert.
    dedup = f"SELECT DISTINCT ON ({pk_list}) {col_list} FROM {stage} ORDER BY {pk_list}, _ord DESC"
    target = table_name
    if not update_columns:
        conflict = f"ON CONFLICT ({pk_list}) DO NOTHING"
    else:
        set_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_columns)
        conflict = f"ON CONFLICT ({pk_list}) DO UPDATE SET {set_clause}"
        if skip_unchanged:
            # The guard needs a row reference to the target: a schema-qualified
            # name is not valid inside DO UPDATE ... WHERE, so alias it. The
            # alias is added only on this branch to keep the default-path SQL
            # byte-identical (pinned by the #377 characterization suite).
            target = f"{table_name} AS _tgt"
            guard = " OR ".join(
                f'_tgt."{c}" IS DISTINCT FROM EXCLUDED."{c}"' for c in update_columns
            )
            conflict = f"{conflict} WHERE {guard}"
    cursor.execute(
        f"INSERT INTO {target} ({col_list}) "  # noqa: S608
        f"SELECT {col_list} FROM ({dedup}) d {conflict}"
    )
    cursor.execute(f"DROP TABLE IF EXISTS {stage}")

    context.log.info(f"Upserted {row_count} rows into {table_name} via staging merge")
    return {"rows_upserted": row_count}


def execute_append(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    df: pl.DataFrame,
    context: LoggingContext,
) -> dict[str, int | str]:
    """Execute append: INSERT only.

    Returns:
        Dict with rows_inserted count and insert_method.
    """
    row_count = len(df)
    if row_count == 0:
        return {"rows_inserted": 0, "insert_method": "none"}

    use_copy = should_use_copy(config, row_count, WriteMode.APPEND)
    if use_copy:
        inserted_count = insert_with_copy(config, cursor, table_name, df, context)
        insert_method = "copy"
    else:
        inserted_count = insert_with_execute_values(
            config, cursor, table_name, df.columns, df, context
        )
        insert_method = "execute_values"

    context.log.info(f"Appended {inserted_count} rows to {table_name}")
    return {"rows_inserted": inserted_count, "insert_method": insert_method}


def execute_partition_scoped(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    df: pl.DataFrame,
    partition_column: str,
    context: LoggingContext,
    *,
    partition_values: list[Any] | None = None,
) -> dict[str, int | str]:
    """Execute partition-scoped write: DELETE WHERE partition_column IN (...) + INSERT.

    When ``partition_values`` is provided (from Dagster partition context), uses
    those values directly for the DELETE scope. Otherwise extracts distinct values
    from the DataFrame's ``partition_column`` (legacy fallback).

    Args:
        config: Writer configuration.
        cursor: Database cursor.
        table_name: Target table.
        df: DataFrame to insert.
        partition_column: Column to partition by.
        context: Dagster context for logging.
        partition_values: Explicit partition values from Dagster context.
            When provided, these drive the DELETE scope rather than DataFrame
            introspection.

    Returns:
        Dict with rows_deleted, rows_inserted, and insert_method.
    """
    if partition_values is None:
        # Legacy path: extract from DataFrame
        partition_values = df[partition_column].unique().to_list()

    if not partition_values:
        return {"rows_deleted": 0, "rows_inserted": 0, "insert_method": "none"}

    # Delete existing rows for these partitions (parameterized for safety)
    placeholders = ", ".join(["%s"] * len(partition_values))
    delete_sql = f'DELETE FROM {table_name} WHERE "{partition_column}" IN ({placeholders})'  # noqa: S608
    cursor.execute(delete_sql, partition_values)
    deleted_count = cursor.rowcount
    context.log.info(
        f"Deleted {deleted_count} rows from {table_name} "
        f"for {len(partition_values)} partition value(s)"
    )

    # Insert new data using standard insert path (supports COPY protocol)
    inserted_count = 0
    insert_method = "none"
    row_count = len(df)
    if row_count > 0:
        use_copy = should_use_copy(config, row_count, WriteMode.FULL_REFRESH)
        if use_copy:
            inserted_count = insert_with_copy(config, cursor, table_name, df, context)
            insert_method = "copy"
        else:
            inserted_count = insert_with_execute_values(
                config, cursor, table_name, df.columns, df, context
            )
            insert_method = "execute_values"

    return {
        "rows_deleted": deleted_count,
        "rows_inserted": inserted_count,
        "insert_method": insert_method,
    }


def _create_staging_bk_index(
    cursor: psycopg.Cursor,
    stage_table: str,
    business_key: list[str],
) -> None:
    """Build a business-key index on the populated SCD2 staging temp table.

    ``scd2_create_staging`` creates the staging table with
    ``CREATE TEMP TABLE (LIKE ... INCLUDING DEFAULTS)``, which copies *no*
    indexes (#361).  Every change-detection statement in
    :func:`scd2_finalize` -- the count LEFT JOIN, the expire UPDATE, the
    Stage-1 anti-join CTAS, and the ``detect_deletes`` anti-join UPDATE --
    joins the target to staging on the business key.  Without a staging-side
    BK index the planner has no index-nested-loop option and is pushed toward
    hashing or seq-scanning staging; on large targets this contributed to the
    pathological full-table read profile observed in the ``npi_address``
    incident.

    Idempotent: ``IF NOT EXISTS`` guards against a second call within the same
    session reusing the fixed temp-table name.  Build this once *after*
    staging is fully populated -- under the batched write path
    :func:`scd2_insert_staging` runs per batch, so the index belongs at
    finalize time, not insert time.

    Args:
        cursor: Database cursor (same transaction as the staging table).
        stage_table: Name of the temp staging table.
        business_key: Business-key columns to index, in order.
    """
    index_name = f"{stage_table}_bk_idx"
    cursor.execute(
        sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {tbl} ({cols})").format(
            idx=sql.Identifier(index_name),
            tbl=sql.Identifier(stage_table),
            cols=sql.SQL(", ").join(sql.Identifier(k) for k in business_key),
        )
    )


_DUPLICATE_KEY_SAMPLE_LIMIT = 5


def _assert_staging_business_keys_unique(
    cursor: psycopg.Cursor,
    stage_table: str,
    table_name: str,
    business_key: list[str],
    partition_column: str | None,
    statement_timeout: str | None,
) -> None:
    """Reject SCD2 staging data containing duplicated business keys (#419).

    An SCD2 write must receive at most one row per business key (per
    partition value, for partition-scoped sinks).  A frame that violates
    this produces *silent* corruption with self-consistent-looking result
    metadata: duplicated keys double-insert as two ``is_current`` rows,
    keys whose duplicate matches an existing row misclassify as
    "unchanged", and nothing errors.  In the #419 incident an upstream
    key-normalization step (``zfill`` applied downstream of a grain
    collapse) merged distinct keys and the write landed 80 doubled / 80
    dropped keys with correct-looking stats.

    Scope: duplicates are counted per ``(business_key, partition_column)``
    group when ``partition_column`` is provided -- the same business key
    across *different* partition values is the designed multi-period
    backfill shape, later stitched by ``reconcile_scd2``.  Without a
    partition column, the business key alone must be unique.

    Placement: runs at finalize time, after staging is fully populated, so
    the batched write path is covered across batches (each batch only sees
    its own frame; staging accumulates all of them).  Cost is a single
    GROUP BY over staging -- the same order as the change-detection count
    join that follows -- bounded by ``change_detection_statement_timeout``.

    Args:
        cursor: Database cursor (same transaction as the staging table).
        stage_table: Name of the populated temp staging table.
        table_name: Fully-qualified target table (for the error message).
        business_key: Business-key columns.
        partition_column: Optional partition column widening the uniqueness
            group.
        statement_timeout: Pre-validated ``statement_timeout`` literal
            bounding the GROUP BY, or ``None``.

    Raises:
        ValueError: If any business key appears more than once (within a
            single partition value, when partitioned).  The write aborts
            before any DML against the target; the caller's rollback
            discards staging.
    """
    group_cols = list(business_key)
    if partition_column:
        group_cols.append(partition_column)
    cols_quoted = ", ".join(f'"{c}"' for c in group_cols)

    dup_sql = f"""
        WITH dups AS (
            SELECT {cols_quoted}, count(*) AS row_copies
            FROM {stage_table}
            GROUP BY {cols_quoted}
            HAVING count(*) > 1
        )
        SELECT (SELECT count(*) FROM dups) AS dup_key_count, dups.*
        FROM dups
        ORDER BY row_copies DESC, {cols_quoted}
        LIMIT {_DUPLICATE_KEY_SAMPLE_LIMIT}
    """  # noqa: S608
    with _bounded_statement_timeout(cursor, statement_timeout):
        cursor.execute(dup_sql)
    rows = cursor.fetchall()
    if not rows:
        return

    dup_key_count = rows[0][0]
    samples = "; ".join(
        "("
        + ", ".join(f"{col}={val!r}" for col, val in zip(group_cols, row[1:-1], strict=True))
        + f") x{row[-1]}"
        for row in rows
    )
    scope = f' within the same "{partition_column}" partition value' if partition_column else ""
    raise ValueError(
        f"SCD2 write to {table_name} aborted: incoming data contains "
        f"{dup_key_count} business key(s){scope} with more than one row. "
        f"An SCD2 write must receive at most one row per business key"
        f"{scope}; duplicates would double-insert current rows and/or "
        f"silently misclassify changes (see #419). "
        f"Sample duplicate keys: {samples}. Common causes: key "
        f"normalization applied downstream of a deduplication or grain "
        f"collapse, duplicate source rows, or an under-specified "
        f"business_key in the contract. Fix the frame upstream of write()."
    )


def _apply_change_detection_work_mem(cursor: psycopg.Cursor, value: str) -> None:
    """Apply a per-transaction ``work_mem`` bump for change-detection planning.

    Mirrors the reconcile path's per-tx bump (``reconcile_work_mem``) for the
    writer's SCD2 change-detection statements (#361).  A larger ``work_mem``
    lowers the planner's estimated cost of the anti-join hash builds, which can
    tip plan choice away from a full-table sequential scan and removes any
    hash/sort spill at execution time.  Applied as ``is_local`` so it reverts
    on commit or rollback and never leaks to concurrent sessions.

    ``value`` is pre-validated and canonicalized by the resource
    (``PostgresResource._resolve_work_mem``); this function applies it
    verbatim via parameterized ``set_config`` (no SQL injection surface).
    """
    cursor.execute("SELECT set_config('work_mem', %s, true)", (value,))


@contextmanager
def _bounded_statement_timeout(cursor: psycopg.Cursor, timeout: str | None) -> Iterator[None]:
    """Bound the wrapped change-detection statement(s) with ``statement_timeout``.

    The ``npi_address`` incident (#361) was a single ``detect_deletes`` UPDATE
    that ran read-bound for ~68h, holding one transaction open the whole time
    and pinning the cluster-wide vacuum xmin horizon as a side effect.  A
    per-statement ``statement_timeout`` bounds the blast radius: a degenerate
    plan aborts in minutes and releases its snapshot instead of grinding for
    days.

    Scope is deliberately narrow -- callers wrap only the *target-reading
    anti-join* statements (count, expire, Stage-1 CTAS, ``detect_deletes``).
    The Stage-2 bulk INSERT, whose runtime scales legitimately with the size
    of a first-ever load, is intentionally left unbounded.

    The prior ``statement_timeout`` is captured and restored on exit so the
    bound does not leak to later statements in the same transaction (e.g. the
    trailing target ANALYZE or the caller's commit-time work).  A ``None``
    timeout is a no-op pass-through.

    The SHOW / ``set_config`` round-trips run on a *separate* cursor on the
    same connection, never on ``cursor`` itself.  ``set_config(..., is_local)``
    is transaction-scoped regardless of which cursor issues it, so a side
    cursor applies the bound just as well -- while leaving ``cursor``'s own
    result set and ``rowcount`` untouched.  Issuing them on ``cursor`` would
    overwrite the wrapped statement's results, so a caller doing
    ``with _bounded_statement_timeout(...): cur.execute(q)`` then
    ``cur.fetchone()`` would read the ``set_config`` row instead of ``q``'s.

    Args:
        cursor: Database cursor (the transaction whose timeout is bounded).
        timeout: Pre-validated Postgres ``statement_timeout`` literal
            (e.g. ``"30min"``), or ``None`` to leave the timeout unchanged.
    """
    if timeout is None:
        yield
        return
    with cursor.connection.cursor() as tcur:
        tcur.execute("SHOW statement_timeout")
        prior_row = tcur.fetchone()
        prior = prior_row[0] if prior_row else "0"
        tcur.execute("SELECT set_config('statement_timeout', %s, true)", (timeout,))
    try:
        yield
    finally:
        with cursor.connection.cursor() as tcur:
            tcur.execute("SELECT set_config('statement_timeout', %s, true)", (prior,))


# Targets already warned about by scd2_preflight_index_shape this process.
# The preflight runs once per finalize; under per-period sequencing a single
# backfill can finalize the same target hundreds of times, so repeat warnings
# would drown the log. Keyed by fully-qualified table name.
_SCD2_INDEX_SHAPE_WARNED: set[str] = set()


def scd2_preflight_index_shape(
    cursor: psycopg.Cursor,
    table_name: str,
    scd2: SCD2Config,
    context: LoggingContext,
) -> None:
    """Warn when the target carries a unique index shape that will break SCD2 expiry.

    The supported shape (docs/scd2-guide.md) is a *partial* unique index::

        CREATE UNIQUE INDEX ... ON {table} ({business_key}) WHERE ({is_current});

    Authoring drift observed in the wild (#401 item 4, data-platform
    dim_diagnosis) is a plain (non-partial) unique index that instead includes
    ``is_current`` as a key column, e.g. ``UNIQUE (business_key, is_current)``.
    That index tolerates the first expiry of any business key but raises
    ``UniqueViolation`` on the second -- there can only be one expired
    ``(key, false)`` row -- so the table is armed to fail on a later,
    unrelated run. Static schema review keeps missing it; a catalog lookup at
    write time does not.

    Diagnostic only: warns at most once per target per process and never
    raises -- catalog access errors are logged at debug level and swallowed
    so a permissions quirk cannot break the write path.
    """
    if table_name in _SCD2_INDEX_SHAPE_WARNED:
        return
    _SCD2_INDEX_SHAPE_WARNED.add(table_name)

    schema, tbl = parse_schema_table(table_name)
    try:
        cursor.execute(
            """
            SELECT ic.relname,
                   ARRAY(
                       SELECT a.attname
                       FROM unnest(ix.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
                       JOIN pg_attribute a
                         ON a.attrelid = ix.indrelid AND a.attnum = k.attnum
                       ORDER BY k.ord
                   ) AS key_cols
            FROM pg_index ix
            JOIN pg_class ic ON ic.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = %s AND t.relname = %s
              AND ix.indisunique
              AND ix.indpred IS NULL
            """,
            (schema, tbl),
        )
        rows = cursor.fetchall()
    except psycopg.Error as exc:
        context.log.debug(f"SCD2 index-shape preflight skipped for {table_name}: {exc}")
        return

    for index_name, key_cols in rows:
        if scd2.is_current_col in (key_cols or []):
            context.log.warning(
                f"SCD2 target {table_name} has a non-partial UNIQUE index "
                f"'{index_name}' that includes '{scd2.is_current_col}' as a key "
                f"column {list(key_cols)}. This allows only one expired row per "
                f"business key, so the write will raise UniqueViolation the "
                f"second time any key changes. Replace it with the partial "
                f"form: CREATE UNIQUE INDEX ... ON {table_name} (<business_key>) "
                f"WHERE ({scd2.is_current_col}); see docs/scd2-guide.md."
            )


def scd2_create_staging(
    cursor: psycopg.Cursor,
    table_name: str,
    scd2: SCD2Config,
) -> str:
    """Create temp staging table from target structure, dropping managed, identity, and generated columns.

    Args:
        cursor: Database cursor.
        table_name: Fully-qualified target table name (schema.table).
        scd2: SCD2 column configuration.

    Returns:
        The staging table name.
    """
    stage_table = "_scd2_staging"

    cursor.execute(f"DROP TABLE IF EXISTS pg_temp.{stage_table}")  # noqa: S608
    cursor.execute(
        f"CREATE TEMP TABLE {stage_table} (LIKE {table_name} INCLUDING DEFAULTS)"  # noqa: S608
    )
    managed_cols = [scd2.effective_from_col, scd2.effective_to_col, scd2.is_current_col]
    if scd2.sequence_col:
        managed_cols.append(scd2.sequence_col)
    for col in managed_cols:
        cursor.execute(
            f'ALTER TABLE {stage_table} DROP COLUMN IF EXISTS "{col}"'  # noqa: S608
        )
    # Also drop identity and GENERATED ... STORED columns (both are computed
    # by the database, never present in our DataFrame). LIKE keeps their
    # NOT NULL constraint but not the identity/generation expression, so
    # leaving them in staging turns the COPY into a NotNullViolation even
    # though the final target INSERT would compute them fine (#400).
    schema, tbl = parse_schema_table(table_name)
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
          AND (is_identity = 'YES' OR is_generated = 'ALWAYS')
        """,
        (schema, tbl),
    )
    for (db_generated_col,) in cursor.fetchall():
        cursor.execute(
            f'ALTER TABLE {stage_table} DROP COLUMN IF EXISTS "{db_generated_col}"'  # noqa: S608
        )

    return stage_table


def scd2_insert_staging(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    stage_table: str,
    df: pl.DataFrame,
    context: LoggingContext,
) -> None:
    """Insert DataFrame rows into the SCD2 staging table using COPY protocol.

    Uses COPY instead of execute_values to avoid materializing the entire
    DataFrame as Python tuples, which would double peak memory for large
    staging loads.  ``config.insert_chunk_size`` is honored so very large
    staging loads can chunk the CSV serialization (#245).

    The staging table itself is created by :func:`scd2_create_staging`; the
    ANALYZE here belongs at the populated site, not the creation site.
    PostgreSQL never autoanalyzes temp tables, so without this explicit
    refresh ``pg_class.reltuples`` for ``_scd2_staging`` stays at ~0 for
    the lifetime of the session. Every subsequent join in
    :func:`scd2_finalize` (count, expire, INSERT-from-diff, and especially
    the ``detect_deletes`` UPDATE) would then see a 0-row estimate on the
    staging side and may pick Nested Loop with seq-scan inner -- O(N*M)
    catastrophic on multi-million-row targets. See #312.

    Args:
        config: Writer configuration.  Forwarded to ``insert_with_copy``
            so chunk-size knobs apply uniformly across write modes.
        cursor: Database cursor.
        stage_table: Name of the temp staging table.
        df: DataFrame to insert (must match staging table columns).
        context: Dagster context for logging.
    """
    insert_with_copy(config, cursor, stage_table, df, context)
    # See #312: refresh planner stats for the just-populated temp table so
    # subsequent joins in scd2_finalize plan against accurate reltuples.
    cursor.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(stage_table)))


def scd2_finalize(
    cursor: psycopg.Cursor,
    table_name: str,
    stage_table: str,
    total_staged_rows: int,
    stage_columns: list[str],
    business_key: list[str],
    scd2: SCD2Config,
    context: LoggingContext,
    *,
    detect_deletes: bool = False,
    partition_column: str | None = None,
    partition_values: list[Any] | None = None,
    effective_date: date | None = None,
    change_detection_work_mem: str | None = None,
    change_detection_statement_timeout: str | None = None,
) -> dict[str, int | str]:
    """Perform SCD2 change detection and apply changes using the populated staging table.

    Counts new and changed records, expires changed records, inserts new
    versions, optionally expires absent business keys, then drops staging.

    When ``partition_column`` and ``partition_values`` are both provided, all
    comparison queries are scoped to the active partition(s) so records from
    other partitions are never compared, expired, or deleted.

    When ``scd2.sequence_col`` is set and the target table contains that
    column, each newly inserted version receives ``MAX(sequence_col) + 1``
    for its business key (starting at 1 for brand-new keys).

    When ``effective_date`` is provided, it is used instead of ``now()`` for
    ``effective_from`` on new inserts and ``effective_to`` on expired rows.
    This enables loading historical data with correct period boundaries.

    Implementation note (see #274): the change-detection branch issues two
    SQL statements -- a CTAS into a temp diff table, then an INSERT-from-diff
    -- rather than a single self-referencing INSERT-with-LEFT-JOIN. The
    single-statement form is O(N^2) on empty targets when the planner picks
    Nested Loop Anti Join, because each heap page extended by the in-flight
    INSERT is re-scanned by the inner Seq Scan. Materializing the diff first
    decouples the read of the target from the write to it, making plan
    choice irrelevant. Do not "simplify" this back to a single statement.

    Implementation note (see #312): a trailing ``ANALYZE <target>`` is issued
    after the writes complete so the *next* finalize call (under per-period
    sequencing, the same target is finalized once per ``load_period`` within
    the same session) plans against accurate target-side stats. The matching
    staging-side refresh lives in :func:`scd2_insert_staging`.

    Implementation note (see #361): three large-table mitigations apply to the
    change-detection statements.  (1) A business-key index is built on the
    populated staging table so the anti-joins have an index-nested-loop option.
    (2) When Stage-1 inserts new rows, the target is re-ANALYZEd *before*
    ``detect_deletes`` so the anti-join plans against the just-inserted
    partition's real cardinality instead of a stale ``rows=1`` estimate.  (3)
    The target-reading anti-join statements (count, expire, Stage-1 CTAS,
    ``detect_deletes``) optionally run under a per-tx ``work_mem`` bump and a
    per-statement ``statement_timeout``; the timeout bounds the blast radius of
    a degenerate plan (the incident pinned the vacuum xmin horizon for 68h).
    The Stage-2 bulk INSERT is deliberately left unbounded.

    Implementation note (see #419): before any change detection, staging is
    checked for duplicated business keys (per partition value, when
    ``partition_column`` is provided) and the write fails with ``ValueError``
    if any are found.  Duplicate keys are never valid SCD2 input -- they
    double-insert current rows and misclassify changes with self-consistent
    result metadata, so the failure must happen here, loudly, before DML.

    Args:
        change_detection_work_mem: Pre-validated per-tx ``work_mem`` literal
            applied to change-detection planning, or ``None`` to leave the
            cluster default.  Resolved by the resource.
        change_detection_statement_timeout: Pre-validated ``statement_timeout``
            literal bounding each target-reading anti-join statement, or
            ``None`` to leave the timeout unchanged.  Resolved by the resource.

    Returns:
        Dict with rows_new, rows_expired, rows_inserted, rows_unchanged,
        and rows_deleted (always present, 0 when detect_deletes is False).
    """
    effective_from_col = scd2.effective_from_col
    effective_to_col = scd2.effective_to_col
    is_current_col = scd2.is_current_col
    hash_col = scd2.hash_col
    sequence_col = scd2.sequence_col

    # Build date expression: parameterized value or database now()
    date_expr = "%s" if effective_date is not None else "now()"
    date_params: list[Any] = [effective_date] if effective_date is not None else []
    bk_join_staging = " AND ".join(f't."{k}" = s."{k}"' for k in business_key)
    bk_first = business_key[0]

    # Parsed once here; reused for the pre-detect_deletes ANALYZE (#361) and
    # the trailing ANALYZE (#312).
    analyze_schema, analyze_tbl = parse_schema_table(table_name)

    # #401 item 4: cheap catalog lookup that flags the non-partial
    # UNIQUE (bk, is_current) authoring anti-pattern before it detonates on a
    # later run. Warns once per target per process; never raises.
    scd2_preflight_index_shape(cursor, table_name, scd2, context)

    # #361: index the staging business key and (optionally) bump work_mem
    # before any change-detection statement plans.  Both shrink the planner's
    # cost for the anti-joins below; the index also gives an index-nested-loop
    # option that the LIKE-cloned staging table otherwise lacks.
    _create_staging_bk_index(cursor, stage_table, business_key)
    if change_detection_work_mem is not None:
        _apply_change_detection_work_mem(cursor, change_detection_work_mem)

    # #419: duplicated business keys in the incoming data corrupt the
    # timeline silently (double-current inserts, misclassified "unchanged"
    # rows) -- fail loudly before any DML.  Runs after the staging BK index
    # so the GROUP BY has an index option.
    _assert_staging_business_keys_unique(
        cursor,
        stage_table,
        table_name,
        business_key,
        partition_column,
        change_detection_statement_timeout,
    )

    # Build partition scope clause (parameterized for safety)
    partition_clause = ""
    partition_params: list[Any] = []
    if partition_column and partition_values:
        placeholders = ", ".join(["%s"] * len(partition_values))
        partition_clause = f' AND t."{partition_column}" IN ({placeholders})'
        partition_params = list(partition_values)

    # Check if the target table has the sequence column (backwards compat:
    # existing tables without the column silently skip sequence population).
    has_sequence_col = False
    if sequence_col:
        _schema, _tbl = parse_schema_table(table_name)
        cursor.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
            """,
            (_schema, _tbl, sequence_col),
        )
        has_sequence_col = cursor.fetchone() is not None

    count_sql = f"""
        SELECT
            count(*) FILTER (
                WHERE t."{bk_first}" IS NULL
            ) AS new_count,
            count(*) FILTER (
                WHERE t."{bk_first}" IS NOT NULL
                  AND t."{hash_col}" <> s."{hash_col}"
            ) AS changed_count
        FROM {stage_table} s
        LEFT JOIN {table_name} t
            ON {bk_join_staging}
            AND t."{is_current_col}" = true{partition_clause}
    """  # noqa: S608
    with _bounded_statement_timeout(cursor, change_detection_statement_timeout):
        cursor.execute(count_sql, partition_params or None)
    stats_row = cursor.fetchone()
    if stats_row is None:
        msg = f"SCD2 count query for {table_name} returned no rows"
        raise ValueError(msg)
    new_count: int = stats_row[0]
    changed_count: int = stats_row[1]

    total_changes = new_count + changed_count
    if total_changes == 0 and not detect_deletes:
        # No changes and no delete detection -- skip remaining work
        cursor.execute(f"DROP TABLE IF EXISTS {stage_table}")  # noqa: S608
        context.log.info(
            f"SCD2: No changes detected for {table_name} ({total_staged_rows} rows unchanged)"
        )
        return {
            "rows_new": 0,
            "rows_expired": 0,
            "rows_inserted": 0,
            "rows_unchanged": total_staged_rows,
            "rows_deleted": 0,
        }

    # Expire changed rows, then insert new/changed rows.
    #
    # These must be two separate statements (not a single CTE) because
    # PostgreSQL executes all CTE sub-statements with the same snapshot.
    # A data-modifying CTE (UPDATE in `expired` + INSERT in main query)
    # would not see the UPDATE's effect on is_current, causing a
    # UniqueViolation on partial unique indexes WHERE (is_current).
    bk_join_update = " AND ".join(f't."{k}" = s."{k}"' for k in business_key)

    if total_changes > 0:
        expire_sql = f"""
            UPDATE {table_name} t
            SET "{effective_to_col}" = {date_expr},
                "{is_current_col}" = false
            FROM {stage_table} s
            WHERE {bk_join_update}
              AND t."{is_current_col}" = true{partition_clause}
              AND t."{hash_col}" <> s."{hash_col}"
        """  # noqa: S608
        with _bounded_statement_timeout(cursor, change_detection_statement_timeout):
            cursor.execute(expire_sql, [*date_params, *partition_params] or None)

        # After the UPDATE, changed rows no longer have is_current=true,
        # so the LEFT JOIN below sees them as "new" (t.bk IS NULL).
        # Unchanged rows still have is_current=true, so they are excluded.
        #
        # Diff-table column-name invariant: Stage 1's SELECT projects bare
        # column references (`s."col"`), so Postgres preserves "col" as the
        # diff table's column name -- which Stage 2 then references via
        # insert_cols_quoted. If select_from_staging ever grows a computed
        # expression (e.g., `s.foo || s.bar`), Postgres would auto-name it
        # `?column?` and the Stage 2 INSERT would break. Any such expression
        # MUST carry an explicit `AS "name"` matching the corresponding entry
        # in insert_cols_quoted.
        insert_cols_quoted = ", ".join(f'"{c}"' for c in stage_columns)
        select_from_staging = ", ".join(f's."{c}"' for c in stage_columns)

        end_of_time = scd2.end_of_time

        # Materialize-then-insert (see #274): Stage 1 CTAS materializes the
        # staging-vs-target diff into a temp table; Stage 2 reads from the
        # diff and inserts into the target with no JOIN against the target.
        # This structurally breaks the self-modification cycle that made the
        # single-statement form O(N^2) on empty targets.
        #
        # Temp-table uniqueness invariant: f"{stage_table}_diff" is
        # collision-free only because (a) stage_table is per-call-unique and
        # (b) scd2_finalize runs in a single transaction, so ON COMMIT DROP
        # fires before any plausible reuse. If either invariant breaks
        # (e.g., shared stage tables across calls, or finalize spanning
        # multiple txs), switch to a UUID/run_id-derived suffix.
        diff_table = f"{stage_table}_diff"

        # now() = transaction_timestamp() invariant: the expire UPDATE above
        # and Stage 1's CTAS below both evaluate {date_expr}. When
        # date_expr=='now()', PostgreSQL guarantees the same value across
        # all statements in a single transaction, which is what makes the
        # SCD2 audit trail symmetric (an expired row's effective_to matches
        # its successor's effective_from). A future "compute _eff_from in
        # Python and parameter-pass it" refactor would silently break this
        # symmetry -- do not move date_expr evaluation to the application
        # layer without updating both call sites in lockstep.
        if has_sequence_col and sequence_col:
            # Per-business-key sequence: COALESCE(MAX(existing), 0) + 1.
            # The MAX subquery is scoped to partition_clause so partition-
            # scoped writes don't pull sequence numbers from unrelated
            # partitions. The subquery references only t2 (the target),
            # which is read-only at Stage 1 time, so it is safe to move
            # into the CTAS SELECT.
            bk_subquery_join = " AND ".join(f't2."{k}" = s."{k}"' for k in business_key)
            # Scope the MAX subquery to the same partition when applicable
            partition_subquery_clause = ""
            if partition_column and partition_values:
                placeholders = ", ".join(["%s"] * len(partition_values))
                partition_subquery_clause = f' AND t2."{partition_column}" IN ({placeholders})'
            ctas_sql = f"""
                CREATE TEMP TABLE {diff_table} ON COMMIT DROP AS
                SELECT {select_from_staging}, {date_expr} AS _eff_from,
                       COALESCE((
                           SELECT MAX(t2."{sequence_col}")
                           FROM {table_name} t2
                           WHERE {bk_subquery_join}{partition_subquery_clause}
                       ), 0) + 1 AS _seq
                FROM {stage_table} s
                LEFT JOIN {table_name} t
                    ON {bk_join_staging}
                    AND t."{is_current_col}" = true{partition_clause}
                WHERE t."{bk_first}" IS NULL
            """  # noqa: S608
            # Extra partition_params for the MAX subquery's IN clause
            seq_partition_params = list(partition_values) if partition_values else []
            ctas_params = [*date_params, *seq_partition_params, *partition_params]
            insert_sql = f"""
                INSERT INTO {table_name}
                    ({insert_cols_quoted}, "{effective_from_col}", "{effective_to_col}",
                     "{is_current_col}", "{sequence_col}")
                SELECT {insert_cols_quoted}, _eff_from, '{end_of_time}'::date, true, _seq
                FROM {diff_table}
            """  # noqa: S608
        else:
            ctas_sql = f"""
                CREATE TEMP TABLE {diff_table} ON COMMIT DROP AS
                SELECT {select_from_staging}, {date_expr} AS _eff_from
                FROM {stage_table} s
                LEFT JOIN {table_name} t
                    ON {bk_join_staging}
                    AND t."{is_current_col}" = true{partition_clause}
                WHERE t."{bk_first}" IS NULL
            """  # noqa: S608
            ctas_params = [*date_params, *partition_params]
            insert_sql = f"""
                INSERT INTO {table_name}
                    ({insert_cols_quoted}, "{effective_from_col}", "{effective_to_col}",
                     "{is_current_col}")
                SELECT {insert_cols_quoted}, _eff_from, '{end_of_time}'::date, true
                FROM {diff_table}
            """  # noqa: S608

        # Stage 1 reads the target via the anti-join -> bound it.  Stage 2
        # writes only the bounded diff (and on a first-ever load that is the
        # entire dataset), so it is deliberately left unbounded (#361).
        with _bounded_statement_timeout(cursor, change_detection_statement_timeout):
            cursor.execute(ctas_sql, ctas_params or None)
        # Stage 2 has no %s placeholders -- _eff_from / _seq come from the
        # diff table, end_of_time is interpolated, true is a literal.
        cursor.execute(insert_sql)

    # Optionally expire records absent from incoming data.
    rows_deleted = 0
    if detect_deletes:
        # #361: Stage-1 above just inserted up to `new_count` fresh
        # is_current=true rows for this partition.  The trailing target
        # ANALYZE (#312) only runs *after* this UPDATE, so without an interim
        # refresh the anti-join below plans against zero stats for the
        # just-inserted load_period -- the planner estimates rows=1 and builds
        # its hash on a side that is actually millions of rows.  Refresh now
        # (only when Stage-1 changed the target) so detect_deletes plans
        # against real cardinalities.
        if total_changes > 0:
            cursor.execute(
                sql.SQL("ANALYZE {}").format(sql.Identifier(analyze_schema, analyze_tbl))
            )
        delete_sql = f"""
            UPDATE {table_name} t
            SET "{effective_to_col}" = {date_expr},
                "{is_current_col}" = false
            WHERE t."{is_current_col}" = true{partition_clause}
              AND NOT EXISTS (
                  SELECT 1 FROM {stage_table} s
                  WHERE {bk_join_update}
              )
        """  # noqa: S608
        with _bounded_statement_timeout(cursor, change_detection_statement_timeout):
            cursor.execute(delete_sql, [*date_params, *partition_params] or None)
        rows_deleted = cursor.rowcount
        if rows_deleted > 0:
            context.log.info(
                f"SCD2 detect_deletes: expired {rows_deleted} absent "
                f"business key(s) in {table_name}"
            )

    # Cleanup
    cursor.execute(f"DROP TABLE IF EXISTS {stage_table}")  # noqa: S608

    # Refresh target table planner stats so back-to-back finalize calls
    # against the same target (one per load_period under per-period
    # sequencing) plan against accurate reltuples / histograms. Without
    # this, the next call's count, expire, and detect_deletes statements
    # still see pg_class stats from before this call's writes. Reaching
    # this point means we passed the early-return guard above, so DML
    # happened. ANALYZE is sample-based (fast) and takes only
    # SHARE UPDATE EXCLUSIVE -- safe to issue inside the write tx. See
    # #312 follow-up comment.
    cursor.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(analyze_schema, analyze_tbl)))

    rows_unchanged = total_staged_rows - total_changes
    context.log.info(
        f"SCD2 complete for {table_name}: "
        f"{new_count} new, {changed_count} changed/expired, "
        f"{rows_unchanged} unchanged, {rows_deleted} deleted"
    )

    return {
        "rows_new": new_count,
        "rows_expired": changed_count,
        "rows_inserted": total_changes,
        "rows_unchanged": rows_unchanged,
        "rows_deleted": rows_deleted,
    }


def execute_scd2(
    config: WriterConfig,
    cursor: psycopg.Cursor,
    table_name: str,
    df: pl.DataFrame,
    business_key: list[str],
    scd2: SCD2Config,
    context: LoggingContext,
    *,
    detect_deletes: bool = False,
    partition_column: str | None = None,
    partition_values: list[Any] | None = None,
    effective_date: date | None = None,
    change_detection_work_mem: str | None = None,
    change_detection_statement_timeout: str | None = None,
) -> dict[str, int | str]:
    """Execute SCD2 write: stage, detect changes, expire old, insert new.

    The incoming DataFrame must already contain the hash column (computed
    by handle_output before dispatching here).

    When ``partition_column`` and ``partition_values`` are both provided,
    change detection is scoped to the active partition(s).

    When ``scd2.sequence_col`` is set, each inserted version receives a
    per-business-key monotonic integer (1, 2, 3, ...).  If the target table
    does not contain the named column, the sequence is silently skipped.

    Returns:
        Dict with rows_new, rows_expired, rows_inserted, rows_unchanged,
        and rows_deleted (always present, 0 when detect_deletes is False).
    """
    row_count = len(df)
    if row_count == 0:
        if detect_deletes:
            raise ValueError(
                f"detect_deletes=True with an empty DataFrame for table "
                f"'{table_name}'. This would expire ALL current records in "
                f"the dimension table. If the source genuinely has zero "
                f"active records, expire them with a targeted UPDATE instead "
                f"of relying on the SCD2 IO manager."
            )
        return {
            "rows_new": 0,
            "rows_expired": 0,
            "rows_inserted": 0,
            "rows_unchanged": 0,
            "rows_deleted": 0,
        }

    stage_table = scd2_create_staging(cursor, table_name, scd2)
    scd2_insert_staging(config, cursor, stage_table, df, context)
    return scd2_finalize(
        cursor,
        table_name,
        stage_table,
        row_count,
        df.columns,
        business_key,
        scd2,
        context,
        detect_deletes=detect_deletes,
        partition_column=partition_column,
        partition_values=partition_values,
        effective_date=effective_date,
        change_detection_work_mem=change_detection_work_mem,
        change_detection_statement_timeout=change_detection_statement_timeout,
    )
