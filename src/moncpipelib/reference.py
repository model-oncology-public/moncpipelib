"""Helpers for non-partitioned, current-snapshot reference silvers.

Some bronze tables are append-partitioned dictionaries -- ICD-O-3
morphology / topography, NDC reference lookups, etc. -- where the silver
consumer never wants history.  The silver answers "what does code X mean
today?", so each materialization should TRUNCATE-and-REPLACE the silver
table from the **latest** bronze partition.

The full pattern for the silver asset is:

1. Read just the latest partition of the bronze.
2. Stream batches through a transform.
3. ``database.write(..., target=..., full_refresh=True)``.

This module owns step (1).  See :mod:`moncpipelib.historical` for the
opposite pattern (per-partition silver tracking SCD2 history) -- that
module is period-registry-aware; this one is its mirror image.

Future siblings (``read_latest_n_partitions``, etc.) belong here too.

I/O at boundaries (CLAUDE.md "streaming by default"):

- :func:`read_latest_partition` returns ``Iterator[pl.DataFrame]`` -- the
  same seam :func:`moncpipelib.resources.read_batched` exposes.  Callers
  chain ``transform_batched(read_latest_partition(...), ...)`` exactly
  like they chain it on top of ``read_batched`` today; peak memory is
  bounded by ``batch_size`` regardless of partition size.
- Identifiers (schema, table, partition column, projection columns) are
  composed via :class:`psycopg.sql.Identifier` -- never f-string
  interpolated -- so reserved words quote correctly and injection
  surface is closed even though the inputs are pipeline-author
  constants (HIPAA / SOC 2 / ISO 27001 / HITRUST).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from psycopg import sql

from moncpipelib.resources.postgres import read_batched

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import polars as pl

    from moncpipelib.resources.postgres import PostgresResource
    from moncpipelib.resources.types import LoggingContext


class EmptyPartitionedTableError(LookupError):
    """The upstream bronze has no rows; there is no latest partition.

    Subclasses :class:`LookupError` so callers with a generic
    ``except LookupError`` still catch the empty-bronze case.  Raised by
    :func:`read_latest_partition` after the pre-check query returns
    ``NULL`` for ``MAX({partition_column})``.
    """


def _split_qualified_table(source_table: str) -> tuple[str, str]:
    """Split a ``schema.table`` literal into its two parts.

    Input is treated as a pipeline-author constant (not user data); the
    helper accepts exactly two dot-separated, non-empty parts and
    rejects everything else.  Three-part ``catalog.schema.table`` is
    refused on purpose: composing ``sql.Identifier(schema, table)``
    requires a single schema literal, and silently merging the catalog
    into the schema would emit a malformed quoted identifier.
    """
    parts = source_table.split(".")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"source_table must be a dotted 'schema.table' identifier, got {source_table!r}"
        )
    return parts[0], parts[1]


def read_latest_partition(
    database: PostgresResource,
    *,
    source_table: str,
    partition_column: str = "load_period",
    columns: Iterable[str] | None = None,
    batch_size: int = 50_000,
    context: LoggingContext | None = None,
) -> Iterator[pl.DataFrame]:
    """Stream every row in ``source_table``'s latest partition.

    The helper composes ``SELECT {cols} FROM {table} WHERE {col} =
    (SELECT MAX({col}) FROM {table})`` and hands it to
    :func:`moncpipelib.resources.read_batched`, so the call site is one
    drop-in substitution away from any existing partitioned-silver that
    uses ``get_period_from_registry`` + ``read_batched``.

    Designed for **non-partitioned, full-refresh reference silvers**
    that re-materialize the whole table from the latest bronze
    partition on every run.  For SCD2 or per-partition silvers, see
    :mod:`moncpipelib.historical` instead.

    Args:
        database: :class:`~moncpipelib.resources.PostgresResource` to
            read from.  Streaming uses a server-side cursor through the
            resource's SQLAlchemy engine.
        source_table: Fully-qualified ``schema.table`` identifier of the
            upstream bronze.  Split on the rightmost ``.``; each side is
            quoted via :class:`psycopg.sql.Identifier` so reserved words
            and mixed case round-trip correctly.
        partition_column: Column holding the partition discriminator.
            Defaults to ``"load_period"`` -- the moncpipelib convention;
            override for sources that partition on a different column.
        columns: Optional column projection.  ``None`` emits
            ``SELECT *``; otherwise each name is quoted via
            :class:`psycopg.sql.Identifier` and emitted in the supplied
            order.  Iterables are materialised once, so a generator that
            yields names is fine.
        batch_size: Rows per yielded :class:`polars.DataFrame`.  Default
            50,000 matches :func:`read_batched`.
        context: Optional :class:`~moncpipelib.resources.types.LoggingContext`
            (Dagster ``OutputContext`` / ``AssetExecutionContext`` both
            satisfy structurally).  Passed through to ``read_batched``
            for per-batch progress logging.  The "Reading latest
            partition ... from ..." line is emitted at INFO after the
            pre-check.

    Yields:
        :class:`polars.DataFrame` for each batch of the latest partition.

    Raises:
        EmptyPartitionedTableError: ``MAX({partition_column})`` returns
            ``NULL`` -- the source table has no rows yet.  The error
            message names both ``source_table`` and ``partition_column``
            so the silver asset fails fast with a useful pointer to the
            upstream gap.  The pre-check runs on the **first ``next()``**
            of the returned iterator (this function is a generator);
            callers that build the iterator without consuming it will
            not observe the error.
        ValueError: ``source_table`` is not a dotted ``schema.table``
            identifier (forwarded from :func:`_split_qualified_table`).
    """
    schema_name, table_name = _split_qualified_table(source_table)
    table_ident = sql.Identifier(schema_name, table_name)
    col_ident = sql.Identifier(partition_column)

    # Pre-check: empty table = no "latest" partition.  ``MAX(col)`` is
    # an index-only scan when ``partition_column`` is indexed (the
    # moncpipelib convention) and a sequential scan otherwise -- callers
    # putting this on a multi-billion-row bronze should ensure the
    # partition column has an index.  Either way, the precheck lets us
    # raise a typed error instead of streaming zero rows for an asset
    # author to debug later.  We intentionally do NOT bind the pre-
    # checked max as a literal to the main SELECT -- the subquery form
    # below is authoritative if a newer partition lands between the two
    # queries, which is the right "latest" semantics.
    precheck = sql.SQL("SELECT MAX({col}) FROM {table}").format(col=col_ident, table=table_ident)
    with database.get_connection() as conn, conn.cursor() as cur:
        cur.execute(precheck)
        row = cur.fetchone()
    latest_partition = row[0] if row is not None else None
    if latest_partition is None:
        raise EmptyPartitionedTableError(
            f"{source_table} has no rows; {partition_column} max is NULL"
        )

    # Projection: SELECT * by default, otherwise quote each column.
    materialized_columns = list(columns) if columns is not None else None
    if materialized_columns is None:
        select_clause: sql.Composable = sql.SQL("*")
    else:
        select_clause = sql.SQL(", ").join(sql.Identifier(c) for c in materialized_columns)

    # Subquery form for the WHERE clause -- re-evaluates at execution
    # time, so any partition that lands between the pre-check and main
    # SELECT is included (the right thing for "latest").
    main_query = sql.SQL(
        "SELECT {cols} FROM {table} WHERE {col} = (SELECT MAX({col}) FROM {table})"
    ).format(cols=select_clause, table=table_ident, col=col_ident)
    query_str = main_query.as_string(None)

    if context is not None:
        context.log.info("Reading latest partition %s from %s", latest_partition, source_table)

    # ``read_batched`` types ``context`` as ``OpExecutionContext`` but
    # only ever calls ``context.log.info(...)`` -- the same surface
    # ``LoggingContext`` declares.  Cast to ``Any`` to keep this helper
    # honest about accepting the wider Protocol type (matching the
    # widened writers.py / reconciliation.py contexts) without lying
    # about ``read_batched``'s annotation.
    yield from read_batched(
        query_str,
        database,
        batch_size=batch_size,
        context=cast(Any, context),
    )
