"""Lineage tracking for data pipelines.

This module provides row-level lineage tracking using UUID7-based lineage keys.
UUID7 provides time-ordering and embedded timestamps for better debuggability.

Lineage metadata is stored in a separate `lineage.data_lineage` table, while
data tables contain a `_lineage_id` foreign key column.

The lineage system supports:
- Source file and system tracking
- Temporal tracking (data dates and date ranges)
- Backfill tracking with history
- Aggregation lineage (many:1 relationships)
- Transformation tracking
- Resilient recovery with composite backup keys
"""

import hashlib
import time
import uuid
from datetime import UTC, date, datetime
from typing import Any

import polars as pl
import sqlalchemy as sa

from moncpipelib.config import LineageDefaults


def generate_uuid7() -> uuid.UUID:
    """Generate a UUID version 7 (time-ordered with embedded timestamp).

    UUID7 format (RFC 9562):
    - 48 bits: Unix timestamp in milliseconds
    - 12 bits: Sub-millisecond precision / sequence counter
    - 2 bits: Version (0b111 = 7)
    - 62 bits: Random data

    Benefits:
    - Chronologically sortable
    - Timestamp extractable for debugging
    - Better database index performance than UUID4

    Returns:
        uuid.UUID: A time-ordered UUID7
    """
    # Get current timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)

    # Generate random bytes for the rest
    random_bytes = uuid.uuid4().bytes[6:]  # Use 10 bytes of random data

    # Build UUID7:
    # - Bytes 0-5: 48-bit timestamp (6 bytes)
    # - Byte 6: version (high nibble = 7) + random (low nibble)
    # - Byte 7: variant (2 bits) + random (6 bits)
    # - Bytes 8-15: random data

    timestamp_bytes = timestamp_ms.to_bytes(6, byteorder="big")

    # Set version to 7 (0b0111xxxx)
    version_byte = 0x70 | (random_bytes[0] & 0x0F)

    # Set variant to RFC 4122 (0b10xxxxxx)
    variant_byte = 0x80 | (random_bytes[1] & 0x3F)

    uuid_bytes = timestamp_bytes + bytes([version_byte, variant_byte]) + random_bytes[2:]

    return uuid.UUID(bytes=uuid_bytes)


def extract_timestamp_from_uuid7(uuid_val: uuid.UUID) -> datetime:
    """Extract the embedded timestamp from a UUID7.

    Args:
        uuid_val: A UUID7 instance

    Returns:
        datetime: The timestamp when the UUID was created (UTC)
    """
    # Extract first 48 bits (6 bytes) as timestamp in milliseconds
    timestamp_ms = int.from_bytes(uuid_val.bytes[:6], byteorder="big")
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)


def generate_lineage_key(
    asset_name: str,
    layer: str,
    run_id: str,
    data_date: date | None = None,
    source_file: str | None = None,
    version: int = 1,
) -> str:
    """Generate a deterministic composite lineage key for recovery scenarios.

    This key provides human-readable context and can help reconstruct lineage
    relationships even if the lineage table is unavailable or corrupted.

    Format: v{version}:{asset}:{layer}:{date_or_hash}:{run_id_prefix}

    Args:
        asset_name: Name of the asset (e.g., "claims_bronze")
        layer: Data layer (bronze/silver/gold)
        run_id: Dagster run ID
        data_date: Optional data date for the load
        source_file: Optional source file name
        version: Lineage key format version (default: 1)

    Returns:
        str: Composite key like "v1:claims:bronze:2024-01-15:abc123" or
             "v1:claims:silver:f3a2:abc123" (hash if no date)

    Example:
        >>> generate_lineage_key("claims_bronze", "bronze", "abc123-def456",
        ...                      data_date=date(2024, 1, 15))
        'v1:claims:bronze:2024-01-15:abc123'
    """
    # Clean asset name (remove layer suffix if present)
    asset_clean = asset_name.replace(f"_{layer}", "")

    # Use data_date if available, otherwise hash source_file for uniqueness
    if data_date:
        date_part = data_date.isoformat()
    elif source_file:
        # Short hash of source file for uniqueness
        file_hash = hashlib.sha256(source_file.encode()).hexdigest()[:8]
        date_part = file_hash
    else:
        # Use timestamp as fallback
        date_part = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    # First 6 chars of run_id for brevity
    run_id_prefix = run_id[:6] if len(run_id) >= 6 else run_id

    return f"v{version}:{asset_clean}:{layer}:{date_part}:{run_id_prefix}"


def parse_lineage_key(lineage_key: str) -> dict[str, str]:
    """Parse a versioned lineage key into its components.

    Args:
        lineage_key: Versioned lineage key (e.g., "v1:claims:bronze:2024-01-15:abc123")

    Returns:
        dict: Parsed components with keys: version, asset, layer, date_or_hash, run_id_prefix

    Raises:
        ValueError: If lineage key format is invalid or version is unsupported

    Example:
        >>> parse_lineage_key("v1:claims:bronze:2024-01-15:abc123")
        {'version': '1', 'asset': 'claims', 'layer': 'bronze',
         'date_or_hash': '2024-01-15', 'run_id_prefix': 'abc123'}
    """
    parts = lineage_key.split(":")

    if len(parts) < 2:
        raise ValueError(f"Invalid lineage key format: {lineage_key}")

    # Extract version
    version_part = parts[0]
    if not version_part.startswith("v"):
        raise ValueError(
            f"Lineage key must start with version prefix (v1, v2, etc.): {lineage_key}"
        )

    version = version_part[1:]  # Remove 'v' prefix

    # Parse based on version
    if version == "1":
        if len(parts) != 5:
            raise ValueError(
                f"Version 1 lineage key must have 5 parts (v1:asset:layer:date:run_id): {lineage_key}"
            )
        return {
            "version": version,
            "asset": parts[1],
            "layer": parts[2],
            "date_or_hash": parts[3],
            "run_id_prefix": parts[4],
        }
    else:
        raise ValueError(f"Unsupported lineage key version: {version}")


class LineageTracker:
    """Manages row-level lineage tracking for data pipelines.

    This tracker creates lineage records in the `lineage.data_lineage` table
    and attaches lineage IDs to dataframes for row-level tracking.

    Attributes:
        engine: SQLAlchemy engine for database operations
    """

    def __init__(self, engine: sa.Engine):
        """Initialize the lineage tracker.

        Args:
            engine: SQLAlchemy engine connected to the target database
        """
        self.engine = engine

    # Bound parameter placeholders used in ``INSERT INTO lineage.data_lineage``.
    # The SQLAlchemy path uses ``:name`` and the psycopg path uses ``%(name)s``;
    # ``_LINEAGE_INSERT_PARAM_NAMES`` is the shared single source of truth so
    # both code paths stay in lock-step when new columns are added in future
    # migrations.
    _LINEAGE_INSERT_PARAM_NAMES: tuple[str, ...] = (
        "lineage_id",
        "lineage_key",
        "run_id",
        "asset_name",
        "pipeline_id",
        "layer",
        "source_file",
        "source_system",
        "data_date",
        "data_date_range",
        "row_count",
        "is_backfill",
        "backfill_reason",
        "backfill_id",
        "replaces_lineage_id",
        "parent_lineage_ids",
        "transformation_type",
        "metadata",
    )

    @staticmethod
    def generate_lineage_ids(
        *,
        asset_name: str,
        layer: str,
        run_id: str,
        data_date: date | None = None,
        source_file: str | None = None,
    ) -> tuple[str, str]:
        """Generate ``(lineage_id, lineage_key)`` client-side without
        touching the database.

        Pure, side-effect-free: callers can call this *before* opening a
        database connection, then attach the ids to the DataFrame and run
        ``write_lineage_record`` inside the same transaction as the data
        DML. This is the entry point migration 018 Phase 3 uses to fold
        the lineage INSERT into the same transaction as the data DML.

        Returns:
            tuple[str, str]: ``(lineage_id, lineage_key)`` -- a UUID7 and
            the composite backup key.
        """
        lineage_id = str(generate_uuid7())
        lineage_key = generate_lineage_key(
            asset_name=asset_name,
            layer=layer,
            run_id=run_id,
            data_date=data_date,
            source_file=source_file,
        )
        return lineage_id, lineage_key

    # Per-column placeholder cast wrappers. psycopg3 has no default
    # ``dict → jsonb`` adapter and the ``metadata`` column needs an
    # explicit ``::jsonb`` cast at bind time. ``_build_lineage_params``
    # ``json.dumps()`` the dict to a ``str``; the cast here is what
    # tells Postgres to parse the string back into jsonb on INSERT.
    # Mirrors the ``scd2_reconciliations`` INSERT which uses the same
    # pattern at ``write_scd2_reconciliation``.
    #
    # The wrapping form ``({placeholder})::jsonb`` is load-bearing for
    # the SA dialect: ``sa.text()``'s named-param parser scans ``:name``
    # greedily and consumes the first ``:`` of a bare ``::jsonb`` suffix
    # into the placeholder name, truncating ``:metadata::jsonb`` to a
    # ``metada`` placeholder + literal ``:jsonb`` -- silently breaks the
    # bind.  Parenthesising the placeholder pins the boundary cleanly
    # on both SA (``:name``) and psycopg (``%(name)s``) dialects.
    _LINEAGE_INSERT_PARAM_CAST_TEMPLATES: dict[str, str] = {
        "metadata": "({placeholder})::jsonb",
    }

    @staticmethod
    def _build_lineage_params(
        *,
        lineage_id: str,
        lineage_key: str,
        run_id: str,
        asset_name: str,
        layer: str,
        source_file: str | None,
        source_system: str | None,
        data_date: date | None,
        data_date_range: tuple[date, date] | None,
        is_backfill: bool,
        backfill_reason: str | None,
        backfill_id: str | None,
        replaces_lineage_id: str | None,
        parent_lineage_ids: list[str] | None,
        transformation_type: str | None,
        row_count: int | None,
        metadata: dict | None,
        pipeline_id: str | None,
    ) -> dict[str, Any]:
        """Build the dict of bound parameters for the lineage INSERT.

        Centralised so the SA-engine path (``create_lineage_record``) and
        the cursor path (``write_lineage_record``) bind identical values
        in identical order. Format-conversion for ``data_date_range``
        and JSON-serialisation for ``metadata`` live here too.
        """
        import json as _json

        date_range_str = None
        if data_date_range:
            start, end = data_date_range
            date_range_str = f"[{start},{end}]"

        # psycopg3 has no default ``dict → jsonb`` adapter; the SQL casts
        # ``%(metadata)s::jsonb`` so we bind a JSON string here.  ``None``
        # passes through as SQL NULL on both dialects.
        metadata_str = _json.dumps(metadata) if metadata is not None else None

        return {
            "lineage_id": lineage_id,
            "lineage_key": lineage_key,
            "run_id": run_id,
            "asset_name": asset_name,
            "pipeline_id": pipeline_id,
            "layer": layer,
            "source_file": source_file,
            "source_system": source_system,
            "data_date": data_date,
            "data_date_range": date_range_str,
            "row_count": row_count,
            "is_backfill": is_backfill,
            "backfill_reason": backfill_reason,
            "backfill_id": backfill_id,
            "replaces_lineage_id": replaces_lineage_id,
            "parent_lineage_ids": parent_lineage_ids,
            "transformation_type": transformation_type,
            "metadata": metadata_str,
        }

    @classmethod
    def _lineage_insert_sql(cls, *, dialect: str) -> str:
        """Return the ``INSERT INTO lineage.data_lineage`` statement.

        ``dialect="sa"`` produces ``:name`` placeholders for SQLAlchemy
        ``text()`` binding; ``dialect="psycopg"`` produces ``%(name)s``
        placeholders for psycopg ``cursor.execute(sql, mapping)``.

        Per-column cast templates from
        ``_LINEAGE_INSERT_PARAM_CAST_TEMPLATES`` wrap the placeholder
        for both dialects (e.g. ``(:metadata)::jsonb``).  Columns
        absent from the map use the bare placeholder.
        """
        if dialect == "sa":
            placeholder = ":{name}"
        elif dialect == "psycopg":
            placeholder = "%({name})s"
        else:
            raise ValueError(f"Unknown dialect: {dialect!r}")

        cols = ",\n            ".join(cls._LINEAGE_INSERT_PARAM_NAMES)

        def render(name: str) -> str:
            ph = placeholder.format(name=name)
            template = cls._LINEAGE_INSERT_PARAM_CAST_TEMPLATES.get(name)
            return template.format(placeholder=ph) if template else ph

        vals = ",\n            ".join(render(p) for p in cls._LINEAGE_INSERT_PARAM_NAMES)
        return (
            "INSERT INTO lineage.data_lineage (\n            "
            + cols
            + "\n        ) VALUES (\n            "
            + vals
            + "\n        )"
        )

    def write_lineage_record(
        self,
        cursor: Any,
        *,
        lineage_id: str,
        lineage_key: str,
        run_id: str,
        asset_name: str,
        layer: str,
        source_file: str | None = None,
        source_system: str | None = None,
        data_date: date | None = None,
        data_date_range: tuple[date, date] | None = None,
        is_backfill: bool = False,
        backfill_reason: str | None = None,
        backfill_id: str | None = None,
        replaces_lineage_id: str | None = None,
        parent_lineage_ids: list[str] | None = None,
        transformation_type: str | None = None,
        row_count: int | None = None,
        metadata: dict | None = None,
        pipeline_id: str | None = None,
    ) -> None:
        """Execute the lineage-record INSERT on the supplied psycopg cursor.

        Does NOT commit; the caller is responsible for ``commit()``  /
        ``rollback()`` on the underlying connection. ``cursor`` is typed
        as ``Any`` to avoid a hard psycopg import at this layer; in
        practice it is a ``psycopg.Cursor``.

        Migration 018 Phase 3 uses this entry point to fold the lineage
        INSERT into the same transaction as the data DML. ``lineage_id``
        and ``lineage_key`` are pre-generated client-side via
        :py:meth:`generate_lineage_ids`.
        """
        sql = self._lineage_insert_sql(dialect="psycopg")
        params = self._build_lineage_params(
            lineage_id=lineage_id,
            lineage_key=lineage_key,
            run_id=run_id,
            asset_name=asset_name,
            layer=layer,
            source_file=source_file,
            source_system=source_system,
            data_date=data_date,
            data_date_range=data_date_range,
            is_backfill=is_backfill,
            backfill_reason=backfill_reason,
            backfill_id=backfill_id,
            replaces_lineage_id=replaces_lineage_id,
            parent_lineage_ids=parent_lineage_ids,
            transformation_type=transformation_type,
            row_count=row_count,
            metadata=metadata,
            pipeline_id=pipeline_id,
        )
        cursor.execute(sql, params)

    def find_prior_lineage_id(
        self,
        cursor: Any,
        *,
        asset_name: str,
        layer: str,
        data_date: date | None = None,
        data_date_range: tuple[date, date] | None = None,
    ) -> str | None:
        """Return the ``lineage_id`` of the most-recent prior ``data_lineage``
        row that matches the in-flight write's ``(asset_name, layer
        [, partition])`` tuple, or ``None`` if no prior row exists.

        Migration 018 Phase 4 helper for populating
        ``replaces_lineage_id`` on ``FULL_REFRESH`` writes. The lookup
        runs on the supplied cursor inside the same transaction as the
        eventual lineage INSERT, so the result is point-in-time
        consistent with the rest of the write.

        Lookup keys by partition shape:
        - **Whole-table** (``data_date is None and data_date_range is None``)
          → matches rows where BOTH ``data_date`` and ``data_date_range``
          are NULL on the prior row. Whole-table writes only chain to
          whole-table writes.
        - **Single-date partition** (``data_date is not None``)
          → matches rows where ``data_date = :data_date``. Partition
          writes only chain to writes of the same partition.
        - **Date-range partition** (``data_date_range is not None``)
          → matches rows where ``data_date_range = :data_date_range``
          (range equality only; overlapping-but-not-identical ranges
          return NULL).

        ``ORDER BY processed_at DESC LIMIT 1`` returns the immediately
        prior row. Under READ COMMITTED, two concurrent ``FULL_REFRESH``
        runs of the same asset may yield either a sibling pair (both
        link to the same predecessor) or a chain (whichever commits
        second links to the first); consumers must not assume one shape.

        Returns:
            str | None: ``lineage_id`` of the prior row as a string, or
            ``None`` if no prior row matched.
        """
        clauses = ["asset_name = %(asset_name)s", "layer = %(layer)s"]
        params: dict[str, Any] = {"asset_name": asset_name, "layer": layer}

        if data_date is not None:
            clauses.append("data_date = %(data_date)s")
            params["data_date"] = data_date
        elif data_date_range is not None:
            clauses.append("data_date_range = %(data_date_range)s")
            start, end = data_date_range
            params["data_date_range"] = f"[{start},{end}]"
        else:
            # Whole-table replacement: prior row must also be whole-table
            # (both NULL). Otherwise a partition write would shadow a
            # later whole-table write -- different shapes, not a chain.
            clauses.append("data_date IS NULL")
            clauses.append("data_date_range IS NULL")

        sql = (
            "SELECT lineage_id FROM lineage.data_lineage WHERE "
            + " AND ".join(clauses)
            + " ORDER BY processed_at DESC LIMIT 1"
        )
        cursor.execute(sql, params)
        row = cursor.fetchone()
        if row is None:
            import logging

            logging.getLogger("moncpipelib.lineage").debug(
                "find_prior_lineage_id: no prior row matched",
                extra={
                    "asset_name": asset_name,
                    "layer": layer,
                    "data_date": str(data_date) if data_date is not None else None,
                    "data_date_range": (
                        f"[{data_date_range[0]},{data_date_range[1]}]"
                        if data_date_range is not None
                        else None
                    ),
                },
            )
            return None
        return str(row[0])

    def write_scd2_reconciliation(
        self,
        cursor: Any,
        *,
        run_id: str,
        asset_name: str,
        target_table: str,
        pipeline_id: str | None,
        work_mem_applied: str | None,
        rows_collapsed: int,
        rows_timeline_updated: int,
        rows_renumbered: int,
        duration_seconds: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist one ``scd2_reconciliations`` audit row.

        Migration 019 (#308) Phase 6: durable audit trail for
        ``PostgresResource.reconcile_scd2`` invocations. Cursor + txn
        lifecycle owned by the caller (no commit here); intended to fire
        on the same cursor as the reconcile DML, before the commit, so
        the audit row is atomic with the reconciliation.

        Args:
            cursor: psycopg cursor in an open reconcile transaction.
            run_id: Dagster run ID (or caller-supplied identifier).
            asset_name: Asset that was reconciled.
            target_table: Fully-qualified ``schema.table`` reconciled.
            pipeline_id: Optional FK to ``pipeline_registry``. ``None``
                is permitted (column is nullable).
            work_mem_applied: Resolved per-tx ``work_mem`` literal, or
                ``None`` when no override was applied.
            rows_collapsed: Count from the reconcile return-dict.
            rows_timeline_updated: Count from the reconcile return-dict.
            rows_renumbered: Count from the reconcile return-dict.
            duration_seconds: Wall-clock duration in seconds.
            metadata: Optional JSONB future-proofing payload.
        """
        import json as _json

        from moncpipelib.config import config

        schema = config.scd2_reconciliations.schema_name
        table = config.scd2_reconciliations.table_name

        sql = (  # noqa: S608
            f"INSERT INTO {schema}.{table} ("
            f"run_id, asset_name, pipeline_id, target_table, "
            f"work_mem_applied, rows_collapsed, rows_timeline_updated, "
            f"rows_renumbered, duration_seconds, metadata) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)"
        )
        cursor.execute(
            sql,
            (
                run_id,
                asset_name,
                pipeline_id,
                target_table,
                work_mem_applied,
                rows_collapsed,
                rows_timeline_updated,
                rows_renumbered,
                duration_seconds,
                _json.dumps(metadata) if metadata is not None else None,
            ),
        )

    def write_validation_runs(
        self,
        cursor: Any,
        *,
        lineage_id: str,
        check_results: list[Any],
    ) -> int:
        """Persist per-check contract validation results.

        Migration 019 (#308) Phase 5: bulk-INSERT one row per
        :class:`~moncpipelib.contracts.models.CheckResultRow` into
        ``lineage.contract_validation_runs``. FKs to ``data_lineage``
        via ``lineage_id``; cursor + transaction lifecycle owned by the
        caller (no commit here).

        ``sample_failures`` JSONB payload is truncated to 20 rows per
        check before persistence to keep the audit-row size bounded.

        Args:
            cursor: psycopg cursor in an open transaction (must match
                the cursor that inserted the parent ``data_lineage`` row
                so the FK resolves via same-txn MVCC).
            lineage_id: UUID of the parent ``data_lineage`` row.
            check_results: List of
                :class:`~moncpipelib.contracts.models.CheckResultRow`.
                Pass-through ``CheckResultRow`` typing kept as ``Any``
                on the signature to avoid a circular import between
                lineage and contracts modules.

        Returns:
            Number of rows inserted (== ``len(check_results)``).
        """
        if not check_results:
            return 0

        import json as _json

        from moncpipelib.config import config

        schema = config.contract_validation_runs.schema_name
        table = config.contract_validation_runs.table_name

        sql = (  # noqa: S608
            f"INSERT INTO {schema}.{table} "
            f"(lineage_id, check_name, severity, passed, "
            f" failed_count, total_count, sample_failures) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)"
        )

        rows: list[tuple[Any, ...]] = []
        for cr in check_results:
            sample = cr.sample_failures
            if sample is not None and len(sample) > 20:
                sample = sample[:20]
            sample_json = _json.dumps(sample) if sample is not None else None
            rows.append(
                (
                    lineage_id,
                    cr.check_name,
                    cr.severity,
                    cr.passed,
                    cr.failed_count,
                    cr.total_count,
                    sample_json,
                )
            )

        cursor.executemany(sql, rows)
        return len(rows)

    def update_parent_lineage_ids(
        self,
        cursor: Any,
        *,
        lineage_id: str,
        parent_lineage_ids: list[str] | None,
    ) -> None:
        """Set ``parent_lineage_ids`` on an already-inserted lineage row.

        Migration 018 Phase 5: the batched-write path cannot know which
        upstream ``_lineage_id`` values to record at lineage-INSERT time
        because the parent set is only complete after every batch has
        been seen. The lineage row is therefore inserted with
        ``parent_lineage_ids = NULL`` (Phase 3 invariant: lineage row
        first so the FK on each batch's ``_lineage_id`` resolves), the
        batched DML loop accumulates the parent set, and this method
        amends the row at the end of the loop. All inside the same
        cursor / transaction; no commit here.

        A ``None`` or empty ``parent_lineage_ids`` is a no-op (the row
        already has ``NULL``); the call is skipped at the call site.
        """
        if not parent_lineage_ids:
            return
        cursor.execute(
            (
                "UPDATE lineage.data_lineage "
                "SET parent_lineage_ids = %(parent_lineage_ids)s "
                "WHERE lineage_id = %(lineage_id)s"
            ),
            {
                "parent_lineage_ids": parent_lineage_ids,
                "lineage_id": lineage_id,
            },
        )

    def create_lineage_record(
        self,
        *,
        run_id: str,
        asset_name: str,
        layer: str,
        source_file: str | None = None,
        source_system: str | None = None,
        data_date: date | None = None,
        data_date_range: tuple[date, date] | None = None,
        is_backfill: bool = False,
        backfill_reason: str | None = None,
        backfill_id: str | None = None,
        replaces_lineage_id: str | None = None,
        parent_lineage_ids: list[str] | None = None,
        transformation_type: str | None = None,
        row_count: int | None = None,
        metadata: dict | None = None,
        pipeline_id: str | None = None,
    ) -> tuple[str, str]:
        """Create a new lineage record and return the lineage_id.

        Args:
            run_id: Dagster run ID.
            asset_name: Dagster asset name (e.g. ``"dim_provider_gold"``).
            layer: Data layer (e.g., 'bronze', 'silver', 'gold')
            source_file: Source file path or name
            source_system: External system identifier (e.g., 'sftp', 'api')
            data_date: Single date for the data (for daily partitions)
            data_date_range: Tuple of (start_date, end_date) for multi-day data
            is_backfill: Whether this is a backfill operation
            backfill_reason: Explanation for the backfill
            backfill_id: Stable identifier of the Dagster backfill batch
                this row belongs to (from ``context.run.backfill_id``).
                NULL for non-backfill runs.
            replaces_lineage_id: UUID of the lineage record being replaced
            parent_lineage_ids: List of UUIDs for parent records (for aggregations)
            transformation_type: Type of transformation (e.g., 'aggregate', 'join', 'filter')
            row_count: Number of rows in the output dataset
            metadata: Additional metadata as JSON
            pipeline_id: Stable UUID identifying the logical pipeline (from contract)

        Returns:
            tuple[str, str]: The (lineage_id, lineage_key) for the created record

        Migration 018 Phase 3 split this method into:
        :py:meth:`generate_lineage_ids` (pure id generation) and
        :py:meth:`write_lineage_record` (cursor-driven INSERT). This
        wrapper is preserved for non-resource callers (tests, ad-hoc
        tools) that want the original "generate + commit" shape; it opens
        its own SA engine transaction. The hot write path in
        ``PostgresResource`` now uses ``generate_lineage_ids`` +
        ``write_lineage_record`` directly so the lineage INSERT shares a
        transaction with the data DML.

        Example:
            >>> tracker = LineageTracker(engine)
            >>> lineage_id = tracker.create_lineage_record(
            ...     run_id="abc123",
            ...     asset_name="claims_bronze",
            ...     layer="bronze",
            ...     source_file="claims_2024_01_15.csv",
            ...     data_date=date(2024, 1, 15),
            ...     row_count=1000
            ... )
        """
        lineage_id, lineage_key = self.generate_lineage_ids(
            asset_name=asset_name,
            layer=layer,
            run_id=run_id,
            data_date=data_date,
            source_file=source_file,
        )

        sql = self._lineage_insert_sql(dialect="sa")
        params = self._build_lineage_params(
            lineage_id=lineage_id,
            lineage_key=lineage_key,
            run_id=run_id,
            asset_name=asset_name,
            layer=layer,
            source_file=source_file,
            source_system=source_system,
            data_date=data_date,
            data_date_range=data_date_range,
            is_backfill=is_backfill,
            backfill_reason=backfill_reason,
            backfill_id=backfill_id,
            replaces_lineage_id=replaces_lineage_id,
            parent_lineage_ids=parent_lineage_ids,
            transformation_type=transformation_type,
            row_count=row_count,
            metadata=metadata,
            pipeline_id=pipeline_id,
        )

        with self.engine.begin() as conn:
            conn.execute(sa.text(sql), params)

        return lineage_id, lineage_key

    def attach_lineage_to_dataframe(
        self,
        df: pl.DataFrame,
        lineage_id: str,
        lineage_key: str,
    ) -> pl.DataFrame:
        """Attach lineage columns to all rows in the dataframe.

        Args:
            df: Input dataframe
            lineage_id: UUID lineage identifier to attach
            lineage_key: Human-readable composite lineage key

        Returns:
            pl.DataFrame: Dataframe with lineage ID and key columns added

        Example:
            >>> df = pl.DataFrame({"col1": [1, 2, 3]})
            >>> df_with_lineage = tracker.attach_lineage_to_dataframe(
            ...     df, lineage_id, lineage_key
            ... )
            >>> assert LineageDefaults.ID_COLUMN in df_with_lineage.columns
            >>> assert LineageDefaults.KEY_COLUMN in df_with_lineage.columns
        """
        return df.with_columns(
            pl.lit(lineage_id).alias(LineageDefaults.ID_COLUMN),
            pl.lit(lineage_key).alias(LineageDefaults.KEY_COLUMN),
        )

    def get_parent_lineage_ids(
        self,
        df: pl.DataFrame,
    ) -> list[str]:
        """Extract unique lineage IDs from a parent dataframe.

        This is useful when creating aggregations or joins where you need
        to track which source records contributed to the output.

        Args:
            df: Parent dataframe containing lineage ID column

        Returns:
            list[str]: List of unique lineage IDs from the parent dataframe

        Raises:
            ValueError: If dataframe does not contain lineage ID column

        Example:
            >>> parent_df = pl.DataFrame({
            ...     "value": [1, 2, 3],
            ...     LineageDefaults.ID_COLUMN: ["uuid1", "uuid2", "uuid3"]
            ... })
            >>> parent_ids = tracker.get_parent_lineage_ids(parent_df)
            >>> lineage_id = tracker.create_lineage_record(
            ...     context=context,
            ...     layer="gold",
            ...     parent_lineage_ids=parent_ids,
            ...     transformation_type="aggregate"
            ... )
        """
        if LineageDefaults.ID_COLUMN not in df.columns:
            raise ValueError(
                f"Parent dataframe must contain {LineageDefaults.ID_COLUMN} column. "
                "Ensure lineage tracking is enabled on upstream assets."
            )

        return df.select(LineageDefaults.ID_COLUMN).unique().to_series().to_list()

    def query_lineage_history(
        self,
        asset_name: str | None = None,
        pipeline_id: str | None = None,
        layer: str | None = None,
        source_file: str | None = None,
        data_date: date | None = None,
        is_backfill: bool | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query lineage records with optional filters.

        Args:
            asset_name: Filter by asset name
            pipeline_id: Filter by stable pipeline UUID (persists across renames)
            layer: Filter by data layer
            source_file: Filter by source file
            data_date: Filter by data date
            is_backfill: Filter by backfill status
            limit: Maximum number of records to return

        Returns:
            list[dict]: List of lineage records as dictionaries

        Example:
            >>> records = tracker.query_lineage_history(
            ...     pipeline_id="550e8400-e29b-41d4-a716-446655440000",
            ...     layer="silver",
            ...     limit=10
            ... )
        """
        query = "SELECT * FROM lineage.data_lineage WHERE 1=1"
        params: dict[str, Any] = {}

        if asset_name:
            query += " AND asset_name = :asset_name"
            params["asset_name"] = asset_name

        if pipeline_id:
            query += " AND pipeline_id = :pipeline_id"
            params["pipeline_id"] = pipeline_id

        if layer:
            query += " AND layer = :layer"
            params["layer"] = layer

        if source_file:
            query += " AND source_file = :source_file"
            params["source_file"] = source_file

        if data_date:
            query += " AND data_date = :data_date"
            params["data_date"] = data_date

        if is_backfill is not None:
            query += " AND is_backfill = :is_backfill"
            params["is_backfill"] = is_backfill

        query += " ORDER BY processed_at DESC LIMIT :limit"
        params["limit"] = limit

        with self.engine.connect() as conn:
            result = conn.execute(sa.text(query), params)
            return [dict(row._mapping) for row in result]
