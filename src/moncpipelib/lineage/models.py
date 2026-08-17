"""SQLAlchemy models for lineage tracking.

This module provides SQLAlchemy ORM models for the lineage tracking system.
The models can be used for:
- Programmatic access to lineage data with type hints
- Generating Alembic migrations (optional)
- Schema validation

The primary model is DataLineage which stores row-level lineage metadata.
Data tables contain _lineage_id and _lineage_key columns that reference
records in the lineage.data_lineage table.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, DATERANGE, JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from moncpipelib.config import REGISTRY_STATUS_REGISTERED, config

# Derive schema name from central config (overridable via env vars)
LINEAGE_SCHEMA = config.lineage.schema_name
lineage_metadata = MetaData(schema=LINEAGE_SCHEMA)


class LineageBase(DeclarativeBase):
    """Base class for lineage models."""

    metadata = lineage_metadata


class DataLineage(LineageBase):
    """Row-level lineage tracking record.

    This table stores metadata about each data load operation, including:
    - Source tracking (files, systems)
    - Temporal information (data dates, ranges)
    - Backfill operations (with history and reasons)
    - Aggregation lineage (parent tracking for many:1 relationships)
    - Transformation types

    Each record has a UUID7 primary key (time-ordered with embedded timestamp)
    and a human-readable composite backup key for recovery scenarios.

    Data tables reference this table via _lineage_id foreign key column,
    with all rows from a single load sharing the same lineage_id.

    Attributes:
        lineage_id: UUID7 primary key (time-ordered, extractable timestamp)
        lineage_key: Human-readable composite key (v1:asset:layer:date:run_id)
        run_id: Dagster run ID
        asset_name: Name of the asset being written
        layer: Data layer (bronze, silver, gold)
        source_file: Source file path or name
        source_system: External system identifier (e.g., 'sftp', 'api')
        data_date: Single date for the data (for daily partitions)
        data_date_range: Date range for multi-day data loads
        processed_at: Timestamp when the record was created
        row_count: Number of rows in the output dataset
        is_backfill: Whether this is a backfill operation
        backfill_reason: Explanation for the backfill
        backfill_id: Stable identifier of the Dagster backfill batch this
            row belongs to, sourced from ``context.run.backfill_id`` (e.g.,
            ``bf_2026_05_22_claims``). NULL for non-backfill runs.
        replaces_lineage_id: UUID of the lineage record being replaced
        parent_lineage_ids: Array of UUIDs for parent records (aggregations)
        transformation_type: Type of transformation (aggregate, join, filter, etc.)
        created_by: Database user who created the record
        metadata: Additional metadata as JSONB
    """

    __tablename__ = config.lineage.table_name

    # Primary key - UUID7 with embedded timestamp
    lineage_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        comment="UUID7 (time-ordered, extractable timestamp)",
    )

    # Composite backup key for recovery scenarios
    lineage_key: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="Composite backup key: v{ver}:{asset}:{layer}:{date}:{run_id_prefix}",
    )

    # Run identification
    run_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )

    # Asset identification
    asset_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )

    # Stable pipeline identity (persists across asset renames).
    # Migration 019 (#308) Phase 4: FK into ``pipeline_registry``. Nullable
    # because ~79% of historical rows (pre-Phase 2) have ``pipeline_id``
    # NULL and are not being backfilled (non-prod reset is the cutover).
    # The production-side ``ALTER TABLE ... ADD CONSTRAINT ... NOT VALID``
    # avoids the validation scan over those historical rows; the FK
    # still enforces on new rows. Post-reset, ``VALIDATE CONSTRAINT``
    # promotes the constraint to a validated state.
    pipeline_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{LINEAGE_SCHEMA}.pipeline_registry.pipeline_id",
            name="fk_data_lineage_pipeline_id",
        ),
        nullable=True,
        index=True,
    )

    layer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )

    # Source tracking
    source_file: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        index=True,
    )

    source_system: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Temporal tracking
    data_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    # Note: DATERANGE requires special handling - use Column for non-mapped types
    data_date_range = Column(
        DATERANGE(),
        nullable=True,
    )

    # Processing metadata
    processed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
        index=True,
    )

    row_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # Backfill tracking
    is_backfill: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("FALSE"),
        index=True,
    )

    backfill_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    backfill_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        index=True,
    )

    # Self-referential foreign key for backfill replacement chain
    replaces_lineage_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{LINEAGE_SCHEMA}.data_lineage.lineage_id"),
        nullable=True,
    )

    # Aggregation lineage - array of parent UUIDs
    # Note: ARRAY with UUID requires Column for proper handling
    parent_lineage_ids: Mapped[list[UUID] | None] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)),
        nullable=True,
    )

    # Transformation tracking
    transformation_type: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Audit tracking
    created_by: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        server_default=text("CURRENT_USER"),
    )

    # Additional metadata as JSONB
    # Note: Use Column for JSONB with proper type handling
    metadata_json = Column(
        "metadata",
        JSONB(astext_type=Text()),
        nullable=True,
    )

    # Relationship to replaced record (self-referential)
    replaced_record: Mapped[DataLineage | None] = relationship(
        "DataLineage",
        remote_side=[lineage_id],
        foreign_keys=[replaces_lineage_id],
    )

    def __repr__(self) -> str:
        """Return string representation of the lineage record."""
        return (
            f"<DataLineage(lineage_id={self.lineage_id!r}, "
            f"asset_name={self.asset_name!r}, layer={self.layer!r})>"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert lineage record to dictionary.

        Returns:
            dict: Dictionary representation of the lineage record.
        """
        return {
            "lineage_id": str(self.lineage_id),
            "lineage_key": self.lineage_key,
            "run_id": self.run_id,
            "asset_name": self.asset_name,
            "pipeline_id": str(self.pipeline_id) if self.pipeline_id else None,
            "layer": self.layer,
            "source_file": self.source_file,
            "source_system": self.source_system,
            "data_date": self.data_date.isoformat() if self.data_date else None,
            "data_date_range": str(self.data_date_range) if self.data_date_range else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "row_count": self.row_count,
            "is_backfill": self.is_backfill,
            "backfill_reason": self.backfill_reason,
            "backfill_id": self.backfill_id,
            "replaces_lineage_id": str(self.replaces_lineage_id)
            if self.replaces_lineage_id
            else None,
            "parent_lineage_ids": [str(uid) for uid in self.parent_lineage_ids]
            if self.parent_lineage_ids
            else None,
            "transformation_type": self.transformation_type,
            "created_by": self.created_by,
            "metadata": self.metadata_json,
        }


# Additional indexes for composite queries (defined outside the class)
# These match the indexes from the Alembic migration
Index(
    "ix_lineage_data_lineage_asset_layer",
    DataLineage.asset_name,
    DataLineage.layer,
)

Index(
    "ix_lineage_data_lineage_asset_date",
    DataLineage.asset_name,
    DataLineage.data_date,
)

Index(
    "ix_lineage_data_lineage_pipeline_layer",
    DataLineage.pipeline_id,
    DataLineage.layer,
)

# Migration 018 Phase 4: composite index for the ``replaces_lineage_id``
# prior-row lookup. The hot query is:
#   WHERE asset_name = ? AND layer = ?
#   ORDER BY processed_at DESC LIMIT 1
# Existing ``ix_lineage_data_lineage_asset_layer`` covers the WHERE
# clause but forces a sort on every lookup; the single-column
# ``processed_at`` index covers the sort but forces a filter on every
# row. ``(asset_name, layer, processed_at DESC)`` is the index-only
# path. Production install uses ``CREATE INDEX CONCURRENTLY`` (Phase 7
# runbook) because ``data_lineage`` is on the write hot path.
Index(
    "ix_lineage_data_lineage_asset_layer_processed",
    DataLineage.asset_name,
    DataLineage.layer,
    DataLineage.processed_at.desc(),
)


class PeriodRegistry(LineageBase):
    """Period registry for cross-code-location partition discovery.

    Stores registered data periods so downstream code locations can
    discover partition definitions without sharing YAML files. Bronze
    pipelines register periods via ``PostgresResource.register_period()``
    (or auto-registration during ``write()``), and silver pipelines
    query the registry via ``get_registry_periods()``.
    """

    __tablename__ = config.period_registry.table_name
    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "partition_key",
            name="uq_period_registry_source_partition",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    # Migration 019 (#308) Phase 1: source_id / pipeline_id migrated from
    # text to uuid for parity with data_lineage.pipeline_id and to allow
    # FK declarations into pipeline_registry (Phase 4) without ::text
    # casts. psycopg3 accepts both str (UUID-form) and uuid.UUID for uuid
    # columns, so existing callers passing string-form UUIDs continue to
    # work unchanged.
    source_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    partition_key: Mapped[str] = mapped_column(Text, nullable=False)
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text(f"'{REGISTRY_STATUS_REGISTERED}'")
    )
    registered_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    source_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    registered_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Migration 019 (#308) Phase 4: FK into ``pipeline_registry``.
    # Enforced (not NOT VALID): all 181 existing rows have non-NULL
    # ``pipeline_id`` values that will be backed by a ``pipeline_registry``
    # row once the Phase 2 upsert has fired for each pipeline. Appendix
    # A5 verifies the absence of orphans before the constraint is
    # declared.
    pipeline_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{LINEAGE_SCHEMA}.pipeline_registry.pipeline_id",
            name="fk_period_registry_pipeline_id",
        ),
        nullable=True,
        index=True,
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)


class PipelineRegistry(LineageBase):
    """Catalogue of all pipelines that have written via ``PostgresResource.write()``.

    Migration 019 (#308) Phase 2 columns: identity + Dagster join handles.
    Phase 3 will extend with contract-identity columns (``contract_hash``,
    ``schema_fingerprint``, etc.); those are deliberately omitted here so
    the join-handle surface can ship without waiting on the upstream
    hashing work in ``contracts/loader.py``.

    Auto-populated on every ``write()`` call from contract metadata +
    Dagster context. Catalogue-only -- no ``last_run_at`` / ``total_runs``
    columns; those are queries over ``data_lineage`` + Dagster runs, not
    denormalised state.

    Attributes:
        pipeline_id: Stable UUID identifying the logical pipeline. PK.
            Persists across asset renames so lineage history remains
            correlated.
        asset_name: Current Dagster asset name (``AssetKey.to_user_string()``).
        layer: Data layer (``bronze``/``silver``/``gold``) or ``None``.
        source_id: Optional reference to the data source ``source_id``
            (UUID) when the contract carries a ``data_source`` block.
        owner_team: ``contract.owner.team`` projected for fast filtering.
        description: ``contract.description`` for catalogue display.
        dagster_asset_key: ``AssetKey`` in JSON-array form (e.g.
            ``["fda_ndc_package_bronze"]``) so a direct join to
            ``dagster.public.asset_keys`` works without JSON parsing.
        dagster_job_name: Dagster job name the run is part of.
        code_location_name: Dagster code-location name.
        registered_at: First-seen timestamp (set once on first INSERT).
        updated_at: Updated on every upsert so consumers can answer
            "when did this pipeline last write" without joining
            ``data_lineage``.
    """

    __tablename__ = config.pipeline_registry.table_name

    pipeline_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        comment="Stable UUID identifying the logical pipeline; PK",
    )
    asset_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    layer: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    source_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    owner_team: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dagster_asset_key: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        index=True,
        comment="AssetKey in JSON-array form for direct join to dagster.public.asset_keys",
    )
    dagster_job_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    code_location_name: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Migration 019 (#308) Phase 3: contract-identity columns. All
    # nullable so the column add is non-blocking and rows written under
    # Phase 2 (skeleton only) do not fail validation.
    contract_hash: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="SHA256 hex digest over the contract's semantic content (Phase 3)",
    )
    schema_fingerprint: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="SHA256 hex digest over the column schema identity fields only (Phase 3)",
    )
    contract_version: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Contract YAML schema version (e.g. '1.0'); see DataContract.version",
    )
    sla_freshness_hours: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tags: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    data_classification: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Rollup of column-level PII flags; 'PHI' if any non-managed column is PII, else 'none'",
    )

    registered_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    def __repr__(self) -> str:
        return (
            f"<PipelineRegistry(pipeline_id={self.pipeline_id!r}, "
            f"asset_name={self.asset_name!r}, layer={self.layer!r})>"
        )


class ContractValidationRun(LineageBase):
    """One row per executed contract validation check per write.

    Migration 019 (#308) Phase 5: durable persistence of per-check
    ``ValidationResult`` outcomes that previously lived only in the
    in-memory ``ContractValidationSummary`` and the materialization-
    event log. Many-to-one to ``data_lineage`` via ``lineage_id``: each
    contract-carrying write produces N rows, where N is the number of
    checks executed (schema + column tests + table expectations).

    Phase 5 happy-path scope: persistence fires after the data DML on
    successful writes. Failed writes (``ContractViolationError`` raised
    under ``enforce_contracts="error"``) do **not** currently persist
    audit rows -- the raise short-circuits before the cursor block
    reaches the persist call. Promoting failed-write persistence to a
    Phase-5b refactor of ``_enforce_contract`` is tracked separately.

    Attributes:
        validation_run_id: UUID PK (server-defaulted to ``gen_random_uuid()``).
        lineage_id: FK to ``data_lineage.lineage_id`` -- the write this
            validation was performed against.
        check_name: Stable name of the check (e.g. ``"schema"``,
            ``"company_id.unique"``, ``"row_count"``).
        severity: ``"error"`` or ``"warn"``.
        passed: Whether the check passed.
        failed_count: Rows that failed (0 for passed checks).
        total_count: Rows the check ran against.
        sample_failures: Up to 20 sample failing rows for debugging,
            truncated at persist time.
        created_at: When the row was inserted.
    """

    __tablename__ = config.contract_validation_runs.table_name

    validation_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    lineage_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{LINEAGE_SCHEMA}.data_lineage.lineage_id",
            name="fk_contract_validation_runs_lineage_id",
        ),
        nullable=False,
        index=True,
    )
    check_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    severity: Mapped[str] = mapped_column(Text, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    total_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    sample_failures: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    def __repr__(self) -> str:
        return (
            f"<ContractValidationRun(validation_run_id={self.validation_run_id!r}, "
            f"lineage_id={self.lineage_id!r}, check_name={self.check_name!r}, "
            f"passed={self.passed!r})>"
        )


class Scd2Reconciliation(LineageBase):
    """Audit row per ``reconcile_scd2()`` invocation.

    Migration 019 (#308) Phase 6: durable persistence of the in-process
    return-dict metadata produced by ``PostgresResource.reconcile_scd2``.

    Standalone -- does **not** FK to ``data_lineage`` because
    ``reconcile_scd2`` is a post-write reconciliation invoked separately
    from ``database.write()`` and does not currently emit a lineage row
    (extending it to do so was rejected during planning as scope creep).
    FKs to ``pipeline_registry`` only when ``pipeline_id`` is known.

    Persisted on the same cursor as the reconcile DML, before commit,
    so the audit row is atomic with the reconciliation it describes.

    Attributes:
        reconciliation_id: UUID PK (server-defaulted to ``gen_random_uuid()``).
        run_id: Dagster run ID (or caller-supplied identifier).
        asset_name: Asset that was reconciled (e.g. ``"silver/dim_provider"``).
        pipeline_id: Optional FK to ``pipeline_registry`` -- populated when
            the caller threads it through (typically via ``contract.pipeline_id``).
            Nullable because ``reconcile_scd2`` can be called with explicit
            ``target`` + ``business_key`` and no contract.
        target_table: Fully-qualified ``schema.table`` that was reconciled.
        applied_at: Server-defaulted timestamp of the persist call.
        work_mem_applied: The per-transaction ``work_mem`` literal applied
            during the reconcile (e.g. ``"256MB"``), or ``None`` when no
            override was applied and the reconcile ran at the cluster default.
        rows_collapsed: Number of consecutive-duplicate versions removed.
        rows_timeline_updated: Number of rows updated by the timeline
            stitching UPDATE.
        rows_renumbered: Number of rows whose sequence column was
            renumbered (zero when no sequence column is configured).
        duration_seconds: Wall-clock duration of the reconcile transaction,
            measured by ``time.perf_counter`` around the cursor block. Stored
            as ``numeric(10, 3)`` to keep millisecond precision without
            float-rounding surprises in audit queries.
        metadata_: Future-proofing JSONB column for additional context
            (e.g. partition keys reconciled, contract version) without a
            DDL change.
    """

    __tablename__ = config.scd2_reconciliations.table_name

    reconciliation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    asset_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    pipeline_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{LINEAGE_SCHEMA}.pipeline_registry.pipeline_id",
            name="fk_scd2_reconciliations_pipeline_id",
        ),
        nullable=True,
        index=True,
    )
    target_table: Mapped[str] = mapped_column(Text, nullable=False)
    applied_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    work_mem_applied: Mapped[str | None] = mapped_column(Text, nullable=True)
    rows_collapsed: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    rows_timeline_updated: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    rows_renumbered: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    duration_seconds: Mapped[float | None] = mapped_column(
        Numeric(10, 3),
        nullable=True,
        comment="Wall-clock reconcile duration in seconds; numeric(10,3) for millisecond precision",
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Scd2Reconciliation(reconciliation_id={self.reconciliation_id!r}, "
            f"asset_name={self.asset_name!r}, target_table={self.target_table!r})>"
        )
