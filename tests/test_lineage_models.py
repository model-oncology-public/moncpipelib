"""SA model shape assertions for ``lineage.*`` tables.

Pins the typed shape of ``PeriodRegistry`` (and other lineage models as
they evolve) against the SQLAlchemy mapping so a future column-type
change does not silently slip past review. These tests do not require a
live database -- they introspect ``Table.c`` for column types,
nullability, and indexed-state.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Date, ForeignKey, Index, Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from moncpipelib.lineage.models import (
    ContractValidationRun,
    DataLineage,
    PeriodRegistry,
    PipelineRegistry,
    Scd2Reconciliation,
)


class TestPeriodRegistryShape:
    """Migration 019 (#308) Phase 1: ``source_id`` / ``pipeline_id`` are ``uuid``."""

    def test_source_id_is_uuid_not_nullable(self) -> None:
        col = PeriodRegistry.__table__.c.source_id
        assert isinstance(col.type, PG_UUID)
        assert col.nullable is False

    def test_pipeline_id_is_uuid_nullable(self) -> None:
        col = PeriodRegistry.__table__.c.pipeline_id
        assert isinstance(col.type, PG_UUID)
        assert col.nullable is True

    def test_pipeline_id_indexed(self) -> None:
        """Phase 1 adds an index on ``pipeline_id`` -- enables the Phase 4
        FK lookup from ``period_registry`` into ``pipeline_registry``
        without a full scan."""
        col = PeriodRegistry.__table__.c.pipeline_id
        assert col.index is True, (
            "PeriodRegistry.pipeline_id should be indexed once Phase 1 lands. "
            "Phase 4 FK validation queries key on this column."
        )

    def test_id_primary_key_unchanged(self) -> None:
        """Sanity check that the surrogate PK column type wasn't accidentally touched."""
        col = PeriodRegistry.__table__.c.id
        assert isinstance(col.type, PG_UUID)
        assert col.primary_key is True

    def test_unique_constraint_unchanged(self) -> None:
        """``(source_id, partition_key)`` uniqueness is the on-conflict key for
        ``_upsert_registry_row``; the type change must not drop the constraint."""
        uniques = [
            c
            for c in PeriodRegistry.__table__.constraints
            if c.__class__.__name__ == "UniqueConstraint"
        ]
        names = {c.name for c in uniques}
        assert "uq_period_registry_source_partition" in names

    def test_non_id_columns_keep_their_types(self) -> None:
        """Regression: the type change targets two columns only. The rest
        of the column types must remain stable so the SA mapping stays in
        lockstep with the production DDL."""
        cols = PeriodRegistry.__table__.c
        assert isinstance(cols.partition_key.type, Text)
        assert isinstance(cols.effective_from.type, Date)
        assert isinstance(cols.effective_to.type, Date)
        assert isinstance(cols.source_uri.type, Text)
        assert isinstance(cols.status.type, Text)
        assert isinstance(cols.registered_at.type, TIMESTAMP)
        assert isinstance(cols.source_name.type, Text)
        assert isinstance(cols.registered_by.type, Text)
        assert isinstance(cols.run_id.type, Text)
        # JSONB column is mapped under the "metadata" SQL name but
        # exposed as ``metadata_`` on the model.
        assert isinstance(cols.metadata.type, JSONB)


class TestDataLineageShape:
    """Sanity checks for adjacent columns the Phase 4 FK will reference.

    Not changed by Phase 1, but pinned here so the model_shape file is
    the single source-of-truth for "what does ``lineage.*`` look like in
    the SA mapping".
    """

    def test_pipeline_id_is_uuid_nullable(self) -> None:
        col = DataLineage.__table__.c.pipeline_id
        assert isinstance(col.type, PG_UUID)
        assert col.nullable is True

    def test_backfill_id_present(self) -> None:
        """Migration 018 Phase 2 added this column. Pin it to catch
        accidental removal."""
        col = DataLineage.__table__.c.backfill_id
        assert isinstance(col.type, Text)
        assert col.nullable is True

    def test_is_backfill_present_and_not_null(self) -> None:
        col = DataLineage.__table__.c.is_backfill
        assert isinstance(col.type, Boolean)
        assert col.nullable is False

    def test_row_count_is_integer(self) -> None:
        col = DataLineage.__table__.c.row_count
        assert isinstance(col.type, Integer)


class TestPeriodRegistryCreateAllShape:
    """``Base.metadata.create_all()`` produces the expected column SQL types.

    Doesn't connect to a real DB -- uses the SQLAlchemy DDL compiler to
    render the ``CREATE TABLE`` for the ``postgresql`` dialect and
    asserts the column types appear as expected.
    """

    def test_period_registry_ddl_uses_uuid_for_source_and_pipeline(self) -> None:
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        compiled = str(CreateTable(PeriodRegistry.__table__).compile(dialect=postgresql.dialect()))
        # source_id should be NOT NULL uuid
        assert "source_id UUID NOT NULL" in compiled or "source_id UUID  NOT NULL" in compiled, (
            f"Expected UUID NOT NULL for source_id, got:\n{compiled}"
        )
        # pipeline_id should be nullable uuid (no NOT NULL)
        assert "pipeline_id UUID" in compiled, f"Expected UUID for pipeline_id, got:\n{compiled}"
        # Sanity: should NOT contain text-typed source_id or pipeline_id
        assert "source_id TEXT" not in compiled
        assert "pipeline_id TEXT" not in compiled


class TestPipelineRegistryShape:
    """Migration 019 (#308) Phase 2: ``PipelineRegistry`` skeleton + Dagster handles."""

    def test_pipeline_id_is_uuid_primary_key(self) -> None:
        col = PipelineRegistry.__table__.c.pipeline_id
        assert isinstance(col.type, PG_UUID)
        assert col.primary_key is True

    def test_asset_name_required_and_indexed(self) -> None:
        col = PipelineRegistry.__table__.c.asset_name
        assert isinstance(col.type, Text)
        assert col.nullable is False
        assert col.index is True

    def test_layer_nullable_and_indexed(self) -> None:
        col = PipelineRegistry.__table__.c.layer
        assert isinstance(col.type, Text)
        assert col.nullable is True
        assert col.index is True

    def test_source_id_is_nullable_uuid(self) -> None:
        col = PipelineRegistry.__table__.c.source_id
        assert isinstance(col.type, PG_UUID)
        assert col.nullable is True

    def test_dagster_asset_key_indexed(self) -> None:
        """The whole point of this column is the join to
        ``dagster.public.asset_keys`` -- needs an index."""
        col = PipelineRegistry.__table__.c.dagster_asset_key
        assert isinstance(col.type, Text)
        assert col.nullable is True
        assert col.index is True

    def test_dagster_handle_columns_present(self) -> None:
        cols = PipelineRegistry.__table__.c
        assert isinstance(cols.dagster_job_name.type, Text)
        assert cols.dagster_job_name.nullable is True
        assert isinstance(cols.code_location_name.type, Text)
        assert cols.code_location_name.nullable is True

    def test_owner_and_description_present(self) -> None:
        cols = PipelineRegistry.__table__.c
        assert isinstance(cols.owner_team.type, Text)
        assert cols.owner_team.nullable is True
        assert isinstance(cols.description.type, Text)
        assert cols.description.nullable is True

    def test_registered_at_and_updated_at_default_now(self) -> None:
        """Both timestamp columns have ``NOW()`` server defaults so
        rows hand-inserted at the SQL layer don't need to specify them."""
        from sqlalchemy.dialects.postgresql import TIMESTAMP as PG_TIMESTAMP

        for col_name in ("registered_at", "updated_at"):
            col = PipelineRegistry.__table__.c[col_name]
            assert isinstance(col.type, PG_TIMESTAMP)
            assert col.nullable is False
            # server_default is a TextClause; coerce to str for the check.
            assert col.server_default is not None
            assert "NOW()" in str(col.server_default.arg)  # type: ignore[union-attr]

    def test_phase3_contract_identity_columns_present(self) -> None:
        """Migration 019 (#308) Phase 3 columns: contract content
        fingerprints, version, SLA freshness, tags, classification."""
        cols = PipelineRegistry.__table__.c
        assert isinstance(cols.contract_hash.type, Text)
        assert cols.contract_hash.nullable is True
        assert isinstance(cols.schema_fingerprint.type, Text)
        assert cols.schema_fingerprint.nullable is True
        assert isinstance(cols.contract_version.type, Text)
        assert cols.contract_version.nullable is True
        assert isinstance(cols.sla_freshness_hours.type, Integer)
        assert cols.sla_freshness_hours.nullable is True
        assert isinstance(cols.tags.type, JSONB)
        assert cols.tags.nullable is True
        assert isinstance(cols.data_classification.type, Text)
        assert cols.data_classification.nullable is True

    def test_no_operational_state_columns(self) -> None:
        """Pinned design decision: ``pipeline_registry`` is catalogue,
        not state. ``last_run_at`` / ``total_runs`` / ``last_status``
        are queries over ``data_lineage`` + Dagster runs, not denormalised
        columns. Regression-guard so a future "convenience" addition
        doesn't slip in unreviewed."""
        forbidden = {"last_run_id", "last_run_at", "last_status", "total_runs"}
        present = set(PipelineRegistry.__table__.c.keys()) & forbidden
        assert not present, (
            f"PipelineRegistry must not carry operational-state columns; "
            f"found: {present}. See migration 019 plan, pinned design decision "
            f"'Catalogue, not state.'"
        )


class TestPipelineRegistryCreateAllShape:
    """``CreateTable`` rendered DDL for ``PipelineRegistry``.

    The Alembic-autogenerate path in data-platform compiles the same
    DDL; this pins it so a typo / accidental column-type drift in the
    SA mapping is caught at unit-test time.
    """

    def test_ddl_includes_pipeline_id_uuid_pk(self) -> None:
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        compiled = str(
            CreateTable(PipelineRegistry.__table__).compile(dialect=postgresql.dialect())
        )
        assert "pipeline_id UUID NOT NULL" in compiled
        assert "PRIMARY KEY (pipeline_id)" in compiled

    def test_ddl_includes_dagster_handles(self) -> None:
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        compiled = str(
            CreateTable(PipelineRegistry.__table__).compile(dialect=postgresql.dialect())
        )
        assert "dagster_asset_key TEXT" in compiled
        assert "dagster_job_name TEXT" in compiled
        assert "code_location_name TEXT" in compiled

    def test_ddl_includes_phase3_columns(self) -> None:
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        compiled = str(
            CreateTable(PipelineRegistry.__table__).compile(dialect=postgresql.dialect())
        )
        assert "contract_hash TEXT" in compiled
        assert "schema_fingerprint TEXT" in compiled
        assert "contract_version TEXT" in compiled
        assert "sla_freshness_hours INTEGER" in compiled
        assert "tags JSONB" in compiled
        assert "data_classification TEXT" in compiled


class TestPhase4ForeignKeyDeclarations:
    """Migration 019 (#308) Phase 4: FK declarations from
    ``data_lineage`` and ``period_registry`` into ``pipeline_registry``.

    The constraint names are pinned because they're referenced in the
    Phase 7 production runbook (``ALTER TABLE ... VALIDATE CONSTRAINT
    fk_data_lineage_pipeline_id``).
    """

    def _foreign_keys(self, column: Any) -> list[ForeignKey]:
        return list(column.foreign_keys)

    def test_data_lineage_pipeline_id_fk_to_pipeline_registry(self) -> None:
        col = DataLineage.__table__.c.pipeline_id
        fks = self._foreign_keys(col)
        assert len(fks) == 1, f"expected exactly one FK on data_lineage.pipeline_id, found {fks}"
        fk = fks[0]
        # Target column is ``pipeline_registry.pipeline_id``
        assert fk.column.table.name == "pipeline_registry"
        assert fk.column.name == "pipeline_id"
        # Constraint name matches Phase 7 runbook
        assert fk.name == "fk_data_lineage_pipeline_id"

    def test_period_registry_pipeline_id_fk_to_pipeline_registry(self) -> None:
        col = PeriodRegistry.__table__.c.pipeline_id
        fks = self._foreign_keys(col)
        assert len(fks) == 1, f"expected exactly one FK on period_registry.pipeline_id, found {fks}"
        fk = fks[0]
        assert fk.column.table.name == "pipeline_registry"
        assert fk.column.name == "pipeline_id"
        assert fk.name == "fk_period_registry_pipeline_id"

    def test_data_lineage_pipeline_id_remains_nullable(self) -> None:
        """The FK is nullable because ~79% of historical rows are NULL
        and are not being backfilled (non-prod reset is the cutover)."""
        col = DataLineage.__table__.c.pipeline_id
        assert col.nullable is True

    def test_data_lineage_pipeline_id_remains_indexed(self) -> None:
        """FK columns should be indexed for efficient validation /
        cascade traversal. Phase 4 must not regress the Phase 2 index."""
        col = DataLineage.__table__.c.pipeline_id
        assert col.index is True

    def test_period_registry_pipeline_id_remains_indexed(self) -> None:
        col = PeriodRegistry.__table__.c.pipeline_id
        assert col.index is True

    def test_no_other_foreign_keys_added_accidentally(self) -> None:
        """Regression guard: Phase 4 only adds the two FKs above.
        Any further FK additions on ``data_lineage`` or ``period_registry``
        must come through their own migration phase."""
        # data_lineage: existing self-FK from replaces_lineage_id +
        # new pipeline_id FK = exactly 2.
        all_fks_dl = list(DataLineage.__table__.foreign_keys)
        assert len(all_fks_dl) == 2, (
            f"data_lineage FK count drifted: {[fk.name for fk in all_fks_dl]}"
        )
        # period_registry: only the new pipeline_id FK.
        all_fks_pr = list(PeriodRegistry.__table__.foreign_keys)
        assert len(all_fks_pr) == 1, (
            f"period_registry FK count drifted: {[fk.name for fk in all_fks_pr]}"
        )


class TestPhase4CreateAllShape:
    """``CreateTable`` rendered DDL includes the Phase 4 FK declarations."""

    def test_data_lineage_ddl_includes_pipeline_registry_fk(self) -> None:
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        compiled = str(CreateTable(DataLineage.__table__).compile(dialect=postgresql.dialect()))
        assert "CONSTRAINT fk_data_lineage_pipeline_id" in compiled
        assert "FOREIGN KEY(pipeline_id) REFERENCES lineage.pipeline_registry" in compiled

    def test_period_registry_ddl_includes_pipeline_registry_fk(self) -> None:
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        compiled = str(CreateTable(PeriodRegistry.__table__).compile(dialect=postgresql.dialect()))
        assert "CONSTRAINT fk_period_registry_pipeline_id" in compiled
        assert "FOREIGN KEY(pipeline_id) REFERENCES lineage.pipeline_registry" in compiled


class TestContractValidationRunShape:
    """Migration 019 (#308) Phase 5: ``ContractValidationRun`` SA model shape."""

    def test_validation_run_id_uuid_primary_key(self) -> None:
        col = ContractValidationRun.__table__.c.validation_run_id
        assert isinstance(col.type, PG_UUID)
        assert col.primary_key is True

    def test_lineage_id_required_indexed_and_fk_to_data_lineage(self) -> None:
        col = ContractValidationRun.__table__.c.lineage_id
        assert isinstance(col.type, PG_UUID)
        assert col.nullable is False
        assert col.index is True
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        fk = fks[0]
        assert fk.column.table.name == "data_lineage"
        assert fk.column.name == "lineage_id"
        assert fk.name == "fk_contract_validation_runs_lineage_id"

    def test_check_name_required_and_indexed(self) -> None:
        col = ContractValidationRun.__table__.c.check_name
        assert isinstance(col.type, Text)
        assert col.nullable is False
        assert col.index is True

    def test_severity_required_text(self) -> None:
        col = ContractValidationRun.__table__.c.severity
        assert isinstance(col.type, Text)
        assert col.nullable is False

    def test_passed_required_boolean(self) -> None:
        col = ContractValidationRun.__table__.c.passed
        assert isinstance(col.type, Boolean)
        assert col.nullable is False

    def test_count_columns_default_zero(self) -> None:
        for col_name in ("failed_count", "total_count"):
            col = ContractValidationRun.__table__.c[col_name]
            assert isinstance(col.type, Integer)
            assert col.nullable is False
            assert col.server_default is not None
            assert "0" in str(col.server_default.arg)  # type: ignore[union-attr]

    def test_sample_failures_nullable_jsonb(self) -> None:
        col = ContractValidationRun.__table__.c.sample_failures
        assert isinstance(col.type, JSONB)
        assert col.nullable is True

    def test_created_at_default_now(self) -> None:
        col = ContractValidationRun.__table__.c.created_at
        assert col.nullable is False
        assert col.server_default is not None
        assert "NOW()" in str(col.server_default.arg)  # type: ignore[union-attr]


class TestContractValidationRunCreateAllShape:
    """``CreateTable`` rendered DDL for ``ContractValidationRun``."""

    def test_ddl_includes_pk_and_fk(self) -> None:
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        compiled = str(
            CreateTable(ContractValidationRun.__table__).compile(dialect=postgresql.dialect())
        )
        assert "validation_run_id UUID DEFAULT gen_random_uuid() NOT NULL" in compiled
        assert "PRIMARY KEY (validation_run_id)" in compiled
        assert "CONSTRAINT fk_contract_validation_runs_lineage_id" in compiled
        assert "FOREIGN KEY(lineage_id) REFERENCES lineage.data_lineage" in compiled
        assert "sample_failures JSONB" in compiled


class TestScd2ReconciliationShape:
    """Migration 019 (#308) Phase 6: ``Scd2Reconciliation`` SA model shape.

    Standalone audit table: no FK to ``data_lineage`` (``reconcile_scd2``
    does not emit a lineage row), but an optional FK to
    ``pipeline_registry`` for catalogue correlation when the caller
    threads ``pipeline_id`` through.
    """

    def test_reconciliation_id_uuid_primary_key(self) -> None:
        col = Scd2Reconciliation.__table__.c.reconciliation_id
        assert isinstance(col.type, PG_UUID)
        assert col.primary_key is True
        assert col.server_default is not None
        assert "gen_random_uuid()" in str(col.server_default.arg)  # type: ignore[union-attr]

    def test_run_id_required_and_indexed(self) -> None:
        col = Scd2Reconciliation.__table__.c.run_id
        assert isinstance(col.type, Text)
        assert col.nullable is False
        assert col.index is True

    def test_asset_name_required_and_indexed(self) -> None:
        col = Scd2Reconciliation.__table__.c.asset_name
        assert isinstance(col.type, Text)
        assert col.nullable is False
        assert col.index is True

    def test_pipeline_id_nullable_uuid_with_fk_to_pipeline_registry(self) -> None:
        col = Scd2Reconciliation.__table__.c.pipeline_id
        assert isinstance(col.type, PG_UUID)
        assert col.nullable is True
        assert col.index is True
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        fk = fks[0]
        assert fk.column.table.name == "pipeline_registry"
        assert fk.column.name == "pipeline_id"
        assert fk.name == "fk_scd2_reconciliations_pipeline_id"

    def test_no_fk_to_data_lineage(self) -> None:
        """Plan-pinned: standalone audit table, NO FK to ``data_lineage``
        because ``reconcile_scd2`` does not emit a lineage row.
        Regression guard against a future "let's link them" PR."""
        all_fks = list(Scd2Reconciliation.__table__.foreign_keys)
        # Only the pipeline_registry FK declared in this model.
        assert len(all_fks) == 1
        assert all_fks[0].column.table.name == "pipeline_registry"

    def test_target_table_required(self) -> None:
        col = Scd2Reconciliation.__table__.c.target_table
        assert isinstance(col.type, Text)
        assert col.nullable is False

    def test_applied_at_default_now(self) -> None:
        col = Scd2Reconciliation.__table__.c.applied_at
        assert col.nullable is False
        assert col.server_default is not None
        assert "NOW()" in str(col.server_default.arg)  # type: ignore[union-attr]

    def test_work_mem_applied_nullable_text(self) -> None:
        col = Scd2Reconciliation.__table__.c.work_mem_applied
        assert isinstance(col.type, Text)
        assert col.nullable is True

    def test_row_count_columns_default_zero(self) -> None:
        for col_name in ("rows_collapsed", "rows_timeline_updated", "rows_renumbered"):
            col = Scd2Reconciliation.__table__.c[col_name]
            assert isinstance(col.type, Integer)
            assert col.nullable is False
            assert col.server_default is not None
            assert "0" in str(col.server_default.arg)  # type: ignore[union-attr]

    def test_duration_seconds_numeric_10_3(self) -> None:
        col = Scd2Reconciliation.__table__.c.duration_seconds
        assert isinstance(col.type, Numeric)
        assert col.type.precision == 10
        assert col.type.scale == 3
        assert col.nullable is True

    def test_metadata_jsonb_nullable(self) -> None:
        col = Scd2Reconciliation.__table__.c.metadata
        assert isinstance(col.type, JSONB)
        assert col.nullable is True


class TestScd2ReconciliationCreateAllShape:
    """``CreateTable`` rendered DDL pins the column types and FK shape."""

    def test_ddl_includes_pk_and_pipeline_registry_fk(self) -> None:
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        compiled = str(
            CreateTable(Scd2Reconciliation.__table__).compile(dialect=postgresql.dialect())
        )
        assert "reconciliation_id UUID DEFAULT gen_random_uuid() NOT NULL" in compiled
        assert "PRIMARY KEY (reconciliation_id)" in compiled
        assert "CONSTRAINT fk_scd2_reconciliations_pipeline_id" in compiled
        assert "FOREIGN KEY(pipeline_id) REFERENCES lineage.pipeline_registry" in compiled
        assert "duration_seconds NUMERIC(10, 3)" in compiled


# ``Index`` re-imported above only so the module imports cleanly; the
# Phase 4 plan will add follow-up index assertions that rely on it.
_ = Index
