"""Period + pipeline/audit registry helpers extracted from PostgresResource.

These are the function bodies of the resource's registry methods.  The
PostgresResource methods of the same names (``_check_period_registry``,
``get_registry_periods``, ``register_period``, ``_upsert_registry_row``,
``update_period_metadata``, plus the four migration-019 audit-table
methods: ``_check_pipeline_registry``, ``_check_contract_validation_runs``,
``_check_scd2_reconciliations``, ``_pipeline_registry_upsert``,
``_pipeline_registry_row_matches``, ``_pipeline_registry_upsert_committed``)
remain on the resource as thin wrappers.

Two helpers (``upsert_registry_row`` and ``pipeline_registry_row_matches``)
are pure functions that take only a cursor and SQL parameters.  The rest
take the resource as their first argument, typed against
:class:`_RegistryResourceProtocol` (declared below) so the wrapper-to-helper
hop is type-checked end-to-end.  Threading the resource preserves the
``patch.object(PostgresResource, ...)`` test-patch pattern: when the helper
calls ``resource._check_pipeline_registry(...)``, the patched method on
the test fixture's resource takes effect.

The four cache flags (``_period_registry_available``, etc.) remain
``PrivateAttr`` declarations on ``PostgresResource``.  The
``_check_table_exists`` probe utility introduced in Phase 1 Move C also
remains on the resource for the same reason: it reads / writes those
caches via ``getattr`` / ``setattr`` and is small enough that decoupling
it would add wrapper noise without a meaningful payoff.
"""

from __future__ import annotations

import json as _json
import logging
from contextlib import AbstractContextManager
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, Protocol

import psycopg

if TYPE_CHECKING:
    from moncpipelib.contracts.models import DataContract
    from moncpipelib.resources.types import WriteContext


class _RegistryResourceProtocol(Protocol):
    """Subset of :class:`PostgresResource` the registry helpers reach into.

    Replaces the prior ``resource: Any`` threading so the call sites in this
    module are type-checked against the real resource interface.  If
    ``PostgresResource`` renames or changes the signature of any method
    listed here, mypy will flag the mismatch on the wrapper-to-helper hop
    (because ``self`` -- typed as ``PostgresResource`` -- no longer
    structurally satisfies this Protocol).

    Lives in this module rather than ``resources/types.py`` so each helper
    module declares exactly what it needs (interface segregation); the
    three-method overlap with :class:`_SCD2ResourceProtocol`
    (``_check_period_registry``, ``_upsert_registry_row``,
    ``get_connection_raw``) is accepted as the cost of keeping the two
    helper modules independent.
    """

    # Lineage-table availability probe (shared by all four _check_* wrappers).
    def _check_table_exists(
        self,
        cursor: psycopg.Cursor,
        *,
        schema: str,
        table: str,
        cache_attr: str,
        wctx: WriteContext | None = ...,
        logger: logging.Logger | None = ...,
        log_level: Literal["warning", "debug"] = ...,
        log_message: str,
    ) -> bool: ...

    # Other resource wrappers the helpers call back into.
    def _check_period_registry(
        self,
        cursor: psycopg.Cursor,
        wctx: WriteContext | None = ...,
        *,
        logger: logging.Logger | None = ...,
    ) -> bool: ...

    def _check_pipeline_registry(
        self,
        cursor: psycopg.Cursor,
        wctx: WriteContext | None = ...,
    ) -> bool: ...

    def _pipeline_registry_row_matches(
        self,
        cursor: psycopg.Cursor,
        loaded_contract: DataContract,
        wctx: WriteContext,
    ) -> bool: ...

    def _pipeline_registry_upsert(
        self,
        cursor: psycopg.Cursor,
        *,
        loaded_contract: DataContract | None,
        wctx: WriteContext,
        layer: str | None,
    ) -> None: ...

    # Connection lifecycle.
    def get_connection_raw(self) -> psycopg.Connection: ...

    def get_connection(self) -> AbstractContextManager[psycopg.Connection]: ...


# ---------------------------------------------------------------------------
# Period registry
# ---------------------------------------------------------------------------


def check_period_registry(
    resource: _RegistryResourceProtocol,
    cursor: psycopg.Cursor,
    wctx: WriteContext | None = None,
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Check if the period registry table exists (cached after first call).

    Returns ``True`` if the table is available. Emits a one-time warning
    if the table does not exist.

    Args:
        resource: ``PostgresResource`` instance owning the cache flag.
        cursor: Database cursor.
        wctx: Optional write context for logging (write-path usage).
        logger: Optional stdlib logger for logging (read-path usage).
            At least one of ``wctx`` or ``logger`` should be provided
            for warning messages.
    """
    from moncpipelib.config import config

    registry_schema = config.period_registry.schema_name
    registry_table = config.period_registry.table_name

    return resource._check_table_exists(
        cursor,
        schema=registry_schema,
        table=registry_table,
        cache_attr="_period_registry_available",
        wctx=wctx,
        logger=logger,
        log_level="warning",
        log_message=(
            f"Period registry table {registry_schema}.{registry_table} "
            f"does not exist. Period registration skipped. "
            f"Create the table to enable cross-location period discovery."
        ),
    )


def get_registry_periods(
    resource: _RegistryResourceProtocol,
    source_id: str,
    status: str | None = "materialized",
) -> list[dict[str, Any]]:
    """Query the period registry for a source's periods.

    Args:
        resource: ``PostgresResource`` instance.
        source_id: Data source identifier.
        status: Filter by status. ``None`` returns all statuses.
            Defaults to ``"materialized"``.

    Returns:
        List of dicts with ``partition_key``, ``effective_from``,
        ``effective_to``, ``source_uri``, ``status``, ``registered_by``,
        and ``registered_at`` keys. Empty list if the registry table
        does not exist.
    """
    from moncpipelib.config import config

    _log = logging.getLogger("moncpipelib.resources")
    registry_schema = config.period_registry.schema_name
    registry_table = config.period_registry.table_name

    conn = resource.get_connection_raw()
    try:
        with conn.cursor() as cursor:
            if not resource._check_period_registry(cursor, logger=_log):
                return []

            if status is not None:
                sql = (
                    f"SELECT partition_key, effective_from, effective_to, "
                    f"source_uri, status, source_name, registered_by, "
                    f"registered_at, run_id, pipeline_id, metadata "
                    f"FROM {registry_schema}.{registry_table} "
                    f"WHERE source_id = %s AND status = %s "
                    f"ORDER BY effective_from"
                )
                cursor.execute(sql, (source_id, status))
            else:
                sql = (
                    f"SELECT partition_key, effective_from, effective_to, "
                    f"source_uri, status, source_name, registered_by, "
                    f"registered_at, run_id, pipeline_id, metadata "
                    f"FROM {registry_schema}.{registry_table} "
                    f"WHERE source_id = %s "
                    f"ORDER BY effective_from"
                )
                cursor.execute(sql, (source_id,))

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        conn.close()


def register_period(
    resource: _RegistryResourceProtocol,
    source_id: str,
    partition_key: str,
    effective_from: date,
    effective_to: date | None = None,
    source_uri: str | None = None,
    status: str = "materialized",
    source_name: str | None = None,
    registered_by: str | None = None,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a period in the period registry.

    Uses ``INSERT ... ON CONFLICT (source_id, partition_key) DO UPDATE``
    to upsert the period. This allows re-registration (e.g., updating
    status from ``'registered'`` to ``'materialized'``) without failing.

    The caller is responsible for ensuring the registry table exists.
    Use this for explicit, standalone registration outside of a write path.
    """
    conn = resource.get_connection_raw()
    try:
        with conn.cursor() as cursor:
            upsert_registry_row(
                cursor,
                source_id=source_id,
                source_name=source_name,
                partition_key=partition_key,
                effective_from=effective_from,
                effective_to=effective_to,
                source_uri=source_uri,
                status=status,
                registered_by=registered_by,
                run_id=run_id,
                pipeline_id=pipeline_id,
                metadata=metadata,
            )
            conn.commit()
    finally:
        conn.close()


def upsert_registry_row(
    cursor: Any,
    *,
    source_id: str,
    source_name: str | None,
    partition_key: str,
    effective_from: date,
    effective_to: date | None,
    source_uri: str | None,
    status: str,
    registered_by: str | None,
    run_id: str | None,
    pipeline_id: str | None,
    metadata: dict[str, Any] | None,
) -> None:
    """Execute the period registry upsert on the given cursor.

    Single source of truth for the ``INSERT ... ON CONFLICT (source_id,
    partition_key) DO UPDATE`` shape. Used by both ``register_period``
    (own connection) and ``_auto_register_period`` (write transaction).

    The caller is responsible for connection lifecycle and commit/rollback;
    this helper executes the SQL on the cursor it is given.

    In test-mode lineage isolation (#420) the upsert is a logged no-op.
    This is the single chokepoint for period-registry INSERT/UPSERT, so the
    gate covers the write path's auto-registration, standalone
    ``register_period`` calls, and sensor-driven registration alike.
    """
    from moncpipelib.config import SKIP_LINEAGE_WRITES_ENV, config, skip_lineage_writes

    if skip_lineage_writes():
        logging.getLogger("moncpipelib.resources").warning(
            "%s is set: skipping period_registry upsert for source_id=%s "
            "partition_key=%s status=%s. Test/ephemeral isolation only (#420).",
            SKIP_LINEAGE_WRITES_ENV,
            source_id,
            partition_key,
            status,
        )
        return

    registry_schema = config.period_registry.schema_name
    registry_table = config.period_registry.table_name

    sql = f"""
        INSERT INTO {registry_schema}.{registry_table}
            (source_id, source_name, partition_key, effective_from, effective_to,
             source_uri, status, registered_by, run_id, pipeline_id, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_id, partition_key) DO UPDATE SET
            status = EXCLUDED.status,
            source_uri = EXCLUDED.source_uri,
            source_name = EXCLUDED.source_name,
            effective_from = EXCLUDED.effective_from,
            effective_to = EXCLUDED.effective_to,
            registered_at = NOW(),
            registered_by = EXCLUDED.registered_by,
            run_id = EXCLUDED.run_id,
            pipeline_id = EXCLUDED.pipeline_id,
            metadata = COALESCE({registry_schema}.{registry_table}.metadata, '{{}}'::jsonb)
                       || COALESCE(EXCLUDED.metadata, '{{}}'::jsonb)
    """

    metadata_json = _json.dumps(metadata) if metadata is not None else None

    cursor.execute(
        sql,
        (
            source_id,
            source_name,
            partition_key,
            effective_from,
            effective_to,
            source_uri,
            status,
            registered_by,
            run_id,
            pipeline_id,
            metadata_json,
        ),
    )


def update_period_metadata(
    resource: _RegistryResourceProtocol,
    source_id: str,
    partition_key: str,
    metadata_updates: dict[str, Any],
) -> None:
    """Merge keys into an existing period registry row's metadata JSONB.

    Uses PostgreSQL's ``||`` operator to merge new keys into existing
    metadata without overwriting other keys.

    In test-mode lineage isolation (#420) the update is a logged no-op,
    covering the reconcile job's ``reconciled_at`` stamping and any other
    metadata merges an ephemeral run would otherwise apply to the real
    registry.
    """
    from moncpipelib.config import SKIP_LINEAGE_WRITES_ENV, config, skip_lineage_writes

    if skip_lineage_writes():
        logging.getLogger("moncpipelib.resources").warning(
            "%s is set: skipping period_registry metadata update for "
            "source_id=%s partition_key=%s (keys=%s). Test/ephemeral "
            "isolation only (#420).",
            SKIP_LINEAGE_WRITES_ENV,
            source_id,
            partition_key,
            sorted(metadata_updates),
        )
        return

    registry_schema = config.period_registry.schema_name
    registry_table = config.period_registry.table_name

    sql = (  # noqa: S608
        f"UPDATE {registry_schema}.{registry_table} "
        f"SET metadata = COALESCE(metadata, '{{}}'::jsonb) || %s::jsonb "
        f"WHERE source_id = %s AND partition_key = %s"
    )

    conn = resource.get_connection_raw()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (_json.dumps(metadata_updates), source_id, partition_key))
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pipeline registry + migration 019 audit tables
# ---------------------------------------------------------------------------


def check_pipeline_registry(
    resource: _RegistryResourceProtocol,
    cursor: psycopg.Cursor,
    wctx: WriteContext | None = None,
) -> bool:
    """Check if ``lineage.pipeline_registry`` exists (cached after first call).

    The pipeline-registry upsert is a no-op until the data-platform
    Alembic migration lands; this check makes the no-op silent and
    self-cancelling rather than a per-write SQL error.
    """
    from moncpipelib.config import config

    registry_schema = config.pipeline_registry.schema_name
    registry_table = config.pipeline_registry.table_name

    return resource._check_table_exists(
        cursor,
        schema=registry_schema,
        table=registry_table,
        cache_attr="_pipeline_registry_available",
        wctx=wctx,
        log_level="debug",
        log_message=(
            f"Pipeline registry table {registry_schema}.{registry_table} "
            f"does not exist yet; skipping upsert. "
            f"Will be enabled once the data-platform Alembic migration lands."
        ),
    )


def check_contract_validation_runs(
    resource: _RegistryResourceProtocol,
    cursor: psycopg.Cursor,
    wctx: WriteContext | None = None,
) -> bool:
    """Check if ``lineage.contract_validation_runs`` exists (cached).

    Migration 019 (#308) Phase 5: same silent-no-op pattern as
    :func:`check_pipeline_registry`. Until the data-platform Alembic
    migration applies the table, persistence is skipped without raising.
    """
    from moncpipelib.config import config

    schema = config.contract_validation_runs.schema_name
    table = config.contract_validation_runs.table_name

    return resource._check_table_exists(
        cursor,
        schema=schema,
        table=table,
        cache_attr="_contract_validation_runs_available",
        wctx=wctx,
        log_level="debug",
        log_message=(
            f"contract_validation_runs table {schema}.{table} does not exist yet; "
            f"per-check persistence skipped. Will engage automatically once "
            f"the data-platform Alembic migration applies."
        ),
    )


def check_scd2_reconciliations(
    resource: _RegistryResourceProtocol,
    cursor: psycopg.Cursor,
    logger: logging.Logger | None = None,
) -> bool:
    """Check if ``lineage.scd2_reconciliations`` exists (cached).

    Migration 019 (#308) Phase 6: same silent-no-op pattern as
    :func:`check_pipeline_registry` / :func:`check_contract_validation_runs`.
    """
    from moncpipelib.config import config

    schema = config.scd2_reconciliations.schema_name
    table = config.scd2_reconciliations.table_name

    return resource._check_table_exists(
        cursor,
        schema=schema,
        table=table,
        cache_attr="_scd2_reconciliations_available",
        logger=logger,
        log_level="debug",
        log_message=(
            f"scd2_reconciliations table {schema}.{table} does not exist yet; "
            f"audit-row persistence skipped. Will engage automatically "
            f"once the data-platform Alembic migration applies."
        ),
    )


def pipeline_registry_upsert(
    resource: _RegistryResourceProtocol,
    cursor: psycopg.Cursor,
    *,
    loaded_contract: DataContract | None,
    wctx: WriteContext,
    layer: str | None,
) -> None:
    """Upsert ``lineage.pipeline_registry`` from contract + Dagster context.

    Migration 019 (#308) Phase 2 + Phase 3: identity, Dagster join
    handles, and contract-identity columns (``contract_hash``,
    ``schema_fingerprint``, ``contract_version``, ``sla_freshness_hours``,
    ``tags``, ``data_classification``).

    Same-cursor SQL helper. Issue #332: production callers no longer
    invoke this method on the data-write cursor -- it would hold the
    registry row's exclusive lock for the entire silver write and
    serialize concurrent same-``pipeline_id`` partitions. Production
    callers go through :func:`pipeline_registry_upsert_committed`,
    which opens its own short-lived autocommit connection so the row
    lock is released before the data-write transaction opens.

    This helper remains the single source of truth for the
    ``INSERT ... ON CONFLICT`` SQL shape and is exercised both
    through the committed wrapper and directly by unit tests with a
    mock cursor.

    Idempotent: re-upserting the same ``pipeline_id`` bumps
    ``updated_at`` and refreshes every projected column in case the
    contract content changed (caught by a different ``contract_hash``).

    Failure mode: no-op if the table does not exist yet; a logged
    warning + re-raise if any other SQL error occurs. The caller owns
    commit / rollback semantics.
    """
    if loaded_contract is None or not loaded_contract.pipeline_id:
        return

    if not resource._check_pipeline_registry(cursor, wctx):
        return

    from moncpipelib.config import config
    from moncpipelib.contracts.hashing import derive_data_classification

    registry_schema = config.pipeline_registry.schema_name
    registry_table = config.pipeline_registry.table_name

    source_id: str | None = None
    if loaded_contract.data_source is not None:
        source_id = loaded_contract.data_source.source_id

    owner_team: str | None = None
    if loaded_contract.owner is not None:
        owner_team = loaded_contract.owner.team

    # Phase 3 projections
    contract_hash: str | None = loaded_contract.contract_hash or None
    schema_fingerprint: str | None = loaded_contract.schema_fingerprint or None
    contract_version: str | None = loaded_contract.version or None
    sla_freshness_hours: int | None = None
    if loaded_contract.sla is not None:
        sla_freshness_hours = loaded_contract.sla.freshness_hours
    contract_tags: dict[str, str] | None = loaded_contract.tags or None
    data_classification = derive_data_classification(loaded_contract)

    sql = f"""
        INSERT INTO {registry_schema}.{registry_table} (
            pipeline_id, asset_name, layer, source_id,
            owner_team, description,
            dagster_asset_key, dagster_job_name, code_location_name,
            contract_hash, schema_fingerprint, contract_version,
            sla_freshness_hours, tags, data_classification,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s::jsonb, %s, NOW())
        ON CONFLICT (pipeline_id) DO UPDATE SET
            asset_name = EXCLUDED.asset_name,
            layer = EXCLUDED.layer,
            source_id = EXCLUDED.source_id,
            owner_team = EXCLUDED.owner_team,
            description = EXCLUDED.description,
            dagster_asset_key = EXCLUDED.dagster_asset_key,
            dagster_job_name = EXCLUDED.dagster_job_name,
            code_location_name = EXCLUDED.code_location_name,
            contract_hash = EXCLUDED.contract_hash,
            schema_fingerprint = EXCLUDED.schema_fingerprint,
            contract_version = EXCLUDED.contract_version,
            sla_freshness_hours = EXCLUDED.sla_freshness_hours,
            tags = EXCLUDED.tags,
            data_classification = EXCLUDED.data_classification,
            updated_at = NOW()
    """  # noqa: S608

    try:
        cursor.execute(
            sql,
            (
                loaded_contract.pipeline_id,
                loaded_contract.asset,
                layer,
                source_id,
                owner_team,
                loaded_contract.description,
                wctx.dagster_asset_key,
                wctx.dagster_job_name,
                wctx.code_location_name,
                contract_hash,
                schema_fingerprint,
                contract_version,
                sla_freshness_hours,
                _json.dumps(contract_tags) if contract_tags is not None else None,
                data_classification,
            ),
        )
    except Exception as upsert_err:
        wctx.log.warning(
            "pipeline_registry upsert failed for pipeline_id=%s asset=%s: %s",
            loaded_contract.pipeline_id,
            loaded_contract.asset,
            upsert_err,
        )
        raise


def pipeline_registry_row_matches(
    cursor: psycopg.Cursor,
    loaded_contract: DataContract,
    wctx: WriteContext,
) -> bool:
    """Return True when the registry row already matches contract + Dagster identity.

    Issue #332 fast path: the steady-state case for repeat writes of
    the same contract from the same Dagster deployment is "nothing
    changed." A single SELECT with no row lock is enough to confirm
    that and skip the UPSERT entirely.

    Drift signal is ``contract_hash`` (covers all serialisable
    contract content, including description, owner, SLA, tags, and
    schema) plus the three Dagster handles (which come from
    :class:`WriteContext` rather than the contract YAML and can
    change independently). When ``contract_hash`` is empty / NULL
    (pre-Phase-3 contracts where the loader did not compute it),
    returns False so the caller falls through to the unconditional
    upsert -- drift cannot be detected without the hash.
    """
    contract_hash = loaded_contract.contract_hash or None
    if contract_hash is None:
        return False

    from moncpipelib.config import config

    schema = config.pipeline_registry.schema_name
    table = config.pipeline_registry.table_name

    cursor.execute(
        f"SELECT contract_hash, dagster_asset_key, dagster_job_name, code_location_name "  # noqa: S608
        f"FROM {schema}.{table} WHERE pipeline_id = %s",
        (loaded_contract.pipeline_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return False

    return bool(
        row[0] == contract_hash
        and row[1] == wctx.dagster_asset_key
        and row[2] == wctx.dagster_job_name
        and row[3] == wctx.code_location_name
    )


def pipeline_registry_upsert_committed(
    resource: _RegistryResourceProtocol,
    *,
    loaded_contract: DataContract | None,
    wctx: WriteContext,
    layer: str | None,
) -> None:
    """Register the pipeline in its own short-lived autocommit connection.

    Issue #332: the prior design ran the registry upsert on the
    data-write cursor inside the main write transaction.  ``INSERT ...
    ON CONFLICT (pipeline_id) DO UPDATE`` acquired a row-exclusive lock
    on the registry row, and Postgres held that lock until the
    transaction committed -- i.e., until after the entire silver
    write completed.  Any second concurrent writer of the same
    ``pipeline_id`` (different partition, same contract) blocked on
    that lock for the full duration of the first write, defeating any
    pool concurrency > 1 for parallel backfills of the same asset.

    This wrapper opens a dedicated autocommit connection, runs the
    registry upsert there, and closes it before the data-write
    transaction opens.  The registry row is committed (visible to FK
    checks on any subsequent transaction's ``data_lineage`` INSERT)
    and the row lock is released immediately after the microsecond-
    scale upsert.

    Includes a SELECT-based fast path: in the steady state (no
    contract drift, no Dagster identity change), the wrapper issues a
    single SELECT, sees a matching row, and skips the upsert
    entirely -- zero row-level locks.

    Failure mode: a no-op when the table does not exist yet (the same
    silent self-cancelling behaviour as the in-cursor variant) or the
    contract is unset.  Any other SQL error is logged and re-raised,
    so the caller aborts the write rather than proceeding with a
    registry row that does not exist for the Phase 4 FK from
    ``data_lineage.pipeline_id``.
    """
    if loaded_contract is None or not loaded_contract.pipeline_id:
        return

    # #420: test-mode lineage isolation -- write-path callers only, so the
    # per-write WARNING from PostgresResource._resolve_skip_lineage already
    # covers this skip; log at DEBUG for traceability.
    from moncpipelib.config import skip_lineage_writes

    if skip_lineage_writes():
        logging.getLogger("moncpipelib.resources").debug(
            "Skipping pipeline_registry upsert for pipeline_id=%s (#420 "
            "test-mode lineage isolation).",
            loaded_contract.pipeline_id,
        )
        return

    with resource.get_connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            if not resource._check_pipeline_registry(cursor, wctx):
                return
            if resource._pipeline_registry_row_matches(cursor, loaded_contract, wctx):
                return
            resource._pipeline_registry_upsert(
                cursor,
                loaded_contract=loaded_contract,
                wctx=wctx,
                layer=layer,
            )
