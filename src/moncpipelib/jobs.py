"""Dagster job factories for moncpipelib.

Provides reusable job definitions for common pipeline operations
like SCD2 reconciliation.

Note: ``from __future__ import annotations`` is intentionally omitted here
because Dagster decorators resolve type annotations eagerly, and the
PEP 563 stringification breaks resolution inside local scopes.
"""

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from moncpipelib.config import DEFAULT_SCHEMA, MetadataKeys

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dagster import (
        AssetsDefinition,
        JobDefinition,
        SensorDefinition,
        UnresolvedAssetJobDefinition,
    )

    from moncpipelib.contracts.models import DataContract
    from moncpipelib.resources.postgres import PostgresResource


def _perform_reconciliation(
    database: "PostgresResource",
    contract: "DataContract",
    source_id: str,
    resolved_name: str,
    collapse_duplicates: bool,
    log: Any,
    context: Any,
) -> dict[str, Any]:
    """Execute SCD2 reconciliation and stamp registry metadata.

    Shared by :func:`make_reconciliation_job` and
    :func:`make_reconciliation_asset`. ``context`` is the Dagster
    ``OpExecutionContext`` / ``AssetExecutionContext``; the resource
    extracts ``run_id`` (and ``asset_name`` when no contract is passed)
    from it so the ``scd2_reconciliations`` audit row carries the
    cohorting key for the run that triggered the reconcile.
    """
    from moncpipelib.resources.postgres import PostgresResource as _PR

    _sink = _PR._resolve_scd2_sink(contract)
    _target = f"{_sink['schema']}.{_sink['table']}"
    log.info(f"Reconciling SCD2 timeline for {_target}")
    result = database.reconcile_scd2(
        contract=contract,
        collapse_duplicates=collapse_duplicates,
        context=context,
    )
    log.info(
        f"Reconciliation complete: "
        f"{result['rows_timeline_updated']} rows updated, "
        f"{result['rows_collapsed']} rows collapsed, "
        f"{result.get('rows_renumbered', 0)} rows renumbered, "
        f"work_mem={result.get('work_mem') or 'cluster default'}"
    )

    try:
        periods = database.get_registry_periods(
            source_id=source_id,
            status="materialized",
        )
        now = datetime.now(UTC).isoformat()
        for period in periods:
            database.update_period_metadata(
                source_id=source_id,
                partition_key=period["partition_key"],
                metadata_updates={
                    MetadataKeys.RECONCILED_AT: now,
                    MetadataKeys.RECONCILED_BY: resolved_name,
                    MetadataKeys.ROWS_TIMELINE_UPDATED: result["rows_timeline_updated"],
                    MetadataKeys.ROWS_COLLAPSED: result["rows_collapsed"],
                    MetadataKeys.ROWS_RENUMBERED: result.get("rows_renumbered", 0),
                },
            )
        log.info(f"Stamped reconciled_at on {len(periods)} period(s)")
    except Exception as e:
        log.warning(f"Failed to stamp reconciliation metadata: {e}")

    return result


def make_reconciliation_job(
    contract: "DataContract",
    source_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    collapse_duplicates: bool = True,
    tags: dict[str, Any] | None = None,
) -> "JobDefinition":
    """Create a Dagster job that reconciles SCD2 timelines and stamps the registry.

    Generates a single-op job that:

    1. Calls ``database.reconcile_scd2(contract=contract)``
    2. Queries the period registry for the given ``source_id``
    3. Stamps ``reconciled_at`` on all materialized periods

    Pair with ``reconciliation_sensor()`` for automated triggering::

        reconcile_job = make_reconciliation_job(
            contract=silver_contract,
            source_id="<source-uuid>",
        )
        reconcile_sensor = reconciliation_sensor(
            source_id="<source-uuid>",
            target_job=reconcile_job,
        )

    Args:
        contract: DataContract with an SCD2 sink (passed to
            ``reconcile_scd2(contract=...)``).
        source_id: Registry source_id UUID for metadata stamping.
        name: Job name. Defaults to
            ``"reconcile_{sanitized_asset_name}"``.
        description: Job description for the Dagster UI.
        collapse_duplicates: Passed to ``reconcile_scd2()``.
        tags: Optional Dagster tags (e.g., K8s resource config).

    Returns:
        A configured ``JobDefinition``.
    """
    from dagster import job, op

    sanitized_asset = re.sub(r"[^A-Za-z0-9_]", "_", contract.asset)
    resolved_name = name or f"reconcile_{sanitized_asset}"
    resolved_desc = description or (
        f"Reconcile SCD2 timeline for {contract.asset} and stamp period registry metadata."
    )

    @op(
        name=f"{resolved_name}_op",
        required_resource_keys={"database"},
        description=f"Reconcile SCD2 timeline for {contract.asset}",
    )
    def _reconcile_op(context):  # type: ignore[no-untyped-def]
        database = context.resources.database
        _perform_reconciliation(
            database,
            contract,
            source_id,
            resolved_name,
            collapse_duplicates,
            context.log,
            context,
        )

    @job(
        name=resolved_name,
        description=resolved_desc,
        tags=tags or {},
    )
    def _reconcile_job():  # type: ignore[no-untyped-def]
        _reconcile_op()

    return _reconcile_job


def make_reconciliation_asset(
    contract: "DataContract",
    source_id: str,
    *,
    key: "Any | None" = None,
    group_name: str | None = None,
    deps: "Sequence[Any] | None" = None,
    collapse_duplicates: bool = True,
    tags: dict[str, Any] | None = None,
) -> "AssetsDefinition":
    """Create a Dagster asset that runs SCD2 reconciliation.

    The generated asset:

    1. Calls ``database.reconcile_scd2(contract=contract)``
    2. Stamps ``reconciled_at`` on all materialized periods
    3. Returns a ``MaterializeResult`` with reconciliation stats

    The asset key defaults to ``[sink_schema, "{sink_table}_reconciled"]``
    derived from the contract's first sink.

    Args:
        contract: DataContract with an SCD2 sink.
        source_id: Registry source_id UUID for metadata stamping.
        key: Asset key. Defaults to key derived from contract sink.
        group_name: Dagster asset group name.
        deps: Asset dependencies (upstream asset keys).
        collapse_duplicates: Passed to ``reconcile_scd2()``.
        tags: Optional Dagster tags.

    Returns:
        A configured ``AssetsDefinition``.
    """
    from dagster import AssetKey, MaterializeResult, asset

    # Derive default key from contract sink
    if key is None:
        sink = contract.sinks[0] if contract.sinks else {}
        schema = sink.get("schema", DEFAULT_SCHEMA)
        table = sink.get("table", contract.asset)
        resolved_key = AssetKey([schema, f"{table}_reconciled"])
    elif isinstance(key, AssetKey):
        resolved_key = key
    elif isinstance(key, (list, tuple)):
        resolved_key = AssetKey(list(key))
    else:
        resolved_key = key

    @asset(
        key=resolved_key,
        group_name=group_name,
        deps=deps,
        required_resource_keys={"database"},
        tags=tags,
    )
    def _reconciliation_asset(context):  # type: ignore[no-untyped-def]
        database = context.resources.database
        result = _perform_reconciliation(
            database,
            contract,
            source_id,
            str(resolved_key.to_user_string()),
            collapse_duplicates,
            context.log,
            context,
        )
        return MaterializeResult(
            metadata={
                MetadataKeys.ROWS_TIMELINE_UPDATED: result["rows_timeline_updated"],
                MetadataKeys.ROWS_COLLAPSED: result["rows_collapsed"],
                MetadataKeys.ROWS_RENUMBERED: result.get("rows_renumbered", 0),
                "work_mem": result.get("work_mem"),
                "source_id": source_id,
            }
        )

    return _reconciliation_asset


def make_reconciliation_bundle(
    contract: "DataContract",
    source_id: str,
    *,
    key: "Any | None" = None,
    group_name: str | None = None,
    deps: "Sequence[Any] | None" = None,
    collapse_duplicates: bool = True,
    sensor_interval_seconds: int = 300,
    tags: dict[str, Any] | None = None,
) -> "tuple[AssetsDefinition, SensorDefinition, UnresolvedAssetJobDefinition]":
    """Create a reconciliation asset, sensor, and job bundle.

    Composes ``make_reconciliation_asset()``, ``define_asset_job()``,
    and ``reconciliation_sensor()`` into a single factory call.

    Returns:
        A ``(asset, sensor, job)`` tuple ready to include in Dagster
        ``Definitions``.
    """
    from dagster import define_asset_job

    from moncpipelib.sensors import reconciliation_sensor

    sanitized_asset = re.sub(r"[^A-Za-z0-9_]", "_", contract.asset)

    reconcile_asset = make_reconciliation_asset(
        contract=contract,
        source_id=source_id,
        key=key,
        group_name=group_name,
        deps=deps,
        collapse_duplicates=collapse_duplicates,
        tags=tags,
    )

    reconcile_job = define_asset_job(
        name=f"reconcile_{sanitized_asset}_job",
        selection=[reconcile_asset],
    )

    reconcile_sensor = reconciliation_sensor(
        source_id=source_id,
        target_job=reconcile_job,
        minimum_interval_seconds=sensor_interval_seconds,
    )

    return (reconcile_asset, reconcile_sensor, reconcile_job)
