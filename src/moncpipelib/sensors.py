"""Dagster sensor factories for moncpipelib.

This module is the **catalogue** for moncpipelib's Dagster sensor
factories.  Implementations may live in domain subpackages (e.g.
:mod:`moncpipelib.ingest.sensors`) and are re-exported here for
discoverability -- one import path covers every sensor moncpipelib
ships.  See the resolved sensor-location decision in
moncpipelib#216.

Provides reusable sensor definitions for cross-code-location period
discovery and materialization triggering.

Note: ``from __future__ import annotations`` is intentionally omitted here
because Dagster's ``@sensor`` decorator resolves type annotations eagerly,
and the PEP 563 stringification breaks resolution inside local scopes.
"""

import re
from typing import TYPE_CHECKING, Any

from moncpipelib.config import MetadataKeys
from moncpipelib.ingest.sensors import build_discovery_sensor

__all__ = [
    "build_discovery_sensor",
    "period_registry_sensor",
    "reconciliation_sensor",
    "registry_sensor",
    "scd2_registry_sensor",
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from dagster import (
        DynamicPartitionsDefinition,
        JobDefinition,
        SensorDefinition,
        UnresolvedAssetJobDefinition,
    )


def registry_sensor(
    source_id: str,
    target_job: "JobDefinition | UnresolvedAssetJobDefinition",
    *,
    partitions_def: "DynamicPartitionsDefinition | None" = None,
    ready_when: "Callable[[dict[str, Any]], bool] | None" = None,
    trigger_mode: str = "per_partition",
    timestamp_key: str = MetadataKeys.REGISTERED_AT,
    name: str | None = None,
    minimum_interval_seconds: int = 300,
    description: str | None = None,
) -> "SensorDefinition":
    """Create a general-purpose registry-driven sensor.

    All sensor logic lives here. The convenience factories
    (``period_registry_sensor``, ``reconciliation_sensor``,
    ``scd2_registry_sensor``) are thin wrappers that pre-configure
    ``ready_when``, ``trigger_mode``, and ``timestamp_key``.

    Uses Dagster's ``context.cursor`` to track a high-water mark
    timestamp. The sensor triggers when registry timestamps advance
    past the cursor, regardless of whether partition keys already
    exist. This enables idempotent re-runs after truncate/reload
    without manually clearing dynamic partitions.

    Args:
        source_id: Registry ``source_id`` UUID to watch.
        target_job: Dagster job to trigger.
        partitions_def: Dynamic partitions definition. Required for
            ``per_partition`` mode, optional for ``all`` mode.
        ready_when: Predicate evaluated against each period's metadata
            dict. Defaults to ``lambda _: True`` (always ready).
        trigger_mode: ``"per_partition"`` emits one ``RunRequest`` per
            ready period key. ``"all"`` waits for every period to be
            ready, then emits a single ``RunRequest``.
        timestamp_key: Field name for cursor-based change detection.
            Checked in the row dict first, then in metadata. Defaults
            to ``"registered_at"``.
        name: Sensor name for the Dagster UI.
        minimum_interval_seconds: Polling interval. Default 300 (5 min).
        description: Sensor description for the Dagster UI.

    Returns:
        A configured ``SensorDefinition``.
    """
    from dagster import RunRequest, SensorResult, SkipReason, sensor

    if ready_when is None:
        ready_when = lambda _: True  # noqa: E731

    if trigger_mode == "per_partition" and partitions_def is None:
        msg = "partitions_def is required when trigger_mode='per_partition'"
        raise ValueError(msg)

    resolved_name = name or f"{re.sub(r'[^A-Za-z0-9_]', '_', source_id)}_registry_sensor"
    resolved_desc = description or (
        f"Registry sensor for source '{source_id}' (mode={trigger_mode})."
    )

    # Capture ready_when in closure to avoid late-binding issues
    _ready_when = ready_when

    @sensor(
        name=resolved_name,
        job=target_job,
        minimum_interval_seconds=minimum_interval_seconds,
        description=resolved_desc,
        required_resource_keys={"database"},
    )
    def _sensor(context):  # type: ignore[no-untyped-def]
        import json as _json

        from moncpipelib.resources.postgres import PostgresResource

        database: PostgresResource = context.resources.database

        # Query registry for materialized periods
        try:
            registry_periods = database.get_registry_periods(
                source_id=source_id,
                status="materialized",
            )
        except Exception as e:
            context.log.warning(f"Failed to query period registry: {e}")
            return SkipReason(f"Registry query failed: {e}")

        if not registry_periods:
            return SkipReason(f"No materialized periods for source '{source_id}'")

        # Parse metadata for each period
        def _parse_meta(row: dict[str, Any]) -> dict[str, Any]:
            meta = row.get("metadata_") or row.get("metadata") or {}
            if isinstance(meta, str):
                meta = _json.loads(meta)
            return meta if isinstance(meta, dict) else {}

        if trigger_mode == "per_partition":
            return _handle_per_partition(
                context,
                registry_periods,
                _parse_meta,
                _ready_when,
                partitions_def,
                source_id,
                timestamp_key,
                RunRequest,
                SensorResult,
                SkipReason,
            )
        else:  # "all" mode
            return _handle_all(
                context,
                registry_periods,
                _parse_meta,
                _ready_when,
                source_id,
                timestamp_key,
                RunRequest,
                SensorResult,
                SkipReason,
            )

    return _sensor


def _get_timestamp(row: dict[str, Any], meta: dict[str, Any], timestamp_key: str) -> str | None:
    """Extract a timestamp from a row or its metadata, returning an ISO string."""
    val = row.get(timestamp_key)
    if val is not None:
        return val.isoformat() if hasattr(val, "isoformat") else str(val)
    val = meta.get(timestamp_key)
    if val is not None:
        return str(val)
    return None


def _handle_per_partition(
    context: Any,
    registry_periods: list[dict[str, Any]],
    parse_meta: Any,
    ready_when: Any,
    partitions_def: Any,
    source_id: str,
    timestamp_key: str,
    RunRequest: type,  # noqa: N803
    SensorResult: type,  # noqa: N803
    SkipReason: type,  # noqa: N803
) -> Any:
    """Handle per_partition trigger mode with cursor-based change detection."""
    cursor = context.cursor  # ISO timestamp string or None

    context.log.info(
        f"[{source_id}] per_partition eval: "
        f"{len(registry_periods)} periods, cursor={cursor!r}, "
        f"timestamp_key={timestamp_key!r}"
    )

    ready_keys: list[str] = []
    actionable_keys: list[str] = []
    pending_count = 0
    max_ts = cursor

    for row in registry_periods:
        meta = parse_meta(row)
        key = row["partition_key"]

        if not ready_when(meta):
            pending_count += 1
            continue
        ready_keys.append(key)

        ts = _get_timestamp(row, meta, timestamp_key)
        if ts and (cursor is None or ts > cursor):
            actionable_keys.append(key)
            if max_ts is None or ts > max_ts:
                max_ts = ts

    context.log.info(
        f"[{source_id}] ready={len(ready_keys)}, pending={pending_count}, "
        f"actionable={len(actionable_keys)}, max_ts={max_ts!r}"
    )

    if not ready_keys:
        reason = f"No ready periods for source '{source_id}' ({pending_count} pending)"
        context.log.info(f"[{source_id}] skip: {reason}")
        return SkipReason(reason)

    if not actionable_keys:
        reason = f"No new activity for source '{source_id}' since last check"
        context.log.info(f"[{source_id}] skip: {reason}")
        return SkipReason(reason)

    sorted_keys = sorted(set(actionable_keys))

    context.log.info(
        f"[{source_id}] triggering {len(sorted_keys)} run(s): {sorted_keys}, "
        f"advancing cursor to {max_ts!r}"
    )

    return SensorResult(
        run_requests=[RunRequest(partition_key=key) for key in sorted_keys],
        dynamic_partitions_requests=[partitions_def.build_add_request(sorted_keys)],
        cursor=max_ts,
        metadata={
            "source_id": source_id,
            "total_periods": len(registry_periods),
            "ready_periods": len(ready_keys),
            "pending_periods": pending_count,
            "actionable_periods": len(sorted_keys),
            "actionable_keys": ", ".join(sorted_keys),
            "cursor": max_ts or "",
        },
    )


def _handle_all(
    context: Any,
    registry_periods: list[dict[str, Any]],
    parse_meta: Any,
    ready_when: Any,
    source_id: str,
    timestamp_key: str,
    RunRequest: type,  # noqa: N803
    SensorResult: type,  # noqa: N803
    SkipReason: type,  # noqa: N803
) -> Any:
    """Handle all trigger mode with cursor-based change detection."""
    cursor = context.cursor  # ISO timestamp string or None
    total = len(registry_periods)

    context.log.info(
        f"[{source_id}] all eval: {total} periods, cursor={cursor!r}, "
        f"timestamp_key={timestamp_key!r}"
    )

    ready: list[dict[str, Any]] = []
    not_ready: list[dict[str, Any]] = []
    max_ts: str | None = None

    for row in registry_periods:
        meta = parse_meta(row)
        if ready_when(meta):
            ready.append(row)
            ts = _get_timestamp(row, meta, timestamp_key)
            if ts and (max_ts is None or ts > max_ts):
                max_ts = ts
        else:
            not_ready.append(row)

    context.log.info(f"[{source_id}] ready={len(ready)}/{total}, max_ts={max_ts!r}")

    if not_ready:
        reason = f"{len(ready)}/{total} periods ready, {len(not_ready)} still pending"
        context.log.info(f"[{source_id}] skip: {reason}")
        return SkipReason(reason)

    # All periods ready -- check if data changed since last trigger
    if max_ts and cursor and max_ts <= cursor:
        reason = "All periods ready but no new activity since last trigger"
        context.log.info(f"[{source_id}] skip: {reason}")
        return SkipReason(reason)

    context.log.info(
        f"[{source_id}] triggering: all {total} periods ready, "
        f"max_ts={max_ts!r} > cursor={cursor!r}, advancing cursor"
    )

    return SensorResult(
        run_requests=[RunRequest(run_key=f"registry-all-{source_id}-{max_ts}")],
        cursor=max_ts,
        metadata={
            "source_id": source_id,
            "total_periods": total,
            "ready_periods": len(ready),
            "pending_periods": 0,
            "cursor": max_ts or "",
        },
    )


def period_registry_sensor(
    source_id: str,
    target_job: "JobDefinition | UnresolvedAssetJobDefinition",
    *,
    partitions_def: "DynamicPartitionsDefinition",
    name: str | None = None,
    minimum_interval_seconds: int = 300,
    description: str | None = None,
) -> "SensorDefinition":
    """Discover new materialized periods and trigger downstream processing.

    Thin wrapper around ``registry_sensor()`` with
    ``ready_when=lambda _: True`` and ``trigger_mode="per_partition"``.

    Queries ``lineage.period_registry`` for rows with
    ``status='materialized'`` for the given ``source_id``. New partition
    keys (not yet in the dynamic partitions set) trigger
    ``RunRequest`` instances and are added to the partitions definition.

    One sensor per ``source_id``. Multiple sources require multiple sensors.

    Args:
        source_id: Registry ``source_id`` UUID to watch.
        target_job: Dagster job to trigger for new periods.
        partitions_def: The ``DynamicPartitionsDefinition`` to add keys to.
        name: Sensor name. Defaults to ``"{sanitized_source_id}_period_sensor"``.
        minimum_interval_seconds: Polling interval. Default 300 (5 min).
        description: Sensor description for the Dagster UI.

    Returns:
        A configured ``SensorDefinition``.
    """
    return registry_sensor(
        source_id=source_id,
        target_job=target_job,
        partitions_def=partitions_def,
        ready_when=lambda _: True,
        trigger_mode="per_partition",
        timestamp_key=MetadataKeys.REGISTERED_AT,
        name=name or f"{re.sub(r'[^A-Za-z0-9_]', '_', source_id)}_period_sensor",
        minimum_interval_seconds=minimum_interval_seconds,
        description=description
        or (
            f"Discovers new materialized periods for source '{source_id}' "
            f"from the period registry and triggers downstream materialization."
        ),
    )


def reconciliation_sensor(
    source_id: str,
    target_job: "JobDefinition | UnresolvedAssetJobDefinition",
    *,
    name: str | None = None,
    minimum_interval_seconds: int = 300,
    description: str | None = None,
) -> "SensorDefinition":
    """Trigger SCD2 reconciliation when all periods are silver-materialized.

    Thin wrapper around ``registry_sensor()`` with
    ``ready_when=lambda m: m.get("silver_materialized_at") is not None``
    and ``trigger_mode="all"``.

    When ALL materialized periods have ``silver_materialized_at`` in
    metadata, emits a single ``RunRequest`` to trigger the reconciliation
    job.

    Args:
        source_id: Registry ``source_id`` UUID to watch.
        target_job: Dagster job that performs the reconciliation.
        name: Sensor name. Defaults to
            ``"{sanitized_source_id}_reconciliation_sensor"``.
        minimum_interval_seconds: Polling interval. Default 300 (5 min).
        description: Sensor description for the Dagster UI.

    Returns:
        A configured ``SensorDefinition``.
    """
    return registry_sensor(
        source_id=source_id,
        target_job=target_job,
        ready_when=lambda m: m.get(MetadataKeys.SILVER_MATERIALIZED_AT) is not None,
        trigger_mode="all",
        timestamp_key=MetadataKeys.SILVER_MATERIALIZED_AT,
        name=name or f"{re.sub(r'[^A-Za-z0-9_]', '_', source_id)}_reconciliation_sensor",
        minimum_interval_seconds=minimum_interval_seconds,
        description=description
        or (
            f"Triggers SCD2 reconciliation for source '{source_id}' when all "
            f"periods are silver-materialized and at least one is unreconciled."
        ),
    )


def scd2_registry_sensor(
    source_id: str,
    target_job: "JobDefinition | UnresolvedAssetJobDefinition",
    *,
    partitions_def: "DynamicPartitionsDefinition",
    name: str | None = None,
    minimum_interval_seconds: int = 300,
    description: str | None = None,
) -> "SensorDefinition":
    """Gate downstream processing on SCD2 reconciliation completion.

    Thin wrapper around ``registry_sensor()`` with
    ``ready_when=lambda m: m.get("reconciled_at") is not None``
    and ``trigger_mode="per_partition"``.

    Use this sensor in gold-layer code locations to wait for reconciliation
    to complete before triggering gold materialization.

    Args:
        source_id: Registry ``source_id`` UUID to watch.
        target_job: Dagster job to trigger after reconciliation.
        partitions_def: The ``DynamicPartitionsDefinition`` to add keys to.
        name: Sensor name. Defaults to
            ``"{sanitized_source_id}_scd2_sensor"``.
        minimum_interval_seconds: Polling interval. Default 300 (5 min).
        description: Sensor description for the Dagster UI.

    Returns:
        A configured ``SensorDefinition``.
    """
    return registry_sensor(
        source_id=source_id,
        target_job=target_job,
        partitions_def=partitions_def,
        ready_when=lambda m: m.get(MetadataKeys.RECONCILED_AT) is not None,
        trigger_mode="per_partition",
        timestamp_key=MetadataKeys.RECONCILED_AT,
        name=name or f"{re.sub(r'[^A-Za-z0-9_]', '_', source_id)}_scd2_sensor",
        minimum_interval_seconds=minimum_interval_seconds,
        description=description
        or (
            f"Triggers downstream processing for source '{source_id}' "
            f"when SCD2 reconciliation is complete."
        ),
    )
