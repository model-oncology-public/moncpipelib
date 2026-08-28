"""Historical SCD2 backfill utilities.

Provides helpers for loading historical data with correct period boundaries
using the ``periods`` section of a ``DataSource``. Includes Dagster partition
generation and a simple loop-based loader.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import polars as pl
from dagster import DynamicPartitionsDefinition

from moncpipelib.contracts.models import DataSource, Period, require_enumerated_periods

if TYPE_CHECKING:
    from dagster import AssetExecutionContext, PartitionsDefinition

    from moncpipelib.resources.postgres import PostgresResource
    from moncpipelib.resources.types import WriteContext, WriteResult


def build_partitions_from_periods(
    source: DataSource,
) -> PartitionsDefinition:
    """Build a Dagster PartitionsDefinition from data source periods.

    If all periods have uniform intervals that map to a known Dagster cron
    schedule (monthly, quarterly, yearly), returns a
    ``TimeWindowPartitionsDefinition``. Otherwise returns a
    ``StaticPartitionsDefinition`` with partition keys named by
    ``effective_from`` date (ISO format).

    Args:
        source: DataSource with a non-empty ``periods`` list.

    Returns:
        PartitionsDefinition suitable for ``@asset(partitions_def=...)``.

    Raises:
        ValueError: If data source has no periods defined.
    """
    from dagster import StaticPartitionsDefinition, TimeWindowPartitionsDefinition

    if not source.periods:
        raise ValueError(
            f"DataSource '{source.source_id}' has no periods defined. "
            f"Add a 'periods' section to the *.source.yaml file."
        )

    periods = require_enumerated_periods(source)

    # If all periods have explicit partition_key, use static partitions.
    # Skip TimeWindowPartitionsDefinition which has completed-window semantics
    # and won't return keys for windows that haven't closed yet.
    if all(p.partition_key is not None for p in periods):
        partition_keys: list[str] = [
            p.partition_key for p in periods if p.partition_key is not None
        ]
        return StaticPartitionsDefinition(partition_keys)

    # No explicit partition_key: try uniform interval detection for TimeWindow
    if len(periods) >= 2:
        intervals = []
        for i in range(1, len(periods)):
            delta = (periods[i].effective_from - periods[i - 1].effective_from).days
            intervals.append(delta)

        if len(set(intervals)) == 1:
            interval_days = intervals[0]
            cron = _interval_to_cron(interval_days)
            if cron is not None:
                return TimeWindowPartitionsDefinition(
                    start=periods[0].effective_from.isoformat(),
                    cron_schedule=cron,
                    fmt="%Y-%m-%d",
                )

    # Fallback: static partitions keyed by ISO date
    return StaticPartitionsDefinition([p.effective_from.isoformat() for p in periods])


def get_period_for_partition(
    source: DataSource,
    partition_key: str,
) -> Period:
    """Look up the Period matching a Dagster partition key.

    Args:
        source: DataSource with periods.
        partition_key: Dagster partition key (partition_key string or ISO date).

    Returns:
        The matching Period.

    Raises:
        KeyError: If no period matches the partition key.
    """
    periods = require_enumerated_periods(source)
    for period in periods:
        if period.partition_key == partition_key:
            return period
        if period.effective_from.isoformat() == partition_key:
            return period
    available = [p.partition_key or p.effective_from.isoformat() for p in periods]
    raise KeyError(f"No period matches partition key '{partition_key}'. Available: {available}")


def load_historical_periods(
    source: DataSource,
    database: PostgresResource,
    context: WriteContext | AssetExecutionContext,
    target: str,
    read_source: Callable[[Period], pl.DataFrame],
    **write_kwargs: Any,
) -> list[WriteResult]:
    """Load all historical periods from a data source in chronological order.

    Iterates periods, calls ``read_source(period)`` to get each DataFrame,
    then writes via ``database.write()`` with the period's ``effective_from``
    as the ``effective_date``.

    Args:
        source: DataSource with periods.
        database: PostgresResource instance.
        context: Dagster context or WriteContext.
        target: Target table (e.g., ``"silver.products"``).
        read_source: Callable that takes a Period and returns a DataFrame.
        **write_kwargs: Additional kwargs passed to ``database.write()``.

    Returns:
        List of WriteResult, one per period (chronological order).

    Raises:
        ValueError: If data source has no periods defined.
    """
    if not source.periods:
        raise ValueError(
            f"DataSource '{source.source_id}' has no periods defined. "
            f"Add a 'periods' section to the *.source.yaml file."
        )

    periods = require_enumerated_periods(source)
    results: list[WriteResult] = []
    for period in periods:
        df = read_source(period)
        result = database.write(
            df,
            target=target,
            context=context,
            effective_date=period.effective_from,
            **write_kwargs,
        )
        results.append(result)

    return results


class RegistryPartitionsDefinition(DynamicPartitionsDefinition):
    """DynamicPartitionsDefinition that preserves the registry ``source_id``.

    Thin subclass enabling the integration test harness to query the
    period registry for valid partition keys without reverse-engineering
    the ``periods_{source_id}`` naming convention.

    Attributes:
        source_id: The data source UUID passed to
            :func:`build_partitions_from_registry`.
    """

    source_id: str

    def __new__(cls, source_id: str, name: str | None = None) -> RegistryPartitionsDefinition:
        resolved_name = name or f"periods_{re.sub(r'[^A-Za-z0-9_]', '_', source_id)}"
        instance: RegistryPartitionsDefinition = super().__new__(cls, name=resolved_name)
        instance.source_id = source_id
        return instance

    def __init__(self, source_id: str, name: str | None = None) -> None:
        # __new__ already handled initialization; avoid duplicate super().__init__
        pass


def build_partitions_from_registry(
    source_id: str,
    name: str | None = None,
) -> RegistryPartitionsDefinition:
    """Build a RegistryPartitionsDefinition for a registry-backed source.

    The returned definition starts empty. Partitions are added at runtime
    by the ``period_registry_sensor`` or via
    ``instance.add_dynamic_partitions()``.

    The ``source_id`` is preserved on the returned object so the
    integration test harness can look up valid partition keys from the
    period registry.

    Args:
        source_id: Data source identifier (used to generate default name).
        name: Explicit partitions definition name. Defaults to
            ``"periods_{source_id}"`` with hyphens replaced by underscores.

    Returns:
        RegistryPartitionsDefinition ready for use with
        ``@asset(partitions_def=...)``.
    """
    return RegistryPartitionsDefinition(source_id=source_id, name=name)


def get_period_from_registry(
    database: PostgresResource,
    source_id: str,
    partition_key: str,
) -> Period:
    """Look up a Period from the registry by source_id and partition_key.

    Queries the period registry for all periods of the given source_id
    and returns the one matching the provided partition_key.

    Args:
        database: PostgresResource for DB access.
        source_id: Data source identifier.
        partition_key: Dagster partition key to resolve.

    Returns:
        Period with source, effective_from, effective_to, partition_key.

    Raises:
        KeyError: If no matching period found in registry.
    """
    rows = database.get_registry_periods(source_id, status=None)
    for row in rows:
        if row["partition_key"] == partition_key:
            return Period(
                source=row["source_uri"] or "",
                effective_from=row["effective_from"],
                effective_to=row["effective_to"],
                partition_key=row["partition_key"],
            )
    raise KeyError(
        f"No period with partition_key '{partition_key}' found in registry "
        f"for source '{source_id}'."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CRON_MAP: dict[int, str] = {
    28: "0 0 1 * *",  # ~monthly (28 days)
    30: "0 0 1 * *",  # ~monthly (30 days)
    31: "0 0 1 * *",  # ~monthly (31 days)
    90: "0 0 1 */3 *",  # ~quarterly
    91: "0 0 1 */3 *",  # ~quarterly
    92: "0 0 1 */3 *",  # ~quarterly
    182: "0 0 1 */6 *",  # ~semi-annual
    183: "0 0 1 */6 *",  # ~semi-annual
    365: "0 0 1 1 *",  # ~yearly
    366: "0 0 1 1 *",  # ~yearly (leap year)
}


def _interval_to_cron(days: int) -> str | None:
    """Map an interval in days to a Dagster cron schedule, or None."""
    return _CRON_MAP.get(days)
