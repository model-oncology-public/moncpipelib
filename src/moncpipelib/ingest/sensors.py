"""Dagster sensor factory for ingest discovery.

Provides :func:`build_discovery_sensor`, which wraps an
:class:`~moncpipelib.contracts.models.IngestContract` in a Dagster
:class:`~dagster.SensorDefinition` that ticks on a configured cadence,
calls ``pattern.discover_partitions(contract, ctx)``, diffs the
returned :class:`PartitionSpec` list against the existing dynamic
partition registry, and adds new keys.

Per the resolved planning decisions in moncpipelib#216:

- **State-based diff** (not cursor-based).  After a Dagster home reset,
  every release looks "new" and is re-added; ``hash_compare`` makes
  re-materialization a no-op.  Documented as "post-DR avalanche,
  idempotent".
- **No load-time side effects**: the resolver call lives inside the
  sensor function body (NOT at factory invocation time).  Importing
  the code location module does not call the network -- a regression
  test pins this.
- **Resolver-failure modes**:

  - ``5xx`` / network / generic transient errors -> log + skip the
    tick.  Does NOT advance any cursor; does NOT remove existing
    partitions.
  - ``401`` / ``403`` -> raises so the failure is visible as a sensor
    error in the Dagster UI.  Existing partitions remain
    materializable (the failure prevents NEW partitions from being
    added; it doesn't unmaterialize anything).
  - Empty result on a previously-non-empty source -> log warning;
    do NOT remove existing partitions.

- **RunRequest emission is opt-in**.  Default behaviour is to add
  partition keys to the registry only; operators wire materialization
  on a separate cadence than discovery.  Setting
  ``emit_run_requests=True`` emits one ``RunRequest`` per newly-added
  partition.

Per the resolved sensor-location decision in moncpipelib#216 (hybrid):
the implementation lives here in the ingest subpackage; top-level
``moncpipelib.sensors`` re-exports it for catalogue discoverability.
"""

# Note: ``from __future__ import annotations`` is intentionally omitted
# here to mirror the existing ``src/moncpipelib/sensors.py`` -- Dagster's
# ``@sensor`` decorator resolves type annotations eagerly and the PEP 563
# stringification breaks resolution inside local scopes.

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dagster import (
        DynamicPartitionsDefinition,
        JobDefinition,
        SensorDefinition,
        UnresolvedAssetJobDefinition,
    )

    from moncpipelib.contracts.models import IngestContract


_DEFAULT_INTERVAL_SECONDS: int = 21_600
"""Default sensor cadence: 6 hours.

UTS rate-limits aggressive polling on the Releases endpoint; 6h is the
documented sweet spot from data-platform's earlier RxNorm pipeline.
Operators can override via ``minimum_interval_seconds``.
"""


def build_discovery_sensor(
    contract: "IngestContract",
    target_job: "JobDefinition | UnresolvedAssetJobDefinition",
    *,
    partitions_def: "DynamicPartitionsDefinition",
    minimum_interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
    secrets_resource_key: str | None = "secrets",
    emit_run_requests: bool = False,
    name: str | None = None,
    description: str | None = None,
    additional_resource_keys: set[str] | None = None,
) -> "SensorDefinition":
    """Build a Dagster sensor that discovers partitions for ``contract``.

    The factory itself does NO network I/O.  ``discover_partitions`` is
    called only inside the sensor function body, at tick time.

    Args:
        contract: The :class:`IngestContract` to discover partitions
            for.  The pattern is looked up via
            :func:`~moncpipelib.ingest.patterns.get_pattern` at
            tick time.
        target_job: Job to trigger via ``RunRequest`` when
            ``emit_run_requests=True``.
        partitions_def: The :class:`DynamicPartitionsDefinition`
            backing this contract's downstream consumers.  The sensor
            adds keys via
            :meth:`DagsterInstance.add_dynamic_partitions` and
            (optionally) emits ``RunRequest`` per new key.
        minimum_interval_seconds: Polling cadence in seconds.  Default
            ``21_600`` (6 hours) -- respects UTS rate limits.
        secrets_resource_key: Resource key under which the
            :class:`~moncpipelib.resources.keyvault.KeyVaultSecretResource`
            is registered in the user's :class:`Definitions`.  Default
            ``"secrets"``.  Pass ``None`` to disable secrets injection
            (only useful for static patterns; ``api_resolver`` requires
            a real resource).
        emit_run_requests: When ``True``, emit one ``RunRequest`` per
            newly-added partition.  Default ``False`` -- operators wire
            materialization on a separate cadence (e.g. a schedule or
            registry sensor downstream) so discovery does not
            accidentally materialize a 5+ GB UMLS archive at every
            tick.
        name: Sensor name for the Dagster UI.  Default
            ``"<source_name>_discovery_sensor"``.
        description: Sensor description for the Dagster UI.
        additional_resource_keys: Additional resource keys this sensor
            should require (e.g. for a custom logger).  ``"secrets"``
            (or whatever ``secrets_resource_key`` resolves to) is
            included automatically when set.

    Returns:
        A configured :class:`SensorDefinition`.
    """
    from dagster import RunRequest, SensorResult, SkipReason, sensor

    resolved_name = name or f"{_safe_identifier(contract.source_name)}_discovery_sensor"
    resolved_desc = description or (
        f"Discovery sensor for ingest contract {contract.source_name!r} "
        f"(pattern={contract.pattern!r}, partitions_def={partitions_def.name!r}). "
        f"State-based diff against existing dynamic partitions; idempotent "
        f"under DR rebuild."
    )

    required: set[str] = set(additional_resource_keys or set())
    if secrets_resource_key is not None:
        required.add(secrets_resource_key)

    @sensor(
        name=resolved_name,
        job=target_job,
        minimum_interval_seconds=minimum_interval_seconds,
        description=resolved_desc,
        required_resource_keys=required if required else None,
    )
    def _sensor(context, **resources):  # type: ignore[no-untyped-def]  # noqa: ARG001
        # Dagster injects required resources as kwargs.  We absorb them
        # via ``**resources`` and use ``context.resources`` for dynamic
        # access keyed by ``secrets_resource_key`` (which can be
        # overridden by the caller).  This keeps the function signature
        # static while supporting variable resource keys.
        del resources

        # Imports inside the sensor body keep load-time side effects
        # to a minimum.  Even if the resolver registry registration
        # eventually grows network calls, this isolates them.
        import httpx

        from moncpipelib.ingest.patterns import get_pattern
        from moncpipelib.ingest.types import IngestContext

        secrets = (
            getattr(context.resources, secrets_resource_key)
            if secrets_resource_key is not None
            else None
        )
        ctx = IngestContext(log=context.log, secrets=secrets, run_id=None)

        try:
            pattern = get_pattern(contract.pattern)
            specs = pattern.discover_partitions(contract, ctx)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (401, 403):
                # Auth failure: surface in UI so an operator notices.
                # Existing partitions remain materializable -- this
                # only blocks NEW partitions from being added.
                context.log.error(
                    "Discovery auth failure for %r: status=%d. Check the api_key in Key Vault.",
                    contract.source_name,
                    status,
                )
                raise
            context.log.warning(
                "Discovery transient HTTP failure for %r: status=%d. "
                "Skipping tick; existing partitions unchanged.",
                contract.source_name,
                status,
            )
            return SkipReason(f"transient HTTP {status}")
        except httpx.RequestError as e:
            context.log.warning(
                "Discovery network failure for %r: %s. "
                "Skipping tick; existing partitions unchanged.",
                contract.source_name,
                e,
            )
            return SkipReason(f"network error: {e}")
        except Exception as e:  # noqa: BLE001 -- intentional broad catch
            context.log.warning(
                "Discovery failed for %r: %s. Skipping tick; existing partitions unchanged.",
                contract.source_name,
                e,
            )
            return SkipReason(f"discovery failed: {e}")

        existing_keys: set[str] = set(context.instance.get_dynamic_partitions(partitions_def.name))

        if not specs:
            if existing_keys:
                # Suspect: previously-populated source returned empty.
                # Could be config drift, auth expiration, or upstream
                # regression.  Log loudly and KEEP existing partitions.
                context.log.warning(
                    "Discovery returned no partitions for %r but %d already "
                    "exist; possible config drift or auth expiration. "
                    "NOT removing existing partitions.",
                    contract.source_name,
                    len(existing_keys),
                )
                return SkipReason("discovery empty; existing kept")
            return SkipReason("no partitions discovered")

        new_keys = [s.key for s in specs if s.key not in existing_keys]
        if not new_keys:
            return SkipReason("no new partitions discovered")

        # State mutation: add the new keys.  Doing this BEFORE returning
        # any RunRequest ensures the partition is registered when Dagster
        # processes the run request.
        context.instance.add_dynamic_partitions(partitions_def.name, new_keys)

        context.log.info(
            "Discovery added %d new partition(s) for %r: %s",
            len(new_keys),
            contract.source_name,
            new_keys,
        )

        if emit_run_requests:
            return SensorResult(
                run_requests=[RunRequest(partition_key=k) for k in new_keys],
            )
        return SkipReason(
            f"added {len(new_keys)} new partition(s); materialization is "
            f"opt-in (set emit_run_requests=True or trigger via a "
            f"separate schedule / sensor)."
        )

    return _sensor


def _safe_identifier(name: str) -> str:
    """Sanitize ``name`` for use as a Dagster sensor name.

    Mirrors the safety pattern in :mod:`moncpipelib.sensors`
    (``re.sub(r'[^A-Za-z0-9_]', '_', name)``) so the Dagster UI
    shows a predictable identifier even when the contract's
    ``source_name`` contains hyphens.
    """
    import re

    return re.sub(r"[^A-Za-z0-9_]", "_", name)


# Avoid unused-import / type-checker noise in non-TYPE_CHECKING contexts
_: Any = None
del _
