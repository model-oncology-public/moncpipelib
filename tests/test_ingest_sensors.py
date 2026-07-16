"""Tests for ``build_discovery_sensor``.

Covers:

- Factory returns a :class:`SensorDefinition` with the expected name,
  description, cadence, and resource keys.
- Factory does NOT call ``discover_partitions`` at definition time
  (regression test pinning the load-time-side-effects rule from #216:
  importing a code location must not touch the network).
- State-based diff: existing keys retained; new keys added.
- Resolver-failure modes per #216:
  - Transient HTTP / network -> SkipReason; existing partitions
    unchanged.
  - 401 / 403 -> raises (sensor error visible in Dagster UI).
  - Empty result on previously-non-empty source -> SkipReason +
    warning; existing partitions kept.
- ``emit_run_requests`` is opt-in (default off).
- Catalogue re-export: top-level ``moncpipelib.sensors`` exports
  ``build_discovery_sensor`` (per the resolved sensor-location decision
  in the #216 roadmap).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest.patterns import (
    INGEST_PATTERNS,
    register_pattern,
)
from moncpipelib.ingest.sensors import build_discovery_sensor
from moncpipelib.ingest.types import IngestContext, PartitionSpec

# ---------------------------------------------------------------------------
# Stub pattern -- deterministic discover_partitions, no I/O
# ---------------------------------------------------------------------------


class _StubPattern:
    """Test pattern: returns a fixed spec list or raises a configured error."""

    name: str = "_test_discovery"

    def __init__(
        self,
        specs: list[PartitionSpec] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._specs = specs or []
        self._raise = raise_exc
        self.calls = 0

    def discover_partitions(
        self, contract: IngestContract, ctx: IngestContext
    ) -> list[PartitionSpec]:
        del contract, ctx
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return list(self._specs)

    def materialize_partition(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError("stub")


@pytest.fixture
def stub_pattern() -> Any:
    """Register / restore a fresh ``_StubPattern`` per test."""
    original = INGEST_PATTERNS.get("_test_discovery")
    stub = _StubPattern()
    register_pattern(stub)  # type: ignore[arg-type]
    yield stub
    if original is not None:
        register_pattern(original)
    else:
        INGEST_PATTERNS.pop("_test_discovery", None)


def _contract() -> IngestContract:
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="umls-meta",
        sensitivity="confidential",
        pattern="_test_discovery",
        prefix_template="umls/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={},
        data_owner="data-platform",
        compliance_review="SECURITY.md#umls",
    )


def _make_job() -> tuple[Any, Any]:
    """Build a dummy job + DynamicPartitionsDefinition for the sensor."""
    from dagster import DynamicPartitionsDefinition, asset, define_asset_job

    partitions = DynamicPartitionsDefinition(name="umls_releases")

    @asset(partitions_def=partitions)
    def dummy() -> None:
        pass

    return define_asset_job("test_job", selection=[dummy]), partitions


# ---------------------------------------------------------------------------
# Factory metadata
# ---------------------------------------------------------------------------


def test_factory_returns_sensor_definition_with_default_name() -> None:
    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    assert sensor_def.name == "umls_meta_discovery_sensor"  # hyphen -> underscore


def test_factory_uses_custom_name_when_provided() -> None:
    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(
        _contract(), job, partitions_def=partitions, name="custom_name"
    )

    assert sensor_def.name == "custom_name"


def test_factory_default_cadence_is_six_hours() -> None:
    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    assert sensor_def.minimum_interval_seconds == 21_600


def test_factory_requires_secrets_resource_by_default() -> None:
    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    assert "secrets" in sensor_def.required_resource_keys


def test_factory_supports_custom_secrets_resource_key() -> None:
    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(
        _contract(),
        job,
        partitions_def=partitions,
        secrets_resource_key="key_vault",
    )

    assert "key_vault" in sensor_def.required_resource_keys
    assert "secrets" not in sensor_def.required_resource_keys


def test_factory_does_not_invoke_discover_partitions_at_load_time(
    stub_pattern: _StubPattern,
) -> None:
    """Pinned regression: importing a code location module must not
    call the network.  The factory builds the sensor function but does
    NOT execute discover_partitions until tick time.

    This is the #216 'load-time side-effects forbidden' rule -- breaking
    it would mean every Dagster code location reload would hit UTS."""
    job, partitions = _make_job()
    build_discovery_sensor(_contract(), job, partitions_def=partitions)

    assert stub_pattern.calls == 0


# ---------------------------------------------------------------------------
# Helper: run the sensor body and return the result
# ---------------------------------------------------------------------------


def _evaluate_sensor(
    sensor_def: Any,
    *,
    secrets: Any | None = None,
) -> Any:
    """Invoke the sensor body once and return whatever it returns."""
    from dagster import DagsterInstance, build_sensor_context

    instance = DagsterInstance.ephemeral()
    context = build_sensor_context(
        instance=instance,
        resources={"secrets": secrets if secrets is not None else MagicMock()},
    )
    return sensor_def(context), instance


# ---------------------------------------------------------------------------
# State-based diff (happy path)
# ---------------------------------------------------------------------------


def test_first_tick_adds_all_discovered_partitions(
    stub_pattern: _StubPattern,
) -> None:
    stub_pattern._specs = [
        PartitionSpec(key="2026AA"),
        PartitionSpec(key="2026AB"),
    ]
    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    _, instance = _evaluate_sensor(sensor_def)

    keys = set(instance.get_dynamic_partitions(partitions.name))
    assert keys == {"2026AA", "2026AB"}


def test_subsequent_tick_adds_only_new_keys(
    stub_pattern: _StubPattern,
) -> None:
    """State-based diff: existing partitions retained; only the
    diff is added.  Mirrors the 'post-DR avalanche, idempotent'
    contract: re-adding existing keys is a no-op."""
    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    from dagster import DagsterInstance, build_sensor_context

    instance = DagsterInstance.ephemeral()
    instance.add_dynamic_partitions(partitions.name, ["2026AA"])

    stub_pattern._specs = [
        PartitionSpec(key="2026AA"),  # already exists
        PartitionSpec(key="2026AB"),  # new
    ]
    context = build_sensor_context(instance=instance, resources={"secrets": MagicMock()})
    sensor_def(context)

    keys = set(instance.get_dynamic_partitions(partitions.name))
    assert keys == {"2026AA", "2026AB"}


# ---------------------------------------------------------------------------
# Resolver-failure modes (per #216)
# ---------------------------------------------------------------------------


def test_5xx_skips_tick_and_keeps_existing_partitions(
    stub_pattern: _StubPattern,
) -> None:
    """Per #216: transient resolver failures -> log + skip.
    Does NOT advance any cursor; does NOT remove existing partitions."""
    request = httpx.Request("GET", "https://upstream.test/x")
    response = httpx.Response(503, request=request)
    stub_pattern._raise = httpx.HTTPStatusError(
        "503 service unavailable", request=request, response=response
    )

    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    from dagster import DagsterInstance, SkipReason, build_sensor_context

    instance = DagsterInstance.ephemeral()
    instance.add_dynamic_partitions(partitions.name, ["existing-key"])

    context = build_sensor_context(instance=instance, resources={"secrets": MagicMock()})
    result = sensor_def(context)

    assert isinstance(result, SkipReason)
    # Existing partitions untouched
    assert set(instance.get_dynamic_partitions(partitions.name)) == {"existing-key"}


def test_network_error_skips_tick(stub_pattern: _StubPattern) -> None:
    stub_pattern._raise = httpx.ConnectError("DNS failure")

    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    from dagster import SkipReason

    result, _ = _evaluate_sensor(sensor_def)
    assert isinstance(result, SkipReason)


def test_401_raises_visible_sensor_error(stub_pattern: _StubPattern) -> None:
    """Per #216: auth failure surfaces as a sensor error in the UI.
    Re-raised so Dagster marks the tick failed; existing partitions
    remain materializable (the failure prevents adding NEW partitions
    but does not remove anything)."""
    request = httpx.Request("GET", "https://upstream.test/x")
    response = httpx.Response(401, request=request)
    stub_pattern._raise = httpx.HTTPStatusError(
        "401 unauthorized", request=request, response=response
    )

    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    with pytest.raises(httpx.HTTPStatusError, match="401"):
        _evaluate_sensor(sensor_def)


def test_403_also_raises(stub_pattern: _StubPattern) -> None:
    request = httpx.Request("GET", "https://upstream.test/x")
    response = httpx.Response(403, request=request)
    stub_pattern._raise = httpx.HTTPStatusError("403 forbidden", request=request, response=response)

    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    with pytest.raises(httpx.HTTPStatusError, match="403"):
        _evaluate_sensor(sensor_def)


def test_empty_result_with_existing_partitions_keeps_them(
    stub_pattern: _StubPattern,
) -> None:
    """Per #216: empty discovery result on a previously-non-empty
    source is suspect (config drift, auth expiration).  Log warning;
    do NOT remove existing partitions."""
    stub_pattern._specs = []  # empty discovery

    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    from dagster import DagsterInstance, SkipReason, build_sensor_context

    instance = DagsterInstance.ephemeral()
    instance.add_dynamic_partitions(partitions.name, ["2026AA", "2026AB"])

    context = build_sensor_context(instance=instance, resources={"secrets": MagicMock()})
    result = sensor_def(context)

    assert isinstance(result, SkipReason)
    # Existing partitions kept
    keys = set(instance.get_dynamic_partitions(partitions.name))
    assert keys == {"2026AA", "2026AB"}


def test_generic_exception_skips_tick(stub_pattern: _StubPattern) -> None:
    """Defensive: an unexpected exception (e.g. ValueError from a
    broken resolver) skips the tick rather than crashing the daemon."""
    stub_pattern._raise = ValueError("resolver returned malformed payload")

    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    from dagster import SkipReason

    result, _ = _evaluate_sensor(sensor_def)
    assert isinstance(result, SkipReason)


# ---------------------------------------------------------------------------
# emit_run_requests opt-in
# ---------------------------------------------------------------------------


def test_emit_run_requests_default_off(stub_pattern: _StubPattern) -> None:
    """Default behavior: discovery adds keys but does NOT trigger
    materialization.  Operators wire materialization separately so a
    discovery tick doesn't accidentally schedule a 5+ GB UMLS download."""
    stub_pattern._specs = [PartitionSpec(key="2026AA")]

    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(_contract(), job, partitions_def=partitions)

    from dagster import SkipReason

    result, _ = _evaluate_sensor(sensor_def)
    assert isinstance(result, SkipReason)


def test_emit_run_requests_true_yields_run_requests(
    stub_pattern: _StubPattern,
) -> None:
    stub_pattern._specs = [
        PartitionSpec(key="2026AA"),
        PartitionSpec(key="2026AB"),
    ]

    job, partitions = _make_job()
    sensor_def = build_discovery_sensor(
        _contract(), job, partitions_def=partitions, emit_run_requests=True
    )

    from dagster import SensorResult

    result, _ = _evaluate_sensor(sensor_def)
    assert isinstance(result, SensorResult)
    assert result.run_requests is not None
    keys = {rr.partition_key for rr in result.run_requests}
    assert keys == {"2026AA", "2026AB"}


# ---------------------------------------------------------------------------
# Catalogue re-export (per resolved decision #4 in #216)
# ---------------------------------------------------------------------------


def test_top_level_sensors_module_re_exports_build_discovery_sensor() -> None:
    """The hybrid sensor-location decision: implementation lives in
    the ingest subpackage; top-level ``moncpipelib.sensors`` re-exports
    for catalogue discoverability."""
    from moncpipelib.sensors import build_discovery_sensor as catalogue_export

    assert catalogue_export is build_discovery_sensor


def test_top_level_package_re_exports_build_discovery_sensor() -> None:
    """The top-level package also surfaces the sensor for one-import
    convenience: ``from moncpipelib import build_discovery_sensor``."""
    import moncpipelib

    assert hasattr(moncpipelib, "build_discovery_sensor")
    assert moncpipelib.build_discovery_sensor is build_discovery_sensor
