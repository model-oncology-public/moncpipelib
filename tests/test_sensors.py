"""Tests for Dagster sensor factories."""

from __future__ import annotations

import pytest

from moncpipelib.sensors import (
    period_registry_sensor,
    reconciliation_sensor,
    registry_sensor,
    scd2_registry_sensor,
)

SOURCE_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def _make_test_job():
    """Create a minimal Dagster job for sensor tests."""
    from dagster import DynamicPartitionsDefinition, asset, define_asset_job

    partitions = DynamicPartitionsDefinition(name="test_periods")

    @asset(partitions_def=partitions)
    def dummy_asset():
        pass

    job = define_asset_job("test_job", selection=[dummy_asset])
    return job, partitions


def _make_reconcile_job():
    """Create a minimal Dagster job for reconciliation sensor tests."""
    from dagster import job, op

    @op
    def reconcile_op():
        pass

    @job
    def reconcile_job():
        reconcile_op()

    return reconcile_job


class TestRegistrySensor:
    """Tests for the core registry_sensor() factory."""

    def test_returns_sensor_definition(self) -> None:
        job, partitions = _make_test_job()
        sensor_def = registry_sensor(
            source_id=SOURCE_ID,
            target_job=job,
            partitions_def=partitions,
        )
        assert sensor_def.name == "a1b2c3d4_e5f6_7890_abcd_ef1234567890_registry_sensor"

    def test_per_partition_requires_partitions_def(self) -> None:
        job, _ = _make_test_job()
        with pytest.raises(ValueError, match="partitions_def is required"):
            registry_sensor(
                source_id=SOURCE_ID,
                target_job=job,
                trigger_mode="per_partition",
            )

    def test_all_mode_no_partitions_def(self) -> None:
        job = _make_reconcile_job()
        sensor_def = registry_sensor(
            source_id=SOURCE_ID,
            target_job=job,
            trigger_mode="all",
        )
        assert "database" in sensor_def.required_resource_keys

    def test_custom_ready_when(self) -> None:
        job, partitions = _make_test_job()
        sensor_def = registry_sensor(
            source_id=SOURCE_ID,
            target_job=job,
            partitions_def=partitions,
            ready_when=lambda m: m.get("reconciled_at") is not None,
            name="custom_sensor",
        )
        assert sensor_def.name == "custom_sensor"

    def test_requires_database_resource(self) -> None:
        job, partitions = _make_test_job()
        sensor_def = registry_sensor(
            source_id=SOURCE_ID,
            target_job=job,
            partitions_def=partitions,
        )
        assert "database" in sensor_def.required_resource_keys

    def test_custom_interval(self) -> None:
        job, partitions = _make_test_job()
        sensor_def = registry_sensor(
            source_id=SOURCE_ID,
            target_job=job,
            partitions_def=partitions,
            minimum_interval_seconds=60,
        )
        assert sensor_def.minimum_interval_seconds == 60


class TestPeriodRegistrySensor:
    """Tests for period_registry_sensor() thin wrapper."""

    def _make_sensor(self, **kwargs):  # type: ignore[no-untyped-def]
        job, partitions = _make_test_job()
        return (
            period_registry_sensor(
                source_id=SOURCE_ID,
                target_job=job,
                partitions_def=partitions,
                **kwargs,
            ),
            partitions,
        )

    def test_returns_sensor_definition(self) -> None:
        sensor_def, _ = self._make_sensor()
        assert sensor_def.name == "a1b2c3d4_e5f6_7890_abcd_ef1234567890_period_sensor"

    def test_custom_name(self) -> None:
        sensor_def, _ = self._make_sensor(name="my_sensor")
        assert sensor_def.name == "my_sensor"

    def test_default_description(self) -> None:
        sensor_def, _ = self._make_sensor()
        assert SOURCE_ID in (sensor_def.description or "")

    def test_custom_description(self) -> None:
        sensor_def, _ = self._make_sensor(description="Custom desc")
        assert sensor_def.description == "Custom desc"

    def test_requires_database_resource(self) -> None:
        sensor_def, _ = self._make_sensor()
        assert "database" in sensor_def.required_resource_keys


class TestReconciliationSensor:
    """Tests for reconciliation_sensor() thin wrapper."""

    def _make_sensor(self, **kwargs):  # type: ignore[no-untyped-def]
        job = _make_reconcile_job()
        return reconciliation_sensor(
            source_id=SOURCE_ID,
            target_job=job,
            **kwargs,
        )

    def test_returns_sensor_definition(self) -> None:
        sensor_def = self._make_sensor()
        assert sensor_def.name == "a1b2c3d4_e5f6_7890_abcd_ef1234567890_reconciliation_sensor"

    def test_custom_name(self) -> None:
        sensor_def = self._make_sensor(name="my_reconcile_sensor")
        assert sensor_def.name == "my_reconcile_sensor"

    def test_requires_database_resource(self) -> None:
        sensor_def = self._make_sensor()
        assert "database" in sensor_def.required_resource_keys


class TestSCD2RegistrySensor:
    """Tests for scd2_registry_sensor() thin wrapper."""

    def _make_sensor(self, **kwargs):  # type: ignore[no-untyped-def]
        job, partitions = _make_test_job()
        return (
            scd2_registry_sensor(
                source_id=SOURCE_ID,
                target_job=job,
                partitions_def=partitions,
                **kwargs,
            ),
            partitions,
        )

    def test_default_name(self) -> None:
        sensor_def, _ = self._make_sensor()
        assert sensor_def.name == "a1b2c3d4_e5f6_7890_abcd_ef1234567890_scd2_sensor"

    def test_custom_name(self) -> None:
        sensor_def, _ = self._make_sensor(name="my_scd2_sensor")
        assert sensor_def.name == "my_scd2_sensor"

    def test_requires_database_resource(self) -> None:
        sensor_def, _ = self._make_sensor()
        assert "database" in sensor_def.required_resource_keys
