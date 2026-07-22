"""Tests for Dagster job factories."""

from __future__ import annotations

from unittest.mock import MagicMock

from moncpipelib.jobs import (
    make_reconciliation_asset,
    make_reconciliation_bundle,
    make_reconciliation_job,
)

SOURCE_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def _make_contract() -> MagicMock:
    contract = MagicMock()
    contract.asset = "cms_asp_ndc_crosswalk"
    contract.sinks = [
        {
            "type": "table",
            "schema": "silver",
            "table": "cms_asp_ndc_crosswalk",
            "mode": "scd2",
            "business_key": ["hcpcs_code", "ndc"],
        }
    ]
    return contract


class TestMakeReconciliationJob:
    """Tests for make_reconciliation_job() factory."""

    def test_returns_job_definition(self) -> None:
        from dagster import JobDefinition

        job = make_reconciliation_job(
            contract=_make_contract(),
            source_id=SOURCE_ID,
        )
        assert isinstance(job, JobDefinition)

    def test_default_name(self) -> None:
        job = make_reconciliation_job(
            contract=_make_contract(),
            source_id=SOURCE_ID,
        )
        assert job.name == "reconcile_cms_asp_ndc_crosswalk"

    def test_custom_name(self) -> None:
        job = make_reconciliation_job(
            contract=_make_contract(),
            source_id=SOURCE_ID,
            name="my_reconcile_job",
        )
        assert job.name == "my_reconcile_job"

    def test_custom_tags(self) -> None:
        job = make_reconciliation_job(
            contract=_make_contract(),
            source_id=SOURCE_ID,
            tags={"team": "data-eng"},
        )
        assert job.tags.get("team") == "data-eng"

    def test_description(self) -> None:
        job = make_reconciliation_job(
            contract=_make_contract(),
            source_id=SOURCE_ID,
        )
        assert "cms_asp_ndc_crosswalk" in (job.description or "")


class TestMakeReconciliationAsset:
    """Tests for make_reconciliation_asset() factory."""

    def test_returns_assets_definition(self) -> None:
        from dagster import AssetsDefinition

        asset_def = make_reconciliation_asset(
            contract=_make_contract(),
            source_id=SOURCE_ID,
        )
        assert isinstance(asset_def, AssetsDefinition)

    def test_default_key_from_contract(self) -> None:
        from dagster import AssetKey

        asset_def = make_reconciliation_asset(
            contract=_make_contract(),
            source_id=SOURCE_ID,
        )
        keys = list(asset_def.keys)
        assert len(keys) == 1
        assert keys[0] == AssetKey(["silver", "cms_asp_ndc_crosswalk_reconciled"])

    def test_custom_key(self) -> None:
        from dagster import AssetKey

        asset_def = make_reconciliation_asset(
            contract=_make_contract(),
            source_id=SOURCE_ID,
            key=AssetKey(["gold", "my_reconciled"]),
        )
        keys = list(asset_def.keys)
        assert keys[0] == AssetKey(["gold", "my_reconciled"])

    def test_custom_key_from_sequence(self) -> None:
        from dagster import AssetKey

        asset_def = make_reconciliation_asset(
            contract=_make_contract(),
            source_id=SOURCE_ID,
            key=["gold", "my_reconciled"],
        )
        keys = list(asset_def.keys)
        assert keys[0] == AssetKey(["gold", "my_reconciled"])


class TestPerformReconciliationContext:
    """Issue #334 Bug 3a + follow-up: ``_perform_reconciliation``
    forwards the Dagster context to ``database.reconcile_scd2`` so the
    resource extracts ``run_id`` and ``asset_name`` from it.  The
    follow-up to the original Bug-3 fix replaced the narrow
    ``run_id: str`` parameter with a full ``context`` parameter to
    match the ``database.write(context=...)`` convention.
    """

    @staticmethod
    def _mock_database() -> MagicMock:
        """Database stub whose ``reconcile_scd2`` returns the shape
        the helper unpacks downstream."""
        database = MagicMock()
        database.reconcile_scd2.return_value = {
            "rows_timeline_updated": 0,
            "rows_collapsed": 0,
            "rows_renumbered": 0,
            "work_mem": None,
            "duration_seconds": 0.001,
        }
        # ``get_registry_periods`` returns no rows so the helper's
        # ``update_period_metadata`` loop is a no-op.
        database.get_registry_periods.return_value = []
        return database

    def test_perform_reconciliation_forwards_context_to_resource(self) -> None:
        """``_perform_reconciliation`` must pass its ``context`` argument
        through as ``reconcile_scd2(context=...)``.  The resource
        extracts ``run_id`` from it -- the helper itself stays
        Dagster-context-shape agnostic."""
        from moncpipelib.jobs import _perform_reconciliation

        database = self._mock_database()
        ctx = MagicMock()
        ctx.run_id = "dagster-run-xyz"

        _perform_reconciliation(
            database=database,
            contract=_make_contract(),
            source_id=SOURCE_ID,
            resolved_name="reconcile_cms_asp_ndc_crosswalk",
            collapse_duplicates=True,
            log=MagicMock(),
            context=ctx,
        )

        database.reconcile_scd2.assert_called_once()
        assert database.reconcile_scd2.call_args.kwargs["context"] is ctx


class TestMakeReconciliationBundle:
    """Tests for make_reconciliation_bundle() factory."""

    def test_returns_tuple(self) -> None:
        from dagster import AssetsDefinition, SensorDefinition

        result = make_reconciliation_bundle(
            contract=_make_contract(),
            source_id=SOURCE_ID,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        asset_def, sensor_def, job_def = result
        assert isinstance(asset_def, AssetsDefinition)
        assert isinstance(sensor_def, SensorDefinition)
        # define_asset_job returns UnresolvedAssetJobDefinition
        assert hasattr(job_def, "name")

    def test_sensor_name(self) -> None:
        _, sensor_def, _ = make_reconciliation_bundle(
            contract=_make_contract(),
            source_id=SOURCE_ID,
        )
        assert "reconciliation_sensor" in sensor_def.name

    def test_job_name(self) -> None:
        _, _, job_def = make_reconciliation_bundle(
            contract=_make_contract(),
            source_id=SOURCE_ID,
        )
        assert job_def.name == "reconcile_cms_asp_ndc_crosswalk_job"
