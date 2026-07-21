"""Unit tests for #420 test-mode lineage isolation.

``MONCPIPELIB_SKIP_LINEAGE_WRITES`` makes every lineage / period-registry
write a logged no-op so integration-test / ephemeral runs cannot mutate the
shared ``lineage`` schema. The sink redirect already isolates the data
write; these gates extend the same isolation to the side-effects:
``data_lineage``, ``contract_validation_runs``, ``column_metadata`` (PII
sync), ``period_registry``, ``pipeline_registry``,
``scd2_reconciliations``, and OpenLineage emission.
"""

from __future__ import annotations

import logging
from datetime import date
from unittest.mock import MagicMock

import polars as pl
import pytest

from moncpipelib.config import (
    SKIP_LINEAGE_WRITES_ENV,
    LineageDefaults,
    skip_lineage_writes,
)
from moncpipelib.resources._registry_helpers import (
    pipeline_registry_upsert_committed,
    update_period_metadata,
    upsert_registry_row,
)
from moncpipelib.resources.postgres import PostgresResource


@pytest.fixture
def resource() -> PostgresResource:
    return PostgresResource(
        host="localhost",
        port=5432,
        user="testuser",
        password="testpass",
        database="testdb",
    )


class TestSkipLineageWritesEnv:
    """``skip_lineage_writes`` parses the env var dynamically."""

    def test_unset_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        assert skip_lineage_writes() is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "On"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, value)
        assert skip_lineage_writes() is True

    @pytest.mark.parametrize("value", ["0", "false", "", "no thanks"])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, value)
        assert skip_lineage_writes() is False

    def test_read_dynamically_not_cached_at_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Harnesses that set the env var after import must be honored."""
        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        assert skip_lineage_writes() is False
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        assert skip_lineage_writes() is True


class TestResolveSkipLineage:
    """``_resolve_skip_lineage`` returns the flag and warns once per write."""

    def test_unset_returns_false_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, resource: PostgresResource
    ) -> None:
        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        wctx = MagicMock()

        assert resource._resolve_skip_lineage(wctx) is False
        wctx.log.warning.assert_not_called()

    def test_set_returns_true_and_warns(
        self, monkeypatch: pytest.MonkeyPatch, resource: PostgresResource
    ) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        wctx = MagicMock()
        wctx.asset_name = "reference_silver/rxclass_class_members"

        assert resource._resolve_skip_lineage(wctx) is True
        wctx.log.warning.assert_called_once()
        msg = wctx.log.warning.call_args.args[0]
        assert SKIP_LINEAGE_WRITES_ENV in msg
        assert "reference_silver/rxclass_class_members" in msg
        assert "#420" in msg


class TestSkipModeLineageColumns:
    """#424/#426: the skip-mode DataFrame shape mirrors a production lineage write."""

    def test_attaches_real_id_and_key(self) -> None:
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        lineage_id = "0197f6f0-0000-7000-8000-000000000042"
        key = "v1:reference/dim_ndc:gold:20260709190204:d428d6"

        out = PostgresResource._attach_skip_mode_lineage_columns(df, lineage_id, key)

        # Both managed columns carry real generated values on every row
        # (String-typed, COPY-compatible), so NOT NULL sink constraints
        # hold -- the pre-#426 NULL id died in NOT NULL sinks and their
        # LIKE-cloned UPSERT staging tables.
        assert out[LineageDefaults.ID_COLUMN].dtype == pl.String
        assert out[LineageDefaults.ID_COLUMN].to_list() == [lineage_id, lineage_id]
        assert out[LineageDefaults.KEY_COLUMN].to_list() == [key, key]
        # No layer metadata columns -- the production lineage path never
        # adds them, and neither may skip mode (the pre-#424 bug).
        assert out.columns == [*df.columns, LineageDefaults.ID_COLUMN, LineageDefaults.KEY_COLUMN]


class TestOpenLineageEmitterGate:
    """The OL emitter resolves to None in test-mode isolation."""

    def test_skip_returns_none_even_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            openlineage_url="http://marquez.example.internal:5000",
        )

        assert resource._get_openlineage_emitter() is None


class TestUpsertRegistryRowGate:
    """The single period-registry upsert chokepoint no-ops with a warning."""

    KWARGS = {
        "source_id": "7047289a-45e5-4ad2-8f82-15ef003ac0c3",
        "source_name": None,
        "partition_key": "2026-07-01",
        "effective_from": date(2026, 7, 1),
        "effective_to": None,
        "source_uri": None,
        "status": "materialized",
        "registered_by": None,
        "run_id": None,
        "pipeline_id": None,
        "metadata": None,
    }

    def test_skip_no_ops_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        cursor = MagicMock()

        with caplog.at_level(logging.WARNING, logger="moncpipelib.resources"):
            upsert_registry_row(cursor, **self.KWARGS)

        cursor.execute.assert_not_called()
        assert SKIP_LINEAGE_WRITES_ENV in caplog.text
        assert "2026-07-01" in caplog.text

    def test_unset_executes_upsert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        cursor = MagicMock()

        upsert_registry_row(cursor, **self.KWARGS)

        cursor.execute.assert_called_once()


class TestUpdatePeriodMetadataGate:
    """Metadata merges (e.g. reconcile stamps) no-op without a connection."""

    def test_skip_never_opens_connection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        resource = MagicMock()

        with caplog.at_level(logging.WARNING, logger="moncpipelib.resources"):
            update_period_metadata(
                resource,
                "7047289a-45e5-4ad2-8f82-15ef003ac0c3",
                "2026-07-01",
                {"reconciled_at": "2026-07-09T00:00:00+00:00"},
            )

        resource.get_connection_raw.assert_not_called()
        assert SKIP_LINEAGE_WRITES_ENV in caplog.text

    def test_unset_opens_connection_and_executes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        resource = MagicMock()

        update_period_metadata(
            resource,
            "7047289a-45e5-4ad2-8f82-15ef003ac0c3",
            "2026-07-01",
            {"reconciled_at": "2026-07-09T00:00:00+00:00"},
        )

        resource.get_connection_raw.assert_called_once()


class TestPipelineRegistryCommittedGate:
    """The write path's pipeline_registry upsert no-ops in skip mode."""

    def test_skip_never_opens_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        resource = MagicMock()
        contract = MagicMock()
        contract.pipeline_id = "0195c1de-0000-7000-8000-000000000001"

        pipeline_registry_upsert_committed(
            resource,
            loaded_contract=contract,
            wctx=MagicMock(),
            layer="silver",
        )

        resource.get_connection.assert_not_called()

    def test_unset_proceeds_to_registry_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        resource = MagicMock()
        # Fast path: registry table missing -> returns after the check.
        resource._check_pipeline_registry.return_value = False
        contract = MagicMock()
        contract.pipeline_id = "0195c1de-0000-7000-8000-000000000001"

        pipeline_registry_upsert_committed(
            resource,
            loaded_contract=contract,
            wctx=MagicMock(),
            layer="silver",
        )

        resource.get_connection.assert_called_once()
