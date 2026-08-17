"""Integration regression tests for the period registry upsert path.

Regression for #268: ``_upsert_registry_row`` previously emitted SQL with
a bare ``metadata`` reference inside the ``ON CONFLICT DO UPDATE SET``
clause. PostgreSQL raises ``42702 column reference is ambiguous`` at
parse time -- so any second upsert against the same
``(source_id, partition_key)`` failed silently (the call site catches
and warns), leaving ``lineage.period_registry`` empty for re-materialized
``from_ingest`` partitions.

These tests exercise the helper against a real PostgreSQL container so
the parser-level error surfaces immediately if the bare reference is
re-introduced. The unit tests in ``tests/test_postgres_resource.py``
mock the cursor and cannot catch this regression.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import date
from typing import Any

import psycopg
import pytest

from moncpipelib.resources.postgres import PostgresResource

REGISTRY_DDL = """
CREATE SCHEMA IF NOT EXISTS lineage;

CREATE TABLE IF NOT EXISTS lineage.period_registry (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       uuid NOT NULL,
    partition_key   text NOT NULL,
    effective_from  date NOT NULL,
    effective_to    date,
    source_uri      text,
    status          text NOT NULL DEFAULT 'registered',
    registered_at   timestamptz NOT NULL DEFAULT now(),
    registered_by   text,
    metadata        jsonb,
    source_name     text,
    run_id          text,
    pipeline_id     uuid,
    CONSTRAINT uq_period_registry_source_partition
        UNIQUE (source_id, partition_key)
);
"""

# Stable UUIDs used as fixture source_ids. Pre-migration these were
# arbitrary strings ("regression-268", "regression-268-merge"); the
# column is now uuid so the values must parse. Generated once and
# pinned for stable test output.
SOURCE_ID_IDEMPOTENT = "0e9a268a-2680-4268-8268-026826802680"
SOURCE_ID_MERGE = "0e9a268a-2680-4268-8268-026826802681"


@pytest.mark.integration()
class TestPeriodRegistryUpsertRegression:
    """Regression: second upsert must not raise ``42702`` (issue #268)."""

    @pytest.fixture(autouse=True)
    def _period_registry_table(
        self,
        pg_connection: psycopg.Connection,
    ) -> Generator[None, None, None]:
        """Create the ``lineage.period_registry`` table for the duration of one test.

        Uses the production-default schema/table names so ``register_period``
        -- which reads ``config.period_registry`` -- finds the table without
        any monkeypatching.
        """
        pg_connection.autocommit = True
        with pg_connection.cursor() as cur:
            cur.execute(REGISTRY_DDL)
            cur.execute("TRUNCATE lineage.period_registry")
        yield
        with pg_connection.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS lineage.period_registry CASCADE")
        pg_connection.autocommit = False

    def test_register_period_idempotent_with_metadata_none(
        self,
        postgres_resource: PostgresResource,
        pg_connection: psycopg.Connection,
    ) -> None:
        """Two ``register_period`` calls with ``metadata=None`` must succeed.

        Reproduces the from_ingest bronze reload pattern: every write
        re-upserts the same ``(source_id, partition_key)`` with
        ``metadata=None``. Pre-fix this raised
        ``column reference "metadata" is ambiguous`` on the second call.
        """
        kwargs: dict[str, Any] = {
            "source_id": SOURCE_ID_IDEMPOTENT,
            "partition_key": "2026-05-04",
            "effective_from": date(2026, 5, 4),
            "effective_to": None,
            "source_uri": "rxnorm/2026-05-04/rrf/RXNCONSO.RRF",
            "status": "materialized",
            "registered_by": "regression_test",
            "metadata": None,
        }

        postgres_resource.register_period(**kwargs)
        postgres_resource.register_period(**kwargs)

        with pg_connection.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM lineage.period_registry "
                "WHERE source_id = %s AND partition_key = %s",
                (kwargs["source_id"], kwargs["partition_key"]),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == 1, "second upsert should update, not insert a duplicate"

    def test_register_period_merges_metadata_on_conflict(
        self,
        postgres_resource: PostgresResource,
        pg_connection: psycopg.Connection,
    ) -> None:
        """The conflict branch merges existing + incoming metadata via ``||``.

        Pins the semantic the f-string formatter encodes: existing
        metadata coalesced with ``{}`` is concatenated with the incoming
        metadata coalesced with ``{}``. A future change that reorders
        the operands or drops the COALESCE would silently lose keys.
        """
        common: dict[str, Any] = {
            "source_id": SOURCE_ID_MERGE,
            "partition_key": "2026-05-04",
            "effective_from": date(2026, 5, 4),
            "effective_to": None,
            "source_uri": "rxnorm/2026-05-04/rrf/RXNCONSO.RRF",
            "status": "materialized",
            "registered_by": "regression_test",
        }

        postgres_resource.register_period(**common, metadata={"first_key": 1})
        postgres_resource.register_period(**common, metadata={"second_key": 2})

        with pg_connection.cursor() as cur:
            cur.execute(
                "SELECT metadata FROM lineage.period_registry "
                "WHERE source_id = %s AND partition_key = %s",
                (common["source_id"], common["partition_key"]),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == {"first_key": 1, "second_key": 2}
