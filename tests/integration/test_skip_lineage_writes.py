"""Integration tests for #420 test-mode lineage isolation.

An integration-test / ephemeral run redirects its *sink* to an isolated
schema, but pre-#420 the write's lineage side-effects still hit the shared
``lineage`` schema -- most damagingly a ``silver_materialized`` stamp on the
real ``period_registry``, which made the environment's sensor silently skip
the first real materialization.

With ``MONCPIPELIB_SKIP_LINEAGE_WRITES=1`` the data write must succeed
unchanged while every lineage / period-registry write is a logged no-op.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from datetime import date
from typing import Any

import polars as pl
import psycopg
import pytest

from moncpipelib.config import SKIP_LINEAGE_WRITES_ENV
from moncpipelib.io_managers.postgres import PostgresIOManager
from moncpipelib.resources.postgres import PostgresResource

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


@pytest.fixture()
def lineage_table(pg_connection: psycopg.Connection) -> str:
    """Minimal ``lineage.data_lineage`` (mirrors the atomicity-test shape)."""
    fqn = "lineage.data_lineage"
    with pg_connection.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS lineage")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lineage.data_lineage (
                lineage_id          uuid PRIMARY KEY,
                lineage_key         text NOT NULL,
                run_id              text NOT NULL,
                asset_name          text NOT NULL,
                pipeline_id         uuid,
                layer               text NOT NULL,
                source_file         text,
                source_system       text,
                data_date           date,
                data_date_range     daterange,
                row_count           integer,
                is_backfill         boolean NOT NULL DEFAULT FALSE,
                backfill_reason     text,
                backfill_id         text,
                replaces_lineage_id uuid,
                parent_lineage_ids  uuid[],
                transformation_type text,
                metadata            jsonb,
                processed_at        timestamptz NOT NULL DEFAULT NOW(),
                created_by          text DEFAULT CURRENT_USER
            )
            """
        )
        cur.execute("DELETE FROM lineage.data_lineage WHERE asset_name LIKE 'skiplineage_%'")
    pg_connection.commit()
    return fqn


@pytest.fixture()
def period_registry_table(pg_connection: psycopg.Connection) -> Iterator[str]:
    """Minimal ``lineage.period_registry`` (mirrors the registry-test shape)."""
    fqn = "lineage.period_registry"
    with pg_connection.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS lineage")
        cur.execute(
            """
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
                CONSTRAINT uq_period_registry_source_partition_skiplineage
                    UNIQUE (source_id, partition_key)
            )
            """
        )
        cur.execute("TRUNCATE lineage.period_registry")
    pg_connection.commit()
    yield fqn
    pg_connection.rollback()
    with pg_connection.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS lineage.period_registry CASCADE")
    pg_connection.commit()


class TestWriteSkipsLineage:
    """A write under the skip env lands data but no lineage row."""

    TABLE_NAME: str = f"skiplineage_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        pg_connection: psycopg.Connection,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        lineage_table: str,  # noqa: ARG002 -- creates lineage.data_lineage
    ) -> Any:
        # No REFERENCES clause: test harnesses clone target tables with FKs
        # stripped (#426), and skip mode attaches a real generated id that
        # references no data_lineage row -- an enforced FK here would model
        # a sink skip mode deliberately does not support.
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "name": "TEXT",
                "_lineage_id": "uuid",
                "_lineage_key": "text",
            },
            primary_key=["id"],
        )
        self.builder = table_builder
        self.pg_conn = pg_connection
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=True,
            add_metadata_columns=False,
        )
        yield
        self.builder.drop(self.fqn)

    def _count_lineage_for_asset(self) -> int:
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM lineage.data_lineage WHERE asset_name = %s",
                (self.TABLE_NAME,),
            )
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    def test_data_written_no_lineage_row_and_warning_logged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh", "layer_override": "bronze"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        self.io_mgr.handle_output(ctx, df)

        # Data landed; lineage did not.
        assert self.builder.count(self.fqn) == 3
        assert self._count_lineage_for_asset() == 0

        # Production shape (#424, #426): every row carries the real
        # generated _lineage_id and _lineage_key; the id references no
        # data_lineage row (asserted empty above).
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) FILTER (WHERE _lineage_id IS NULL), "  # noqa: S608
                f"count(*) FILTER (WHERE _lineage_key IS NULL) "
                f"FROM {self.fqn}"
            )
            row = cur.fetchone()
            assert row is not None
            assert row == (0, 0)

        # The skip is announced at WARNING level in the run log.
        warnings = [str(c.args[0]) for c in ctx.log.warning.call_args_list if c.args]
        assert any(SKIP_LINEAGE_WRITES_ENV in w for w in warnings)

    def test_without_env_lineage_row_is_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control: the same write without the env produces a lineage row."""
        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh", "layer_override": "bronze"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 3
        assert self._count_lineage_for_asset() == 1


class TestSkipModeKeepsProductionWriteShape:
    """#424/#426: skip mode keeps byte-for-byte production write shape.

    Mirrors the consumer integration-harness failures (data-platform#1017):
    a lineage-enabled resource with ``add_metadata_columns=True`` (the
    default) writing to a production-shaped sink -- ``_lineage_id`` and
    ``_lineage_key`` both NOT NULL (the dim_ndc posture), no ``_{layer}_*``
    metadata columns, FK stripped as consumer harnesses clone tables.

    Pre-#424, skip mode injected ``_gold_run_id`` / ``_gold_processed_at``
    / ``_source_file`` and column validation rejected the write ("Columns
    in DataFrame but not in table"). Pre-#426, skip mode attached a NULL
    ``_lineage_id``, which NOT NULL sinks (directly, or via the UPSERT
    staging table's LIKE-clone of the target) rejected with
    ``NotNullViolation ... relation "_ups_stage"``.
    """

    TABLE_NAME: str = f"skiplineage_shape_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(autouse=True)
    def setup(
        self,
        pg_connection: psycopg.Connection,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
        lineage_table: str,  # noqa: ARG002 -- creates lineage.data_lineage
    ) -> Any:
        self.fqn = table_builder.create_table(
            self.TABLE_NAME,
            columns={
                "id": "INTEGER NOT NULL",
                "name": "TEXT",
                "_lineage_id": "uuid NOT NULL",
                "_lineage_key": "text NOT NULL",
            },
            primary_key=["id"],
        )
        self.builder = table_builder
        self.pg_conn = pg_connection
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=True,
            add_metadata_columns=True,
        )
        yield
        self.builder.drop(self.fqn)

    def _shape_counts(self) -> tuple[int, int]:
        """(rows with null _lineage_id, rows with null _lineage_key)."""
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) FILTER (WHERE _lineage_id IS NULL), "  # noqa: S608
                f"count(*) FILTER (WHERE _lineage_key IS NULL) "
                f"FROM {self.fqn}"
            )
            row = cur.fetchone()
            assert row is not None
            return int(row[0]), int(row[1])

    def _count_lineage_for_asset(self) -> int:
        self.pg_conn.rollback()
        with self.pg_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM lineage.data_lineage WHERE asset_name = %s",
                (self.TABLE_NAME,),
            )
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    def test_skip_write_succeeds_with_lineage_columns_not_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh", "layer_override": "gold"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        # Pre-#424 this raised: Column mismatch ... Columns in DataFrame
        # but not in table: ['_gold_processed_at', '_gold_run_id',
        # '_source_file']. Pre-#426 it raised NotNullViolation on
        # _lineage_id.
        self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 3
        assert self._shape_counts() == (0, 0)
        assert self._count_lineage_for_asset() == 0

    def test_skip_upsert_exercises_like_cloned_staging(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """UPSERT stages through _ups_stage (LIKE target, NOT NULL cloned).

        Pins the exact data-platform#1016 failure path: the staging table
        inherits the target's NOT NULL _lineage_id, so a NULL id dies in
        the stage even before touching the target.
        """
        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={
                "write_mode": "upsert",
                "primary_key": ["id"],
                "layer_override": "gold",
            },
        )

        self.io_mgr.handle_output(ctx, pl.DataFrame({"id": [1, 2], "name": ["a", "b"]}))
        self.io_mgr.handle_output(ctx, pl.DataFrame({"id": [2, 3], "name": ["B", "c"]}))

        assert self.builder.count(self.fqn) == 3
        assert self._shape_counts() == (0, 0)
        assert self._count_lineage_for_asset() == 0

    def test_control_without_env_writes_real_ids_and_lineage_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: the same write without the env lands ids + lineage row."""
        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        ctx = make_mock_output_context(
            asset_name=self.TABLE_NAME,
            metadata={"write_mode": "full_refresh", "layer_override": "gold"},
        )
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        self.io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 3
        assert self._shape_counts() == (0, 0)
        assert self._count_lineage_for_asset() == 1


class TestRegisterPeriodSkipped:
    """Standalone ``register_period`` is a no-op under the skip env."""

    SOURCE_ID = "0195c1de-4242-7000-8000-00000000feed"

    def test_skip_leaves_registry_empty_then_control_writes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        postgres_resource: PostgresResource,
        pg_connection: psycopg.Connection,
        period_registry_table: str,  # noqa: ARG002 -- creates the table
    ) -> None:
        kwargs: dict[str, Any] = {
            "source_id": self.SOURCE_ID,
            "partition_key": "2026-07-01",
            "effective_from": date(2026, 7, 1),
            "effective_to": None,
            "source_uri": None,
            "status": "materialized",
            "registered_by": "skiplineage_test",
            "metadata": None,
        }

        def _count() -> int:
            pg_connection.rollback()
            with pg_connection.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM lineage.period_registry WHERE source_id = %s",
                    (self.SOURCE_ID,),
                )
                row = cur.fetchone()
                assert row is not None
                return int(row[0])

        monkeypatch.setenv(SKIP_LINEAGE_WRITES_ENV, "1")
        postgres_resource.register_period(**kwargs)
        assert _count() == 0, "skip mode must not touch period_registry"

        monkeypatch.delenv(SKIP_LINEAGE_WRITES_ENV, raising=False)
        postgres_resource.register_period(**kwargs)
        assert _count() == 1, "control write must land once env is unset"
