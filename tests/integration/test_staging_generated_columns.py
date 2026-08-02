"""Integration tests for staging behavior on tables with database-computed columns (#400 / #401).

``CREATE TEMP TABLE (LIKE target)`` copies NOT NULL constraints but not
GENERATED expressions, identity, or (without INCLUDING DEFAULTS) default
expressions. Any target column the DataFrame does not supply therefore staged
as a plain NOT NULL column with no way to receive a value, failing the COPY
with NotNullViolation even though the final INSERT into the target computes
or defaults it fine. #400's repro is reference_gold.dim_histology's
``eligibility_section`` (a total CASE over a NOT NULL boolean); the identity
and server-default variants are the same failure family.

Also covers the SCD2 index-shape preflight (#401 item 4): a plain unique
index on (business_key, is_current) instead of the partial
``UNIQUE ... WHERE (is_current)`` form is armed to raise UniqueViolation the
second time any key expires; the writer now warns at first write.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import polars as pl
import psycopg
import pytest

from moncpipelib.io_managers.postgres import PostgresIOManager

from .conftest import SCD2TableBuilder, SCD2Verifier, TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Upsert staging vs database-computed columns (#400)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUpsertStagingGeneratedColumns:
    """Upsert staging drops target columns the DataFrame does not supply."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
        )
        self.fqns: list[str] = []
        yield
        for fqn in self.fqns:
            self.builder.drop(fqn)

    def _create(self, name: str, columns: dict[str, str], **kwargs: Any) -> str:
        fqn = self.builder.create_table(name, columns=columns, **kwargs)
        self.fqns.append(fqn)
        return fqn

    def test_upsert_with_stored_generated_not_null_column(self) -> None:
        """The #400 repro: GENERATED ALWAYS ... STORED, NOT NULL target column.

        Before the fix, staging cloned the column as plain NOT NULL with no
        generation expression and the COPY failed with NotNullViolation.
        """
        name = f"ups_gen_{uuid.uuid4().hex[:8]}"
        fqn = self._create(
            name,
            columns={
                "id": "INTEGER NOT NULL",
                "eligible": "BOOLEAN NOT NULL",
                "eligibility_section": (
                    "TEXT GENERATED ALWAYS AS "
                    "(CASE WHEN eligible THEN 'ELIGIBLE' ELSE 'INELIGIBLE' END) "
                    "STORED NOT NULL"
                ),
            },
            primary_key=["id"],
        )

        ctx = make_mock_output_context(
            asset_name=name,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": [1, 2], "eligible": [True, False]})
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(fqn, order_by="id")
        assert [r["eligibility_section"] for r in rows] == ["ELIGIBLE", "INELIGIBLE"]

        # Second materialization exercises the ON CONFLICT UPDATE path; the
        # generated value must recompute from the updated dependency.
        ctx2 = make_mock_output_context(
            asset_name=name,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df2 = pl.DataFrame({"id": [1, 2], "eligible": [False, False]})
        self.io_mgr.handle_output(ctx2, df2)

        rows = self.builder.read_all(fqn, order_by="id")
        assert [r["eligibility_section"] for r in rows] == ["INELIGIBLE", "INELIGIBLE"]

    def test_upsert_with_identity_surrogate_and_natural_key_conflict(self) -> None:
        """Identity surrogate not in the DataFrame; conflict on a unique natural key."""
        name = f"ups_ident_{uuid.uuid4().hex[:8]}"
        fqn = self._create(
            name,
            columns={
                "id": "BIGINT GENERATED ALWAYS AS IDENTITY",
                "code": "TEXT NOT NULL UNIQUE",
                "val": "TEXT",
            },
        )

        ctx = make_mock_output_context(
            asset_name=name,
            metadata={"write_mode": "upsert", "primary_key": ["code"]},
        )
        df = pl.DataFrame({"code": ["a", "b"], "val": ["one", "two"]})
        self.io_mgr.handle_output(ctx, df)

        ctx2 = make_mock_output_context(
            asset_name=name,
            metadata={"write_mode": "upsert", "primary_key": ["code"]},
        )
        df2 = pl.DataFrame({"code": ["b", "c"], "val": ["TWO", "three"]})
        self.io_mgr.handle_output(ctx2, df2)

        rows = self.builder.read_all(fqn, order_by="code")
        assert [(r["code"], r["val"]) for r in rows] == [("a", "one"), ("b", "TWO"), ("c", "three")]
        # Surrogate ids were assigned by the target's identity, never staged.
        assert all(r["id"] is not None for r in rows)

    def test_upsert_with_not_null_default_column_omitted(self) -> None:
        """A NOT NULL DEFAULT column absent from the DataFrame defaults on insert."""
        name = f"ups_dflt_{uuid.uuid4().hex[:8]}"
        fqn = self._create(
            name,
            columns={
                "id": "INTEGER NOT NULL",
                "name": "TEXT",
                "created_at": "TIMESTAMPTZ NOT NULL DEFAULT now()",
            },
            primary_key=["id"],
        )

        ctx = make_mock_output_context(
            asset_name=name,
            metadata={"write_mode": "upsert", "primary_key": ["id"]},
        )
        df = pl.DataFrame({"id": [1], "name": ["x"]})
        self.io_mgr.handle_output(ctx, df)

        rows = self.builder.read_all(fqn)
        assert rows[0]["created_at"] is not None

    def test_upsert_null_primary_key_rejected(self) -> None:
        """NULL conflict keys never match ON CONFLICT; the write must fail fast."""
        name = f"ups_nullpk_{uuid.uuid4().hex[:8]}"
        self._create(
            name,
            columns={"code": "TEXT", "from_date": "DATE", "val": "TEXT"},
        )

        ctx = make_mock_output_context(
            asset_name=name,
            metadata={"write_mode": "upsert", "primary_key": ["code", "from_date"]},
        )
        df = pl.DataFrame(
            {
                "code": ["a", "b"],
                "from_date": [None, "2026-01-01"],
                "val": ["x", "y"],
            }
        )
        with pytest.raises(ValueError, match="NULL"):
            self.io_mgr.handle_output(ctx, df)


# ---------------------------------------------------------------------------
# SCD2 staging vs generated columns + index-shape preflight (#401)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestScd2GeneratedColumnsAndIndexShape:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        scd2_table_builder: SCD2TableBuilder,
        scd2_verifier: SCD2Verifier,
        pg_connection: psycopg.Connection,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.builder = scd2_table_builder
        self.verifier = scd2_verifier
        self.conn = pg_connection
        self.io_mgr_factory = io_manager_factory
        self.fqns: list[str] = []
        yield
        for fqn in self.fqns:
            self.builder.drop(fqn)

    def _scd2_write(self, table_name: str, df: pl.DataFrame) -> Any:
        io_mgr = self.io_mgr_factory()
        ctx = make_mock_output_context(
            asset_name=table_name,
            metadata={"write_mode": "scd2", "business_key": ["product_id"]},
        )
        io_mgr.handle_output(ctx, df)
        return ctx

    def test_scd2_target_with_stored_generated_not_null_column(self) -> None:
        """SCD2 staging drops GENERATED columns like it already drops identity."""
        table_name = f"dim_gen_{uuid.uuid4().hex[:8]}"
        fqn = self.builder.create_table(
            table_name=table_name,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"price": "NUMERIC"},
            extra_columns={
                "price_band": (
                    "TEXT GENERATED ALWAYS AS "
                    "(CASE WHEN price > 100 THEN 'high' ELSE 'low' END) STORED NOT NULL"
                )
            },
        )
        self.fqns.append(fqn)

        df = pl.DataFrame({"product_id": ["P1", "P2"], "price": [50.0, 150.0]})
        self._scd2_write(table_name, df)
        assert self.verifier.count_current(fqn) == 2

        row = self.verifier.get_current_row(fqn, "product_id", "P2")
        assert row is not None
        assert row["price_band"] == "high"

        # A change run exercises expire + insert against the generated column.
        df2 = pl.DataFrame({"product_id": ["P1", "P2"], "price": [500.0, 150.0]})
        self._scd2_write(table_name, df2)

        row = self.verifier.get_current_row(fqn, "product_id", "P1")
        assert row is not None
        assert row["price_band"] == "high"

    def test_index_shape_preflight_warns_on_non_partial_unique(self) -> None:
        """UNIQUE (bk, is_current) without a WHERE clause draws the #401 warning."""
        table_name = f"dim_badidx_{uuid.uuid4().hex[:8]}"
        fqn = self.builder.create_table(
            table_name=table_name,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"price": "NUMERIC"},
            include_unique_index=False,
        )
        self.fqns.append(fqn)
        with self.conn.cursor() as cur:
            cur.execute(
                f"CREATE UNIQUE INDEX uq_{table_name}_bad ON {fqn} (product_id, is_current)"
            )
        self.conn.commit()

        df = pl.DataFrame({"product_id": ["P1"], "price": [1.0]})
        ctx = self._scd2_write(table_name, df)

        warnings = [str(c.args[0]) for c in ctx.log.warning.call_args_list]
        assert any("non-partial UNIQUE index" in w for w in warnings), warnings

    def test_index_shape_preflight_silent_on_partial_unique(self) -> None:
        """The documented partial-index shape draws no warning."""
        table_name = f"dim_goodidx_{uuid.uuid4().hex[:8]}"
        fqn = self.builder.create_table(
            table_name=table_name,
            business_key_columns={"product_id": "TEXT"},
            tracked_columns={"price": "NUMERIC"},
            include_unique_index=True,
        )
        self.fqns.append(fqn)

        df = pl.DataFrame({"product_id": ["P1"], "price": [1.0]})
        ctx = self._scd2_write(table_name, df)

        warnings = [str(c.args[0]) for c in ctx.log.warning.call_args_list]
        assert not any("non-partial UNIQUE index" in w for w in warnings), warnings
