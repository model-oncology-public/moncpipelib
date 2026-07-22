"""Integration tests for presence-only SCD2 (#432).

A junction/reference table whose business key is the full source tuple has
nothing to hash for attribute change detection. With ``tracked_columns``
empty (or omitted when the key covers every column), ``_prepare_scd2``
hashes the business key itself: the writer's change predicate can never
fire within a key, so versioning reduces to presence tracking -- a tuple's
appearance opens an ``effective_from`` span and its absence (via
``detect_deletes``) closes it.

These tests exercise the resource-first path (``database.write(...)``).

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import psycopg
import pytest

from moncpipelib.resources.postgres import PostgresResource
from moncpipelib.resources.types import WriteContext

from .conftest import SCD2TableBuilder, SCD2Verifier

pytestmark = pytest.mark.integration

TUPLES_V1 = pl.DataFrame(
    {
        "npi": ["1001", "1001", "1002"],
        "other_name": ["ACME CLINIC", "ACME MEDICAL", "BETA HEALTH"],
    }
)

# 1001/ACME MEDICAL vanishes; 1003/GAMMA CARE appears.
TUPLES_V2 = pl.DataFrame(
    {
        "npi": ["1001", "1002", "1003"],
        "other_name": ["ACME CLINIC", "BETA HEALTH", "GAMMA CARE"],
    }
)

# 1001/ACME MEDICAL reappears.
TUPLES_V3 = pl.DataFrame(
    {
        "npi": ["1001", "1001", "1002", "1003"],
        "other_name": ["ACME CLINIC", "ACME MEDICAL", "BETA HEALTH", "GAMMA CARE"],
    }
)

BUSINESS_KEY = ["npi", "other_name"]


class TestSCD2PresenceOnly:
    """Presence semantics: spans open on appearance, close on absence."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        scd2_table_builder: SCD2TableBuilder,
        scd2_verifier: SCD2Verifier,
        pg_connection: psycopg.Connection,
        postgres_resource: PostgresResource,
    ) -> Any:
        self.table_name = f"npi_other_name_{uuid.uuid4().hex[:8]}"
        self.fqn = scd2_table_builder.create_table(
            table_name=self.table_name,
            business_key_columns={"npi": "TEXT", "other_name": "TEXT"},
            tracked_columns={},
        )
        self.builder = scd2_table_builder
        self.verifier = scd2_verifier
        self.conn = pg_connection
        self.resource = postgres_resource
        yield
        self.builder.drop(self.fqn)

    # -- helpers ---------------------------------------------------------

    def _write(
        self,
        df: pl.DataFrame,
        *,
        tracked_columns: list[str] | None = None,
        detect_deletes: bool = True,
    ) -> MagicMock:
        wctx = WriteContext(
            asset_name=self.table_name,
            run_id=f"presence-only-{uuid.uuid4().hex[:8]}",
            log=MagicMock(),
        )
        self.resource.write(
            df,
            target=self.fqn,
            context=wctx,
            write_mode="scd2",
            business_key=BUSINESS_KEY,
            tracked_columns=tracked_columns,
            detect_deletes=detect_deletes,
            contract=None,
        )
        return wctx.log

    def _key_history(self, npi: str, other_name: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {self.fqn} "  # noqa: S608
                f"WHERE npi = %s AND other_name = %s ORDER BY effective_from, id",
                (npi, other_name),
            )
            cols = [d[0] for d in cur.description or []]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    # -- tests -----------------------------------------------------------

    def test_initial_load_opens_spans(self) -> None:
        """Explicit tracked_columns=[] writes successfully; all tuples current."""
        log = self._write(TUPLES_V1, tracked_columns=[])

        assert self.verifier.count_current(self.fqn) == 3
        assert self.verifier.count_expired(self.fqn) == 0
        info_msgs = [c.args[0] for c in log.info.call_args_list]
        assert any("presence-only" in m for m in info_msgs)

    def test_omitted_tracked_columns_also_presence_only(self) -> None:
        """tracked_columns omitted with key covering all columns also writes."""
        self._write(TUPLES_V1, tracked_columns=None)

        assert self.verifier.count_current(self.fqn) == 3

    def test_reload_same_tuples_is_noop(self) -> None:
        """Re-loading identical tuples creates no versions and expires nothing."""
        self._write(TUPLES_V1, tracked_columns=[])
        self._write(TUPLES_V1, tracked_columns=[])

        assert self.verifier.count_current(self.fqn) == 3
        assert self.verifier.count_expired(self.fqn) == 0
        assert self.verifier.count_total(self.fqn) == 3

    def test_absence_closes_span_and_appearance_opens_one(self) -> None:
        """A vanished tuple is expired; a new tuple opens a current span."""
        self._write(TUPLES_V1, tracked_columns=[])
        self._write(TUPLES_V2, tracked_columns=[])

        vanished = self._key_history("1001", "ACME MEDICAL")
        assert len(vanished) == 1
        assert vanished[0]["is_current"] is False
        assert vanished[0]["effective_to"] is not None

        appeared = self._key_history("1003", "GAMMA CARE")
        assert len(appeared) == 1
        assert appeared[0]["is_current"] is True

        # Tuples present in both loads are untouched (still one row, current).
        stable = self._key_history("1001", "ACME CLINIC")
        assert len(stable) == 1
        assert stable[0]["is_current"] is True

    def test_reappearance_opens_new_span(self) -> None:
        """A tuple that vanishes and returns gets a second, current span."""
        self._write(TUPLES_V1, tracked_columns=[])
        self._write(TUPLES_V2, tracked_columns=[])
        self._write(TUPLES_V3, tracked_columns=[])

        history = self._key_history("1001", "ACME MEDICAL")
        assert len(history) == 2
        closed, reopened = history
        assert closed["is_current"] is False
        assert reopened["is_current"] is True

    def test_without_detect_deletes_warns_and_never_expires(self) -> None:
        """Presence-only without detect_deletes inserts new keys but closes nothing."""
        log = self._write(TUPLES_V1, tracked_columns=[], detect_deletes=False)
        warning_msgs = [c.args[0] for c in log.warning.call_args_list]
        assert any("spans will never close" in m for m in warning_msgs)

        self._write(TUPLES_V2, tracked_columns=[], detect_deletes=False)

        # 1001/ACME MEDICAL is absent from V2 but stays current.
        assert self.verifier.count_expired(self.fqn) == 0
        assert self.verifier.count_current(self.fqn) == 4
