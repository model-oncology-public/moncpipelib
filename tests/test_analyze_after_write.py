"""Unit tests for the post-write ANALYZE step.

Tracks public mirror issue model-oncology-public/moncpipelib#1: autovacuum
never autoanalyzes partitioned parents, so the write path refreshes their
aggregate statistics post-commit. These tests cover the gating matrix, the
PG18 ``ANALYZE ONLY`` version gate, and the never-fail-the-write contract,
all without a live database.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from psycopg.sql import Composed

from moncpipelib.io_managers.enums import WriteMode
from moncpipelib.resources._analyze_helpers import (
    VALID_ANALYZE_AFTER_WRITE,
    _write_changed,
    analyze_after_write,
    resolve_analyze_after_write,
)

PG18 = 180004
PG16 = 160002


def _make_conn(*, relkind: str | None, server_version: int = PG18) -> tuple[MagicMock, MagicMock]:
    """Mock psycopg connection whose first fetchone returns the relkind row."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    cursor.fetchone.return_value = None if relkind is None else (relkind,)
    conn.info.server_version = server_version
    return conn, cursor


def _make_context() -> MagicMock:
    context = MagicMock()
    context.log = MagicMock()
    return context


def _analyze_statements(cursor: MagicMock) -> list[str]:
    """Rendered ANALYZE statements executed on the cursor (excludes the relkind SELECT)."""
    return [
        c.args[0].as_string(None)
        for c in cursor.execute.call_args_list
        if isinstance(c.args[0], Composed)
    ]


CHANGED_STATS: dict[str, Any] = {"rows_inserted": 10, "rows_deleted": 0}


class TestResolveAnalyzeAfterWrite:
    """Effective-mode resolution: override wins, invalid values fail fast."""

    def test_defaults_to_resource_setting(self) -> None:
        assert resolve_analyze_after_write("partitioned", {}) == "partitioned"
        assert resolve_analyze_after_write("never", {"analyze_after_write": None}) == "never"

    def test_write_config_override_wins(self) -> None:
        assert resolve_analyze_after_write("never", {"analyze_after_write": "always"}) == "always"

    def test_invalid_override_raises(self) -> None:
        with pytest.raises(ValueError, match="analyze_after_write"):
            resolve_analyze_after_write("partitioned", {"analyze_after_write": "sometimes"})

    def test_invalid_resource_default_raises(self) -> None:
        with pytest.raises(ValueError, match="analyze_after_write"):
            resolve_analyze_after_write("yes", {})

    def test_valid_values_are_the_documented_three(self) -> None:
        assert {"never", "partitioned", "always"} == VALID_ANALYZE_AFTER_WRITE


class TestWriteChanged:
    """Change gate: writer counters when present, row count as fallback."""

    def test_all_zero_counters_is_unchanged(self) -> None:
        assert _write_changed({"rows_inserted": 0, "rows_deleted": 0}, row_count=500) is False

    def test_any_positive_counter_is_changed(self) -> None:
        assert _write_changed({"rows_inserted": 0, "rows_deleted": 3}, row_count=0) is True

    def test_no_counters_falls_back_to_row_count(self) -> None:
        assert _write_changed({}, row_count=1) is True
        assert _write_changed({}, row_count=0) is False

    def test_non_int_counters_are_ignored(self) -> None:
        assert _write_changed({"rows_inserted": "n/a"}, row_count=0) is False


class TestAnalyzeAfterWriteGating:
    """Skip paths: never / SCD2 / unchanged / ordinary table under 'partitioned'."""

    def test_mode_never_skips_without_touching_connection(self) -> None:
        conn, _ = _make_conn(relkind="p")
        action = analyze_after_write(
            conn,
            schema="reference_bronze",
            bare_table="npi_npidata",
            mode="never",
            write_mode=WriteMode.APPEND,
            stats=CHANGED_STATS,
            row_count=10,
            context=_make_context(),
        )
        assert action is None
        conn.cursor.assert_not_called()

    def test_scd2_skips_even_under_always(self) -> None:
        conn, _ = _make_conn(relkind="p")
        action = analyze_after_write(
            conn,
            schema="silver",
            bare_table="dim_provider",
            mode="always",
            write_mode=WriteMode.SCD2,
            stats={"rows_new": 5, "rows_expired": 2},
            row_count=7,
            context=_make_context(),
        )
        assert action is None
        conn.cursor.assert_not_called()

    def test_unchanged_write_skips(self) -> None:
        conn, _ = _make_conn(relkind="p")
        action = analyze_after_write(
            conn,
            schema="bronze",
            bare_table="t",
            mode="always",
            write_mode=WriteMode.APPEND,
            stats={"rows_inserted": 0},
            row_count=100,
            context=_make_context(),
        )
        assert action is None
        conn.cursor.assert_not_called()

    def test_ordinary_table_under_partitioned_mode_skips(self) -> None:
        conn, cursor = _make_conn(relkind="r")
        action = analyze_after_write(
            conn,
            schema="bronze",
            bare_table="t",
            mode="partitioned",
            write_mode=WriteMode.APPEND,
            stats=CHANGED_STATS,
            row_count=10,
            context=_make_context(),
        )
        assert action is None
        assert _analyze_statements(cursor) == []
        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()

    def test_missing_table_warns_and_skips(self) -> None:
        conn, cursor = _make_conn(relkind=None)
        context = _make_context()
        action = analyze_after_write(
            conn,
            schema="bronze",
            bare_table="ghost",
            mode="partitioned",
            write_mode=WriteMode.APPEND,
            stats=CHANGED_STATS,
            row_count=10,
            context=context,
        )
        assert action is None
        assert _analyze_statements(cursor) == []
        context.log.warning.assert_called_once()


class TestAnalyzeAfterWriteExecution:
    """Statement shape and version gate on the executing paths."""

    def test_partitioned_parent_pg18_uses_analyze_only(self) -> None:
        conn, cursor = _make_conn(relkind="p", server_version=PG18)
        action = analyze_after_write(
            conn,
            schema="reference_bronze",
            bare_table="npi_npidata",
            mode="partitioned",
            write_mode=WriteMode.FULL_REFRESH,
            stats=CHANGED_STATS,
            row_count=10,
            context=_make_context(),
        )
        assert action == "parent"
        assert _analyze_statements(cursor) == ['ANALYZE ONLY "reference_bronze"."npi_npidata"']
        conn.commit.assert_called_once()

    def test_partitioned_parent_pre_pg18_falls_back_to_recursive(self) -> None:
        conn, cursor = _make_conn(relkind="p", server_version=PG16)
        action = analyze_after_write(
            conn,
            schema="bronze",
            bare_table="events",
            mode="partitioned",
            write_mode=WriteMode.APPEND,
            stats=CHANGED_STATS,
            row_count=10,
            context=_make_context(),
        )
        assert action == "recursive"
        assert _analyze_statements(cursor) == ['ANALYZE "bronze"."events"']

    def test_ordinary_table_under_always_analyzes_plain(self) -> None:
        conn, cursor = _make_conn(relkind="r", server_version=PG18)
        action = analyze_after_write(
            conn,
            schema="bronze",
            bare_table="t",
            mode="always",
            write_mode=WriteMode.UPSERT,
            stats={"rows_upserted": 4},
            row_count=4,
            context=_make_context(),
        )
        assert action == "table"
        assert _analyze_statements(cursor) == ['ANALYZE "bronze"."t"']

    def test_relkind_lookup_is_parameterized(self) -> None:
        conn, cursor = _make_conn(relkind="p")
        analyze_after_write(
            conn,
            schema="bronze",
            bare_table="t",
            mode="partitioned",
            write_mode=WriteMode.APPEND,
            stats=CHANGED_STATS,
            row_count=10,
            context=_make_context(),
        )
        lookup = cursor.execute.call_args_list[0]
        assert lookup.args[1] == ("bronze", "t")


class TestAnalyzeAfterWriteNeverFails:
    """A failed ANALYZE warns and returns None -- the committed write stands."""

    def test_execute_failure_is_swallowed_and_rolls_back(self) -> None:
        conn, cursor = _make_conn(relkind="p")
        cursor.execute.side_effect = [None, RuntimeError("permission denied for table")]
        context = _make_context()
        action = analyze_after_write(
            conn,
            schema="bronze",
            bare_table="t",
            mode="partitioned",
            write_mode=WriteMode.APPEND,
            stats=CHANGED_STATS,
            row_count=10,
            context=context,
        )
        assert action is None
        conn.rollback.assert_called_once()
        context.log.warning.assert_called_once()

    def test_rollback_failure_is_also_swallowed(self) -> None:
        conn, cursor = _make_conn(relkind="p")
        cursor.execute.side_effect = RuntimeError("connection lost")
        conn.rollback.side_effect = RuntimeError("connection lost")
        context = _make_context()
        action = analyze_after_write(
            conn,
            schema="bronze",
            bare_table="t",
            mode="partitioned",
            write_mode=WriteMode.APPEND,
            stats=CHANGED_STATS,
            row_count=10,
            context=context,
        )
        assert action is None
        context.log.warning.assert_called_once()
