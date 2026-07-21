"""Unit tests for the upsert write-time guards.

The NULL-primary-key preflight (#401) runs before any SQL is issued, so it is
testable without a database: a NULL in any conflict-key column means the row
can never match ``ON CONFLICT`` (SQL NULL never equals NULL) and would insert
a fresh duplicate on every re-materialization (data-platform dim_hcpcs).

The skip_unchanged change-guard (mirror issue
model-oncology-public/moncpipelib#3) is likewise testable without a database:
its entire contract is the shape of the single merge statement, which a mock
cursor captures. Behavioral coverage against a real PostgreSQL lives in
``tests/integration/test_write_upsert_skip_unchanged.py``.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import polars as pl
import pytest

from moncpipelib.io_managers.enums import BulkInsertMethod, FullRefreshMethod
from moncpipelib.io_managers.writers import WriterConfig, execute_upsert


def _config() -> WriterConfig:
    return WriterConfig(
        bulk_insert_method=BulkInsertMethod.COPY,
        bulk_insert_threshold=1000,
        full_refresh_method=FullRefreshMethod.DELETE,
        full_refresh_threshold=100_000,
        insert_chunk_size=None,
    )


class TestUpsertNullPrimaryKeyGuard:
    def test_null_in_primary_key_raises_before_any_sql(self) -> None:
        cursor = MagicMock()
        df = pl.DataFrame(
            {
                "hcpcs_code": ["J1234", "J5678", "J9012"],
                "from_date": ["2026-01-01", None, "2026-03-01"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="from_date: 1 row"):
            execute_upsert(
                _config(),
                cursor,
                "reference_gold.dim_hcpcs",
                df,
                ["hcpcs_code", "from_date"],
                None,
                MagicMock(),
            )
        cursor.execute.assert_not_called()

    def test_multiple_null_key_columns_all_reported(self) -> None:
        cursor = MagicMock()
        df = pl.DataFrame({"a": [None, 1], "b": [None, None], "v": [1, 2]})
        with pytest.raises(ValueError) as exc:
            execute_upsert(_config(), cursor, "s.t", df, ["a", "b"], None, MagicMock())
        assert "a: 1 row(s)" in str(exc.value)
        assert "b: 2 row(s)" in str(exc.value)

    def test_non_null_keys_pass_the_guard(self) -> None:
        """Guard lets clean keys through; the mock cursor then receives SQL."""
        cursor = MagicMock()
        df = pl.DataFrame({"id": [1, 2], "v": [None, "x"]})  # NULLs in non-key ok
        # Downstream mock-driven failure is irrelevant to the guard.
        with contextlib.suppress(Exception):
            execute_upsert(_config(), cursor, "s.t", df, ["id"], None, MagicMock())
        assert cursor.execute.called

    def test_empty_dataframe_short_circuits(self) -> None:
        cursor = MagicMock()
        df = pl.DataFrame({"id": [], "v": []})
        result = execute_upsert(_config(), cursor, "s.t", df, ["id"], None, MagicMock())
        assert result == {"rows_upserted": 0}
        cursor.execute.assert_not_called()


def _merge_sql(cursor: MagicMock) -> str:
    """Return the single ``INSERT ... ON CONFLICT`` merge statement issued."""
    stmts = [c.args[0] for c in cursor.execute.call_args_list if "INSERT INTO" in c.args[0]]
    assert len(stmts) == 1, f"expected exactly one merge statement, got {stmts}"
    return stmts[0]


class TestUpsertSkipUnchangedGuard:
    """SQL shape of the opt-in skip_unchanged change-guard.

    The guard's contract is entirely in the merge statement: default-path SQL
    must stay byte-identical (no alias, no WHERE), and the guarded path must
    compare exactly the update columns with NULL-safe ``IS DISTINCT FROM``.
    """

    @staticmethod
    def _run(
        *,
        skip_unchanged: bool,
        update_columns: list[str] | None = None,
    ) -> MagicMock:
        cursor = MagicMock()
        # Stage introspection (SELECT * ... WHERE false) reads
        # cursor.description; an empty list means "no extra columns to drop".
        cursor.description = []
        df = pl.DataFrame({"id": [1, 2], "v1": ["a", None], "v2": [1.0, 2.0]})
        execute_upsert(
            _config(),
            cursor,
            "s.t",
            df,
            ["id"],
            update_columns,
            MagicMock(),
            skip_unchanged=skip_unchanged,
        )
        return cursor

    def test_default_merge_sql_is_byte_identical(self) -> None:
        """skip_unchanged=False (the default) pins today's exact merge SQL."""
        cursor = self._run(skip_unchanged=False)
        assert _merge_sql(cursor) == (
            'INSERT INTO s.t ("id", "v1", "v2") '
            'SELECT "id", "v1", "v2" FROM ('
            'SELECT DISTINCT ON ("id") "id", "v1", "v2" FROM _ups_stage '
            'ORDER BY "id", _ord DESC) d '
            'ON CONFLICT ("id") DO UPDATE SET "v1" = EXCLUDED."v1", "v2" = EXCLUDED."v2"'
        )

    def test_guard_adds_alias_and_null_safe_where(self) -> None:
        """skip_unchanged=True aliases the target and guards every update column."""
        cursor = self._run(skip_unchanged=True)
        assert _merge_sql(cursor) == (
            'INSERT INTO s.t AS _tgt ("id", "v1", "v2") '
            'SELECT "id", "v1", "v2" FROM ('
            'SELECT DISTINCT ON ("id") "id", "v1", "v2" FROM _ups_stage '
            'ORDER BY "id", _ord DESC) d '
            'ON CONFLICT ("id") DO UPDATE SET "v1" = EXCLUDED."v1", "v2" = EXCLUDED."v2" '
            'WHERE _tgt."v1" IS DISTINCT FROM EXCLUDED."v1" '
            'OR _tgt."v2" IS DISTINCT FROM EXCLUDED."v2"'
        )

    def test_guard_scoped_to_explicit_update_columns(self) -> None:
        """Guard compares only the declared update columns, never the key."""
        cursor = self._run(skip_unchanged=True, update_columns=["v1"])
        sql = _merge_sql(cursor)
        assert 'WHERE _tgt."v1" IS DISTINCT FROM EXCLUDED."v1"' in sql
        assert '"v2" IS DISTINCT FROM' not in sql
        assert '"id" IS DISTINCT FROM' not in sql

    def test_guard_inert_on_do_nothing(self) -> None:
        """Empty update_columns is DO NOTHING; the guard adds no alias/WHERE."""
        cursor = self._run(skip_unchanged=True, update_columns=[])
        sql = _merge_sql(cursor)
        assert sql.endswith('ON CONFLICT ("id") DO NOTHING')
        assert "AS _tgt" not in sql
        assert "WHERE" not in sql

    def test_rows_upserted_unaffected_by_guard(self) -> None:
        """The incoming-row-count stat keeps its meaning under the guard."""
        cursor = MagicMock()
        cursor.description = []
        df = pl.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]})
        result = execute_upsert(
            _config(), cursor, "s.t", df, ["id"], None, MagicMock(), skip_unchanged=True
        )
        assert result == {"rows_upserted": 3}
