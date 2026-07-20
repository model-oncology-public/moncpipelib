"""Unit tests for full-refresh clear-method selection.

``FullRefreshMethod.AUTO`` picks TRUNCATE or DELETE, and until
model-oncology-public/moncpipelib#4 that choice was unreachable on the batched
write path: the caller passed ``batched.total_rows_hint or 0``, so AUTO
evaluated ``0 >= full_refresh_threshold`` and resolved to DELETE at any volume.

Two properties are pinned here:

- ``None`` (unknown) and ``0`` (a real, measured zero) are distinct inputs. The
  old ``or 0`` collapsed them, which is what made the bug silent.
- Under AUTO with an unknown hint the decision falls back to the target's
  ``pg_class.reltuples``, because the cost TRUNCATE-vs-DELETE trades away is
  O(rows already in the target), not O(rows arriving). ``reltuples = -1``
  (never analyzed -- notably every partitioned parent) is *not* an estimate of
  zero, so it must fall back to DELETE rather than silently truncating.

These are statement-shape tests against a mock cursor, matching
``tests/test_writers_upsert_guards.py``. Behavioral coverage against a real
PostgreSQL lives in ``tests/integration/test_write_full_refresh.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from moncpipelib.io_managers.enums import BulkInsertMethod, FullRefreshMethod
from moncpipelib.io_managers.writers import WriterConfig, clear_table, should_use_truncate

_TABLE = "reference_gold.dim_hcpcs"


def _config(
    full_refresh_method: FullRefreshMethod = FullRefreshMethod.AUTO,
    full_refresh_threshold: int = 10_000,
) -> WriterConfig:
    return WriterConfig(
        bulk_insert_method=BulkInsertMethod.COPY,
        bulk_insert_threshold=1000,
        full_refresh_method=full_refresh_method,
        full_refresh_threshold=full_refresh_threshold,
        insert_chunk_size=None,
    )


def _cursor(reltuples: float | None = None, *, found: bool = True) -> MagicMock:
    """Mock cursor whose catalog probe reports ``reltuples`` for the target."""
    cursor = MagicMock()
    cursor.rowcount = 42
    cursor.fetchone.return_value = (reltuples,) if found else None
    return cursor


def _statements(cursor: MagicMock) -> list[str]:
    return [call.args[0] for call in cursor.execute.call_args_list]


def _probed_catalog(cursor: MagicMock) -> bool:
    return any("pg_class" in stmt for stmt in _statements(cursor))


class TestShouldUseTruncate:
    """The pure predicate: no cursor, no catalog access."""

    @pytest.mark.parametrize("row_count", [None, 0, 9_999, 10_000, 1_000_000])
    def test_explicit_truncate_always_truncates(self, row_count: int | None) -> None:
        assert should_use_truncate(_config(FullRefreshMethod.TRUNCATE), row_count) is True

    @pytest.mark.parametrize("row_count", [None, 0, 9_999, 10_000, 1_000_000])
    def test_explicit_delete_never_truncates(self, row_count: int | None) -> None:
        assert should_use_truncate(_config(FullRefreshMethod.DELETE), row_count) is False

    def test_auto_at_threshold_truncates(self) -> None:
        assert should_use_truncate(_config(), 10_000) is True

    def test_auto_below_threshold_deletes(self) -> None:
        assert should_use_truncate(_config(), 9_999) is False

    def test_auto_unknown_row_count_deletes(self) -> None:
        """``None`` means "no estimate available" -- stay on the safer path."""
        assert should_use_truncate(_config(), None) is False


class TestClearTableExplicitMethods:
    """An explicit method short-circuits: no catalog probe, hint irrelevant."""

    def test_truncate_method_does_not_probe_catalog(self) -> None:
        cursor = _cursor()
        deleted, method = clear_table(
            _config(FullRefreshMethod.TRUNCATE), cursor, _TABLE, None, MagicMock()
        )
        assert (deleted, method) == (0, "truncate")
        assert _statements(cursor) == [f"TRUNCATE {_TABLE}"]
        assert not _probed_catalog(cursor)

    def test_delete_method_does_not_probe_catalog(self) -> None:
        cursor = _cursor()
        deleted, method = clear_table(
            _config(FullRefreshMethod.DELETE), cursor, _TABLE, None, MagicMock()
        )
        assert (deleted, method) == (42, "delete")
        assert _statements(cursor) == [f"DELETE FROM {_TABLE}"]
        assert not _probed_catalog(cursor)


class TestClearTableAutoWithHint:
    """A caller-supplied hint is authoritative; the catalog is not consulted."""

    def test_hint_at_threshold_truncates(self) -> None:
        cursor = _cursor()
        _, method = clear_table(_config(), cursor, _TABLE, 10_000, MagicMock())
        assert method == "truncate"
        assert not _probed_catalog(cursor)

    def test_hint_below_threshold_deletes(self) -> None:
        cursor = _cursor()
        _, method = clear_table(_config(), cursor, _TABLE, 9_999, MagicMock())
        assert method == "delete"
        assert not _probed_catalog(cursor)

    def test_explicit_zero_is_honoured_as_a_real_count(self) -> None:
        """``0`` is a measured "no rows arriving", not "unknown".

        It must be taken at face value and must NOT trigger the catalog
        fallback -- the distinction the old ``or 0`` destroyed.
        """
        cursor = _cursor(reltuples=5_000_000.0)
        _, method = clear_table(_config(), cursor, _TABLE, 0, MagicMock())
        assert method == "delete"
        assert not _probed_catalog(cursor)


class TestClearTableAutoWithoutHint:
    """The #4 regression surface: AUTO with an unknown incoming row count."""

    def test_large_existing_table_truncates(self) -> None:
        """The batched path passes no hint; a 5.6M-row target must TRUNCATE."""
        cursor = _cursor(reltuples=5_694_493.0)
        deleted, method = clear_table(_config(), cursor, _TABLE, None, MagicMock())
        assert (deleted, method) == (0, "truncate")
        assert _probed_catalog(cursor)
        assert _statements(cursor)[-1] == f"TRUNCATE {_TABLE}"

    def test_small_existing_table_deletes(self) -> None:
        cursor = _cursor(reltuples=100.0)
        _, method = clear_table(_config(), cursor, _TABLE, None, MagicMock())
        assert method == "delete"
        assert _probed_catalog(cursor)

    def test_never_analyzed_table_deletes(self) -> None:
        """``reltuples = -1`` means unknown, not empty.

        Every partitioned parent reports -1 until an explicit ANALYZE, so
        treating it as 0 would be both wrong and biased toward the more
        disruptive lock.
        """
        cursor = _cursor(reltuples=-1.0)
        _, method = clear_table(_config(), cursor, _TABLE, None, MagicMock())
        assert method == "delete"

    def test_missing_relation_deletes(self) -> None:
        """``to_regclass`` yields NULL for an unknown name -> no estimate."""
        cursor = _cursor(found=False)
        _, method = clear_table(_config(), cursor, _TABLE, None, MagicMock())
        assert method == "delete"

    def test_null_reltuples_deletes(self) -> None:
        cursor = _cursor(reltuples=None)
        _, method = clear_table(_config(), cursor, _TABLE, None, MagicMock())
        assert method == "delete"

    def test_catalog_probe_is_parameterized(self) -> None:
        """The target name is bound, never interpolated into the probe."""
        cursor = _cursor(reltuples=5_000_000.0)
        clear_table(_config(), cursor, _TABLE, None, MagicMock())
        probe = next(c for c in cursor.execute.call_args_list if "pg_class" in c.args[0])
        assert _TABLE not in probe.args[0]
        assert probe.args[1] == (_TABLE,)
