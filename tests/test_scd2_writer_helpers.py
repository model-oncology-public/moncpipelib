"""Unit tests for the SCD2 change-detection writer helpers (#361).

These cover the three large-table mitigations added to ``scd2_finalize``'s
change-detection path without requiring a live database:

- ``_create_staging_bk_index``: business-key index on the staging temp table.
- ``_apply_change_detection_work_mem``: per-tx ``work_mem`` bump.
- ``_bounded_statement_timeout``: per-statement ``statement_timeout`` bound
  that captures and restores the prior value.
- ``_assert_staging_business_keys_unique``: duplicate-business-key guard
  (#419) that fails the write before any DML.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from psycopg.sql import Composed

from moncpipelib.io_managers.writers import (
    _apply_change_detection_work_mem,
    _assert_staging_business_keys_unique,
    _bounded_statement_timeout,
    _create_staging_bk_index,
)


class TestCreateStagingBkIndex:
    """``_create_staging_bk_index`` composes a safe CREATE INDEX on staging."""

    def test_single_key_index_sql(self) -> None:
        cursor = MagicMock()
        _create_staging_bk_index(cursor, "_scd2_staging", ["npi"])

        cursor.execute.assert_called_once()
        composed = cursor.execute.call_args.args[0]
        assert isinstance(composed, Composed)
        rendered = composed.as_string(None)
        assert rendered == (
            'CREATE INDEX IF NOT EXISTS "_scd2_staging_bk_idx" ON "_scd2_staging" ("npi")'
        )

    def test_composite_key_quotes_each_column(self) -> None:
        cursor = MagicMock()
        _create_staging_bk_index(cursor, "_scd2_staging", ["npi", "address_type", "sequence"])

        rendered = cursor.execute.call_args.args[0].as_string(None)
        assert rendered == (
            'CREATE INDEX IF NOT EXISTS "_scd2_staging_bk_idx" '
            'ON "_scd2_staging" ("npi", "address_type", "sequence")'
        )

    def test_identifiers_are_quoted_not_interpolated(self) -> None:
        """A hostile business-key name is quoted, never interpolated raw."""
        cursor = MagicMock()
        _create_staging_bk_index(cursor, "_scd2_staging", ['evil";DROP TABLE x--'])

        rendered = cursor.execute.call_args.args[0].as_string(None)
        # Embedded quote is doubled (Postgres identifier escaping), so the
        # statement remains a single safe CREATE INDEX.
        assert '"evil"";DROP TABLE x--"' in rendered


class TestApplyChangeDetectionWorkMem:
    """``_apply_change_detection_work_mem`` sets a local work_mem via set_config."""

    def test_uses_parameterized_set_config_local(self) -> None:
        cursor = MagicMock()
        _apply_change_detection_work_mem(cursor, "256MB")

        cursor.execute.assert_called_once_with(
            "SELECT set_config('work_mem', %s, true)", ("256MB",)
        )


class TestBoundedStatementTimeout:
    """``_bounded_statement_timeout`` bounds then restores statement_timeout.

    The SHOW / set_config round-trips run on a side cursor
    (``cursor.connection.cursor()``) so the wrapped statement's own result set
    and rowcount on ``cursor`` are never disturbed.
    """

    @staticmethod
    def _with_side_cursor(prior: tuple[str] | None) -> tuple[MagicMock, MagicMock]:
        """Build a main cursor whose ``connection.cursor()`` yields a tracked
        side cursor with ``fetchone`` -> ``prior``.
        """
        cursor = MagicMock()
        tcur = MagicMock()
        tcur.fetchone.return_value = prior
        cursor.connection.cursor.return_value.__enter__.return_value = tcur
        return cursor, tcur

    def test_none_is_a_no_op(self) -> None:
        cursor = MagicMock()
        with _bounded_statement_timeout(cursor, None):
            pass
        cursor.execute.assert_not_called()
        cursor.connection.cursor.assert_not_called()

    def test_sets_then_restores_prior_value(self) -> None:
        cursor, tcur = self._with_side_cursor(("0",))

        with _bounded_statement_timeout(cursor, "30min"):
            # Inside the block the bound is applied (SHOW + set_config) on the
            # side cursor; the main cursor is left untouched.
            cursor.execute.assert_not_called()
            assert tcur.execute.call_args_list == [
                call("SHOW statement_timeout"),
                call("SELECT set_config('statement_timeout', %s, true)", ("30min",)),
            ]

        # On exit the prior value is restored.
        assert tcur.execute.call_args_list[-1] == call(
            "SELECT set_config('statement_timeout', %s, true)", ("0",)
        )
        cursor.execute.assert_not_called()

    def test_restores_even_when_body_raises(self) -> None:
        cursor, tcur = self._with_side_cursor(("1min",))

        with (
            pytest.raises(RuntimeError),
            _bounded_statement_timeout(cursor, "30min"),
        ):
            raise RuntimeError("boom")

        # Last call restores the captured prior timeout despite the exception.
        assert tcur.execute.call_args_list[-1] == call(
            "SELECT set_config('statement_timeout', %s, true)", ("1min",)
        )

    def test_missing_prior_defaults_to_zero(self) -> None:
        cursor, tcur = self._with_side_cursor(None)

        with _bounded_statement_timeout(cursor, "30min"):
            pass

        assert tcur.execute.call_args_list[-1] == call(
            "SELECT set_config('statement_timeout', %s, true)", ("0",)
        )


class TestAssertStagingBusinessKeysUnique:
    """``_assert_staging_business_keys_unique`` fails loudly on duplicate keys (#419)."""

    def test_no_duplicates_is_a_no_op(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = []

        _assert_staging_business_keys_unique(
            cursor, "_scd2_staging", "ref.dim_x", ["npi"], None, None
        )

        executed = cursor.execute.call_args.args[0]
        assert 'GROUP BY "npi"' in executed
        assert "HAVING count(*) > 1" in executed

    def test_partition_column_widens_the_uniqueness_group(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = []

        _assert_staging_business_keys_unique(
            cursor, "_scd2_staging", "ref.dim_x", ["npi"], "load_period", None
        )

        executed = cursor.execute.call_args.args[0]
        assert 'GROUP BY "npi", "load_period"' in executed

    def test_composite_key_groups_all_columns(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = []

        _assert_staging_business_keys_unique(
            cursor, "_scd2_staging", "ref.dim_x", ["bla_number", "product_number"], None, None
        )

        executed = cursor.execute.call_args.args[0]
        assert 'GROUP BY "bla_number", "product_number"' in executed

    def test_duplicates_raise_with_count_and_samples(self) -> None:
        cursor = MagicMock()
        # Row shape: (dup_key_count, *group_values, row_copies)
        cursor.fetchall.return_value = [
            (80, "017016", 2),
            (80, "017054", 2),
        ]

        with pytest.raises(ValueError, match="80 business key") as excinfo:
            _assert_staging_business_keys_unique(
                cursor,
                "_scd2_staging",
                "reference_gold.fda_purplebook_bla",
                ["bla_number"],
                None,
                None,
            )

        msg = str(excinfo.value)
        assert "reference_gold.fda_purplebook_bla" in msg
        assert "bla_number='017016'" in msg
        assert "x2" in msg
        assert "#419" in msg

    def test_partitioned_duplicates_name_the_partition_column(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            (1, "017016", "2026-04-01", 3),
        ]

        with pytest.raises(ValueError, match="partition value") as excinfo:
            _assert_staging_business_keys_unique(
                cursor, "_scd2_staging", "ref.dim_x", ["bla_number"], "load_period", None
            )

        msg = str(excinfo.value)
        assert '"load_period" partition value' in msg
        assert "load_period='2026-04-01'" in msg
        assert "x3" in msg
