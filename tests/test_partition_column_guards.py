"""Unit tests for the partition_column write-time guard (#480).

A partition-scoped write clears the target with ``DELETE FROM t WHERE
"<partition_column>" IN (%s, ...)``. SQL ``NULL IN (...)`` evaluates to
UNKNOWN, never TRUE, so a row whose partition column is NULL is permanently
unreachable by that clear: it survives every re-land, duplicates on each one,
and is invisible to every partition-scoped guard while all of them report
success.

The guard is testable without a database: its entire contract is the shape
of the declared ``partition_column`` and the content of the DataFrame column,
both of which a plain ``pl.DataFrame`` captures. The resource-path tests
proving the guard is wired on both the single-DataFrame and batched write
paths live in ``tests/test_postgres_resource.py`` (see
``TestPartitionColumnNullGuardWiring``), following that module's existing
mock-cursor harness.
"""

from __future__ import annotations

import polars as pl
import pytest

from moncpipelib.contracts.exceptions import ContractViolationError
from moncpipelib.resources._contract_helpers import validate_partition_column_values

ASSET_NAME = "reference_bronze/some_asset"
TABLE_NAME = "reference_bronze.some_table"


class TestPartitionColumnValueGuard:
    def test_none_partition_column_is_noop(self) -> None:
        df = pl.DataFrame({"id": [1, 2]})
        validate_partition_column_values(df, None, ASSET_NAME, TABLE_NAME)

    def test_empty_string_partition_column_is_noop(self) -> None:
        df = pl.DataFrame({"id": [1, 2]})
        validate_partition_column_values(df, "", ASSET_NAME, TABLE_NAME)

    def test_clean_column_passes(self) -> None:
        df = pl.DataFrame({"load_period": ["2026-01", "2026-01"], "id": [1, 2]})
        validate_partition_column_values(df, "load_period", ASSET_NAME, TABLE_NAME)

    def test_missing_column_raises(self) -> None:
        df = pl.DataFrame({"id": [1, 2]})
        with pytest.raises(ContractViolationError) as exc:
            validate_partition_column_values(df, "load_period", ASSET_NAME, TABLE_NAME)
        message = str(exc.value)
        assert "does not contain that column" in message
        assert "load_period" in message
        assert ASSET_NAME in message
        assert TABLE_NAME in message
        assert "partition_key" in message
        assert "Dagster partition key" in message

    def test_null_values_raise_with_counts(self) -> None:
        df = pl.DataFrame({"load_period": ["2026-01", None, None], "id": [1, 2, 3]})
        with pytest.raises(ContractViolationError) as exc:
            validate_partition_column_values(df, "load_period", ASSET_NAME, TABLE_NAME)
        message = str(exc.value)
        assert "contains NULLs" in message
        assert "2 of 3 row(s)" in message

    def test_all_null_column_raises(self) -> None:
        df = pl.DataFrame({"load_period": [None, None, None], "id": [1, 2, 3]})
        with pytest.raises(ContractViolationError) as exc:
            validate_partition_column_values(df, "load_period", ASSET_NAME, TABLE_NAME)
        assert "3 of 3 row(s)" in str(exc.value)

    def test_empty_dataframe_with_column_passes(self) -> None:
        df = pl.DataFrame({"load_period": [], "id": []})
        validate_partition_column_values(df, "load_period", ASSET_NAME, TABLE_NAME)

    def test_empty_dataframe_without_column_raises(self) -> None:
        """Fail closed: an empty frame is not exempt from the presence requirement."""
        df = pl.DataFrame({"id": []})
        with pytest.raises(ContractViolationError):
            validate_partition_column_values(df, "load_period", ASSET_NAME, TABLE_NAME)

    def test_raised_error_carries_asset_name(self) -> None:
        df = pl.DataFrame({"id": [1]})
        with pytest.raises(ContractViolationError) as exc:
            validate_partition_column_values(df, "load_period", ASSET_NAME, TABLE_NAME)
        assert exc.value.asset_name == ASSET_NAME
