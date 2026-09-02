"""Tests for SCD2 change detection utilities."""

import polars as pl
import pytest

from moncpipelib.scd import SCD2ChangeResult, detect_changes
from moncpipelib.transforms import compute_row_hash


def _make_hashed(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Helper to add row_hash to a DataFrame."""
    return df.with_columns(compute_row_hash(columns))


class TestDetectChanges:
    """Tests for detect_changes utility."""

    def test_all_new_records(self) -> None:
        """When current is empty, all incoming records are new."""
        incoming = _make_hashed(
            pl.DataFrame({"pk": ["a", "b"], "val": ["x", "y"]}),
            ["val"],
        )
        current = pl.DataFrame(
            {
                "pk": pl.Series([], dtype=pl.String),
                "val": pl.Series([], dtype=pl.String),
                "row_hash": pl.Series([], dtype=pl.String),
            }
        )
        result = detect_changes(incoming, current, business_key="pk")
        assert len(result.new_records) == 2
        assert len(result.changed_records) == 0
        assert len(result.unchanged_records) == 0
        assert result.summary["new"] == 2

    def test_all_unchanged(self) -> None:
        """When data is identical, all records are unchanged."""
        df = _make_hashed(
            pl.DataFrame({"pk": ["a", "b"], "val": ["x", "y"]}),
            ["val"],
        )
        result = detect_changes(df, df, business_key="pk")
        assert len(result.new_records) == 0
        assert len(result.changed_records) == 0
        assert len(result.unchanged_records) == 2
        assert result.summary["unchanged"] == 2

    def test_mixed_new_changed_unchanged(self) -> None:
        """Mix of new, changed, and unchanged records."""
        incoming = _make_hashed(
            pl.DataFrame(
                {
                    "pk": ["a", "b", "c"],
                    "val": ["same", "updated", "brand_new"],
                }
            ),
            ["val"],
        )
        current = _make_hashed(
            pl.DataFrame(
                {
                    "pk": ["a", "b"],
                    "val": ["same", "old_value"],
                }
            ),
            ["val"],
        )
        result = detect_changes(incoming, current, business_key="pk")
        assert result.summary["new"] == 1
        assert result.summary["changed"] == 1
        assert result.summary["unchanged"] == 1
        assert result.new_records["pk"].to_list() == ["c"]
        assert result.changed_records["pk"].to_list() == ["b"]
        assert result.unchanged_records["pk"].to_list() == ["a"]

    def test_deleted_keys_detected(self) -> None:
        """With detect_deletes=True, keys in current but not incoming are found."""
        incoming = _make_hashed(
            pl.DataFrame({"pk": ["a"], "val": ["x"]}),
            ["val"],
        )
        current = _make_hashed(
            pl.DataFrame({"pk": ["a", "b"], "val": ["x", "y"]}),
            ["val"],
        )
        result = detect_changes(incoming, current, business_key="pk", detect_deletes=True)
        assert result.summary["deleted"] == 1
        assert result.deleted_keys["pk"].to_list() == ["b"]

    def test_deleted_keys_not_detected_by_default(self) -> None:
        """By default, deleted_keys is empty."""
        incoming = _make_hashed(
            pl.DataFrame({"pk": ["a"], "val": ["x"]}),
            ["val"],
        )
        current = _make_hashed(
            pl.DataFrame({"pk": ["a", "b"], "val": ["x", "y"]}),
            ["val"],
        )
        result = detect_changes(incoming, current, business_key="pk")
        assert len(result.deleted_keys) == 0
        assert result.summary["deleted"] == 0

    def test_composite_business_key(self) -> None:
        """Composite (multi-column) business key works correctly."""
        incoming = _make_hashed(
            pl.DataFrame(
                {
                    "pk1": ["a", "a", "b"],
                    "pk2": [1, 2, 1],
                    "val": ["x", "y", "z"],
                }
            ),
            ["val"],
        )
        current = _make_hashed(
            pl.DataFrame(
                {
                    "pk1": ["a", "a"],
                    "pk2": [1, 2],
                    "val": ["x", "changed"],
                }
            ),
            ["val"],
        )
        result = detect_changes(incoming, current, business_key=["pk1", "pk2"])
        assert result.summary["new"] == 1
        assert result.summary["changed"] == 1
        assert result.summary["unchanged"] == 1

    def test_summary_counts(self) -> None:
        """Summary dict has all expected keys and correct total."""
        incoming = _make_hashed(
            pl.DataFrame({"pk": ["a", "b", "c"], "val": ["x", "y", "z"]}),
            ["val"],
        )
        current = _make_hashed(
            pl.DataFrame({"pk": ["a"], "val": ["x"]}),
            ["val"],
        )
        result = detect_changes(incoming, current, business_key="pk")
        assert set(result.summary.keys()) == {
            "new",
            "changed",
            "unchanged",
            "deleted",
            "total_incoming",
        }
        assert result.summary["total_incoming"] == 3

    def test_empty_incoming(self) -> None:
        """Empty incoming DataFrame produces all-zero results."""
        incoming = pl.DataFrame(
            {
                "pk": pl.Series([], dtype=pl.String),
                "val": pl.Series([], dtype=pl.String),
                "row_hash": pl.Series([], dtype=pl.String),
            }
        )
        current = _make_hashed(
            pl.DataFrame({"pk": ["a"], "val": ["x"]}),
            ["val"],
        )
        result = detect_changes(incoming, current, business_key="pk")
        assert result.summary["new"] == 0
        assert result.summary["changed"] == 0
        assert result.summary["unchanged"] == 0
        assert result.summary["total_incoming"] == 0

    def test_custom_hash_col(self) -> None:
        """Non-default hash column name is respected."""
        incoming = pl.DataFrame({"pk": ["a"], "val": ["x"]}).with_columns(
            compute_row_hash(["val"], alias="my_hash")
        )
        current = pl.DataFrame({"pk": ["a"], "val": ["x"]}).with_columns(
            compute_row_hash(["val"], alias="my_hash")
        )
        result = detect_changes(incoming, current, business_key="pk", hash_col="my_hash")
        assert result.summary["unchanged"] == 1

    def test_missing_hash_column_raises(self) -> None:
        """Error raised when hash column is missing from either DataFrame."""
        df_no_hash = pl.DataFrame({"pk": ["a"], "val": ["x"]})
        df_with_hash = _make_hashed(df_no_hash, ["val"])

        with pytest.raises(ValueError, match="hash_col 'row_hash' not found in incoming"):
            detect_changes(df_no_hash, df_with_hash, business_key="pk")

        with pytest.raises(ValueError, match="hash_col 'row_hash' not found in current"):
            detect_changes(df_with_hash, df_no_hash, business_key="pk")

    def test_missing_business_key_raises(self) -> None:
        """Error raised when business key column is missing."""
        df = _make_hashed(
            pl.DataFrame({"pk": ["a"], "val": ["x"]}),
            ["val"],
        )
        with pytest.raises(ValueError, match="business_key column"):
            detect_changes(df, df, business_key="nonexistent")

    def test_returns_scd2_change_result(self) -> None:
        """Return type is SCD2ChangeResult."""
        df = _make_hashed(
            pl.DataFrame({"pk": ["a"], "val": ["x"]}),
            ["val"],
        )
        result = detect_changes(df, df, business_key="pk")
        assert isinstance(result, SCD2ChangeResult)
