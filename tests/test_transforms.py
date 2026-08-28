"""Tests for data transformation utilities."""

from datetime import date

import polars as pl
import pytest

from moncpipelib.transforms import (
    clean_text,
    compute_row_hash,
    normalize_ndc,
    safe_bool,
    safe_date,
    safe_decimal,
    safe_int,
)


class TestNormalizeNdc:
    """Tests for normalize_ndc transform."""

    # --- Segment-aware (dashed input) ---

    def test_5_4_2_package_ndc(self) -> None:
        """Standard 11-digit NDC with dashes passes through."""
        assert normalize_ndc("50242-0918-01") == "50242-0918-01"

    def test_4_4_2_package_ndc(self) -> None:
        """4-digit labeler padded to 5."""
        assert normalize_ndc("0536-1327-01") == "00536-1327-01"

    def test_5_3_2_package_ndc(self) -> None:
        """3-digit product padded to 4."""
        assert normalize_ndc("50242-918-01") == "50242-0918-01"

    def test_4_3_2_package_ndc(self) -> None:
        """Both labeler and product padded."""
        assert normalize_ndc("0536-327-01") == "00536-0327-01"

    def test_5_4_product_ndc(self) -> None:
        """Product NDC (no package) returns 5-4 only."""
        assert normalize_ndc("50242-0918") == "50242-0918"

    def test_5_3_product_ndc(self) -> None:
        """Product NDC with 3-digit product padded, no package appended."""
        assert normalize_ndc("50242-918") == "50242-0918"

    def test_4_4_product_ndc(self) -> None:
        """Product NDC with 4-digit labeler padded, no package appended."""
        assert normalize_ndc("0536-1327") == "00536-1327"

    # --- force_package mode ---

    def test_force_package_appends_suffix(self) -> None:
        """force_package=True appends the package suffix to 2-segment NDCs."""
        assert normalize_ndc("50242-0918", force_package=True) == "50242-0918-00"

    def test_force_package_custom_suffix(self) -> None:
        """Custom package_suffix is used when force_package=True."""
        assert (
            normalize_ndc("50242-0918", force_package=True, package_suffix="99") == "50242-0918-99"
        )

    def test_force_package_no_effect_on_3_segment(self) -> None:
        """force_package has no effect on 3-segment NDCs."""
        assert normalize_ndc("50242-0918-01", force_package=True) == "50242-0918-01"

    def test_force_package_pads_suffix(self) -> None:
        """Single-digit package_suffix is zero-padded to 2 digits."""
        assert (
            normalize_ndc("50242-0918", force_package=True, package_suffix="1") == "50242-0918-01"
        )

    # --- with_hyphens mode ---

    def test_without_hyphens_3_segment(self) -> None:
        """with_hyphens=False strips dashes but preserves leading zeros."""
        assert normalize_ndc("0536-1327-01", with_hyphens=False) == "00536132701"

    def test_without_hyphens_2_segment(self) -> None:
        """with_hyphens=False on product NDC returns digit-only string."""
        assert normalize_ndc("50242-918", with_hyphens=False) == "502420918"

    def test_without_hyphens_pure_digits(self) -> None:
        """with_hyphens=False on pure digit input."""
        assert normalize_ndc("00536132701", with_hyphens=False) == "00536132701"

    def test_without_hyphens_with_force_package(self) -> None:
        """with_hyphens=False combined with force_package."""
        assert normalize_ndc("50242-0918", with_hyphens=False, force_package=True) == "50242091800"

    def test_without_hyphens_none_passthrough(self) -> None:
        """with_hyphens=False still returns None for None input."""
        assert normalize_ndc(None, with_hyphens=False) is None

    # --- Pure digit (no dashes) ---

    def test_formats_11_digit_ndc(self) -> None:
        assert normalize_ndc("00536132701") == "00536-1327-01"

    def test_pads_10_digit_ndc_with_leading_zero(self) -> None:
        assert normalize_ndc("0536132701") == "00536-1327-01"

    def test_pads_short_ndc(self) -> None:
        assert normalize_ndc("12345") == "00000-0123-45"

    def test_truncates_long_ndc(self) -> None:
        assert normalize_ndc("123456789012") == "12345-6789-01"

    # --- Type handling ---

    def test_handles_integer_input(self) -> None:
        assert normalize_ndc(536132701) == "00536-1327-01"

    def test_handles_float_input(self) -> None:
        assert normalize_ndc(536132701.0) == "00536-1327-01"

    def test_handles_none(self) -> None:
        assert normalize_ndc(None) is None

    def test_handles_empty_string(self) -> None:
        assert normalize_ndc("") is None

    def test_handles_non_numeric_string(self) -> None:
        assert normalize_ndc("abc") is None

    def test_works_with_polars_map_elements(self) -> None:
        df = pl.DataFrame({"ndc": ["00536132701", "0536132701", None, ""]})
        result = df.with_columns(
            pl.col("ndc")
            .map_elements(normalize_ndc, return_dtype=pl.String)
            .alias("ndc_normalized")
        )
        assert result["ndc_normalized"].to_list() == [
            "00536-1327-01",
            "00536-1327-01",
            None,
            None,
        ]


class TestSafeDecimal:
    """Tests for safe_decimal transform."""

    def test_parses_valid_numbers(self) -> None:
        df = pl.DataFrame({"value": ["1.5", "2.0", "3.14159"]})
        result = df.select(safe_decimal("value"))
        assert result["value"].to_list() == [1.5, 2.0, 3.14159]

    def test_handles_null(self) -> None:
        df = pl.DataFrame({"value": ["1.0", None, "3.0"]})
        result = df.select(safe_decimal("value"))
        assert result["value"].to_list() == [1.0, None, 3.0]

    def test_handles_empty_string(self) -> None:
        df = pl.DataFrame({"value": ["1.0", "", "3.0"]})
        result = df.select(safe_decimal("value"))
        assert result["value"].to_list() == [1.0, None, 3.0]

    def test_handles_whitespace(self) -> None:
        df = pl.DataFrame({"value": ["  1.5  ", "   ", "3.0"]})
        result = df.select(safe_decimal("value"))
        assert result["value"].to_list() == [1.5, None, 3.0]

    def test_handles_negative_numbers(self) -> None:
        df = pl.DataFrame({"value": ["-1.5", "-0.5", "0"]})
        result = df.select(safe_decimal("value"))
        assert result["value"].to_list() == [-1.5, -0.5, 0.0]


class TestSafeInt:
    """Tests for safe_int transform."""

    def test_parses_valid_integers(self) -> None:
        df = pl.DataFrame({"value": ["1", "2", "100"]})
        result = df.select(safe_int("value"))
        assert result["value"].to_list() == [1, 2, 100]

    def test_handles_null(self) -> None:
        df = pl.DataFrame({"value": ["1", None, "3"]})
        result = df.select(safe_int("value"))
        assert result["value"].to_list() == [1, None, 3]

    def test_handles_empty_string(self) -> None:
        df = pl.DataFrame({"value": ["1", "", "3"]})
        result = df.select(safe_int("value"))
        assert result["value"].to_list() == [1, None, 3]


class TestSafeBool:
    """Tests for safe_bool transform."""

    def test_parses_true_values(self) -> None:
        df = pl.DataFrame({"value": ["t", "true", "1", "yes", "y", "TRUE", "Yes"]})
        result = df.select(safe_bool("value"))
        assert all(v is True for v in result["value"].to_list())

    def test_parses_false_values(self) -> None:
        df = pl.DataFrame({"value": ["f", "false", "0", "no", "n", "FALSE", "No"]})
        result = df.select(safe_bool("value"))
        assert all(v is False for v in result["value"].to_list())

    def test_handles_null(self) -> None:
        df = pl.DataFrame({"value": ["true", None, "false"]})
        result = df.select(safe_bool("value"))
        assert result["value"].to_list() == [True, None, False]

    def test_handles_empty_string(self) -> None:
        df = pl.DataFrame({"value": ["true", "", "false"]})
        result = df.select(safe_bool("value"))
        assert result["value"].to_list() == [True, None, False]

    def test_handles_unrecognized_values(self) -> None:
        df = pl.DataFrame({"value": ["maybe", "unknown", "2"]})
        result = df.select(safe_bool("value"))
        assert all(v is None for v in result["value"].to_list())


class TestCleanText:
    """Tests for clean_text transform."""

    def test_strips_whitespace(self) -> None:
        df = pl.DataFrame({"value": ["  hello  ", "world  ", "  test"]})
        result = df.select(clean_text("value"))
        assert result["value"].to_list() == ["hello", "world", "test"]

    def test_handles_null(self) -> None:
        df = pl.DataFrame({"value": ["hello", None, "world"]})
        result = df.select(clean_text("value"))
        assert result["value"].to_list() == ["hello", None, "world"]

    def test_converts_empty_to_null(self) -> None:
        df = pl.DataFrame({"value": ["hello", "", "world"]})
        result = df.select(clean_text("value"))
        assert result["value"].to_list() == ["hello", None, "world"]

    def test_converts_whitespace_only_to_null(self) -> None:
        df = pl.DataFrame({"value": ["hello", "   ", "world"]})
        result = df.select(clean_text("value"))
        assert result["value"].to_list() == ["hello", None, "world"]


class TestSafeDate:
    """Tests for safe_date transform."""

    def test_parses_iso_dates(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["2024-01-15", "2024-12-31"]})
        result = df.select(safe_date("value"))
        assert result["value"].to_list() == [date(2024, 1, 15), date(2024, 12, 31)]

    def test_handles_null(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["2024-01-15", None, "2024-12-31"]})
        result = df.select(safe_date("value"))
        assert result["value"].to_list() == [date(2024, 1, 15), None, date(2024, 12, 31)]

    def test_handles_empty_string(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["2024-01-15", "", "2024-12-31"]})
        result = df.select(safe_date("value"))
        assert result["value"].to_list() == [date(2024, 1, 15), None, date(2024, 12, 31)]

    def test_custom_format(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["01/15/2024", "12/31/2024"]})
        result = df.select(safe_date("value", format="%m/%d/%Y"))
        assert result["value"].to_list() == [date(2024, 1, 15), date(2024, 12, 31)]

    def test_auto_detect_iso(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["2024-01-15", "2024-12-31"]})
        result = df.select(safe_date("value"))
        assert result["value"].to_list() == [date(2024, 1, 15), date(2024, 12, 31)]

    def test_auto_detect_yyyymmdd(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["20240115", "20241231"]})
        result = df.select(safe_date("value"))
        assert result["value"].to_list() == [date(2024, 1, 15), date(2024, 12, 31)]

    def test_auto_detect_dd_mon_yy(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["15-Jan-24", "31-Dec-24"]})
        result = df.select(safe_date("value"))
        assert result["value"].to_list() == [date(2024, 1, 15), date(2024, 12, 31)]

    def test_auto_detect_mixed_formats(self) -> None:
        """Mixed date formats in same column resolve correctly."""
        from datetime import date

        df = pl.DataFrame({"value": ["20240115", "15-Jan-24", "2024-06-30"]})
        result = df.select(safe_date("value"))
        assert result["value"].to_list() == [
            date(2024, 1, 15),
            date(2024, 1, 15),
            date(2024, 6, 30),
        ]

    def test_explicit_formats_list(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["20240115", "15-Jan-24"]})
        result = df.select(safe_date("value", formats=["%Y%m%d", "%d-%b-%y"]))
        assert result["value"].to_list() == [date(2024, 1, 15), date(2024, 1, 15)]

    def test_format_and_formats_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both"):
            safe_date("value", format="%Y-%m-%d", formats=["%Y%m%d"])

    def test_auto_detect_with_nulls(self) -> None:
        from datetime import date

        df = pl.DataFrame({"value": ["20240115", None, "", "15-Jan-24"]})
        result = df.select(safe_date("value"))
        assert result["value"].to_list() == [date(2024, 1, 15), None, None, date(2024, 1, 15)]


class TestComputeRowHash:
    """Tests for compute_row_hash transform."""

    def test_produces_64_char_hex_string(self) -> None:
        df = pl.DataFrame({"a": ["hello"], "b": ["world"]})
        result = df.with_columns(compute_row_hash(["a", "b"]))
        hash_val = result["row_hash"][0]
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_deterministic_same_input(self) -> None:
        df = pl.DataFrame({"a": ["hello", "foo"], "b": ["world", "bar"]})
        result1 = df.with_columns(compute_row_hash(["a", "b"]))
        result2 = df.with_columns(compute_row_hash(["a", "b"]))
        assert result1["row_hash"].to_list() == result2["row_hash"].to_list()

    def test_different_data_different_hash(self) -> None:
        df = pl.DataFrame({"a": ["hello", "changed"], "b": ["world", "world"]})
        result = df.with_columns(compute_row_hash(["a", "b"]))
        hashes = result["row_hash"].to_list()
        assert hashes[0] != hashes[1]

    def test_null_handling(self) -> None:
        df = pl.DataFrame({"a": ["hello", None], "b": [None, "world"]})
        result = df.with_columns(compute_row_hash(["a", "b"]))
        hashes = result["row_hash"].to_list()
        # Both rows should have valid hashes (not None)
        assert all(h is not None for h in hashes)
        # Nulls in different positions should produce different hashes
        assert hashes[0] != hashes[1]

    def test_null_consistency(self) -> None:
        """Same null pattern produces the same hash across runs."""
        df1 = pl.DataFrame({"a": [None], "b": ["world"]})
        df2 = pl.DataFrame({"a": [None], "b": ["world"]})
        h1 = df1.with_columns(compute_row_hash(["a", "b"]))["row_hash"][0]
        h2 = df2.with_columns(compute_row_hash(["a", "b"]))["row_hash"][0]
        assert h1 == h2

    def test_mixed_types(self) -> None:
        df = pl.DataFrame(
            {
                "str_col": ["hello"],
                "int_col": [42],
                "float_col": [3.14],
                "date_col": [date(2024, 1, 15)],
            }
        )
        result = df.with_columns(compute_row_hash(["str_col", "int_col", "float_col", "date_col"]))
        assert result["row_hash"][0] is not None
        assert len(result["row_hash"][0]) == 64

    def test_custom_alias(self) -> None:
        df = pl.DataFrame({"a": ["hello"]})
        result = df.with_columns(compute_row_hash(["a"], alias="my_hash"))
        assert "my_hash" in result.columns
        assert "row_hash" not in result.columns

    def test_custom_separator(self) -> None:
        df = pl.DataFrame({"a": ["hello"], "b": ["world"]})
        result_default = df.with_columns(compute_row_hash(["a", "b"]))
        result_custom = df.with_columns(
            compute_row_hash(["a", "b"], separator="::", alias="custom_hash")
        )
        # Different separator should produce different hash
        assert result_default["row_hash"][0] != result_custom["custom_hash"][0]

    def test_single_column(self) -> None:
        df = pl.DataFrame({"a": ["hello", "world"]})
        result = df.with_columns(compute_row_hash(["a"]))
        assert len(result["row_hash"].to_list()) == 2
        assert all(h is not None for h in result["row_hash"].to_list())

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.String)})
        result = df.with_columns(compute_row_hash(["a"]))
        assert len(result) == 0
        assert "row_hash" in result.columns

    def test_only_specified_columns_in_hash(self) -> None:
        """Changing an unspecified column should not change the hash."""
        df1 = pl.DataFrame({"a": ["hello"], "b": ["world"], "c": ["v1"]})
        df2 = pl.DataFrame({"a": ["hello"], "b": ["world"], "c": ["v2"]})
        h1 = df1.with_columns(compute_row_hash(["a", "b"]))["row_hash"][0]
        h2 = df2.with_columns(compute_row_hash(["a", "b"]))["row_hash"][0]
        assert h1 == h2

    def test_column_order_matters(self) -> None:
        """Different column order should produce different hashes."""
        df = pl.DataFrame({"a": ["hello"], "b": ["world"]})
        h_ab = df.with_columns(compute_row_hash(["a", "b"]))["row_hash"][0]
        h_ba = df.with_columns(compute_row_hash(["b", "a"], alias="row_hash_ba"))["row_hash_ba"][0]
        assert h_ab != h_ba

    def test_empty_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="columns must be a non-empty list"):
            compute_row_hash([])
