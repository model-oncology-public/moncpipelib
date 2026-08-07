"""Tests for polars_to_md rendering utility."""

from __future__ import annotations

import polars as pl

from moncpipelib.contracts.models import Column, ColumnType, DataContract, Schema
from moncpipelib.rendering import polars_to_md


class TestBasicRendering:
    """Tests for basic markdown table rendering without PII masking."""

    def test_simple_dataframe(self) -> None:
        """Test rendering a simple DataFrame to markdown."""
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = polars_to_md(df)
        lines = result.strip().split("\n")
        assert lines[0] == "| a | b |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 1 | x |"
        assert lines[3] == "| 2 | y |"

    def test_empty_dataframe(self) -> None:
        """Test rendering an empty DataFrame shows only header."""
        df = pl.DataFrame(
            {"col_a": pl.Series([], dtype=pl.Utf8), "col_b": pl.Series([], dtype=pl.Int64)}
        )
        result = polars_to_md(df)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "col_a" in lines[0]
        assert "---" in lines[1]

    def test_null_values(self) -> None:
        """Test that null values render as empty strings."""
        df = pl.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
        result = polars_to_md(df)
        lines = result.strip().split("\n")
        # Row with nulls should have empty cells
        assert "|  |" in lines[3]

    def test_max_rows_truncation(self) -> None:
        """Test that max_rows truncates output and adds note."""
        df = pl.DataFrame({"a": list(range(20))})
        result = polars_to_md(df, max_rows=5)
        lines = result.strip().split("\n")
        # header + separator + 5 data rows + blank + truncation note
        data_lines = [
            ln for ln in lines if ln.startswith("| ") and "---" not in ln and "a" not in ln
        ]
        assert len(data_lines) == 5
        assert "Showing 5 of 20 rows" in result

    def test_max_rows_none_shows_all(self) -> None:
        """Test that max_rows=None shows all rows."""
        df = pl.DataFrame({"a": list(range(15))})
        result = polars_to_md(df, max_rows=None)
        assert "Showing" not in result
        data_lines = [
            ln
            for ln in result.split("\n")
            if ln.startswith("| ") and "---" not in ln and "a" not in ln
        ]
        assert len(data_lines) == 15

    def test_no_truncation_when_within_limit(self) -> None:
        """Test no truncation note when rows <= max_rows."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = polars_to_md(df, max_rows=10)
        assert "Showing" not in result


class TestPiiMaskingFromContract:
    """Tests for PII masking based on data contract."""

    def _make_contract(self, columns: list[Column]) -> DataContract:
        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns),
        )

    def test_pii_columns_masked(self) -> None:
        """Test that PII columns are masked based on contract."""
        contract = self._make_contract(
            [
                Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
                Column(name="event_type", type=ColumnType.STRING, nullable=False, pii=False),
            ]
        )
        df = pl.DataFrame({"patient_id": ["PAT-001"], "event_type": ["admission"]})
        result = polars_to_md(df, contract=contract)
        assert "***" in result
        assert "PAT-001" not in result
        assert "admission" in result

    def test_default_pii_masked(self) -> None:
        """Test that columns defaulting to pii=True are masked."""
        contract = self._make_contract(
            [
                Column(name="ssn", type=ColumnType.STRING, nullable=False),  # default pii=True
                Column(name="status", type=ColumnType.STRING, nullable=False, pii=False),
            ]
        )
        df = pl.DataFrame({"ssn": ["123-45-6789"], "status": ["active"]})
        result = polars_to_md(df, contract=contract)
        assert "123-45-6789" not in result
        assert "***" in result
        assert "active" in result

    def test_all_non_pii_no_masking(self) -> None:
        """Test that all non-PII columns are shown unmasked."""
        contract = self._make_contract(
            [
                Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                Column(name="count", type=ColumnType.INTEGER, nullable=False, pii=False),
            ]
        )
        df = pl.DataFrame({"id": [1], "count": [42]})
        result = polars_to_md(df, contract=contract)
        assert "***" not in result
        assert "1" in result
        assert "42" in result


class TestPiiMaskingFromExplicitList:
    """Tests for PII masking based on explicit pii_columns list."""

    def test_explicit_pii_columns(self) -> None:
        """Test masking from explicit pii_columns parameter."""
        df = pl.DataFrame({"name": ["Alice"], "age": [30]})
        result = polars_to_md(df, pii_columns=["name"])
        assert "Alice" not in result
        assert "***" in result
        assert "30" in result

    def test_pii_column_not_in_dataframe(self) -> None:
        """Test that non-existent pii columns are silently ignored."""
        df = pl.DataFrame({"a": [1], "b": [2]})
        result = polars_to_md(df, pii_columns=["nonexistent"])
        assert "***" not in result
        assert "1" in result


class TestPiiMaskingUnion:
    """Tests for union of contract + explicit PII columns."""

    def test_union_masking(self) -> None:
        """Test that contract and explicit pii_columns are unioned."""
        columns = [
            Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
            Column(name="email", type=ColumnType.STRING, nullable=False, pii=False),
            Column(name="status", type=ColumnType.STRING, nullable=False, pii=False),
        ]
        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test",
            layer="bronze",
            schema=Schema(columns=columns),
        )
        df = pl.DataFrame(
            {
                "patient_id": ["PAT-001"],
                "email": ["test@example.com"],
                "status": ["active"],
            }
        )
        # Contract masks patient_id; explicit list masks email
        result = polars_to_md(df, contract=contract, pii_columns=["email"])
        assert "PAT-001" not in result
        assert "test@example.com" not in result
        assert "active" in result


class TestCustomMaskValue:
    """Tests for custom mask value."""

    def test_custom_mask(self) -> None:
        """Test rendering with a custom mask value."""
        df = pl.DataFrame({"secret": ["hidden"]})
        result = polars_to_md(df, pii_columns=["secret"], mask_value="[REDACTED]")
        assert "[REDACTED]" in result
        assert "hidden" not in result


class TestTopLevelImport:
    """Tests for importing polars_to_md from top-level package."""

    def test_import(self) -> None:
        """Test polars_to_md is importable from moncpipelib."""
        from moncpipelib import polars_to_md as fn

        assert callable(fn)
