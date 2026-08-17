"""Tests for contract-driven query builder."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from moncpipelib.contracts.models import (
    Column,
    ColumnType,
    DataContract,
    Schema,
    TableReference,
)
from moncpipelib.testing.query_builder import AssetQueryBuilder


@pytest.fixture
def sample_contract() -> DataContract:
    """Contract with sources and sinks."""
    return DataContract(
        version="1.0",
        pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        asset="fda_ndc_package_silver",
        layer="silver",
        schema=Schema(
            columns=[
                Column(name="id", type=ColumnType.INTEGER, nullable=False),
            ]
        ),
        sources=[
            {
                "type": "table",
                "database": "analytics",
                "schema": "synthetic_bronze",
                "table": "fda_ndc_package_raw",
            }
        ],
        sinks=[
            {
                "type": "table",
                "database": "analytics",
                "schema": "synthetic_silver",
                "table": "fda_ndc_package",
            }
        ],
    )


@pytest.fixture
def contract_no_sources() -> DataContract:
    """Contract with no source tables."""
    return DataContract(
        version="1.0",
        pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        asset="test",
        layer="silver",
        schema=Schema(
            columns=[
                Column(name="id", type=ColumnType.INTEGER, nullable=False),
            ]
        ),
        sources=[],
    )


class TestAssetQueryBuilderProductionMode:
    """Tests for AssetQueryBuilder in production mode (no env vars)."""

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_production_mode_select_star(self, mock_load, sample_contract):
        mock_load.return_value = sample_contract

        with patch.dict("os.environ", {}, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")
            query = builder.get_source_query()

        assert query == "SELECT * FROM synthetic_bronze.fda_ndc_package_raw"

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_production_mode_with_columns(self, mock_load, sample_contract):
        mock_load.return_value = sample_contract

        with patch.dict("os.environ", {}, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")
            query = builder.get_source_query(columns=["id", "name"])

        assert query == "SELECT id, name FROM synthetic_bronze.fda_ndc_package_raw"

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_is_test_mode_false_by_default(self, mock_load, sample_contract):
        mock_load.return_value = sample_contract

        with patch.dict("os.environ", {}, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")

        assert builder.is_test_mode is False


class TestAssetQueryBuilderTestMode:
    """Tests for AssetQueryBuilder in test mode (env vars set)."""

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_test_mode_query(self, mock_load, sample_contract):
        mock_load.return_value = sample_contract

        env = {
            "ONCALYTICS_TEST_SCHEMA": "integration_tests",
            "ONCALYTICS_TEST_TABLE_PREFIX": "johndoe_abc123_",
        }
        with patch.dict("os.environ", env, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")
            query = builder.get_source_query()

        expected = (
            "SELECT * FROM integration_tests."
            "johndoe_abc123_synthetic_bronze_fda_ndc_package_raw_source"
        )
        assert query == expected

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_test_mode_with_columns(self, mock_load, sample_contract):
        mock_load.return_value = sample_contract

        env = {
            "ONCALYTICS_TEST_SCHEMA": "integration_tests",
            "ONCALYTICS_TEST_TABLE_PREFIX": "test_",
        }
        with patch.dict("os.environ", env, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")
            query = builder.get_source_query(columns=["id"])

        expected = (
            "SELECT id FROM integration_tests.test_synthetic_bronze_fda_ndc_package_raw_source"
        )
        assert query == expected

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_test_mode_no_prefix(self, mock_load, sample_contract):
        mock_load.return_value = sample_contract

        env = {"ONCALYTICS_TEST_SCHEMA": "test_schema"}
        with patch.dict("os.environ", env, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")
            query = builder.get_source_query()

        expected = "SELECT * FROM test_schema.synthetic_bronze_fda_ndc_package_raw_source"
        assert query == expected

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_is_test_mode_true_with_env(self, mock_load, sample_contract):
        mock_load.return_value = sample_contract

        with patch.dict("os.environ", {"ONCALYTICS_TEST_SCHEMA": "test"}, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")

        assert builder.is_test_mode is True


class TestAssetQueryBuilderErrors:
    """Tests for error handling in AssetQueryBuilder."""

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_missing_contract_raises(self, mock_load):
        mock_load.return_value = None

        with patch.dict("os.environ", {}, clear=True):
            builder = AssetQueryBuilder("nonexistent", "silver")

        with pytest.raises(ValueError, match="No contract found"):
            builder.get_source_query()

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_no_sources_raises(self, mock_load, contract_no_sources):
        mock_load.return_value = contract_no_sources

        with patch.dict("os.environ", {}, clear=True):
            builder = AssetQueryBuilder("test", "silver")

        with pytest.raises(ValueError, match="no source tables"):
            builder.get_source_query()

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_missing_contract_reference_raises(self, mock_load):
        mock_load.return_value = None

        with patch.dict("os.environ", {}, clear=True):
            builder = AssetQueryBuilder("nonexistent", "silver")

        with pytest.raises(ValueError, match="No contract found"):
            builder.get_source_table_reference()

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_no_sources_reference_raises(self, mock_load, contract_no_sources):
        mock_load.return_value = contract_no_sources

        with patch.dict("os.environ", {}, clear=True):
            builder = AssetQueryBuilder("test", "silver")

        with pytest.raises(ValueError, match="no source tables"):
            builder.get_source_table_reference()


class TestAssetQueryBuilderTableReference:
    """Tests for get_source_table_reference."""

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_returns_production_reference(self, mock_load, sample_contract):
        mock_load.return_value = sample_contract

        with patch.dict("os.environ", {}, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")
            ref = builder.get_source_table_reference()

        assert isinstance(ref, TableReference)
        assert ref.database == "analytics"
        assert ref.schema == "synthetic_bronze"
        assert ref.table == "fda_ndc_package_raw"

    @patch("moncpipelib.testing.query_builder.load_contract_for_asset")
    def test_returns_production_reference_even_in_test_mode(self, mock_load, sample_contract):
        """get_source_table_reference always returns the REAL table, not the test table."""
        mock_load.return_value = sample_contract

        env = {
            "ONCALYTICS_TEST_SCHEMA": "integration_tests",
            "ONCALYTICS_TEST_TABLE_PREFIX": "test_",
        }
        with patch.dict("os.environ", env, clear=True):
            builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")
            ref = builder.get_source_table_reference()

        # Should still return the real source table
        assert ref.schema == "synthetic_bronze"
        assert ref.table == "fda_ndc_package_raw"
