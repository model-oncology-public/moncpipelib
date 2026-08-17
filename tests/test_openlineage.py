"""Tests for OpenLineage integration."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from moncpipelib.lineage import (
    ColumnClassificationFacet,
    DataPartitionFacet,
    MoncpipelibLineageFacet,
    OpenLineageConfig,
    OpenLineageEmitter,
    SourceFileFacet,
)


class TestCustomFacets:
    """Tests for custom facet dataclasses."""

    def test_moncpipelib_lineage_facet_to_dict(self):
        """Test MoncpipelibLineageFacet serialization."""
        facet = MoncpipelibLineageFacet(
            lineage_id="abc-123",
            lineage_key="v1:asset:bronze:2024-01-15:run123",
            layer="bronze",
            is_backfill=True,
            parent_lineage_ids=["parent-1", "parent-2"],
        )

        result = facet.to_dict()

        assert result["lineage_id"] == "abc-123"
        assert result["lineage_key"] == "v1:asset:bronze:2024-01-15:run123"
        assert result["layer"] == "bronze"
        assert result["is_backfill"] is True
        assert result["parent_lineage_ids"] == ["parent-1", "parent-2"]
        assert "_schemaURL" in result

    def test_moncpipelib_lineage_facet_defaults(self):
        """Test MoncpipelibLineageFacet with default values."""
        facet = MoncpipelibLineageFacet(
            lineage_id="abc-123",
            lineage_key="v1:asset:bronze:2024-01-15:run123",
            layer="bronze",
        )

        result = facet.to_dict()

        assert result["is_backfill"] is False
        assert result["parent_lineage_ids"] == []

    def test_data_partition_facet_to_dict(self):
        """Test DataPartitionFacet serialization."""
        facet = DataPartitionFacet(
            data_date="2024-01-15",
            data_date_start="2024-01-01",
            data_date_end="2024-01-31",
        )

        result = facet.to_dict()

        assert result["data_date"] == "2024-01-15"
        assert result["data_date_start"] == "2024-01-01"
        assert result["data_date_end"] == "2024-01-31"
        assert "_schemaURL" in result

    def test_data_partition_facet_omits_none_values(self):
        """Test DataPartitionFacet omits None values."""
        facet = DataPartitionFacet(data_date="2024-01-15")

        result = facet.to_dict()

        assert result["data_date"] == "2024-01-15"
        assert "data_date_start" not in result
        assert "data_date_end" not in result

    def test_source_file_facet_to_dict(self):
        """Test SourceFileFacet serialization."""
        facet = SourceFileFacet(
            source_file="blob://container/file.csv",
            source_system="azure_blob",
            file_format="csv",
        )

        result = facet.to_dict()

        assert result["source_file"] == "blob://container/file.csv"
        assert result["source_system"] == "azure_blob"
        assert result["file_format"] == "csv"
        assert "_schemaURL" in result

    def test_source_file_facet_omits_none_values(self):
        """Test SourceFileFacet omits None values."""
        facet = SourceFileFacet(source_file="blob://container/file.csv")

        result = facet.to_dict()

        assert result["source_file"] == "blob://container/file.csv"
        assert "source_system" not in result
        assert "file_format" not in result

    def test_column_classification_facet_to_dict(self):
        """Test ColumnClassificationFacet serializes pii and phi columns (#391)."""
        facet = ColumnClassificationFacet(
            pii_columns=["patient_id", "ssn", "provider_npi"],
            phi_columns=["patient_id", "ssn"],
        )

        result = facet.to_dict()

        assert result["pii_columns"] == ["patient_id", "ssn", "provider_npi"]
        assert result["phi_columns"] == ["patient_id", "ssn"]
        assert "_schemaURL" in result

    def test_column_classification_facet_defaults(self):
        """Test ColumnClassificationFacet defaults to empty lists."""
        facet = ColumnClassificationFacet()

        result = facet.to_dict()

        assert result["pii_columns"] == []
        assert result["phi_columns"] == []


class TestOpenLineageConfig:
    """Tests for OpenLineageConfig resource."""

    def test_config_with_required_fields(self):
        """Test creating config with required fields only."""
        config = OpenLineageConfig(url="http://marquez:5000")

        assert config.url == "http://marquez:5000"
        assert config.namespace == "moncpipelib"
        assert config.api_key is None
        assert config.timeout == 10.0
        assert config.enabled is True

    def test_config_with_all_fields(self):
        """Test creating config with all fields."""
        config = OpenLineageConfig(
            url="http://marquez:5000",
            namespace="my-pipeline",
            api_key="secret-key",
            timeout=30.0,
            enabled=False,
        )

        assert config.url == "http://marquez:5000"
        assert config.namespace == "my-pipeline"
        assert config.api_key == "secret-key"
        assert config.timeout == 30.0
        assert config.enabled is False


class TestOpenLineageEmitter:
    """Tests for OpenLineageEmitter."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return OpenLineageConfig(
            url="http://marquez:5000",
            namespace="test-namespace",
        )

    @pytest.fixture
    def disabled_config(self):
        """Create a disabled config."""
        return OpenLineageConfig(
            url="http://marquez:5000",
            enabled=False,
        )

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
                "value": [1.0, 2.0, 3.0],
                "created_at": [pl.date(2024, 1, 1)] * 3,
            }
        )

    def test_emit_start_returns_run_id(self, config):
        """Test emit_start returns a run ID."""
        emitter = OpenLineageEmitter(config)

        with patch.object(emitter, "_emit_event"):
            run_id = emitter.emit_start(job_name="test_job")

            assert run_id is not None
            assert len(run_id) > 0

    def test_emit_start_uses_provided_run_id(self, config):
        """Test emit_start uses provided run ID."""
        emitter = OpenLineageEmitter(config)
        # Use a valid UUID format
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"

        with patch.object(emitter, "_emit_event"):
            run_id = emitter.emit_start(job_name="test_job", run_id=valid_uuid)

            assert run_id == valid_uuid

    def test_emit_start_disabled_skips_emission(self, disabled_config):
        """Test emit_start skips emission when disabled."""
        emitter = OpenLineageEmitter(disabled_config)
        emitter._client = MagicMock()

        run_id = emitter.emit_start(job_name="test_job")

        assert run_id is not None
        emitter._client.emit.assert_not_called()

    def test_emit_complete_disabled_skips_emission(self, disabled_config):
        """Test emit_complete skips emission when disabled."""
        emitter = OpenLineageEmitter(disabled_config)
        emitter._client = MagicMock()

        emitter.emit_complete(job_name="test_job", run_id="550e8400-e29b-41d4-a716-446655440000")

        emitter._client.emit.assert_not_called()

    def test_emit_fail_disabled_skips_emission(self, disabled_config):
        """Test emit_fail skips emission when disabled."""
        emitter = OpenLineageEmitter(disabled_config)
        emitter._client = MagicMock()

        emitter.emit_fail(job_name="test_job", run_id="550e8400-e29b-41d4-a716-446655440000")

        emitter._client.emit.assert_not_called()

    def test_build_schema_facet(self, config, sample_df):
        """Test schema facet is built from DataFrame."""
        emitter = OpenLineageEmitter(config)

        facet = emitter._build_schema_facet(sample_df)

        assert len(facet.fields) == 4
        field_names = [f.name for f in facet.fields]
        assert "id" in field_names
        assert "name" in field_names
        assert "value" in field_names
        assert "created_at" in field_names

    def test_build_custom_facets_with_lineage(self, config):
        """Test custom facets include lineage when provided."""
        emitter = OpenLineageEmitter(config)

        facets = emitter._build_custom_facets(
            lineage_id="abc-123",
            lineage_key="v1:asset:bronze:2024-01-15:run123",
            layer="bronze",
        )

        assert "moncpipelibLineage" in facets

    def test_build_custom_facets_with_source_file(self, config):
        """Test custom facets include source file when provided."""
        emitter = OpenLineageEmitter(config)

        facets = emitter._build_custom_facets(source_file="blob://file.csv")

        assert "sourceFile" in facets

    def test_build_custom_facets_with_data_date(self, config):
        """Test custom facets include data partition when provided."""
        emitter = OpenLineageEmitter(config)

        facets = emitter._build_custom_facets(data_date="2024-01-15")

        assert "dataPartition" in facets

    def test_build_custom_facets_with_pii_and_phi_columns(self, config):
        """Test classification facet carries both pii and phi columns (#391)."""
        emitter = OpenLineageEmitter(config)

        facets = emitter._build_custom_facets(
            pii_columns=["patient_id", "provider_npi"],
            phi_columns=["patient_id"],
        )

        facet = facets["columnClassification"]
        assert facet.pii_columns == ["patient_id", "provider_npi"]
        assert facet.phi_columns == ["patient_id"]

    def test_build_custom_facets_with_phi_columns_only(self, config):
        """Test phi columns alone are enough to build the classification facet."""
        emitter = OpenLineageEmitter(config)

        facets = emitter._build_custom_facets(phi_columns=["lab_value"])

        facet = facets["columnClassification"]
        assert facet.pii_columns == []
        assert facet.phi_columns == ["lab_value"]

    def test_emit_event_handles_errors_gracefully(self, config):
        """Test _emit_event logs warning on error but doesn't raise."""
        emitter = OpenLineageEmitter(config)
        mock_client = MagicMock()
        mock_client.emit.side_effect = Exception("Connection failed")
        emitter._client = mock_client

        # Should not raise
        with patch("moncpipelib.lineage.openlineage.logger") as mock_logger:
            mock_event = MagicMock()
            emitter._emit_event(mock_event)

            # Should have logged a warning
            mock_logger.warning.assert_called_once()

    def test_emit_complete_with_dataframe(self, config, sample_df):
        """Test emit_complete includes schema when DataFrame provided."""
        emitter = OpenLineageEmitter(config)

        with patch.object(emitter, "_emit_event") as mock_emit:
            emitter.emit_complete(
                job_name="test_job",
                run_id="550e8400-e29b-41d4-a716-446655440000",
                output_dataset="bronze.test_table",
                df=sample_df,
                row_count=3,
            )

            mock_emit.assert_called_once()
            event = mock_emit.call_args[0][0]
            assert len(event.outputs) == 1
            assert event.outputs[0].name == "bronze.test_table"

    def test_emit_complete_with_all_custom_facets(self, config, sample_df):
        """Test emit_complete includes all custom facets."""
        emitter = OpenLineageEmitter(config)

        with patch.object(emitter, "_emit_event") as mock_emit:
            emitter.emit_complete(
                job_name="test_job",
                run_id="550e8400-e29b-41d4-a716-446655440000",
                output_dataset="bronze.test_table",
                df=sample_df,
                row_count=3,
                lineage_id="abc-123",
                lineage_key="v1:asset:bronze:2024-01-15:run123",
                layer="bronze",
                is_backfill=True,
                parent_lineage_ids=["parent-1"],
                source_file="blob://file.csv",
                data_date="2024-01-15",
            )

            mock_emit.assert_called_once()

    def test_emit_fail_with_error_message(self, config):
        """Test emit_fail includes error message facet."""
        emitter = OpenLineageEmitter(config)

        with patch.object(emitter, "_emit_event") as mock_emit:
            emitter.emit_fail(
                job_name="test_job",
                run_id="550e8400-e29b-41d4-a716-446655440000",
                error_message="Something went wrong",
            )

            mock_emit.assert_called_once()
            event = mock_emit.call_args[0][0]
            assert event.run.facets is not None

    def test_emit_start_with_input_datasets(self, config):
        """Test emit_start includes input datasets."""
        emitter = OpenLineageEmitter(config)

        with patch.object(emitter, "_emit_event") as mock_emit:
            emitter.emit_start(
                job_name="test_job",
                input_datasets=["source.table1", "source.table2"],
            )

            mock_emit.assert_called_once()
            event = mock_emit.call_args[0][0]
            assert len(event.inputs) == 2

    def test_emit_complete_with_input_datasets(self, config):
        """Test emit_complete includes input datasets."""
        emitter = OpenLineageEmitter(config)

        with patch.object(emitter, "_emit_event") as mock_emit:
            emitter.emit_complete(
                job_name="test_job",
                run_id="550e8400-e29b-41d4-a716-446655440000",
                output_dataset="bronze.test_table",
                input_datasets=["source.table1"],
            )

            mock_emit.assert_called_once()
            event = mock_emit.call_args[0][0]
            assert len(event.inputs) == 1
