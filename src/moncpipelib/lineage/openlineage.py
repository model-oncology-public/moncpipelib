"""OpenLineage event emission for external lineage backends.

This module provides integration with OpenLineage-compatible backends
like Marquez and DataHub, enabling lineage events to be emitted during
asset materialization.

Note: This module requires the 'openlineage' optional dependency.
Install with: pip install moncpipelib[openlineage]
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dagster import ConfigurableResource

from moncpipelib.config import config as moncpipelib_config

if TYPE_CHECKING:
    import polars as pl
    from openlineage.client import OpenLineageClient
    from openlineage.client.event_v2 import InputDataset, Job, OutputDataset, Run, RunEvent
    from openlineage.client.facet_v2 import DatasetFacet, JobFacet, RunFacet


def _get_schema_url_base() -> str:
    """Get the schema URL base from configuration.

    This allows the URL to be overridden via environment variable
    MONCPIPELIB_OPENLINEAGE_SCHEMA_URL.
    """
    return moncpipelib_config.openlineage.schema_url_base


# For backwards compatibility, provide module-level constant
# Note: This is evaluated at import time; use _get_schema_url_base() for dynamic access
SCHEMA_URL_BASE = _get_schema_url_base()

logger = logging.getLogger(__name__)


def _check_openlineage_available() -> bool:
    """Check if openlineage-python is installed."""
    try:
        import openlineage.client  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class MoncpipelibLineageFacet:
    """Custom facet for moncpipelib lineage metadata.

    Contains lineage tracking information specific to moncpipelib pipelines.
    """

    lineage_id: str
    lineage_key: str
    layer: str
    is_backfill: bool = False
    pipeline_id: str | None = None
    parent_lineage_ids: list[str] = field(default_factory=list)

    _schemaURL: str = field(
        default=f"{SCHEMA_URL_BASE}MoncpipelibLineageFacet/1-0-0/MoncpipelibLineageFacet.json",
        init=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenLineage-compatible dictionary."""
        result: dict[str, Any] = {
            "_schemaURL": self._schemaURL,
            "lineage_id": self.lineage_id,
            "lineage_key": self.lineage_key,
            "layer": self.layer,
            "is_backfill": self.is_backfill,
            "parent_lineage_ids": self.parent_lineage_ids,
        }
        if self.pipeline_id:
            result["pipeline_id"] = self.pipeline_id
        return result


@dataclass
class DataPartitionFacet:
    """Custom facet for data partition information."""

    data_date: str | None = None
    data_date_start: str | None = None
    data_date_end: str | None = None

    _schemaURL: str = field(
        default=f"{SCHEMA_URL_BASE}DataPartitionFacet/1-0-0/DataPartitionFacet.json",
        init=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenLineage-compatible dictionary."""
        result: dict[str, Any] = {"_schemaURL": self._schemaURL}
        if self.data_date:
            result["data_date"] = self.data_date
        if self.data_date_start:
            result["data_date_start"] = self.data_date_start
        if self.data_date_end:
            result["data_date_end"] = self.data_date_end
        return result


@dataclass
class SourceFileFacet:
    """Custom facet for source file information."""

    source_file: str | None = None
    source_system: str | None = None
    file_format: str | None = None

    _schemaURL: str = field(
        default=f"{SCHEMA_URL_BASE}SourceFileFacet/1-0-0/SourceFileFacet.json",
        init=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenLineage-compatible dictionary."""
        result: dict[str, Any] = {"_schemaURL": self._schemaURL}
        if self.source_file:
            result["source_file"] = self.source_file
        if self.source_system:
            result["source_system"] = self.source_system
        if self.file_format:
            result["file_format"] = self.file_format
        return result


@dataclass
class ColumnClassificationFacet:
    """Custom facet for column-level PII / PHI classification.

    Records which columns are classified as PII and which as PHI so that
    downstream consumers (Marquez, DataHub, governance tools) can surface
    data sensitivity metadata alongside lineage. Distinguishing the two
    lets HIPAA-context consumers affirmatively clear a column of PHI
    instead of treating every PII column as PHI-suspect (#391).
    """

    pii_columns: list[str] = field(default_factory=list)
    phi_columns: list[str] = field(default_factory=list)

    _schemaURL: str = field(
        default=f"{SCHEMA_URL_BASE}ColumnClassificationFacet/1-0-0/ColumnClassificationFacet.json",
        init=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenLineage-compatible dictionary."""
        return {
            "_schemaURL": self._schemaURL,
            "pii_columns": self.pii_columns,
            "phi_columns": self.phi_columns,
        }


class OpenLineageConfig(ConfigurableResource):
    """Configuration for OpenLineage event emission.

    This Dagster resource configures how OpenLineage events are emitted
    to external lineage backends.

    Attributes:
        url: The OpenLineage API endpoint URL
        namespace: The namespace for jobs and datasets. Defaults to value from
                   MONCPIPELIB_OPENLINEAGE_NAMESPACE env var or "moncpipelib"
        api_key: Optional API key for authentication
        timeout: HTTP timeout in seconds (default: 10)
        enabled: Whether to emit events (default: True)
    """

    url: str
    namespace: str = moncpipelib_config.openlineage.namespace
    api_key: str | None = None
    timeout: float = 10.0
    enabled: bool = True


class OpenLineageEmitter:
    """Emits OpenLineage events to external lineage backends.

    This class handles the emission of OpenLineage-compatible events
    during asset materialization. It creates START/COMPLETE/FAIL run events
    with appropriate facets for input and output datasets.

    Example:
        ```python
        from moncpipelib.lineage import OpenLineageEmitter, OpenLineageConfig

        config = OpenLineageConfig(
            url="http://marquez:5000",
            namespace="my-pipeline",
        )
        emitter = OpenLineageEmitter(config)

        # Emit start event
        run_id = emitter.emit_start(job_name="orders_bronze")

        # Emit complete event with dataset info
        emitter.emit_complete(
            job_name="orders_bronze",
            run_id=run_id,
            output_dataset="bronze.orders",
            row_count=1000,
        )
        ```
    """

    def __init__(self, config: OpenLineageConfig) -> None:
        """Initialize the emitter with configuration.

        Args:
            config: OpenLineage configuration resource
        """
        self.config = config
        self._client: OpenLineageClient | None = None

    @property
    def client(self) -> "OpenLineageClient":
        """Lazy-load the OpenLineage client."""
        if self._client is None:
            if not _check_openlineage_available():
                raise ImportError(
                    "openlineage-python is not installed. "
                    "Install with: pip install moncpipelib[openlineage]"
                )
            from openlineage.client import OpenLineageClient
            from openlineage.client.transport.http import (
                ApiKeyTokenProvider,
                HttpConfig,
                HttpTransport,
            )

            transport_config = HttpConfig(
                url=self.config.url,
                timeout=self.config.timeout,
            )
            if self.config.api_key:
                transport_config.auth = ApiKeyTokenProvider({"apiKey": self.config.api_key})

            transport = HttpTransport(transport_config)
            self._client = OpenLineageClient(transport=transport)
        return self._client

    def _create_job(
        self,
        job_name: str,
        facets: "dict[str, JobFacet] | None" = None,
    ) -> "Job":
        """Create an OpenLineage Job object."""
        from openlineage.client.event_v2 import Job

        return Job(
            namespace=self.config.namespace,
            name=job_name,
            facets=facets,
        )

    def _create_run(
        self,
        run_id: str,
        facets: "dict[str, RunFacet] | None" = None,
    ) -> "Run":
        """Create an OpenLineage Run object."""
        from openlineage.client.event_v2 import Run

        return Run(
            runId=run_id,
            facets=facets,
        )

    def _create_input_dataset(
        self,
        name: str,
        namespace: str | None = None,
        facets: "dict[str, DatasetFacet] | None" = None,
    ) -> "InputDataset":
        """Create an OpenLineage InputDataset object."""
        from openlineage.client.event_v2 import InputDataset

        return InputDataset(
            namespace=namespace or self.config.namespace,
            name=name,
            facets=facets,
        )

    def _create_output_dataset(
        self,
        name: str,
        namespace: str | None = None,
        facets: "dict[str, DatasetFacet] | None" = None,
    ) -> "OutputDataset":
        """Create an OpenLineage OutputDataset object."""
        from openlineage.client.event_v2 import OutputDataset

        return OutputDataset(
            namespace=namespace or self.config.namespace,
            name=name,
            facets=facets,
        )

    def _build_schema_facet(self, df: "pl.DataFrame") -> Any:
        """Build a schema facet from a Polars DataFrame."""
        from openlineage.client.facet_v2 import schema_dataset

        polars_to_ol_type: dict[str, str] = {
            "Int8": "INTEGER",
            "Int16": "INTEGER",
            "Int32": "INTEGER",
            "Int64": "BIGINT",
            "UInt8": "INTEGER",
            "UInt16": "INTEGER",
            "UInt32": "INTEGER",
            "UInt64": "BIGINT",
            "Float32": "FLOAT",
            "Float64": "DOUBLE",
            "Boolean": "BOOLEAN",
            "String": "VARCHAR",
            "Utf8": "VARCHAR",
            "Date": "DATE",
            "Datetime": "TIMESTAMP",
            "Time": "TIME",
            "Duration": "INTERVAL",
            "Decimal": "DECIMAL",
            "Binary": "BINARY",
            "Null": "NULL",
        }

        fields = []
        for col_name in df.columns:
            dtype = str(df[col_name].dtype)
            # Extract base type (handle parametrized types like Datetime(us, ...))
            base_type = dtype.split("(")[0] if "(" in dtype else dtype
            ol_type = polars_to_ol_type.get(base_type, "VARCHAR")
            fields.append(
                schema_dataset.SchemaDatasetFacetFields(
                    name=col_name,
                    type=ol_type,
                )
            )

        return schema_dataset.SchemaDatasetFacet(fields=fields)

    def _build_custom_facets(
        self,
        lineage_id: str | None = None,
        lineage_key: str | None = None,
        layer: str | None = None,
        is_backfill: bool = False,
        pipeline_id: str | None = None,
        parent_lineage_ids: list[str] | None = None,
        source_file: str | None = None,
        data_date: str | None = None,
        pii_columns: list[str] | None = None,
        phi_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build custom facets as a dictionary."""
        facets: dict[str, Any] = {}

        if lineage_id and lineage_key and layer:
            facets["moncpipelibLineage"] = MoncpipelibLineageFacet(
                lineage_id=lineage_id,
                lineage_key=lineage_key,
                layer=layer,
                is_backfill=is_backfill,
                pipeline_id=pipeline_id,
                parent_lineage_ids=parent_lineage_ids or [],
            )

        if source_file:
            facets["sourceFile"] = SourceFileFacet(source_file=source_file)

        if pii_columns or phi_columns:
            facets["columnClassification"] = ColumnClassificationFacet(
                pii_columns=pii_columns or [],
                phi_columns=phi_columns or [],
            )

        if data_date:
            facets["dataPartition"] = DataPartitionFacet(data_date=data_date)

        return facets

    def emit_start(
        self,
        job_name: str,
        run_id: str | None = None,
        input_datasets: list[str] | None = None,
    ) -> str:
        """Emit a START run event.

        Args:
            job_name: Name of the job (typically asset name)
            run_id: Optional run ID (generated if not provided)
            input_datasets: List of input dataset names

        Returns:
            The run ID used for this event
        """
        if not self.config.enabled:
            return run_id or str(uuid4())

        from openlineage.client.event_v2 import RunEvent, RunState

        # Generate a valid UUID if not provided
        run_id = run_id or str(uuid4())

        inputs = []
        if input_datasets:
            inputs = [self._create_input_dataset(name) for name in input_datasets]

        event = RunEvent(
            eventTime=datetime.now(UTC).isoformat(),
            producer=f"{SCHEMA_URL_BASE}producer",
            run=self._create_run(run_id),
            job=self._create_job(job_name),
            eventType=RunState.START,
            inputs=inputs,
            outputs=[],
        )

        self._emit_event(event)
        return run_id

    def emit_complete(
        self,
        job_name: str,
        run_id: str,
        output_dataset: str | None = None,
        row_count: int | None = None,
        df: "pl.DataFrame | None" = None,
        lineage_id: str | None = None,
        lineage_key: str | None = None,
        layer: str | None = None,
        is_backfill: bool = False,
        pipeline_id: str | None = None,
        parent_lineage_ids: list[str] | None = None,
        source_file: str | None = None,
        data_date: str | None = None,
        input_datasets: list[str] | None = None,
        pii_columns: list[str] | None = None,
        phi_columns: list[str] | None = None,
    ) -> None:
        """Emit a COMPLETE run event.

        Args:
            job_name: Name of the job (typically asset name)
            run_id: Run ID from the start event
            output_dataset: Name of the output dataset (e.g., "bronze.orders")
            row_count: Number of rows written
            df: Optional DataFrame for schema extraction
            lineage_id: Lineage ID for custom facet
            lineage_key: Lineage key for custom facet
            layer: Data layer (bronze/silver/gold)
            is_backfill: Whether this is a backfill run
            parent_lineage_ids: Parent lineage IDs for lineage tracking
            source_file: Source file path
            data_date: Data date for partition facet
            input_datasets: List of input dataset names
            pii_columns: Column names classified as PII for classification facet
            phi_columns: Column names classified as PHI for classification facet
        """
        if not self.config.enabled:
            return

        from openlineage.client.event_v2 import RunEvent, RunState
        from openlineage.client.facet_v2 import output_statistics_output_dataset

        outputs = []
        if output_dataset:
            output_facets: dict[str, Any] = {}

            # Add schema facet if DataFrame provided
            if df is not None:
                output_facets["schema"] = self._build_schema_facet(df)

            # Add row count facet
            if row_count is not None:
                output_facets["outputStatistics"] = (
                    output_statistics_output_dataset.OutputStatisticsOutputDatasetFacet(
                        rowCount=row_count,
                    )
                )

            # Add custom facets
            custom_facets = self._build_custom_facets(
                lineage_id=lineage_id,
                lineage_key=lineage_key,
                layer=layer,
                is_backfill=is_backfill,
                pipeline_id=pipeline_id,
                parent_lineage_ids=parent_lineage_ids,
                source_file=source_file,
                data_date=data_date,
                pii_columns=pii_columns,
                phi_columns=phi_columns,
            )
            output_facets.update(custom_facets)

            outputs.append(
                self._create_output_dataset(output_dataset, facets=output_facets or None)
            )

        inputs = []
        if input_datasets:
            inputs = [self._create_input_dataset(name) for name in input_datasets]

        event = RunEvent(
            eventTime=datetime.now(UTC).isoformat(),
            producer=f"{SCHEMA_URL_BASE}producer",
            run=self._create_run(run_id),
            job=self._create_job(job_name),
            eventType=RunState.COMPLETE,
            inputs=inputs,
            outputs=outputs,
        )

        self._emit_event(event)

    def emit_fail(
        self,
        job_name: str,
        run_id: str,
        error_message: str | None = None,
        input_datasets: list[str] | None = None,
    ) -> None:
        """Emit a FAIL run event.

        Args:
            job_name: Name of the job (typically asset name)
            run_id: Run ID from the start event
            error_message: Optional error message
            input_datasets: List of input dataset names
        """
        if not self.config.enabled:
            return

        from openlineage.client.event_v2 import RunEvent, RunState
        from openlineage.client.facet_v2 import error_message_run

        run_facets: dict[str, Any] = {}
        if error_message:
            run_facets["errorMessage"] = error_message_run.ErrorMessageRunFacet(
                message=error_message,
                programmingLanguage="python",
            )

        inputs = []
        if input_datasets:
            inputs = [self._create_input_dataset(name) for name in input_datasets]

        event = RunEvent(
            eventTime=datetime.now(UTC).isoformat(),
            producer=f"{SCHEMA_URL_BASE}producer",
            run=self._create_run(run_id, facets=run_facets or None),
            job=self._create_job(job_name),
            eventType=RunState.FAIL,
            inputs=inputs,
            outputs=[],
        )

        self._emit_event(event)

    def _emit_event(self, event: "RunEvent") -> None:
        """Emit an event, handling errors gracefully.

        Emission failures are logged but don't fail the asset.
        """
        try:
            self.client.emit(event)
        except Exception as e:
            # Non-blocking: log warning but don't fail
            logger.warning(
                f"Failed to emit OpenLineage event: {e}. Continuing without lineage emission."
            )
