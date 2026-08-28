"""Row-level lineage tracking for data pipelines."""

from moncpipelib.config import LineageDefaults
from moncpipelib.lineage.column_metadata import ColumnMetadata
from moncpipelib.lineage.models import (
    ContractValidationRun,
    DataLineage,
    LineageBase,
    PeriodRegistry,
    PipelineRegistry,
    Scd2Reconciliation,
    lineage_metadata,
)
from moncpipelib.lineage.tracker import (
    LineageTracker,
    extract_timestamp_from_uuid7,
    generate_lineage_key,
    generate_uuid7,
    parse_lineage_key,
)

# OpenLineage integration (optional dependency)
try:
    from moncpipelib.lineage.openlineage import (
        ColumnClassificationFacet,
        DataPartitionFacet,
        MoncpipelibLineageFacet,
        OpenLineageConfig,
        OpenLineageEmitter,
        SourceFileFacet,
    )

    _OPENLINEAGE_AVAILABLE = True
except ImportError:
    _OPENLINEAGE_AVAILABLE = False
    # Define placeholder classes for type checking
    OpenLineageConfig = None  # type: ignore[assignment, misc]
    OpenLineageEmitter = None  # type: ignore[assignment, misc]
    MoncpipelibLineageFacet = None  # type: ignore[assignment, misc]
    ColumnClassificationFacet = None  # type: ignore[assignment, misc]
    DataPartitionFacet = None  # type: ignore[assignment, misc]
    SourceFileFacet = None  # type: ignore[assignment, misc]

__all__ = [
    # Configuration
    "LineageDefaults",
    # Models
    "ColumnMetadata",
    "ContractValidationRun",
    "DataLineage",
    "LineageBase",
    "PeriodRegistry",
    "PipelineRegistry",
    "Scd2Reconciliation",
    "lineage_metadata",
    # Tracker
    "LineageTracker",
    "generate_uuid7",
    "extract_timestamp_from_uuid7",
    "generate_lineage_key",
    "parse_lineage_key",
    # OpenLineage (optional)
    "OpenLineageConfig",
    "OpenLineageEmitter",
    "MoncpipelibLineageFacet",
    "ColumnClassificationFacet",
    "DataPartitionFacet",
    "SourceFileFacet",
]
