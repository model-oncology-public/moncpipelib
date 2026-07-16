"""Model Oncology Pipeline Library.

Shared utilities for Dagster data pipelines including database resources,
IO managers, and common data transformation helpers.
"""

from importlib.metadata import version

from moncpipelib.config import (
    MoncpipelibConfig,
    config,
    set_verbose_metadata,
    verbose_metadata,
)
from moncpipelib.contracts import (
    ContractCorpus,
    ContractEnforcementMode,
    DataContract,
    DataSource,
    FromIngestTemplate,
    IngestContract,
    LineageConfig,
    TableReference,
    TestingConfig,
    generate_asset_checks_from_contract,
    load_all_contracts,
    load_contract,
    load_contract_checks,
    load_contract_for_asset,
    load_data_source,
    load_ingest_contract,
)
from moncpipelib.contracts.models import Period
from moncpipelib.diagnostics import PodResourceSampler, SamplerConfig, SamplerMode
from moncpipelib.historical import (
    RegistryPartitionsDefinition,
    build_partitions_from_periods,
    build_partitions_from_registry,
    get_period_for_partition,
    get_period_from_registry,
    load_historical_periods,
)
from moncpipelib.ingest import (
    ApiResolverPattern,
    BlobRef,
    HttpUrlsPattern,
    IngestContext,
    IngestManifest,
    IngestPattern,
    IngestResolutionError,
    IngestResult,
    ManifestFileEntry,
    PartitionSpec,
    RawUrl,
    ReleaseResolver,
    UtsReleaseResolver,
    get_pattern,
    get_resolver,
    materialize_with_manifest,
    register_pattern,
    register_resolver,
    resolve_source_for_partition,
)
from moncpipelib.io_managers import (
    BulkInsertMethod,
    FullRefreshMethod,
    PostgresIOManager,
    WriteMode,
)
from moncpipelib.jobs import (
    make_reconciliation_asset,
    make_reconciliation_bundle,
    make_reconciliation_job,
)
from moncpipelib.lineage import LineageTracker
from moncpipelib.reference import EmptyPartitionedTableError, read_latest_partition
from moncpipelib.rendering import polars_to_md
from moncpipelib.resources import (
    BlobStorageResource,
    KeyVaultSecretResource,
    PostgresResource,
    WriteContext,
    WriteResult,
    read_batched,
    read_batched_to_dataframe,
)
from moncpipelib.scd import SCD2ChangeResult, detect_changes
from moncpipelib.sensors import (
    build_discovery_sensor,
    period_registry_sensor,
    reconciliation_sensor,
    registry_sensor,
    scd2_registry_sensor,
)
from moncpipelib.streaming import BatchedDataFrame, transform_batched
from moncpipelib.tags import ContractTags, RunTags
from moncpipelib.testing import (
    AssetQueryBuilder,
    SafeWhereClauseBuilder,
    SQLSafetyError,
)
from moncpipelib.transforms import (
    TextNormalizer,
    clean_text,
    compute_row_hash,
    normalize_ndc,
    safe_bool,
    safe_date,
    safe_datetime,
    safe_decimal,
    safe_int,
)
from moncpipelib.versioning import code_hash

__version__ = version("moncpipelib")

__all__ = [
    # Configuration
    "config",
    "MoncpipelibConfig",
    "set_verbose_metadata",
    "verbose_metadata",
    # Resources
    "BlobStorageResource",
    "KeyVaultSecretResource",
    "PostgresResource",
    "WriteContext",
    "WriteResult",
    "read_batched",
    "read_batched_to_dataframe",
    # Ingest (universal blob-landing boundary)
    "ApiResolverPattern",
    "BlobRef",
    "HttpUrlsPattern",
    "IngestContext",
    "IngestManifest",
    "IngestPattern",
    "IngestResolutionError",
    "IngestResult",
    "ManifestFileEntry",
    "PartitionSpec",
    "RawUrl",
    "ReleaseResolver",
    "UtsReleaseResolver",
    "get_pattern",
    "get_resolver",
    "materialize_with_manifest",
    "register_pattern",
    "register_resolver",
    "resolve_source_for_partition",
    # Streaming
    "BatchedDataFrame",
    "transform_batched",
    # IO Managers
    "BulkInsertMethod",
    "FullRefreshMethod",
    "PostgresIOManager",
    "WriteMode",
    # Historical
    "DataSource",
    "Period",
    "RegistryPartitionsDefinition",
    "build_partitions_from_periods",
    "build_partitions_from_registry",
    "get_period_for_partition",
    "get_period_from_registry",
    "load_data_source",
    "load_historical_periods",
    # Diagnostics
    "PodResourceSampler",
    "SamplerConfig",
    "SamplerMode",
    # Lineage
    "LineageTracker",
    # Reference (non-partitioned silver helpers)
    "EmptyPartitionedTableError",
    "read_latest_partition",
    # Rendering
    "polars_to_md",
    # SCD2
    "SCD2ChangeResult",
    "detect_changes",
    # Jobs
    "make_reconciliation_asset",
    "make_reconciliation_bundle",
    "make_reconciliation_job",
    # Sensors
    "build_discovery_sensor",
    "period_registry_sensor",
    "reconciliation_sensor",
    "registry_sensor",
    "scd2_registry_sensor",
    # Tags
    "ContractTags",
    "RunTags",
    # Contracts
    "ContractCorpus",
    "ContractEnforcementMode",
    "DataContract",
    "FromIngestTemplate",
    "IngestContract",
    "LineageConfig",
    "TableReference",
    "TestingConfig",
    "load_all_contracts",
    "load_contract",
    "load_contract_for_asset",
    "load_ingest_contract",
    "generate_asset_checks_from_contract",
    "load_contract_checks",
    # Testing
    "AssetQueryBuilder",
    "SafeWhereClauseBuilder",
    "SQLSafetyError",
    # Versioning
    "code_hash",
    # Transforms
    "TextNormalizer",
    "clean_text",
    "compute_row_hash",
    "normalize_ndc",
    "safe_bool",
    "safe_date",
    "safe_datetime",
    "safe_decimal",
    "safe_int",
]
