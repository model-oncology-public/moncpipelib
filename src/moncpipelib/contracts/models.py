"""Data contract model definitions.

This module contains dataclasses representing data contracts, including
schema definitions, column tests, and table expectations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, overload

from moncpipelib.config import DEFAULT_DATABASE

if TYPE_CHECKING:
    from moncpipelib.contracts.checks_types import (
        AcceptedValues,
        Between,
        Freshness,
        GreaterThan,
        GreaterThanOrEqual,
        LessThan,
        LessThanOrEqual,
        MaxLength,
        MinLength,
        NotIn,
        NotInFuture,
        NotNull,
        NullPercentage,
        Pattern,
        RowCount,
        Unique,
        UniqueCombination,
        WithinDays,
    )

    ColumnTestType: TypeAlias = (
        "ColumnTest"
        | NotNull
        | Unique
        | AcceptedValues
        | NotIn
        | Pattern
        | GreaterThan
        | GreaterThanOrEqual
        | LessThan
        | LessThanOrEqual
        | Between
        | MinLength
        | MaxLength
        | NotInFuture
        | WithinDays
    )

    TableExpectationType: TypeAlias = (
        "TableExpectation" | RowCount | Freshness | NullPercentage | UniqueCombination
    )

_T = TypeVar("_T")
_UNSET: Any = object()

logger = logging.getLogger(__name__)


class Severity(StrEnum):
    """Validation severity level."""

    ERROR = "error"
    WARN = "warn"


class ColumnType(StrEnum):
    """Supported column data types."""

    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    UUID = "uuid"
    JSON = "json"
    JSONB = "jsonb"


class ContractEnforcementMode(StrEnum):
    """How the IO Manager enforces contracts at write time."""

    ERROR = "error"  # Raise exception on validation failure
    WARN = "warn"  # Log warnings but continue
    SILENT = "silent"  # Skip validation entirely


@dataclass
class ColumnTest:
    """A validation test for a column.

    Attributes:
        test_type: The type of test (e.g., "not_null", "unique", "pattern")
        parameters: Test-specific parameters (e.g., {"values": ["a", "b"]} for accepted_values)
        severity: Whether failure is an error or warning
        when: Optional condition (e.g., "not_null" to only test non-null values)
    """

    test_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.ERROR
    when: str | None = None


@dataclass
class Column:
    """Column definition in a data contract.

    Attributes:
        name: Column name
        type: Data type (string, integer, decimal, etc.)
        nullable: Whether nulls are allowed
        description: Human-readable description
        primary_key: Whether this column is part of the primary key
        managed: Whether this column is auto-managed by moncpipelib
        pii: Whether this column contains personally identifiable information.
            Defaults to True (safe by default). Engineers must explicitly set
            pii: false to opt out of PII protections. This controls masking in
            polars_to_md(), PII column comments in PostgreSQL, OpenLineage
            classification facets, and Dagster output metadata.
        phi: Whether this column contains protected health information under
            HIPAA. Defaults to the column's pii value when unset, so existing
            contracts stay valid and fail-safe (after __post_init__ it is
            always a bool). PII and PHI diverge for e.g. provider or business
            identifiers (PII but not PHI) and de-identified clinical values
            (neither). Synced to lineage.column_metadata tags and the
            OpenLineage classification facet alongside pii, so HIPAA-context
            consumers can affirmatively clear columns (phi: false) instead of
            falling back to schema-name heuristics.
        tests: List of validation tests for this column. Can be ColumnTest instances
               or typed test classes from checks_types (NotNull, Unique, etc.)
    """

    name: str
    type: ColumnType
    nullable: bool
    description: str | None = None
    primary_key: bool = False
    managed: bool = False
    pii: bool = True
    phi: bool | None = None
    tests: Sequence[ColumnTestType] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.phi is None:
            self.phi = self.pii


@dataclass
class TableExpectation:
    """Table-level validation expectation.

    Attributes:
        expectation_type: Type of expectation (e.g., "row_count", "freshness")
        parameters: Expectation-specific parameters
        severity: Whether failure is an error or warning
    """

    expectation_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.ERROR


@dataclass
class Owner:
    """Ownership metadata for an asset.

    Attributes:
        team: Name of the owning team
        contact: Email contact for the team
        slack_channel: Slack channel for alerts
    """

    team: str
    contact: str | None = None
    slack_channel: str | None = None


@dataclass
class UpstreamDependency:
    """Documentation of upstream data source.

    Attributes:
        name: Name of the upstream source
        type: Type of dependency ("asset" or "external")
        system: System name for external dependencies
        description: Human-readable description
    """

    name: str
    type: str  # "asset" or "external"
    system: str | None = None
    description: str | None = None


@dataclass
class SLA:
    """Service level agreement metadata.

    Attributes:
        freshness_hours: Maximum age of data in hours
        update_frequency: How often data should be updated (e.g., "daily")
        availability_percent: Target availability percentage
    """

    freshness_hours: int | None = None
    update_frequency: str | None = None
    availability_percent: float | None = None


@dataclass
class TableReference:
    """Reference to a database table.

    Attributes:
        database: Database name (e.g., 'analytics')
        schema: Schema name (e.g., 'synthetic_bronze')
        table: Table name (e.g., 'fda_ndc_package_raw')
    """

    database: str
    schema: str
    table: str

    @property
    def fully_qualified_name(self) -> str:
        """Return database.schema.table."""
        return f"{self.database}.{self.schema}.{self.table}"

    @property
    def schema_qualified_name(self) -> str:
        """Return schema.table."""
        return f"{self.schema}.{self.table}"


@dataclass
class LineageConfig:
    """Row-level lineage configuration for a data contract.

    Controls how moncpipelib creates lineage records when writing data for
    this asset.  When omitted from the contract YAML, lineage uses default
    behavior (enabled, no source_system or transformation_type).

    Attributes:
        enabled: Whether row-level lineage tracking is enabled for this asset.
            Defaults to True.  Set False to skip lineage even when the
            resource has ``enable_row_lineage=True``.
        source_system: External system identifier (e.g., ``"openfda"``,
            ``"sftp"``, ``"api"``).  Stored in the lineage record for
            provenance tracking.
        transformation_type: Type of transformation applied (e.g.,
            ``"ingest"``, ``"aggregate"``, ``"join"``, ``"filter"``).
            Stored in the lineage record.
    """

    enabled: bool = True
    source_system: str | None = None
    transformation_type: str | None = None


@dataclass
class TestingConfig:
    """Integration testing configuration for a data contract.

    Defines how to set up and validate integration tests for this pipeline.

    Attributes:
        enabled: Whether integration testing is enabled for this contract.
        source_row_limit: Maximum rows to copy from source for testing.
        source_where_clause: Optional WHERE clause filter for test data selection.
        expected_min_rows: Minimum expected output row count.
        expected_max_rows: Maximum expected output row count.
        timeout_seconds: Test execution timeout in seconds.
    """

    enabled: bool = True
    source_row_limit: int = 1000
    source_where_clause: str | None = None
    expected_min_rows: int | None = None
    expected_max_rows: int | None = None
    timeout_seconds: int = 300


@dataclass
class Schema:
    """Schema definition for a data contract.

    Attributes:
        columns: List of column definitions
        strict: If True, fail on unexpected columns in the DataFrame
    """

    columns: list[Column]
    strict: bool = True


@dataclass(frozen=True)
class Period:
    """A historical data period with source and effective date boundaries.

    Defines a single time period and its data source for SCD2 historical
    backfill. Used in the ``periods`` section of a data contract.

    Attributes:
        source: Path or URL to the source data for this period.
        effective_from: Start date of this period (inclusive).
        effective_to: End date of this period (exclusive). None for
            the current/open-ended period.
        partition_key: Optional partition key for this period. When set,
            moncpipelib injects this value as a column into the DataFrame
            during write, using the sink's ``partition_column`` name.
    """

    source: str
    effective_from: date
    effective_to: date | None = None
    partition_key: str | None = None


@dataclass(frozen=True)
class FromIngestTemplate:
    """Template applied to every partition produced by a linked ingest contract.

    Used when a downstream ``*.source.yaml`` declares ``periods.mode: from_ingest``
    instead of an enumerated period list. At resolution time each ingest-produced
    partition is hydrated into a concrete ``Period`` using this template.

    Attributes:
        source: Blob-relative path or glob under the ingest prefix. May
            reference manifest fields via ``{field_name}`` placeholders
            (hydrated by the resolver from the per-partition manifest).
        effective_from_field: Name of the ingest manifest field that
            supplies ``effective_from`` for each dynamically-discovered
            partition.
    """

    source: str
    effective_from_field: str


@dataclass
class DataSource:
    """External data source definition.

    Loaded from ``*.source.yaml`` files. Defines where data comes from
    and its historical period boundaries. Referenced by pipeline
    contracts via the ``data_source`` field.

    Attributes:
        source_id: Stable UUID identifying this data source. Persists
            across renames so registry history remains correlated.
        source_name: Human-readable name for display and logging.
        periods: Ordered list of historical data periods, or a
            ``FromIngestTemplate`` when periods are discovered at runtime
            from a linked ingest contract.
        ingest_source: Optional ``source_name`` of a sibling ingest
            contract (``*.ingest.yaml``) that lands this data in blob.
            When set, consumers resolve blob refs via
            ``resolve_source_for_partition`` instead of fetching the
            ``Period.source`` URL directly.
        description: Optional free-text description.
    """

    source_id: str  # UUID string
    source_name: str
    periods: list[Period] | FromIngestTemplate
    ingest_source: str | None = None
    description: str | None = None


def require_enumerated_periods(source: DataSource) -> list[Period]:
    """Narrow ``source.periods`` to ``list[Period]`` or raise.

    Call sites that predate the ingest boundary (partition generation,
    historical backfill, contract-period injection) require an
    enumerated period list. A ``FromIngestTemplate``-backed source is
    resolved through ``resolve_source_for_partition`` and a Phase 2
    manifest reader, not through these code paths.
    """
    if isinstance(source.periods, FromIngestTemplate):
        raise ValueError(
            f"DataSource {source.source_name!r} uses FromIngestTemplate. "
            f"Use resolve_source_for_partition (ingest boundary) instead of "
            f"the enumerated-periods API."
        )
    return source.periods


@dataclass(frozen=True)
class IngestContract:
    """External data ingest contract.

    Loaded from ``*.ingest.yaml`` files. Declares how a single external
    source is pulled into the blob-landing boundary. One ingest contract
    feeds one or more downstream ``DataSource`` contracts via
    ``DataSource.ingest_source``.

    Attributes:
        source_id: Stable UUID identifying this ingest source.
        source_name: Human-readable name; referenced by downstream
            ``DataSource.ingest_source``.
        sensitivity: Data sensitivity class. Drives container selection
            in ``BlobStorageResource``.
        pattern: Discriminator selecting the ingest pattern
            implementation (``http_urls``, ``api_resolver``, ...).
            Resolved through the pattern registry.
        prefix_template: Blob prefix template with bounded
            interpolation (``{partition_key}``, ``{source_name}``).
        extract: Ordered tuple of archive formats to expand per
            downloaded payload (e.g. ``("zip",)``). Nested archives are
            applied in order.
        strip_extensions: Tuple of file extensions to strip from
            extracted filenames before upload.
        extract_filter: Optional ``fnmatch`` glob patterns. When
            non-empty, only files whose post-strip path matches at
            least one glob are extracted at every archive level.
            Archives matching ``extract`` extensions are recursed into
            regardless of the filter; the filter applies to terminal
            (non-archive) members. See ADR-1 in
            ``docs/migrations/20260426_phase2-ingest-decisions.md``.
            Empty tuple means "no filter" (Phase 1 default behavior).
        pattern_config: Pattern-specific inner config block parsed by
            the selected ``IngestPattern`` implementation.
        data_owner: Required when ``sensitivity in {"phi", "confidential"}``.
            Identifies the accountable team / individual.
        compliance_review: Required when
            ``sensitivity in {"phi", "confidential"}``. Pointer to the
            entry documenting this source in ``SECURITY.md``.
        description: Optional free-text description.
        payload_filename_template: Optional filename template for
            non-archive (``extract: []``) payloads, rendered through the
            same bounded placeholder set as ``prefix_template``
            (``{partition_key}``, ``{source_name}``). When set, takes
            highest precedence in the non-archive filename derivation
            chain (template -> resolver hint -> Content-Disposition ->
            sanitized URL basename -> raise). Ignored when ``extract``
            is non-empty -- archive expansion produces filenames from
            the archive's member names. See #270.
    """

    source_id: str
    source_name: str
    sensitivity: Literal["public", "confidential", "phi"]
    pattern: str
    prefix_template: str
    extract: tuple[str, ...]
    strip_extensions: tuple[str, ...]
    pattern_config: dict[str, Any]
    extract_filter: tuple[str, ...] = ()
    data_owner: str | None = None
    compliance_review: str | None = None
    description: str | None = None
    payload_filename_template: str | None = None


@dataclass
class ContractCorpus:
    """Lookup-able bundle of loaded ingest and source contracts.

    Returned by ``load_all_contracts``. Holds all ``IngestContract`` and
    ``DataSource`` objects discovered under a code location's contract
    root after cross-contract validation has passed.

    Attributes:
        ingests: Map of ``source_name`` -> ``IngestContract``.
        sources: Map of ``source_name`` -> ``DataSource``.
    """

    ingests: dict[str, IngestContract] = field(default_factory=dict)
    sources: dict[str, DataSource] = field(default_factory=dict)

    def get_ingest(self, source_name: str) -> IngestContract:
        """Return the ingest contract for ``source_name``.

        Raises:
            KeyError: If no ingest contract with that name is loaded.
        """
        return self.ingests[source_name]

    def get_source(self, source_name: str) -> DataSource:
        """Return the data source contract for ``source_name``.

        Raises:
            KeyError: If no data source with that name is loaded.
        """
        return self.sources[source_name]


@dataclass
class DataContract:
    """Complete data contract definition.

    Attributes:
        version: Contract schema version (e.g., "1.0")
        pipeline_id: Stable UUID identifying the logical pipeline. Persists across
            asset renames so lineage history remains correlated.
        asset: Name of the asset this contract applies to
        layer: Data layer (bronze, silver, gold)
        schema: Schema definition with columns
        description: Human-readable description
        owner: Ownership metadata
        expectations: Table-level validation rules. Can be TableExpectation instances
                      or typed expectation classes from checks_types (RowCount, etc.)
        upstream: Upstream dependencies documentation
        sla: Service level agreement metadata
        tags: User-defined string tags for Dagster job/op tagging
        data_source: Optional reference to an external data source definition
            loaded from a ``*.source.yaml`` file.
    """

    version: str
    pipeline_id: str
    asset: str
    layer: str
    schema: Schema
    description: str | None = None
    owner: Owner | None = None
    expectations: Sequence[TableExpectationType] = field(default_factory=list)
    upstream: list[UpstreamDependency] = field(default_factory=list)
    sla: SLA | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    sinks: list[dict[str, Any]] = field(default_factory=list)
    testing: TestingConfig | None = None
    lineage: LineageConfig | None = None
    tags: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    data_source: DataSource | None = None

    # Migration 019 (#308) Phase 3: stable content fingerprints populated
    # by ``load_contract`` after parsing. ``contract_hash`` covers the
    # full semantic content; ``schema_fingerprint`` covers column-schema
    # identity only. Both are excluded from their own hash computation
    # via ``contracts.hashing._HASH_EXCLUDED_FIELDS``.
    contract_hash: str = ""
    schema_fingerprint: str = ""

    @overload
    def get_parameter(self, key: str) -> Any: ...
    @overload
    def get_parameter(self, key: str, default: _T) -> Any | _T: ...

    def get_parameter(self, key: str, default: Any = _UNSET) -> Any:
        """Retrieve a business-logic parameter from the contract.

        Args:
            key: Parameter name to look up.
            default: Optional fallback value. When provided, a missing key
                logs a warning instead of raising.

        Returns:
            The parameter value, or *default* if supplied and key is missing.

        Raises:
            KeyError: If *key* is missing and no *default* was given.
        """
        if key in self.parameters:
            return self.parameters[key]

        has_default = default is not _UNSET

        if self.parameters:
            msg = (
                f"Parameter '{key}' not found in contract '{self.asset}'. "
                f"Available parameters: {sorted(self.parameters.keys())}. "
                f"Check for typos."
            )
        else:
            msg = (
                f"Parameter '{key}' requested but no parameters are defined "
                f"in contract '{self.asset}'. Add a 'parameters:' section to "
                f"the contract YAML to configure business logic values."
            )

        if has_default:
            logger.warning("%s Using default value: %r", msg, default)
            return default

        raise KeyError(msg)

    def get_primary_key_columns(self) -> list[str]:
        """Return list of primary key column names."""
        return [c.name for c in self.schema.columns if c.primary_key]

    def get_column(self, name: str) -> Column | None:
        """Get column definition by name.

        Args:
            name: Column name to find

        Returns:
            Column definition if found, None otherwise
        """
        for col in self.schema.columns:
            if col.name == name:
                return col
        return None

    def get_non_managed_columns(self) -> list[Column]:
        """Get columns that are not managed by moncpipelib.

        Returns:
            List of non-managed columns
        """
        return [c for c in self.schema.columns if not c.managed]

    def get_pii_columns(self) -> list[Column]:
        """Return columns marked as PII (includes unannotated columns).

        Columns default to pii=True, so any column without an explicit
        pii: false annotation is included.
        """
        return [c for c in self.schema.columns if c.pii]

    def get_pii_column_names(self) -> list[str]:
        """Return names of columns marked as PII (includes unannotated columns)."""
        return [c.name for c in self.schema.columns if c.pii]

    def get_non_pii_column_names(self) -> list[str]:
        """Return names of columns explicitly marked as non-PII."""
        return [c.name for c in self.schema.columns if not c.pii]

    def get_phi_columns(self) -> list[Column]:
        """Return columns marked as PHI (includes unannotated columns).

        phi defaults to the column's pii value, which itself defaults to
        True -- so any column without an explicit annotation is included.
        """
        return [c for c in self.schema.columns if c.phi]

    def get_phi_column_names(self) -> list[str]:
        """Return names of columns marked as PHI (includes unannotated columns)."""
        return [c.name for c in self.schema.columns if c.phi]

    def get_non_phi_column_names(self) -> list[str]:
        """Return names of columns explicitly cleared of PHI (phi: false)."""
        return [c.name for c in self.schema.columns if not c.phi]

    def get_source_tables(self) -> list[TableReference]:
        """Extract source table references from sources section.

        Returns table references for all sources with type='table'.
        Sources missing 'database' default to :data:`config.DEFAULT_DATABASE`
        (``MONCPIPELIB_DEFAULT_DATABASE`` env var).
        """
        refs: list[TableReference] = []
        for source in self.sources:
            if source.get("type") == "table":
                refs.append(
                    TableReference(
                        database=source.get("database", DEFAULT_DATABASE),
                        schema=source["schema"],
                        table=source["table"],
                    )
                )
        return refs

    def get_sink_tables(self) -> list[TableReference]:
        """Extract sink table references from sinks section.

        Returns table references for all sinks with type='table'.
        Sinks missing 'database' default to :data:`config.DEFAULT_DATABASE`
        (``MONCPIPELIB_DEFAULT_DATABASE`` env var).
        """
        refs: list[TableReference] = []
        for sink in self.sinks:
            if sink.get("type") == "table":
                refs.append(
                    TableReference(
                        database=sink.get("database", DEFAULT_DATABASE),
                        schema=sink["schema"],
                        table=sink["table"],
                    )
                )
        return refs

    def validate_for_integration_testing(self) -> list[str]:
        """Validate contract has required fields for integration testing.

        Returns:
            List of error messages (empty if valid).

        Checks:
        - At least one source of type 'table' with schema/table fields
        - At least one sink of type 'table' with schema/table fields
        - Testing config validity if present
        """
        errors: list[str] = []

        # Check sources
        table_sources = [s for s in self.sources if s.get("type") == "table"]
        if not table_sources:
            errors.append("At least one source of type 'table' is required for integration testing")
        for i, source in enumerate(table_sources):
            if "schema" not in source:
                errors.append(f"Source {i}: 'schema' is required")
            if "table" not in source:
                errors.append(f"Source {i}: 'table' is required")

        # Check sinks
        table_sinks = [s for s in self.sinks if s.get("type") == "table"]
        if not table_sinks:
            errors.append("At least one sink of type 'table' is required for integration testing")
        for i, sink in enumerate(table_sinks):
            if "schema" not in sink:
                errors.append(f"Sink {i}: 'schema' is required")
            if "table" not in sink:
                errors.append(f"Sink {i}: 'table' is required")

        # Validate testing config
        if self.testing is not None:
            if self.testing.source_row_limit <= 0:
                errors.append("testing.source_row_limit must be positive")
            if self.testing.timeout_seconds <= 0:
                errors.append("testing.timeout_seconds must be positive")
            if (
                self.testing.expected_min_rows is not None
                and self.testing.expected_max_rows is not None
                and self.testing.expected_min_rows > self.testing.expected_max_rows
            ):
                errors.append("testing.expected_min_rows cannot exceed testing.expected_max_rows")

        return errors


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        passed: Whether the validation passed
        message: Human-readable message describing the result
        failed_count: Number of rows that failed validation
        total_count: Total number of rows checked
        sample_failures: Optional sample of failing rows for debugging
    """

    passed: bool
    message: str
    failed_count: int = 0
    total_count: int = 0
    sample_failures: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class CheckResultRow:
    """One executed validation check, retained for ``contract_validation_runs`` persistence.

    Migration 019 (#308) Phase 5: the in-process
    :class:`ContractValidationSummary` collapses per-check
    :class:`ValidationResult` objects into a count + message-list shape
    and loses per-check granularity. ``CheckResultRow`` is the per-check
    audit row that flows from ``_enforce_contract`` into
    ``LineageTracker.write_validation_runs`` and finally into
    ``lineage.contract_validation_runs``.

    One row per check executed per write (schema check + each column
    test + each table expectation).

    Attributes:
        check_name: Stable name for the check (e.g. ``"schema"``,
            ``"company_id.unique"``, ``"row_count"``). Matches the
            string used in ``_log_validation_result``.
        severity: ``"error"`` or ``"warn"`` per the contract spec.
        passed: Whether this individual check passed.
        failed_count: Rows that failed this check (0 for passed checks).
        total_count: Rows the check ran against.
        sample_failures: Up to 20 sample failing rows for debugging,
            truncated at persist time to keep the JSONB payload bounded.
            ``None`` for passed checks.
    """

    check_name: str
    severity: str  # "error" | "warn"
    passed: bool
    failed_count: int = 0
    total_count: int = 0
    sample_failures: list[dict[str, Any]] | None = None


@dataclass
class ContractValidationSummary:
    """Summary of contract validation results for Dagster metadata reporting.

    Aggregates individual ValidationResult outcomes into a single summary
    that can be surfaced in the Dagster UI as materialization metadata.

    Attributes:
        contract_version: Version string from the contract (e.g., "1.0")
        contract_asset: Asset name the contract applies to
        status: Overall result — "passed", "failed", or "skipped"
        total_checks: Total number of checks executed
        passed_checks: Number of checks that passed
        failed_checks: Number of error-severity checks that failed
        warned_checks: Number of warn-severity checks that failed
        violations: Error messages from error-severity failures
        warnings: Messages from warn-severity failures
        check_results: Per-check audit rows for persistence into
            ``lineage.contract_validation_runs`` (migration 019 Phase 5).
            One entry per executed check (passed or failed). Empty when
            no contract validation ran (status ``"skipped"``).
    """

    contract_version: str
    contract_asset: str
    status: str  # "passed", "failed", "skipped"
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warned_checks: int = 0
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    check_results: list[CheckResultRow] = field(default_factory=list)
