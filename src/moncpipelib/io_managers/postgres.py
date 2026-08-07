"""PostgreSQL IO Manager for Dagster pipelines.

Thin Dagster adapter that delegates all writes to ``PostgresResource``.
Handles target resolution (schema cascade, table prefix, suffix stripping)
and metadata extraction from ``OutputContext``, while the resource handles
contract enforcement, reconciliation, lineage, metadata columns, PII sync,
OpenLineage, and database writes.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dagster import AssetChecksDefinition

    from moncpipelib.streaming import BatchedDataFrame

import polars as pl
from dagster import ConfigurableIOManager, InputContext, OutputContext

from moncpipelib.config import CONTRACT_FILE_PATTERN, VALID_LAYERS, SCD2Config
from moncpipelib.contracts import (
    ContractEnforcementMode,
    DataContract,
    load_contract_for_asset,
)
from moncpipelib.contracts.reconciliation import ContractReconciler
from moncpipelib.io_managers.enums import (
    BulkInsertMethod,
    FullRefreshMethod,
    ResolvedTarget,
    WriteMode,
)
from moncpipelib.resources._app_name import bind_run_id
from moncpipelib.resources.postgres import PostgresPolarsSchema, PostgresResource
from moncpipelib.resources.types import WriteContext

# Re-export enums for backwards compatibility (import from this module still works)
__all__ = [
    "BulkInsertMethod",
    "FullRefreshMethod",
    "PostgresIOManager",
    "ResolvedTarget",
    "SCD2Config",
    "VALID_LAYERS",
    "WriteMode",
]


class PostgresIOManager(ConfigurableIOManager):
    """IO Manager for writing Polars DataFrames to PostgreSQL tables.

    Delegates all writes to a required ``PostgresResource`` instance.
    The IO Manager handles Dagster-specific concerns (target resolution from
    asset keys, metadata extraction from ``OutputContext``) while the Resource
    handles database operations, contract enforcement, lineage, and PII sync.

    Supports multiple write modes for different data loading patterns:
    - full_refresh: DELETE + INSERT (default)
    - upsert: INSERT ON CONFLICT UPDATE (idempotent incremental)
    - append: INSERT only (event logs, audit trails)
    - scd2: Slowly Changing Dimension Type 2 (versioned history)

    Write settings can be configured at the IO Manager level (defaults) or
    overridden per-asset using output metadata.

    Example:
        ```python
        from dagster import asset, Definitions, EnvVar
        from moncpipelib import PostgresIOManager, PostgresResource, WriteMode

        database = PostgresResource(
            host=EnvVar("DB_HOST"),
            port=EnvVar.int("DB_PORT"),
            user=EnvVar("DB_USER"),
            password=EnvVar("DB_PASSWORD"),
            database=EnvVar("DB_NAME"),
        )

        defs = Definitions(
            assets=[orders_silver, customers_silver],
            resources={
                "database": database,
                "silver_io_manager": PostgresIOManager(
                    postgres_resource=database,
                    default_schema="silver",
                ),
            },
        )
        ```
    """

    # Required resource for all database operations
    postgres_resource: PostgresResource
    """``PostgresResource`` for database operations.

    All writes and reads are delegated to this resource. Connection config,
    lineage, OpenLineage, performance tuning, and metadata columns are all
    configured on the resource.
    """

    # Schema/target resolution
    db_schema: str = ""
    """Deprecated: use ``default_schema`` instead. Target schema for tables."""

    default_schema: str | None = None
    """Fallback schema when not specified per-asset via ``target_schema`` metadata
    or a contract sink ``schema`` field. Replaces ``db_schema``."""

    layer: str | None = None
    """Deprecated: layer is now derived automatically from the resolved target schema.
    If set, used as a fallback layer hint for contract loading."""

    table_suffix_to_strip: str = ""
    """Deprecated: rename assets to match target tables directly.
    Suffix to strip from asset name when deriving table name (e.g., '_silver')."""

    table_prefix: str | None = None
    """Prefix to add to table names for test isolation (e.g., 'johndoe_abc123_')."""

    schema_override: str | None = None
    """Override schema for integration testing. When set, all table operations
    use this schema instead of the resolved target schema."""

    # Write mode configuration (defaults, can be overridden per-asset)
    write_mode: WriteMode = WriteMode.FULL_REFRESH
    """Default write strategy. Can be overridden per-asset via metadata."""

    primary_key: list[str] | None = None
    """Default primary key column(s) for upsert. Can be overridden per-asset."""

    update_columns: list[str] | None = None
    """Columns to update on upsert conflict. None = all non-key columns (default)."""

    partition_column: str | None = None
    """Default partition column for partition-scoped writes. Can be overridden per-asset."""

    # Contract enforcement configuration
    enforce_contracts: ContractEnforcementMode = ContractEnforcementMode.ERROR
    """How to handle contract validation at write time.
    - error: Raise ContractViolationError on validation failure (default)
    - warn: Log warnings but continue write
    - silent: Skip validation entirely
    """

    contract_search_paths: list[str] | None = None
    """Paths to search for contract YAML files. If None, auto-discovers from asset location."""

    def setup_for_execution(self, context: Any) -> None:  # noqa: ARG002
        """Dagster post-init hook: validate config and emit deprecation warnings."""
        if self.db_schema:
            warnings.warn(
                "PostgresIOManager(db_schema=...) is deprecated. "
                "Use default_schema instead, or set target_schema per-asset in metadata.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not self.default_schema:
                object.__setattr__(self, "default_schema", self.db_schema)

        if self.layer is not None:
            warnings.warn(
                "PostgresIOManager(layer=...) is deprecated. "
                "Layer is now derived automatically from the resolved target schema.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.table_suffix_to_strip:
            warnings.warn(
                "PostgresIOManager(table_suffix_to_strip=...) is deprecated. "
                "Rename assets to match target table names directly.",
                DeprecationWarning,
                stacklevel=2,
            )

    def for_testing(
        self,
        *,
        test_schema: str,
        table_prefix: str | None = None,
        contract_search_paths: list[str] | None = None,
        **overrides: Any,
    ) -> PostgresIOManager:
        """Create a test-isolated clone of this IO manager.

        Returns a copy with test-specific overrides applied. All other
        configuration (write behavior, contract enforcement, deprecated fields)
        is automatically preserved from the original instance.

        Args:
            test_schema: Target schema for test isolation (sets ``schema_override``).
            table_prefix: Optional prefix for table names (e.g., ``'johndoe_abc123_'``).
            contract_search_paths: Explicit paths to search for contract YAML files.
                When ``None``, the original instance's ``contract_search_paths`` is
                preserved.
            **overrides: Any additional field overrides to apply to the clone.

        Returns:
            A new ``PostgresIOManager`` instance with test overrides applied.
        """
        update: dict[str, Any] = {
            "schema_override": test_schema,
        }
        if table_prefix is not None:
            update["table_prefix"] = table_prefix
        if contract_search_paths is not None:
            update["contract_search_paths"] = contract_search_paths
        update.update(overrides)
        return self.model_copy(update=update)

    # ------------------------------------------------------------------
    # Target resolution (schema, table, layer)
    # ------------------------------------------------------------------

    def _resolve_canonical_table_name(self, context: OutputContext | InputContext) -> str:
        """Derive the canonical table name from the asset key.

        Applies ``table_suffix_to_strip`` but NOT ``table_prefix``. This is
        the name that matches contract sink ``table`` fields.
        """
        asset_key = context.asset_key.path[-1]
        table_name = asset_key
        if self.table_suffix_to_strip and asset_key.endswith(self.table_suffix_to_strip):
            table_name = asset_key[: -len(self.table_suffix_to_strip)]
        return table_name

    def _resolve_bare_table_name(self, context: OutputContext | InputContext) -> str:
        """Derive the bare table name from the asset key.

        Applies ``table_suffix_to_strip`` (deprecated) and ``table_prefix``
        (test isolation) but does NOT prepend a schema.
        """
        table_name = self._resolve_canonical_table_name(context)
        if self.table_prefix:
            table_name = f"{self.table_prefix}{table_name}"
        return table_name

    def _resolve_schema(
        self,
        context: OutputContext | InputContext,
        contract: DataContract | None = None,
    ) -> str:
        """Resolve the target PostgreSQL schema using a priority cascade.

        Resolution order:
        1. ``schema_override`` (integration testing)
        2. Contract sink ``schema`` (if contract has a matching sink)
        3. ``target_schema`` from asset metadata
        4. ``default_schema`` (constructor fallback)
        5. ``db_schema`` (deprecated backwards-compat)
        6. Error
        """
        if self.schema_override:
            return self.schema_override

        canonical_table = self._resolve_canonical_table_name(context)

        # When asset metadata names a target_schema, pass it to sink matching
        # so a sink declaring a DIFFERENT schema is never applied (#405). The
        # cascade order is unchanged when metadata and sink agree (or metadata
        # is absent); on a mismatch the sink is skipped with a warning and the
        # metadata schema wins.
        metadata = self._get_context_metadata(context)
        metadata_schema: str | None = None
        if metadata and "target_schema" in metadata:
            metadata_schema = str(metadata["target_schema"])

        if contract is not None:
            sink = ContractReconciler.find_matching_sink(
                contract, canonical_table, context, target_schema=metadata_schema
            )
            if sink is not None and "schema" in sink:
                return str(sink["schema"])

        if metadata_schema is not None:
            return metadata_schema

        if self.default_schema:
            return self.default_schema

        if self.db_schema:
            return self.db_schema

        asset_name = context.asset_key.to_user_string()
        raise ValueError(
            f"No target schema resolved for asset '{asset_name}'. "
            "Set target_schema in asset metadata, declare a contract sink with "
            "a schema field, or set default_schema on the IO manager."
        )

    @staticmethod
    def _get_context_metadata(
        context: OutputContext | InputContext,
    ) -> Mapping[str, Any] | None:
        """Extract metadata from either output or input context.

        Type-strict on the return: only ``Mapping`` results are surfaced.
        ``isinstance(context, OutputContext)`` is False on a ``MagicMock``
        without ``spec=OutputContext`` (the common integration-test
        shape), so we also accept a duck-typed ``context.metadata``
        attribute that is a real ``Mapping``. Without this fallback, the
        ``hasattr(context, "upstream_output")`` check would always pick
        up an auto-generated child mock and return non-``Mapping`` data,
        causing downstream callers (e.g., ``_resolve_layer``) to silently
        receive ``None`` and disable layer-dependent features like
        lineage tracking.
        """
        if isinstance(context, OutputContext):
            metadata = context.metadata
            return metadata if isinstance(metadata, Mapping) else None
        if hasattr(context, "upstream_output") and context.upstream_output:
            upstream_metadata = getattr(context.upstream_output, "metadata", None)
            if isinstance(upstream_metadata, Mapping):
                return upstream_metadata
        # Duck-type fallback for test harnesses (bare ``MagicMock``) and
        # any future context shape that exposes ``.metadata`` directly
        # without satisfying ``isinstance(context, OutputContext)``.
        direct_metadata = getattr(context, "metadata", None)
        if isinstance(direct_metadata, Mapping):
            return direct_metadata
        return None

    def _resolve_layer(
        self,
        context: OutputContext | InputContext,
        schema: str,
    ) -> str | None:
        """Derive the data layer from the resolved schema.

        Resolution order:
        1. ``layer_override`` from asset metadata
        2. Schema name if it matches a valid layer (bronze/silver/gold)
        3. ``self.layer`` (deprecated constructor arg, backwards compat)
        4. None
        """
        metadata = self._get_context_metadata(context)
        if metadata and "layer_override" in metadata:
            return str(metadata["layer_override"])
        if schema in VALID_LAYERS:
            return schema
        if self.layer is not None:
            return self.layer
        return None

    def _resolve_target(
        self,
        context: OutputContext | InputContext,
        contract: DataContract | None = None,
    ) -> ResolvedTarget:
        """Resolve the full write/read target: table name, schema, and layer.

        When a contract sink declares a ``table`` field, that value becomes the
        physical table name used for SQL.
        """
        canonical_table = self._resolve_canonical_table_name(context)
        schema = self._resolve_schema(context, contract)
        layer = self._resolve_layer(context, schema)

        physical_table = canonical_table
        if contract is not None:
            # Under schema_override (test isolation) the resolved schema is
            # not the contract's logical schema, so skip the comparison (#405).
            sink = ContractReconciler.find_matching_sink(
                contract,
                canonical_table,
                context,
                target_schema=None if self.schema_override else schema,
            )
            if sink is not None and sink.get("table"):
                physical_table = str(sink["table"])

        bare_table = physical_table
        if self.table_prefix:
            bare_table = f"{self.table_prefix}{physical_table}"

        return ResolvedTarget(
            table_name=f"{schema}.{bare_table}",
            schema=schema,
            bare_table=bare_table,
            layer=layer,
            canonical_table=canonical_table,
        )

    @staticmethod
    def _extract_partition_values(
        context: OutputContext | InputContext,
    ) -> list[str] | None:
        """Extract partition values from Dagster partition context."""
        if not context.has_partition_key:
            return None
        try:
            return list(context.asset_partition_keys)
        except Exception:
            return [context.partition_key]

    def _get_table_name(
        self,
        context: OutputContext | InputContext,
        contract: DataContract | None = None,
    ) -> str:
        """Derive fully-qualified table name from asset key.

        Thin wrapper around ``_resolve_target()`` for backwards compatibility.
        """
        return self._resolve_target(context, contract).table_name

    def _derive_table_name(self, asset_name: str) -> str:
        """Derive fully-qualified table name from an asset name string."""
        table_name = asset_name
        if self.table_suffix_to_strip and asset_name.endswith(self.table_suffix_to_strip):
            table_name = asset_name[: -len(self.table_suffix_to_strip)]
        if self.table_prefix:
            table_name = f"{self.table_prefix}{table_name}"
        schema = self.schema_override or self.default_schema or self.db_schema
        return f"{schema}.{table_name}"

    # ------------------------------------------------------------------
    # Contract checks
    # ------------------------------------------------------------------

    def make_contract_checks(
        self,
        contracts_dir: str | Path,
        asset_key_prefix: Sequence[str] | None = None,
        *,
        batched: bool = True,
        op_tags: dict[str, Any] | None = None,
    ) -> list[AssetChecksDefinition]:
        """Generate Dagster asset checks for all contracts in a directory.

        Recursively scans ``contracts_dir`` for ``*.contract.yaml`` files and
        generates Dagster asset checks for each contract found.  Connection
        credentials are resolved at check execution time (not at definition
        time), so this method works correctly with Dagster ``EnvVar``.

        When ``batched=True`` (default), checks run as SQL queries directly
        in PostgreSQL -- no data is loaded into Python. Use ``op_tags`` to
        configure k8s pod resources if needed.

        Args:
            contracts_dir: Directory to recursively scan for contract files.
            asset_key_prefix: Optional prefix for asset keys (e.g., ["bronze"]).
                When set, checks attach to ``[*prefix, asset]``; otherwise they
                attach to ``[schema, table]`` derived from the resolved sink
                table, matching the IO manager's asset key convention.
            batched: Bundle all checks per contract into a single op (default True).
            op_tags: Dagster op tags (e.g., ``dagster-k8s/config`` for pod resources).

        Returns:
            List of AssetChecksDefinitions ready for Dagster Definitions.
        """
        from moncpipelib.contracts.checks import (
            _derive_check_asset_key,
            _make_deferred_df_loader,
            _resolve_check_table,
            generate_asset_checks_from_contract,
        )
        from moncpipelib.contracts.loader import load_contract

        contracts_path = Path(contracts_dir)

        if not self.contract_search_paths:
            resolved_dir = str(contracts_path.resolve())
            object.__setattr__(self, "contract_search_paths", [resolved_dir])

        # Build a connection factory from the resource's fields, resolving
        # EnvVars at execution time (asset check ops bypass Dagster's resource
        # lifecycle). Delegated to the resource so the two make_contract_checks
        # implementations share one connection-identity path (#365) and cannot
        # drift.
        _connection_factory = self.postgres_resource._make_check_connection_factory()

        all_checks: list[AssetChecksDefinition] = []

        for contract_file in contracts_path.rglob(CONTRACT_FILE_PATTERN):
            contract = load_contract(contract_file)

            fq_table = _resolve_check_table(
                contract,
                schema_override=self.schema_override,
                default_schema=self.default_schema,
                db_schema=self.db_schema,
                table_suffix_to_strip=self.table_suffix_to_strip,
                table_prefix=self.table_prefix,
            )

            df_loader = _make_deferred_df_loader(
                connection_factory=_connection_factory,
                fq_table=fq_table,
            )

            asset_key = _derive_check_asset_key(
                contract,
                fq_table=fq_table,
                asset_key_prefix=asset_key_prefix,
            )

            checks = generate_asset_checks_from_contract(
                contract,
                asset_key,
                df_loader,
                batched=batched,
                connection_factory=_connection_factory,
                fq_table=fq_table,
                op_tags=op_tags,
            )
            all_checks.extend(checks)

        return all_checks

    # ------------------------------------------------------------------
    # Metadata key validation
    # ------------------------------------------------------------------

    # Metadata keys recognized by PostgresIOManager.
    # Any key NOT in this set will raise an error to prevent silent misconfiguration.
    _RECOGNIZED_METADATA_KEYS: frozenset[str] = frozenset(
        {
            # Schema routing (new universal IO manager)
            "target_schema",
            "layer_override",
            # Write configuration
            "write_mode",
            "primary_key",
            "update_columns",
            "partition_column",
            "source_file",
            "data_date",
            "analyze_after_write",
            # SCD2-specific keys
            "business_key",
            "tracked_columns",
            "effective_from_col",
            "effective_to_col",
            "is_current_col",
            "hash_col",
            "detect_deletes",
            "sequence_column",
        }
    )

    def _validate_metadata_keys(self, context: OutputContext) -> None:
        """Validate that all metadata keys are recognized by PostgresIOManager.

        Raises ValueError for unrecognized keys to prevent silent misconfiguration.
        """
        if not context.metadata:
            return

        unrecognized = set(context.metadata.keys()) - self._RECOGNIZED_METADATA_KEYS
        if unrecognized:
            asset_name = context.asset_key.to_user_string()
            raise ValueError(
                f"Unrecognized metadata key(s) {unrecognized} on asset '{asset_name}'. "
                f"Recognized keys are: {sorted(self._RECOGNIZED_METADATA_KEYS)}. "
                f"Check for typos (e.g., 'mode' should be 'write_mode')."
            )

    # ------------------------------------------------------------------
    # Write configuration
    # ------------------------------------------------------------------

    def _get_write_config(self, context: OutputContext) -> dict[str, Any]:
        """Get write configuration, merging IO Manager defaults with asset overrides.

        Asset metadata takes precedence over IO Manager configuration.
        """
        self._validate_metadata_keys(context)

        config: dict[str, Any] = {
            "write_mode": self.write_mode,
            "write_mode_explicit": False,
            "primary_key": self.primary_key,
            "primary_key_explicit": False,
            "update_columns": self.update_columns,
            "partition_column": self.partition_column,
            "partition_column_explicit": False,
            # SCD2 defaults
            "business_key": None,
            "business_key_explicit": False,
            "tracked_columns": None,
            "tracked_columns_explicit": False,
            "scd2": SCD2Config(),
            "effective_from_col": SCD2Config().effective_from_col,
            "effective_to_col": SCD2Config().effective_to_col,
            "is_current_col": SCD2Config().is_current_col,
            "hash_col": SCD2Config().hash_col,
            "detect_deletes": False,
            "detect_deletes_explicit": False,
            "skip_unchanged": False,
            "skip_unchanged_explicit": False,
            "sequence_col": SCD2Config().sequence_col,
            "sequence_col_explicit": False,
            # None = defer to the resource-level analyze_after_write setting
            "analyze_after_write": None,
        }

        # Override from asset metadata if present
        if context.metadata:
            if "write_mode" in context.metadata:
                mode_value = context.metadata["write_mode"]
                if isinstance(mode_value, str):
                    config["write_mode"] = WriteMode(mode_value)
                elif isinstance(mode_value, WriteMode):
                    config["write_mode"] = mode_value
                config["write_mode_explicit"] = True

            if "primary_key" in context.metadata:
                pk = context.metadata["primary_key"]
                if isinstance(pk, str):
                    config["primary_key"] = [pk]
                else:
                    config["primary_key"] = list(pk)
                config["primary_key_explicit"] = True

            if "update_columns" in context.metadata:
                uc = context.metadata["update_columns"]
                if isinstance(uc, str):
                    config["update_columns"] = [uc]
                elif uc is not None:
                    config["update_columns"] = list(uc)

            if "partition_column" in context.metadata:
                config["partition_column"] = str(context.metadata["partition_column"])
                config["partition_column_explicit"] = True

            if "analyze_after_write" in context.metadata:
                config["analyze_after_write"] = str(context.metadata["analyze_after_write"])

            # SCD2 metadata overrides
            if "business_key" in context.metadata:
                bk = context.metadata["business_key"]
                if isinstance(bk, str):
                    config["business_key"] = [bk]
                else:
                    config["business_key"] = list(bk)
                config["business_key_explicit"] = True

            if "tracked_columns" in context.metadata:
                tc = context.metadata["tracked_columns"]
                if isinstance(tc, str):
                    config["tracked_columns"] = [tc]
                elif tc is not None:
                    config["tracked_columns"] = list(tc)
                config["tracked_columns_explicit"] = True

            # Collect SCD2 column overrides, then rebuild SCD2Config
            scd2_overrides: dict[str, str] = {}
            for scd2_key in (
                "effective_from_col",
                "effective_to_col",
                "is_current_col",
                "hash_col",
            ):
                if scd2_key in context.metadata:
                    val = str(context.metadata[scd2_key])
                    config[scd2_key] = val
                    scd2_overrides[scd2_key] = val

            if "detect_deletes" in context.metadata:
                config["detect_deletes"] = bool(context.metadata["detect_deletes"])
                config["detect_deletes_explicit"] = True

            if "sequence_column" in context.metadata:
                raw_sc = context.metadata["sequence_column"]
                config["sequence_col"] = str(raw_sc) if raw_sc is not None else None
                config["sequence_col_explicit"] = True
                scd2_overrides["sequence_col"] = config["sequence_col"]

            # Rebuild SCD2Config if any fields were overridden via metadata
            if scd2_overrides:
                base = SCD2Config()
                config["scd2"] = SCD2Config(
                    effective_from_col=scd2_overrides.get(
                        "effective_from_col", base.effective_from_col
                    ),
                    effective_to_col=scd2_overrides.get("effective_to_col", base.effective_to_col),
                    is_current_col=scd2_overrides.get("is_current_col", base.is_current_col),
                    hash_col=scd2_overrides.get("hash_col", base.hash_col),
                    sequence_col=scd2_overrides.get("sequence_col", base.sequence_col),
                )

        return config

    # ------------------------------------------------------------------
    # Contract loading
    # ------------------------------------------------------------------

    def _get_contract_search_paths(self) -> list[Path | str] | None:
        """Return contract search paths for write-time contract discovery."""
        if self.contract_search_paths:
            return [Path(p) for p in self.contract_search_paths]
        return None

    def _load_contract(
        self,
        context: OutputContext,
        *,
        layer: str | None = None,
    ) -> DataContract | None:
        """Load the data contract for the current asset, if one exists."""
        if self.enforce_contracts == ContractEnforcementMode.SILENT:
            return None

        effective_layer = layer if layer is not None else self.layer
        asset_name = context.asset_key.to_user_string()
        search_paths = self._get_contract_search_paths()

        return load_contract_for_asset(
            asset_name=asset_name,
            layer=effective_layer,
            search_paths=search_paths,
        )

    # ------------------------------------------------------------------
    # Write path (delegates to PostgresResource)
    # ------------------------------------------------------------------

    def handle_output(
        self,
        context: OutputContext,
        obj: pl.DataFrame | BatchedDataFrame | None,
    ) -> None:
        """Write DataFrame or BatchedDataFrame to table using configured write mode.

        Accepts either:
        - pl.DataFrame: Single DataFrame written in one operation
        - BatchedDataFrame: Iterator of DataFrames written batch-by-batch for
          memory-efficient processing of large datasets

        Write mode and related settings can be overridden per-asset via metadata:
            @asset(metadata={"write_mode": "upsert", "primary_key": ["id"]})

        Supported metadata keys:
            - write_mode: "full_refresh", "upsert", "append", "scd2"
            - primary_key: Column(s) for upsert conflict detection
            - update_columns: Column(s) to update on upsert (default: all non-key)
            - partition_column: Column for partition-scoped writes
            - source_file: Source file path for lineage tracking
            - business_key: Column(s) for SCD2 business entity identification
            - tracked_columns: Column(s) to hash for SCD2 change detection
            - effective_from_col: SCD2 effective-from column name (default: effective_from)
            - effective_to_col: SCD2 effective-to column name (default: effective_to)
            - is_current_col: SCD2 current-flag column name (default: is_current)
            - hash_col: SCD2 row hash column name (default: row_hash)
            - detect_deletes: SCD2 flag to expire current records whose business
              key is absent from incoming data (default: false)
            - sequence_column: Column name for per-business-key version sequence
              (default: "seq_id"). Set to null to opt out.
            - analyze_after_write: Post-commit ANALYZE behavior for this asset
              ("partitioned" / "always" / "never"). Defaults to the
              resource-level setting.
        """
        if obj is None:
            context.log.warning("Received None, skipping write")
            return

        from moncpipelib.streaming import BatchedDataFrame

        if not isinstance(obj, (pl.DataFrame, BatchedDataFrame)):
            asset_name = context.asset_key.to_user_string()
            raise TypeError(
                f"PostgresIOManager.handle_output expected a Polars DataFrame or BatchedDataFrame, "
                f"but received {type(obj).__name__} from asset '{asset_name}'. "
                f"Ensure your asset returns a pl.DataFrame or BatchedDataFrame."
            )

        # IO-manager-specific: validate metadata keys for typo detection
        self._validate_metadata_keys(context)

        # Build write config from IO manager defaults + asset metadata
        write_config = self._get_write_config(context)

        # Extract source_file from metadata
        source_file: str | None = None
        if context.metadata:
            sf = context.metadata.get("source_file")
            if sf is not None:
                source_file = str(sf)

        # Load contract for target resolution (without full validation)
        layer_hint = self.layer or self.default_schema
        _preloaded_contract = self._load_contract(context, layer=layer_hint)
        pipeline_id = _preloaded_contract.pipeline_id if _preloaded_contract else None

        # Resolve target using IO manager's schema cascade
        target = self._resolve_target(context, contract=_preloaded_contract)

        # Create WriteContext from OutputContext
        wctx = WriteContext.from_output_context(context)

        # #365: bind the run_id so the delegated write connections carry it as
        # application_name (this path bypasses ``PostgresResource.write``, which
        # binds on the direct-resource path).
        bind_run_id(wctx.run_id)

        resource = self.postgres_resource

        # Under schema_override (test isolation) the resolved schema is not
        # the contract's logical schema, so skip sink-schema comparison (#405).
        reconcile_schema = None if self.schema_override else target.schema

        if isinstance(obj, BatchedDataFrame):
            result = resource._write_batched(
                batched=obj,
                table_name=target.table_name,
                schema=target.schema,
                bare_table=target.bare_table,
                layer=target.layer,
                wctx=wctx,
                write_config=write_config,
                loaded_contract=_preloaded_contract,
                source_file=source_file,
                pipeline_id=pipeline_id,
                target_schema=reconcile_schema,
            )
        else:
            result = resource._write_single(
                df=obj,
                table_name=target.table_name,
                schema=target.schema,
                bare_table=target.bare_table,
                layer=target.layer,
                wctx=wctx,
                write_config=write_config,
                loaded_contract=_preloaded_contract,
                source_file=source_file,
                pipeline_id=pipeline_id,
                target_schema=reconcile_schema,
            )

        # Convert WriteResult to Dagster output metadata
        context.add_output_metadata(result.to_dagster_metadata())

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def _check_pii_drift(self, context: InputContext) -> None:
        """Warn if upstream PII columns flow into a downstream asset without tracking.

        Compares upstream contract PII annotations against the downstream
        contract. Logs warnings for PII columns that exist in the downstream
        schema but are not marked as PII there.
        """
        if self.enforce_contracts == ContractEnforcementMode.SILENT:
            return
        if not context.upstream_output:
            return

        search_paths = self._get_contract_search_paths()

        upstream_key = context.upstream_output.asset_key
        if upstream_key is None:
            return
        upstream_name = upstream_key.to_user_string()

        upstream_contract = load_contract_for_asset(upstream_name, search_paths=search_paths)
        if upstream_contract is None:
            return

        upstream_pii = set(upstream_contract.get_pii_column_names())
        if not upstream_pii:
            return

        downstream_key = context.asset_key
        if downstream_key is None:
            return
        downstream_name = downstream_key.to_user_string()

        downstream_contract = load_contract_for_asset(downstream_name, search_paths=search_paths)
        if downstream_contract is None:
            context.log.warning(
                f"Upstream asset '{upstream_name}' has PII columns {sorted(upstream_pii)} "
                f"but downstream asset '{downstream_name}' has no data contract. "
                "PII may flow without being tracked."
            )
            return

        downstream_col_names = {c.name for c in downstream_contract.schema.columns}
        downstream_pii = set(downstream_contract.get_pii_column_names())

        drifted = upstream_pii & downstream_col_names - downstream_pii
        if drifted:
            context.log.warning(
                f"PII drift detected: columns {sorted(drifted)} are PII in upstream "
                f"'{upstream_name}' but NOT marked as PII in downstream "
                f"'{downstream_name}'. Review downstream contract PII annotations."
            )

    def load_input(self, context: InputContext) -> pl.DataFrame:
        """Read DataFrame from table with optional partition and column filtering.

        Supports optional column projection via upstream asset metadata to reduce
        data transfer and memory usage when only a subset of columns are needed.

        When a Dagster partition context is active and the upstream asset's
        metadata includes ``partition_column``, a WHERE clause is added to
        scope the read to the active partition(s).
        """
        # #365: bind the run_id so read connections (streaming engine /
        # raw connection) carry it as application_name for run-to-backend
        # correlation. ``InputContext.run_id`` is the active run.
        bind_run_id(getattr(context, "run_id", None))

        target = self._resolve_target(context)
        table_name = target.table_name

        try:
            self._check_pii_drift(context)
        except Exception as drift_err:
            context.log.warning(f"PII drift check failed: {drift_err}")

        # Check for column projection in upstream metadata
        columns: list[str] | None = None
        if context.upstream_output and context.upstream_output.metadata:
            cols_meta = context.upstream_output.metadata.get("columns")
            if cols_meta is not None:
                columns = [cols_meta] if isinstance(cols_meta, str) else list(cols_meta)

        # Build base query with optional column projection
        if columns:
            for col in columns:
                if not col.replace("_", "").isalnum():
                    raise ValueError(f"Invalid column name: {col}")
            columns_str = ", ".join(f'"{col}"' for col in columns)
            query = f"SELECT {columns_str} FROM {table_name}"  # noqa: S608
        else:
            query = f"SELECT * FROM {table_name}"  # noqa: S608

        # Partition filtering
        query_params: list[Any] = []
        partition_column: str | None = None
        if context.upstream_output and context.upstream_output.metadata:
            pc = context.upstream_output.metadata.get("partition_column")
            if pc is not None:
                partition_column = str(pc)

        partition_values = self._extract_partition_values(context)

        base_query = query

        if partition_column and partition_values:
            if not partition_column.replace("_", "").isalnum():
                raise ValueError(f"Invalid partition_column name: {partition_column}")
            placeholders = ", ".join(["%s"] * len(partition_values))
            query += f' WHERE "{partition_column}" IN ({placeholders})'
            query_params = list(partition_values)

        conn = self.postgres_resource.get_connection_raw()
        try:
            schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(conn, base_query)

            # Both branches now go through ``pl.read_database`` so the
            # parameterized path stops materializing each column twice
            # (once as tuples in ``cursor.fetchall()``, once as lists in
            # the dict-of-lists DataFrame constructor).  Migration 012
            # Phase C / #244.
            #
            # ``execute_options={"params": ...}`` matches psycopg3's
            # ``Cursor.execute(query, params=None)`` signature; polars
            # introspects the cursor and forwards options as kwargs.
            df = pl.read_database(
                query=query,
                connection=conn,
                schema_overrides=schema_overrides,
                infer_schema_length=0,
                execute_options=({"params": list(query_params)} if query_params else None),
            )

            partition_info = ""
            if partition_values and partition_column:
                partition_info = f" (partition: {partition_column} IN {partition_values})"
            cols_info = f" ({len(columns)} columns)" if columns else ""
            context.log.info(f"Loaded {len(df)} rows from {table_name}{cols_info}{partition_info}")

            return df
        finally:
            conn.close()
