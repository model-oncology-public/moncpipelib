"""Dagster asset check generation from data contracts.

This module provides functions to generate Dagster asset checks from
data contracts, enabling validation to run as part of asset materialization
pipelines with results visible in the Dagster UI.
"""

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import psycopg
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetChecksDefinition,
    AssetCheckSeverity,
    AssetCheckSpec,
    AssetKey,
    asset_check,
    multi_asset_check,
)

from moncpipelib.config import CONTRACT_FILE_PATTERN, SCD2_DEFAULTS
from moncpipelib.contracts.loader import load_contract
from moncpipelib.contracts.models import DataContract, Severity
from moncpipelib.contracts.validators import (
    run_column_test,
    run_table_expectation,
    validate_schema,
)
from moncpipelib.resources.postgres import PostgresPolarsSchema


def _severity_to_dagster(severity: Severity) -> AssetCheckSeverity:
    """Convert contract severity to Dagster AssetCheckSeverity."""
    if severity == Severity.ERROR:
        return AssetCheckSeverity.ERROR
    return AssetCheckSeverity.WARN


def _resolve_current_scope_col(contract: DataContract) -> str | None:
    """Return the current-flag column when the check sink is SCD2, else None.

    Checks run against the contract's first table sink (the same sink
    ``_resolve_check_table`` resolves). An SCD2 table legitimately repeats
    business keys across history rows, so unscoped full-table checks fail
    ``unique`` on the first change wave and degrade every other test type
    as history accrues (issue #418). Scoping to current rows restores
    snapshot semantics: checks validate the table as of now.

    Sinks cannot override SCD2 column names (``KNOWN_SINK_TABLE_FIELDS``
    in the loader has no such field), so the writer default from
    ``SCD2_DEFAULTS`` applies.
    """
    if contract.sinks:
        for s in contract.sinks:
            if s.get("type") == "table":
                if s.get("mode") == "scd2":
                    return str(SCD2_DEFAULTS["is_current_col"])
                return None
    return None


def _scope_df_to_current(df: pl.DataFrame, current_col: str) -> pl.DataFrame:
    """Filter a loaded table frame to SCD2 current rows.

    Raises instead of silently falling back to full-history checks: a
    missing current-flag column means the table was not written by the
    SCD2 writer the contract's ``mode: scd2`` sink declares.
    """
    if current_col not in df.columns:
        raise ValueError(
            f"Contract sink declares mode 'scd2' but the loaded table has no "
            f"'{current_col}' column; cannot scope checks to current rows."
        )
    return df.filter(pl.col(current_col))


def _wrap_df_loader_current_only(
    df_loader: Callable[[AssetCheckExecutionContext], pl.DataFrame],
    current_col: str,
) -> Callable[[AssetCheckExecutionContext], pl.DataFrame]:
    """Wrap a df_loader so every check sees only SCD2 current rows."""

    def scoped_loader(context: AssetCheckExecutionContext) -> pl.DataFrame:
        return _scope_df_to_current(df_loader(context), current_col)

    return scoped_loader


def generate_asset_check(
    contract: DataContract,
    asset_key: AssetKey | str | Sequence[str],
) -> AssetChecksDefinition:
    """Generate a metadata-only Dagster asset check from a data contract.

    .. deprecated::
        This function does not perform actual validation. Use
        ``make_contract_checks()`` or ``generate_asset_checks_from_contract()``
        for real validation as Dagster asset checks.

    Args:
        contract: The data contract to validate against
        asset_key: The Dagster asset key to associate with the check

    Returns:
        An AssetChecksDefinition that can be added to Dagster Definitions
    """
    warnings.warn(
        "generate_asset_check() is deprecated and does not perform validation. "
        "Use make_contract_checks() or generate_asset_checks_from_contract() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Normalize asset key
    resolved_asset_key: AssetKey
    if isinstance(asset_key, AssetKey):
        resolved_asset_key = asset_key
    elif isinstance(asset_key, str):
        resolved_asset_key = AssetKey([asset_key])
    else:
        resolved_asset_key = AssetKey(list(asset_key))

    check_name = f"{'_'.join(resolved_asset_key.path)}_contract_validation"

    @asset_check(
        asset=resolved_asset_key,
        name=check_name,
        description=f"Validates {contract.asset} against contract v{contract.version}",
    )
    def contract_check(_context: AssetCheckExecutionContext) -> AssetCheckResult:
        """Validate the asset against its data contract."""
        return AssetCheckResult(
            passed=True,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "contract_version": contract.version,
                "asset": contract.asset,
                "layer": contract.layer,
                "message": "Contract check executed - see IO Manager logs for validation details",
            },
        )

    return contract_check


def _generate_batched_checks(
    contract: DataContract,
    resolved_asset_key: AssetKey,
    df_loader: Callable[[AssetCheckExecutionContext], pl.DataFrame] | None = None,
    *,
    connection_factory: Callable[[], "psycopg.Connection"] | None = None,
    fq_table: str | None = None,
    op_tags: dict[str, Any] | None = None,
    check_prefix: str = "",
    current_col: str | None = None,
) -> AssetChecksDefinition:
    """Generate a single multi-check op for all contract rules.

    When ``connection_factory`` and ``fq_table`` are provided, checks run as
    SQL aggregation queries directly in PostgreSQL (no data loaded into
    Python). Schema validation uses ``LIMIT 0`` to probe column types.

    When only ``df_loader`` is provided, falls back to the Polars-based
    validation path (loads the full table into memory).

    ``current_col`` scopes column tests and table expectations to SCD2
    current rows (see ``_resolve_current_scope_col``).
    """

    use_sql = connection_factory is not None and fq_table is not None

    specs: list[AssetCheckSpec] = []

    # Schema check spec
    schema_check_name = f"{check_prefix}_schema"
    specs.append(
        AssetCheckSpec(
            name=schema_check_name,
            asset=resolved_asset_key,
            description="Validates DataFrame schema matches contract",
        )
    )

    # Column test specs
    col_check_meta: list[tuple[str, str, str, dict[str, Any], str | None, Severity, bool]] = []
    for col in contract.schema.columns:
        if col.managed:
            continue
        for test in col.tests:
            check_name = f"{check_prefix}_{col.name}_{test.test_type}"
            specs.append(
                AssetCheckSpec(
                    name=check_name,
                    asset=resolved_asset_key,
                    description=f"Validates {col.name} {test.test_type}",
                )
            )
            col_check_meta.append(
                (
                    check_name,
                    col.name,
                    test.test_type,
                    test.parameters,
                    test.when,
                    test.severity,
                    col.pii,
                )
            )

    # Table expectation specs
    exp_check_meta: list[tuple[str, str, dict[str, Any], Severity]] = []
    for exp in contract.expectations:
        column_suffix = f"_{exp.parameters['column']}" if "column" in exp.parameters else ""
        check_name = f"{check_prefix}_{exp.expectation_type}{column_suffix}"
        specs.append(
            AssetCheckSpec(
                name=check_name,
                asset=resolved_asset_key,
                description=f"Validates table {exp.expectation_type}",
            )
        )
        exp_check_meta.append(
            (
                check_name,
                exp.expectation_type,
                exp.parameters,
                exp.severity,
            )
        )

    # Capture in closure-safe locals
    _contract = contract
    _schema_check_name = schema_check_name
    _col_checks = col_check_meta
    _exp_checks = exp_check_meta
    _use_sql = use_sql
    _connection_factory = connection_factory
    _fq_table = fq_table
    _df_loader = df_loader
    _current_col = current_col

    # Unique op name per contract -- uses check_prefix (schema_table) to
    # avoid collisions when the same asset name exists in multiple schemas.
    op_name = f"{check_prefix}_contract_checks"

    @multi_asset_check(specs=specs, name=op_name, op_tags=op_tags or {})
    def batched_contract_checks(
        context: AssetCheckExecutionContext,
    ) -> Any:
        if _use_sql:
            assert _connection_factory is not None
            assert _fq_table is not None
            yield from _run_checks_sql(
                context,
                _connection_factory,
                _fq_table,
                _contract,
                _schema_check_name,
                _col_checks,
                _exp_checks,
                current_col=_current_col,
            )
        else:
            assert _df_loader is not None
            yield from _run_checks_polars(
                context,
                _df_loader,
                _contract,
                _schema_check_name,
                _col_checks,
                _exp_checks,
                current_col=_current_col,
            )

    return batched_contract_checks


def _run_checks_sql(
    _context: AssetCheckExecutionContext,
    connection_factory: Callable[[], "psycopg.Connection"],
    fq_table: str,
    contract: DataContract,
    schema_check_name: str,
    col_checks: list[tuple[str, str, str, dict[str, Any], str | None, Severity, bool]],
    exp_checks: list[tuple[str, str, dict[str, Any], Severity]],
    *,
    current_col: str | None = None,
) -> Any:
    """Execute all checks via SQL pushdown."""
    from moncpipelib.contracts.sql_checks import (
        run_column_test_sql,
        run_table_expectation_sql,
    )

    scope_metadata: dict[str, Any] = (
        {"scope": f"current rows only ({current_col} = TRUE)"} if current_col else {}
    )

    conn = connection_factory()
    try:
        # Schema check: LIMIT 0 probe (zero rows, column metadata only)
        PostgresPolarsSchema.register_uuid_adapter(conn)
        PostgresPolarsSchema.register_json_adapters(conn)
        probe_query = f"SELECT * FROM {fq_table} LIMIT 0"  # noqa: S608
        schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(conn, probe_query)
        probe_df = pl.read_database(
            query=probe_query,
            connection=conn,
            schema_overrides=schema_overrides,
            infer_schema_length=0,
        )
        schema_result = validate_schema(probe_df, contract)
        yield AssetCheckResult(
            check_name=schema_check_name,
            passed=schema_result.passed,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "message": schema_result.message,
                "column_count": len(contract.schema.columns),
                "mode": "sql_pushdown",
            },
        )

        # Column tests + table expectations: SQL queries
        with conn.cursor() as cursor:
            for name, column, test_type, params, when, severity, pii in col_checks:
                result = run_column_test_sql(
                    cursor,
                    fq_table,
                    column,
                    test_type,
                    params,
                    when=when,
                    pii=pii,
                    current_col=current_col,
                )
                yield AssetCheckResult(
                    check_name=name,
                    passed=result.passed,
                    severity=_severity_to_dagster(severity),
                    metadata={
                        "column": column,
                        "test_type": test_type,
                        "message": result.message,
                        "failed_count": result.failed_count,
                        "total_count": result.total_count,
                        "mode": "sql_pushdown",
                        **scope_metadata,
                    },
                )

            for name, exp_type, exp_params, severity in exp_checks:
                result = run_table_expectation_sql(
                    cursor,
                    fq_table,
                    exp_type,
                    exp_params,
                    current_col=current_col,
                )
                yield AssetCheckResult(
                    check_name=name,
                    passed=result.passed,
                    severity=_severity_to_dagster(severity),
                    metadata={
                        "expectation_type": exp_type,
                        "message": result.message,
                        "mode": "sql_pushdown",
                        **scope_metadata,
                    },
                )
    finally:
        conn.close()


def _run_checks_polars(
    context: AssetCheckExecutionContext,
    df_loader: Callable[[AssetCheckExecutionContext], pl.DataFrame],
    contract: DataContract,
    schema_check_name: str,
    col_checks: list[tuple[str, str, str, dict[str, Any], str | None, Severity, bool]],
    exp_checks: list[tuple[str, str, dict[str, Any], Severity]],
    *,
    current_col: str | None = None,
) -> Any:
    """Execute all checks via Polars (full DataFrame load). Fallback path."""
    df = df_loader(context)

    # Schema validation sees the unfiltered frame (columns/dtypes are
    # unaffected by row scoping); tests and expectations see current rows.
    schema_result = validate_schema(df, contract)
    yield AssetCheckResult(
        check_name=schema_check_name,
        passed=schema_result.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "message": schema_result.message,
            "column_count": len(contract.schema.columns),
        },
    )

    scope_metadata: dict[str, Any] = {}
    if current_col is not None:
        df = _scope_df_to_current(df, current_col)
        scope_metadata = {"scope": f"current rows only ({current_col} = TRUE)"}

    for name, column, test_type, params, when, severity, pii in col_checks:
        result = run_column_test(
            df=df,
            column=column,
            test_type=test_type,
            parameters=params,
            when=when,
            pii=pii,
        )
        yield AssetCheckResult(
            check_name=name,
            passed=result.passed,
            severity=_severity_to_dagster(severity),
            metadata={
                "column": column,
                "test_type": test_type,
                "message": result.message,
                "failed_count": result.failed_count,
                "total_count": result.total_count,
                **scope_metadata,
            },
        )

    for name, exp_type, exp_params, severity in exp_checks:
        result = run_table_expectation(
            df=df,
            expectation_type=exp_type,
            parameters=exp_params,
        )
        yield AssetCheckResult(
            check_name=name,
            passed=result.passed,
            severity=_severity_to_dagster(severity),
            metadata={
                "expectation_type": exp_type,
                "message": result.message,
                **scope_metadata,
            },
        )


def generate_asset_checks_from_contract(
    contract: DataContract,
    asset_key: "AssetKey | str | Sequence[str]",
    df_loader: Callable[[AssetCheckExecutionContext], pl.DataFrame] | None = None,
    *,
    batched: bool = True,
    connection_factory: Callable[[], "psycopg.Connection"] | None = None,
    fq_table: str | None = None,
    op_tags: dict[str, Any] | None = None,
) -> "list[AssetChecksDefinition]":
    """Generate Dagster asset checks for each contract rule.

    Creates asset checks for schema validation, each column test,
    and each table expectation.

    Args:
        contract: The data contract to validate against.
        asset_key: The Dagster asset key to associate with the checks.
        df_loader: A function that loads the DataFrame given a check context.
            Used as fallback when ``connection_factory`` is not provided.
        batched: When True (default), bundles all checks for this contract
            into a single ``@multi_asset_check`` op. When False, each check
            is a separate op for granular UI visibility.
        connection_factory: Callable returning a psycopg connection. When
            provided with ``fq_table``, checks run as SQL queries directly
            in PostgreSQL instead of loading data into Python.
        fq_table: Fully-qualified table name for SQL pushdown checks.
        op_tags: Dagster op tags for the generated check ops. Use to set
            k8s pod resource limits via ``dagster-k8s/config``.

    Returns:
        List of AssetChecksDefinitions. When ``batched=True``, the list
        contains a single element.
    """
    # Normalize asset key
    resolved_asset_key: AssetKey
    if isinstance(asset_key, AssetKey):
        resolved_asset_key = asset_key
    elif isinstance(asset_key, str):
        resolved_asset_key = AssetKey([asset_key])
    else:
        resolved_asset_key = AssetKey(list(asset_key))

    # Derive check name prefix. Use fq_table (schema.table) when available
    # for guaranteed uniqueness across schemas. Fall back to asset key path.
    check_prefix = fq_table.replace(".", "_") if fq_table else "_".join(resolved_asset_key.path)

    # SCD2 sinks: scope tests and expectations to current rows so history
    # rows don't fail snapshot-semantics checks like unique (issue #418).
    current_col = _resolve_current_scope_col(contract)

    if batched:
        return [
            _generate_batched_checks(
                contract,
                resolved_asset_key,
                df_loader,
                connection_factory=connection_factory,
                fq_table=fq_table,
                op_tags=op_tags,
                check_prefix=check_prefix,
                current_col=current_col,
            )
        ]

    # Unbatched path requires df_loader
    assert df_loader is not None, "df_loader is required when batched=False"

    if current_col is not None:
        df_loader = _wrap_df_loader_current_only(df_loader, current_col)
    checks: list[AssetChecksDefinition] = []

    # 1. Schema check
    @asset_check(
        asset=resolved_asset_key,
        name=f"{check_prefix}_schema",
        description="Validates DataFrame schema matches contract",
    )
    def schema_check(context: AssetCheckExecutionContext) -> AssetCheckResult:
        df = df_loader(context)
        result = validate_schema(df, contract)
        return AssetCheckResult(
            passed=result.passed,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "message": result.message,
                "column_count": len(contract.schema.columns),
            },
        )

    checks.append(schema_check)

    # 2. Column tests
    for col in contract.schema.columns:
        if col.managed:
            continue
        for test in col.tests:
            check_name = f"{check_prefix}_{col.name}_{test.test_type}"

            # Capture variables for closure
            def make_column_check(
                column_name: str,
                test_type: str,
                test_params: "dict[str, Any]",
                test_when: "str | None",
                test_severity: Severity,
                name: str,
                is_pii: bool,
            ) -> AssetChecksDefinition:
                @asset_check(
                    asset=resolved_asset_key,
                    name=name,
                    description=f"Validates {column_name} {test_type}",
                )
                def column_check(context: AssetCheckExecutionContext) -> AssetCheckResult:
                    df = df_loader(context)
                    result = run_column_test(
                        df=df,
                        column=column_name,
                        test_type=test_type,
                        parameters=test_params,
                        when=test_when,
                        pii=is_pii,
                    )
                    return AssetCheckResult(
                        passed=result.passed,
                        severity=_severity_to_dagster(test_severity),
                        metadata={
                            "column": column_name,
                            "test_type": test_type,
                            "message": result.message,
                            "failed_count": result.failed_count,
                            "total_count": result.total_count,
                        },
                    )

                return column_check

            checks.append(
                make_column_check(
                    col.name,
                    test.test_type,
                    test.parameters,
                    test.when,
                    test.severity,
                    check_name,
                    col.pii,
                )
            )

    # 3. Table expectations
    for exp in contract.expectations:
        # Include the column parameter in the name when present to avoid
        # collisions (e.g. two null_percentage checks on different columns).
        column_suffix = f"_{exp.parameters['column']}" if "column" in exp.parameters else ""
        check_name = f"{check_prefix}_{exp.expectation_type}{column_suffix}"

        def make_expectation_check(
            exp_type: str,
            exp_params: "dict[str, Any]",
            exp_severity: Severity,
            name: str,
        ) -> AssetChecksDefinition:
            @asset_check(
                asset=resolved_asset_key,
                name=name,
                description=f"Validates table {exp_type}",
            )
            def expectation_check(context: AssetCheckExecutionContext) -> AssetCheckResult:
                df = df_loader(context)
                result = run_table_expectation(
                    df=df,
                    expectation_type=exp_type,
                    parameters=exp_params,
                )
                return AssetCheckResult(
                    passed=result.passed,
                    severity=_severity_to_dagster(exp_severity),
                    metadata={
                        "expectation_type": exp_type,
                        "message": result.message,
                    },
                )

            return expectation_check

        checks.append(
            make_expectation_check(
                exp.expectation_type,
                exp.parameters,
                exp.severity,
                check_name,
            )
        )

    return checks


def load_contract_checks(
    contract_path: "str | Path",
    asset_key: "AssetKey | str | Sequence[str]",
    df_loader: Callable[[AssetCheckExecutionContext], pl.DataFrame],
) -> "list[AssetChecksDefinition]":
    """Load a contract from file and generate asset checks.

    Convenience function that combines load_contract and generate_asset_checks_from_contract.

    Args:
        contract_path: Path to the contract YAML file
        asset_key: The Dagster asset key to associate with the checks
        df_loader: A function that loads the DataFrame given a check context

    Returns:
        List of AssetChecksDefinitions for the contract
    """
    contract = load_contract(contract_path)
    return generate_asset_checks_from_contract(contract, asset_key, df_loader)


def discover_contract_checks(
    contracts_dir: "str | Path",
    df_loader_factory: Callable[[str], Callable[[AssetCheckExecutionContext], pl.DataFrame]],
    asset_key_prefix: "Sequence[str] | None" = None,
    *,
    batched: bool = True,
    op_tags: dict[str, Any] | None = None,
) -> "list[AssetChecksDefinition]":
    """Discover all contracts in a directory and generate checks.

    Scans a directory for *.contract.yaml files and generates asset checks
    for each contract found.

    By default each contract's checks attach to ``[sink_schema, sink_table]``
    derived from the contract's first table sink (falling back to the flat
    ``[asset]`` key when the contract has no sink schema), matching the
    ``[schema, table]`` convention used for assets persisted through
    ``PostgresIOManager``.

    Args:
        contracts_dir: Directory to scan for contract files
        df_loader_factory: Function that creates a df_loader given an asset name
        asset_key_prefix: Optional prefix for asset keys. When set, overrides
            sink-derived keys with ``[*prefix, asset]``.

    Returns:
        List of all AssetChecksDefinitions for discovered contracts

    Example:
        ```python
        from moncpipelib.contracts import discover_contract_checks

        def make_loader(asset_name):
            def loader(context):
                return pl.read_database(f"SELECT * FROM {asset_name}")
            return loader

        checks = discover_contract_checks(
            "contracts/",
            make_loader,
            asset_key_prefix=["bronze"],
        )

        defs = Definitions(asset_checks=checks)
        ```
    """
    contracts_path = Path(contracts_dir)
    all_checks: list[AssetChecksDefinition] = []

    for contract_file in contracts_path.rglob(CONTRACT_FILE_PATTERN):
        contract = load_contract(contract_file)
        asset_name = contract.asset

        asset_key = _derive_check_asset_key(contract, asset_key_prefix=asset_key_prefix)

        # Get df_loader for this asset
        df_loader = df_loader_factory(asset_name)

        # Generate checks
        checks = generate_asset_checks_from_contract(
            contract,
            asset_key,
            df_loader,
            batched=batched,
            op_tags=op_tags,
        )
        all_checks.extend(checks)

    return all_checks


def _derive_table_name(
    asset_name: str,
    *,
    db_schema: str,
    table_suffix_to_strip: str = "",
    table_prefix: str | None = None,
    schema_override: str | None = None,
) -> str:
    """Derive fully-qualified table name from an asset name.

    Applies the same naming logic as PostgresIOManager._get_table_name
    without requiring a Dagster context object.

    Args:
        asset_name: The asset name to derive the table name from.
        db_schema: Target database schema.
        table_suffix_to_strip: Suffix to remove from asset name.
        table_prefix: Prefix to prepend to table name.
        schema_override: Override schema (for testing).

    Returns:
        Fully-qualified table name (schema.table).
    """
    table_name = asset_name
    if table_suffix_to_strip and asset_name.endswith(table_suffix_to_strip):
        table_name = asset_name[: -len(table_suffix_to_strip)]
    if table_prefix:
        table_name = f"{table_prefix}{table_name}"
    schema = schema_override or db_schema
    return f"{schema}.{table_name}"


def _resolve_check_table(
    contract: DataContract,
    *,
    schema_override: str | None = None,
    default_schema: str | None = None,
    db_schema: str = "",
    table_suffix_to_strip: str = "",
    table_prefix: str | None = None,
) -> str:
    """Resolve the fully-qualified table name for a contract check.

    Reads the contract's first table sink for schema and table name,
    falling back to constructor-level defaults.  Mirrors the IO manager's
    ``_resolve_schema()`` priority cascade (minus context-dependent parts).

    Table name resolution:
        1. First table sink ``table`` field
        2. ``contract.asset`` (with ``table_suffix_to_strip`` applied)

    Schema resolution:
        1. ``schema_override`` (integration test isolation)
        2. First table sink ``schema`` field
        3. ``default_schema``
        4. ``db_schema`` (deprecated fallback)
        5. Raise ``ValueError``

    Args:
        contract: The data contract to resolve for.
        schema_override: Override schema (for testing).
        default_schema: Default schema from IO manager constructor.
        db_schema: Deprecated schema fallback.
        table_suffix_to_strip: Suffix to remove from asset name.
        table_prefix: Prefix to prepend to table name.

    Returns:
        Fully-qualified table name (schema.table).

    Raises:
        ValueError: When no schema can be resolved from any source.
    """
    # Find the first table sink (if any)
    sink: dict[str, Any] | None = None
    if contract.sinks:
        for s in contract.sinks:
            if s.get("type") == "table":
                sink = s
                break

    # Resolve table name
    if sink and sink.get("table"):
        table_name = str(sink["table"])
    else:
        table_name = contract.asset
        if table_suffix_to_strip and table_name.endswith(table_suffix_to_strip):
            table_name = table_name[: -len(table_suffix_to_strip)]

    if table_prefix:
        table_name = f"{table_prefix}{table_name}"

    # Resolve schema (priority cascade)
    if schema_override:
        schema = schema_override
    elif sink and sink.get("schema"):
        schema = str(sink["schema"])
    elif default_schema:
        schema = default_schema
    elif db_schema:
        schema = db_schema
    else:
        raise ValueError(
            f"No schema resolved for contract '{contract.asset}'. "
            "Set a sink schema in the contract, default_schema on the IO manager, "
            "or db_schema as a fallback."
        )

    return f"{schema}.{table_name}"


def _derive_check_asset_key(
    contract: DataContract,
    *,
    fq_table: str | None = None,
    asset_key_prefix: "Sequence[str] | None" = None,
) -> AssetKey:
    """Derive the asset key a contract's checks should attach to.

    Checks must target the exact key of a defined asset; any other key
    makes Dagster auto-create a stub ``AssetSpec`` in the ``default``
    group, and the check's status never surfaces on the real asset.
    Assets that persist through ``PostgresIOManager`` are keyed
    ``[schema, table]``, so the default here follows the resolved sink
    table rather than the bare contract asset name.

    Priority:
        1. ``asset_key_prefix`` -- explicit override, preserves the
           legacy ``[*prefix, contract.asset]`` shape.
        2. ``fq_table`` -- the resolved ``schema.table`` from
           ``_resolve_check_table()`` (reflects ``schema_override`` /
           ``default_schema`` routing), split into ``[schema, table]``.
        3. The contract's first table sink: ``[schema, table]``.
        4. Flat ``[contract.asset]`` when no sink schema exists.
    """
    if asset_key_prefix:
        return AssetKey(list(asset_key_prefix) + [contract.asset])

    if fq_table and "." in fq_table:
        schema, table_name = fq_table.split(".", 1)
        return AssetKey([schema, table_name])

    sink: dict[str, Any] | None = None
    if contract.sinks:
        for s in contract.sinks:
            if s.get("type") == "table":
                sink = s
                break
    if sink and sink.get("schema"):
        table_name = str(sink["table"]) if sink.get("table") else contract.asset
        return AssetKey([str(sink["schema"]), table_name])

    return AssetKey([contract.asset])


def _make_deferred_df_loader(
    *,
    connection_factory: Callable[[], "psycopg.Connection"],
    fq_table: str,
) -> Callable[[AssetCheckExecutionContext], pl.DataFrame]:
    """Create a df_loader that defers connection creation to execution time.

    Unlike ``_make_df_loader_factory`` which captures raw credentials in a
    closure (breaking Dagster's ``EnvVar`` pattern), this function captures
    a ``connection_factory`` callable.  The factory is invoked only when the
    check actually runs, by which time Dagster has resolved all ``EnvVar``
    values.

    Args:
        connection_factory: Callable that returns a new database connection.
            Typically ``PostgresIOManager._get_connection`` (a bound method).
        fq_table: Fully-qualified table name (schema.table).

    Returns:
        A df_loader function: (context) -> pl.DataFrame
    """

    def loader(_context: AssetCheckExecutionContext) -> pl.DataFrame:
        conn = connection_factory()
        try:
            PostgresPolarsSchema.register_uuid_adapter(conn)
            PostgresPolarsSchema.register_json_adapters(conn)
            query = f"SELECT * FROM {fq_table}"  # noqa: S608
            schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(conn, query)
            return pl.read_database(
                query=query,
                connection=conn,
                schema_overrides=schema_overrides,
                infer_schema_length=0,
            )
        finally:
            conn.close()

    return loader


def _make_df_loader_factory(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    db_schema: str,
    sslmode: str = "require",
    table_suffix_to_strip: str = "",
    table_prefix: str | None = None,
    schema_override: str | None = None,
) -> Callable[[str], Callable[[AssetCheckExecutionContext], pl.DataFrame]]:
    """Create a df_loader factory that reads tables from Postgres.

    Builds closures that capture connection config and table naming conventions.
    Each generated loader opens a fresh connection, reads the full table, and
    closes the connection.

    Args:
        host: Database server hostname.
        port: Database server port.
        user: Database username.
        password: Database password.
        database: Database name.
        db_schema: Target schema for tables.
        sslmode: SSL mode for connections.
        table_suffix_to_strip: Suffix to strip from asset names.
        table_prefix: Prefix to prepend to table names.
        schema_override: Override schema (for testing).

    Returns:
        A factory function: (asset_name) -> (context) -> pl.DataFrame
    """

    def factory(asset_name: str) -> Callable[[AssetCheckExecutionContext], pl.DataFrame]:
        fq_table = _derive_table_name(
            asset_name,
            db_schema=db_schema,
            table_suffix_to_strip=table_suffix_to_strip,
            table_prefix=table_prefix,
            schema_override=schema_override,
        )

        def loader(_context: AssetCheckExecutionContext) -> pl.DataFrame:
            conn = psycopg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=database,
                sslmode=sslmode,
            )
            try:
                PostgresPolarsSchema.register_uuid_adapter(conn)
                PostgresPolarsSchema.register_json_adapters(conn)
                query = f"SELECT * FROM {fq_table}"  # noqa: S608
                schema_overrides = PostgresPolarsSchema.from_psycopg2_connection(conn, query)
                return pl.read_database(
                    query=query,
                    connection=conn,
                    schema_overrides=schema_overrides,
                    infer_schema_length=0,
                )
            finally:
                conn.close()

        return loader

    return factory


def make_contract_checks(
    contracts_dir: str | Path,
    *,
    connection_factory: Callable[[], "psycopg.Connection"] | None = None,
    host: str = "",
    port: int = 5432,
    user: str = "",
    password: str = "",
    database: str = "",
    db_schema: str = "",
    default_schema: str | None = None,
    sslmode: str = "require",
    table_suffix_to_strip: str = "",
    table_prefix: str | None = None,
    schema_override: str | None = None,
    asset_key_prefix: Sequence[str] | None = None,
    batched: bool = True,
    op_tags: dict[str, Any] | None = None,
) -> list[AssetChecksDefinition]:
    """Generate Dagster asset checks from contracts using Postgres connection config.

    Recursively scans ``contracts_dir`` for ``*.contract.yaml`` files and
    generates asset checks for each contract.  Each contract's sink ``schema``
    field drives schema routing, so a single call can generate checks for
    contracts spanning multiple database schemas.

    Checks attach to ``[schema, table]`` derived from the same resolved
    sink table the checks run against, matching the key convention used
    for assets persisted through ``PostgresIOManager``.  Pass
    ``asset_key_prefix`` to override with ``[*prefix, asset]`` instead.

    Prefer ``PostgresIOManager.make_contract_checks()`` when an IO Manager
    instance is available -- it handles ``EnvVar`` resolution automatically.

    Args:
        contracts_dir: Directory to recursively scan for contract files.
        connection_factory: Callable returning a new ``psycopg`` connection.
            Preferred over individual credential arguments because it defers
            credential resolution to check execution time.
        host: Database server hostname (legacy; prefer ``connection_factory``).
        port: Database server port.
        user: Database username (legacy; prefer ``connection_factory``).
        password: Database password (legacy; prefer ``connection_factory``).
        database: Database name (legacy; prefer ``connection_factory``).
        db_schema: Fallback schema for contracts without a sink schema.
        default_schema: Default schema (takes priority over ``db_schema``).
        sslmode: SSL mode for connections.
        table_suffix_to_strip: Suffix to strip from asset names.
        table_prefix: Prefix to prepend to table names.
        schema_override: Override schema (for testing).
        asset_key_prefix: Optional prefix for asset keys (e.g., ["bronze"]).

    Returns:
        List of AssetChecksDefinitions ready for Dagster Definitions.

    Example:
        ```python
        from moncpipelib.contracts.checks import make_contract_checks

        # Preferred: connection_factory defers credential resolution
        checks = make_contract_checks(
            "defs/",
            connection_factory=lambda: psycopg.connect(
                host=os.environ["DB_HOST"],
                user=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
                dbname=os.environ["DB_NAME"],
            ),
            default_schema="silver",
        )

        # Legacy: individual credential arguments (resolved at call time)
        checks = make_contract_checks(
            "contracts/bronze/",
            host="db.example.com",
            user="reader",
            password=os.environ["DB_PASSWORD"],
            database="analytics",
            db_schema="bronze",
        )
        ```
    """
    if connection_factory is None:
        # Legacy path: wrap individual credentials in a factory
        _host, _port, _user = host, port, user
        _password, _database, _sslmode = password, database, sslmode

        def connection_factory() -> psycopg.Connection:
            conn: psycopg.Connection = psycopg.connect(
                host=_host,
                port=_port,
                user=_user,
                password=_password,
                dbname=_database,
                sslmode=_sslmode,
            )
            return conn

    contracts_path = Path(contracts_dir)
    all_checks: list[AssetChecksDefinition] = []

    for contract_file in contracts_path.rglob(CONTRACT_FILE_PATTERN):
        contract = load_contract(contract_file)

        fq_table = _resolve_check_table(
            contract,
            schema_override=schema_override,
            default_schema=default_schema,
            db_schema=db_schema,
            table_suffix_to_strip=table_suffix_to_strip,
            table_prefix=table_prefix,
        )

        df_loader = _make_deferred_df_loader(
            connection_factory=connection_factory,
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
            connection_factory=connection_factory,
            fq_table=fq_table,
            op_tags=op_tags,
        )
        all_checks.extend(checks)

    return all_checks
